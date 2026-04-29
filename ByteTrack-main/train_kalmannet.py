import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import warnings
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN.")
    exit()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 1e-3
    VAL_SPLIT = 0.1   # 10% 用于验证，保存最优权重
    data_file_path = 'mot_train_data.pt'

    if os.path.exists(data_file_path):
        print(f"INFO ✅| Found real data file: {data_file_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_obs, train_gt = torch.load(data_file_path)
    else:
        print("Error: Training data not found.")
        return

    # 数据归一化
    scale = torch.tensor([1920, 1080, 1, 1080], dtype=torch.float32)
    train_obs[:, :, :4] /= scale
    train_gt /= scale

    scale = scale.to(device)

    # 划分训练集 / 验证集（固定随机种子保证可复现）
    total_size = len(train_obs)
    val_size   = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    full_dataset = TensorDataset(train_obs, train_gt)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    dataloader     = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {train_size} samples  |  Val: {val_size} samples")

    model = KalmanNetNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    dt = 1.0
    F = torch.eye(8).to(device)
    for i in range(4):
        F[i, 4 + i] = dt
    H = torch.eye(4, 8).to(device)

    print("Start Training KalmanNet (13D: F2+F4+Conf + Velocity Loss) ...")

    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    best_val_loss = float('inf')   # 用验证集损失决定保存哪一轮权重

    for epoch in range(EPOCHS):
        # -------- 训练阶段 --------
        model.train()
        total_loss = 0
        for b_obs, b_gt in dataloader:
            b_obs = b_obs.to(device)
            b_gt  = b_gt.to(device)
            optimizer.zero_grad()

            current_state = torch.zeros(b_obs.size(0), 8, device=device)
            current_state[:, :4] = b_gt[:, 0, :]
            hidden = None
            batch_loss = 0

            prev_update = torch.zeros(b_obs.size(0), 8, device=device)

            seq_len = b_obs.size(1)
            for t in range(1, seq_len):
                pred_state = torch.matmul(current_state, F.T)

                z_meas = b_obs[:, t, :4]
                conf = b_obs[:, t, 4:5]

                # 适度的数据增强，强迫模型学会根据低 Conf 忽略突变误差
                if model.training and torch.rand(1).item() < 0.15:
                    conf = conf * 0.01
                    z_meas = z_meas + torch.randn_like(z_meas) * 0.02

                pred_meas = torch.matmul(pred_state, H.T)
                innovation = z_meas - pred_meas

                f2 = innovation
                f4 = prev_update

                net_input = torch.cat([f2, f4, conf], dim=1).unsqueeze(1)
                k_gain, hidden = model(net_input, hidden)

                update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
                current_state = pred_state + update_term
                prev_update = update_term

                # 位置监督
                loss_pos = criterion(current_state[:, :4], b_gt[:, t, :])
                # 速度监督（权重从 5.0 降至 2.0，避免速度项过度主导梯度）
                gt_vel = b_gt[:, t, :] - b_gt[:, t - 1, :]
                loss_vel = criterion(current_state[:, 4:8], gt_vel)
                batch_loss += loss_pos + 2.0 * loss_vel

            batch_loss = batch_loss / (seq_len - 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(dataloader)

        # -------- 验证阶段 --------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for b_obs, b_gt in val_dataloader:
                b_obs = b_obs.to(device)
                b_gt  = b_gt.to(device)

                current_state = torch.zeros(b_obs.size(0), 8, device=device)
                current_state[:, :4] = b_gt[:, 0, :]
                hidden      = None
                prev_update = torch.zeros(b_obs.size(0), 8, device=device)
                seq_len     = b_obs.size(1)
                val_loss    = 0

                for t in range(1, seq_len):
                    pred_state = torch.matmul(current_state, F.T)
                    z_meas     = b_obs[:, t, :4]
                    conf       = b_obs[:, t, 4:5]
                    pred_meas  = torch.matmul(pred_state, H.T)
                    innovation = z_meas - pred_meas

                    net_input = torch.cat([innovation, prev_update, conf], dim=1).unsqueeze(1)
                    k_gain, hidden = model(net_input, hidden)

                    update_term   = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
                    current_state = pred_state + update_term
                    prev_update   = update_term

                    val_loss += criterion(current_state[:, :4], b_gt[:, t, :]).item()

                total_val_loss += val_loss / (seq_len - 1)

        avg_val_loss = total_val_loss / len(val_dataloader)

        # 保存验证集损失最优的权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 2 == 0:
            current_lr = scheduler.get_last_lr()[0]
            best_flag  = " ← best" if avg_val_loss == best_val_loss else ""
            print(f"Epoch [{epoch + 1}/{EPOCHS}]  "
                  f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  "
                  f"LR: {current_lr:.6f}{best_flag}")

    print(f"✅ Training Complete. Best val_loss={best_val_loss:.6f}, saved to {save_path}")


if __name__ == "__main__":
    train()