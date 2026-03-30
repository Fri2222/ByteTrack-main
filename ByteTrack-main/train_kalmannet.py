import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import warnings
from torch.utils.data import DataLoader, TensorDataset

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN.")
    exit()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    BATCH_SIZE = 32
    EPOCHS = 60  # 提高到 60 轮充分收敛
    LR = 1e-3
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

    train_obs = train_obs.to(device)
    train_gt = train_gt.to(device)
    scale = scale.to(device)

    dataset = TensorDataset(train_obs, train_gt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 载入 13 维模型
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
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for b_obs, b_gt in dataloader:
            optimizer.zero_grad()

            current_state = torch.zeros(b_obs.size(0), 8).to(device)
            current_state[:, :4] = b_gt[:, 0, :]
            hidden = None
            batch_loss = 0

            # 只需要记录 F4 的前一帧记忆 (F1 已被抛弃)
            prev_update = torch.zeros(b_obs.size(0), 8).to(device)

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

                # 提取 13 维黄金特征组合
                f2 = innovation
                f4 = prev_update

                # 拼接: [F2(4) + F4(8) + Conf(1)] = 13
                net_input = torch.cat([f2, f4, conf], dim=1).unsqueeze(1)
                k_gain, hidden = model(net_input, hidden)

                innovation_expanded = innovation.unsqueeze(2)
                update_term = torch.bmm(k_gain, innovation_expanded).squeeze(2)
                current_state = pred_state + update_term

                # 更新 F4 记忆供下一帧循环使用
                prev_update = update_term

                # === [核心防线：双重物理 Loss，杜绝框乱飞] ===
                # 1. 位置监督
                gt_pos = b_gt[:, t, :]
                loss_pos = criterion(current_state[:, :4], gt_pos)

                # 2. 速度监督 (当前帧真实坐标 - 上一帧真实坐标)
                gt_vel = b_gt[:, t, :] - b_gt[:, t - 1, :]
                loss_vel = criterion(current_state[:, 4:8], gt_vel)

                # 速度权重放大 5 倍，严禁乱跑
                batch_loss += loss_pos + 5.0 * loss_vel

            batch_loss = batch_loss / (seq_len - 1)
            batch_loss.backward()

            # 梯度防爆
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()

        # 更新学习率
        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 2 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete. Model saved to {save_path}")


if __name__ == "__main__":
    train()