import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset


from yolox.tracker.kalmannet_model import KalmanNetNN, KF_INPUT_DIM, KF_STATE_DIM, KF_OBS_DIM


try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetArch2. Make sure 'yolox/tracker/kalmannet_model.py' exists and contains the class.")
    exit()

# ... (generate_nonlinear_data 保持你的原样不变) ...

def train():
    # 1. 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3

    # 2. 准备数据
    data_file_path = 'mot_train_data.pt'

    if os.path.exists(data_file_path):
        print(f"INFO ✅| Found real data file: {data_file_path}")
        print("Loading real MOT data...")
        train_obs, train_gt = torch.load(data_file_path)
    else:
        # ... (数据生成保持你的原样不变) ...
        pass

    # 数据归一化
    scale = torch.tensor([1920, 1080, 1, 1080], dtype=torch.float32)
    train_obs[:, :, :4] /= scale
    train_gt /= scale

    train_obs = train_obs.to(device)
    train_gt = train_gt.to(device)
    scale = scale.to(device)

    dataset = TensorDataset(train_obs, train_gt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # [修复 2] 使用 KalmanNetArch2，并显式指定 input_dim=5
    # input_dim=5 (x,y,a,h,conf), state_dim=8, obs_dim=4
    model = KalmanNetNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    dt = 1.0
    F = torch.eye(8).to(device)
    for i in range(4):
        F[i, 4 + i] = dt

    H = torch.eye(4, 8).to(device)

    print("Start Training KalmanNet ...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for b_obs, b_gt in dataloader:
            optimizer.zero_grad()

            current_state = torch.zeros(b_obs.size(0), 8).to(device)
            current_state[:, :4] = b_gt[:, 0, :]

            # 初始隐状态设为 None，架构 #2 内部会自动生成 3 个 GRU 的零状态元组
            hidden = None
            batch_loss = 0

            # 👇 [新增]: 初始化 F1 和 F4 的“前一帧记忆”
            # 第一帧没有上一帧，所以设为初始坐标和 0 修正量
            prev_z_meas = b_obs[:, 0, :4]
            prev_update = torch.zeros(b_obs.size(0), 8).to(device)
            #  [新增]: 初始化 F1 和 F4 的“前一帧记忆”

            seq_len = b_obs.size(1)
            for t in range(1, seq_len):
                pred_state = torch.matmul(current_state, F.T)

                z_meas = b_obs[:, t, :4]
                conf = b_obs[:, t, 4:5]

                # F2: 新息差值 (原 innovation)
                pred_meas = torch.matmul(pred_state, H.T)
                innovation = z_meas - pred_meas

                # 👇 [新增]: 计算 F1 和 F4
                f1 = z_meas - prev_z_meas  # 观测差值
                f2 = innovation  # 新息差值
                f4 = prev_update  # 前向更新差值

                # 👇 [修改]: 拼接 17 维输入
                # [Batch, 4+4+8+1 = 17] -> [Batch, 1, 17]
                net_input = torch.cat([f1, f2, f4, conf], dim=1).unsqueeze(1)

                k_gain, hidden = model(net_input, hidden)

                innovation_expanded = innovation.unsqueeze(2)
                update_term = torch.bmm(k_gain, innovation_expanded).squeeze(2)
                current_state = pred_state + update_term

                # 👇 [新增]: 关键！更新记忆，供下一帧循环使用
                prev_z_meas = z_meas
                prev_update = update_term

                gt_pos = b_gt[:, t, :]
                batch_loss += criterion(current_state[:, :4], gt_pos)

            batch_loss = batch_loss / (seq_len - 1)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()