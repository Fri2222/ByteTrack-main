# train_kalmannet.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# 尝试导入模型，确保 kalmannet_model.py 存在
try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN. Make sure 'yolox/tracker/kalmannet_model.py' exists.")
    exit()


def generate_nonlinear_data(num_samples=2000, seq_len=40):
    """
    生成非线性轨迹数据 (模拟论文中的复杂运动场景)
    包括：正弦运动、急转弯、变速运动
    """
    X = []  # 总观测数据: [Batch, Seq, 5]
    Y = []  # 总GT数据:  [Batch, Seq, 4]

    print(f"Generating {num_samples} nonlinear trajectories...")

    for _ in range(num_samples):
        # 初始状态
        x, y = np.random.rand(2) * 1000
        vx, vy = np.random.randn(2) * 5

        traj_gt = []
        traj_obs = []

        # 随机选择运动模式
        mode = np.random.choice(['sin', 'turn', 'accelerate'])

        for t in range(seq_len):
            # 1. 更新真实位置 (非线性逻辑)
            if mode == 'sin':
                x += vx
                y += vy + 3.0 * np.sin(t / 5.0)  # 正弦波动
            elif mode == 'turn':
                # 缓慢转弯
                angle = 0.1 * t
                vx_rot = vx * np.cos(angle) - vy * np.sin(angle)
                vy_rot = vx * np.sin(angle) + vy * np.cos(angle)
                x += vx_rot
                y += vy_rot
            else:  # accelerate
                x += vx * (1 + t / 50.0)  # 加速
                y += vy * (1 + t / 50.0)

            # 构造 GT [x, y, a, h]
            gt_state = np.array([x, y, 0.5, 100.0])  # 简化 a, h 固定
            traj_gt.append(gt_state)

            # 2. 生成观测 (带置信度的噪声)
            # 随机生成置信度
            conf = np.random.rand() * 0.9 + 0.1  # 0.1~1.0

            # 论文逻辑：低分 -> 高噪
            noise_std = (1.0 / conf) * 5.0
            noise = np.random.randn(4) * noise_std

            obs = gt_state + noise

            # Input: [x, y, a, h, conf]
            obs_input = np.concatenate([obs, [conf]])
            traj_obs.append(obs_input)

        # [关键修复]: 将生成的这一条轨迹加入总列表
        X.append(np.array(traj_obs))
        Y.append(np.array(traj_gt))

    # 转换为 Tensor (Batch, Seq, Feature)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


def train():
    # 1. 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    BATCH_SIZE = 32
    EPOCHS = 20  # 训练轮数
    LR = 1e-3

    # 2. 准备数据 (逻辑修改：优先加载文件，否则生成)
    data_file_path = 'mot_train_data.pt'

    if os.path.exists(data_file_path):
        print(f"INFO ✅| Found real data file: {data_file_path}")
        print("Loading real MOT data...")
        # 加载数据 (我们之前的脚本保存的就是这个名字)
        train_obs, train_gt = torch.load(data_file_path)
    else:
        print(f"INFO ✅| File {data_file_path} not found.")
        print("Generating synthetic nonlinear data instead...")
        train_obs, train_gt = generate_nonlinear_data(num_samples=2000, seq_len=40)

    print(f"Data Shape - Obs: {train_obs.shape}, GT: {train_gt.shape}")

    # 数据归一化 (关键！)
    # 无论数据来源是真实的(坐标很大)还是生成的(坐标很大)，都需要归一化
    # 输入 Obs: [x, y, a, h, conf] -> 前4维需要归一化
    # GT: [x, y, a, h] -> 全部需要归一化
    scale = torch.tensor([1920, 1080, 1, 1080], dtype=torch.float32)

    # 注意：obs 的第5维是 confidence (0~1)，不需要除以 scale
    # 这里的切片操作 [:, :, :4] 现在对 3维 Tensor 是合法的
    train_obs[:, :, :4] /= scale
    train_gt /= scale

    # 移动到设备
    train_obs = train_obs.to(device)
    train_gt = train_gt.to(device)
    scale = scale.to(device)  # 训练中如需反归一化用到

    # 创建 DataLoader
    dataset = TensorDataset(train_obs, train_gt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化模型
    # input_dim=5 (x,y,a,h,conf), state_dim=8
    model = KalmanNetNN(input_dim=5, state_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 定义卡尔曼滤波的基础矩阵 (用于训练中的状态预测)
    # F: 状态转移矩阵 (8x8)
    dt = 1.0
    F = torch.eye(8).to(device)
    for i in range(4):
        F[i, 4 + i] = dt

    # H: 观测矩阵 (4x8)
    H = torch.eye(4, 8).to(device)

    print("Start Training KalmanNet...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for b_obs, b_gt in dataloader:
            # b_obs: [B, Seq, 5]
            # b_gt:  [B, Seq, 4]

            optimizer.zero_grad()

            # 初始化状态 (使用第一帧 GT 初始化，速度设为0)
            current_state = torch.zeros(b_obs.size(0), 8).to(device)
            current_state[:, :4] = b_gt[:, 0, :]

            hidden = None  # GRU 隐状态初始化
            batch_loss = 0

            # 时间步循环 (Time Loop)
            # 从第1帧开始预测 (第0帧用于初始化)
            seq_len = b_obs.size(1)
            for t in range(1, seq_len):
                # --- Kalman Predict ---
                # x_pred = F * x_prev
                pred_state = torch.matmul(current_state, F.T)

                # --- Prepare Neural Input ---
                # 获取当前观测和置信度
                z_meas = b_obs[:, t, :4]
                conf = b_obs[:, t, 4:5]

                # 计算残差 (Innovation): y = z - H * x_pred
                pred_meas = torch.matmul(pred_state, H.T)
                innovation = z_meas - pred_meas

                # 拼接输入: [Innovation, Confidence] -> [B, 1, 5]
                # 注意需要 unsqueeze 增加序列维度适应 GRU
                net_input = torch.cat([innovation, conf], dim=1).unsqueeze(1)

                # --- Neural Forward ---
                # 预测卡尔曼增益 K
                k_gain, hidden = model(net_input, hidden)
                # k_gain shape: [B, 8, 4]

                # --- Kalman Update ---
                # x_new = x_pred + K * y
                # 手动实现批量矩阵乘法: K [B,8,4] * y [B,4,1]
                innovation_expanded = innovation.unsqueeze(2)  # [B, 4, 1]
                update_term = torch.bmm(k_gain, innovation_expanded).squeeze(2)  # [B, 8]

                current_state = pred_state + update_term

                # --- Calculate Loss ---
                # 监督信号：当前估计位置 vs 真实位置
                gt_pos = b_gt[:, t, :]
                batch_loss += criterion(current_state[:, :4], gt_pos)

            # 反向传播 (Backprop through time)
            # 平均每个时间步的 Loss
            batch_loss = batch_loss / (seq_len - 1)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # 4. 保存模型
    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete. Model saved to {save_path}")


if __name__ == "__main__":
    train()