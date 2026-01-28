# train_kalmannet.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from yolox.tracker.kalmannet_model import KalmanNetNN
import os

def generate_synthetic_data(num_samples=1000, seq_len=50):
    """生成模拟轨迹数据 (模拟行人的匀速运动 + 随机噪声)"""
    # 状态: [x, y, a, h, vx, vy, va, vh]
    X = [] # 输入：观测值
    Y = [] # 标签：真实位置 (GT)
    
    for _ in range(num_samples):
        # 随机初始状态
        x0 = np.random.rand(8) * 100
        x0[2] = 0.5 # aspect ratio
        x0[3] = 100 # height
        # 随机速度
        x0[4:6] = np.random.randn(2) * 2 
        x0[6:] = 0
        
        traj_obs = []
        traj_gt = []
        
        curr_state = x0.copy()
        for t in range(seq_len):
            # 更新真实位置 (x = x + v)
            curr_state[:4] += curr_state[4:] 
            traj_gt.append(curr_state[:4].copy())
            
            # 生成观测值 (真实位置 + 噪声)
            noise = np.random.randn(4) * 5.0 # 假设噪声
            obs = curr_state[:4] + noise
            traj_obs.append(obs)
            
        X.append(traj_obs)
        Y.append(traj_gt)
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def train():
    model = KalmanNetNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 生成数据
# === 修改这里：加载真实数据 ===
    if os.path.exists('mot_train_data.pt'):
        print("Loading real MOT data...")
        train_obs, train_gt = torch.load('mot_train_data.pt')
    else:
        print("Real data not found, using synthetic...")
        train_obs, train_gt = generate_synthetic_data()
    # ==========================
    
    # 增加数据归一化！非常重要！
    # 因为 MOT 的坐标是 1000 多，直接进网络会梯度爆炸
    # 我们只训练“残差”，所以我们在送入网络前要手动 Normalize
    # 但最简单的办法是：把数据除以图片宽高 (1920, 1080) 归一化到 0~1
    
    # 简单归一化 (粗略值)
    scale = torch.tensor([1920, 1080, 1, 1080], dtype=torch.float32)
    train_obs /= scale
    train_gt /= scale
    
    print("Start Training KalmanNet...")
    # === 保存模型 ===
    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"KalmanNet weights saved to {save_path}")

if __name__ == "__main__":
    #使用MOT数据集的真实数据进行训练
    train()