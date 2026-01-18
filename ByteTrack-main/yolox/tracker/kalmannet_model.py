# yolox/tracker/kalmannet_model.py
import torch
import torch.nn as nn


class KalmanNetNN(nn.Module):
    def __init__(self, input_dim=4, state_dim=8, hidden_dim=64):
        """
        input_dim: 观测维度 (x, y, a, h) = 4
        state_dim: 状态维度 (x, y, a, h, vx, vy, va, vh) = 8
        """
        super(KalmanNetNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.input_dim = input_dim

        # GRU 单元：用于记忆历史轨迹特征
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # 全连接层：将 GRU 的隐状态映射为 卡尔曼增益 K
        # K 的形状应该是 (state_dim, input_dim)，即 8x4
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim * input_dim)  # 输出 32 个值
        )

    def forward(self, innovation, hidden_state=None):
        """
        innovation: 残差 (measurement - prediction), shape [Batch, 1, 4]
        """
        # GRU 前向传播
        gru_out, new_hidden = self.gru(innovation, hidden_state)

        # 计算增益 K
        k_flat = self.fc(gru_out[:, -1, :])  # 取最后一个时间步

        # 重塑为矩阵形式 [Batch, 8, 4]
        k_gain = k_flat.view(-1, self.state_dim, self.input_dim)

        return k_gain, new_hidden