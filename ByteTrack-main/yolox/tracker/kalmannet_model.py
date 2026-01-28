import torch
import torch.nn as nn


class KalmanNetNN(nn.Module):
    def __init__(self, input_dim=5, state_dim=8, hidden_dim=64):
        super(KalmanNetNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.input_dim = input_dim

        # 观测维度固定为 4 (x, y, a, h)
        self.obs_dim = 4

        # GRU 单元
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            # [关键修改]: 这里必须乘以 4 (self.obs_dim), 不能用 input_dim (5)!
            nn.Linear(128, state_dim * self.obs_dim)
        )

    def forward(self, inputs, hidden_state=None):
        # GRU 前向传播
        gru_out, new_hidden = self.gru(inputs, hidden_state)

        # 计算增益 K
        k_flat = self.fc(gru_out[:, -1, :])

        # [关键修改]: 重塑为 [Batch, 8, 4]
        k_gain = k_flat.view(-1, self.state_dim, self.obs_dim)

        return k_gain, new_hidden