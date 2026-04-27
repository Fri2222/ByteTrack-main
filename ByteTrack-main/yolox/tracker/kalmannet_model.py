import torch
import torch.nn as nn

# ==========================================
# 🎯 全局维度宏定义 (Macros)
# ==========================================
# 彻底移除 F1。只保留 F2(4维) + F4(8维) + conf(1维) = 13 维
KF_INPUT_DIM = 13
KF_STATE_DIM = 8  # 状态空间维度 [x, y, a, h, vx, vy, va, vh]
KF_OBS_DIM = 4  # 观测空间维度 [x, y, a, h]
KF_HIDDEN_DIM = 80  # 隐藏状态大小


class KalmanNetNN(nn.Module):
    """
    13维输入版 (F2+F4+Conf)：去除剧毒的 F1，保留新息残差 F2 与历史修正记忆 F4。
    fc_in 层使用 Tanh 激活：既保留残差正负符号（不截断负值），又引入非线性，
    与 KalmanNet 论文原始实现一致。ReLU 仅用在输出层（特征高度抽象后）。
    """

    def __init__(self, input_dim=KF_INPUT_DIM, state_dim=KF_STATE_DIM, obs_dim=KF_OBS_DIM, hidden_dim=KF_HIDDEN_DIM):
        super(KalmanNetNN, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.h1_dim = state_dim * state_dim  # 64
        self.h2_dim = state_dim * state_dim  # 64
        self.h3_dim = obs_dim * obs_dim  # 16

        # --- 特征放大器 (Linear层) ---
        self.fc_in1 = nn.Linear(input_dim, self.h1_dim)
        self.fc_in2 = nn.Linear(input_dim, self.h2_dim)
        self.fc_in3 = nn.Linear(input_dim, self.h3_dim)

        # --- 核心记忆网络 ---
        self.gru_q = nn.GRU(self.h1_dim, self.h1_dim, batch_first=True)
        self.gru_sigma = nn.GRU(self.h2_dim + self.h1_dim, self.h2_dim, batch_first=True)

        self.fc_sigma_to_s = nn.Linear(self.h2_dim, self.h3_dim)
        self.gru_s = nn.GRU(self.h3_dim + self.h3_dim, self.h3_dim, batch_first=True)

        # --- 增益输出层 ---
        self.fc_out = nn.Sequential(
            nn.Linear(self.h2_dim + self.h3_dim, hidden_dim * 2),
            nn.ReLU(),  # 只有特征高度抽象后才使用 ReLU
            nn.Linear(hidden_dim * 2, state_dim * obs_dim)
        )

    def forward(self, inputs, hidden_states=None):
        batch_size = inputs.size(0)

        if hidden_states is None:
            device = inputs.device
            h_q_0 = torch.zeros(1, batch_size, self.h1_dim, device=device)
            h_sigma_0 = torch.zeros(1, batch_size, self.h2_dim, device=device)
            h_s_0 = torch.zeros(1, batch_size, self.h3_dim, device=device)
        else:
            h_q_0, h_sigma_0, h_s_0 = hidden_states

        # [第一级] Tanh 放大信号：保留残差正负符号的同时引入非线性（论文原始实现）
        x1 = torch.tanh(self.fc_in1(inputs))
        out_q, h_q_n = self.gru_q(x1, h_q_0)

        # [第二级]
        x2 = torch.tanh(self.fc_in2(inputs))
        gru_sigma_input = torch.cat([x2, out_q], dim=-1)
        out_sigma, h_sigma_n = self.gru_sigma(gru_sigma_input, h_sigma_0)

        # [第三级]
        x3 = torch.tanh(self.fc_in3(inputs))
        sigma_mapped = self.fc_sigma_to_s(out_sigma)
        gru_s_input = torch.cat([x3, sigma_mapped], dim=-1)
        out_s, h_s_n = self.gru_s(gru_s_input, h_s_0)

        # --- 计算卡尔曼增益 K ---
        last_out_sigma = out_sigma[:, -1, :]
        last_out_s = out_s[:, -1, :]

        k_input = torch.cat([last_out_sigma, last_out_s], dim=-1)
        k_flat = self.fc_out(k_input)

        k_gain = k_flat.view(-1, self.state_dim, self.obs_dim)
        new_hidden_states = (h_q_n, h_sigma_n, h_s_n)

        return k_gain, new_hidden_states