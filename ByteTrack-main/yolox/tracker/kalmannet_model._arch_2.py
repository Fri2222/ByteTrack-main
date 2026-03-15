import torch
import torch.nn as nn

# ==========================================
# 🎯 全局维度宏定义 (Macros)
# 保证 Train, Model, Track 三端绝对对齐！
# ==========================================
KF_INPUT_DIM = 5   # 4维物理残差 (x, y, a, h) + 1维检测置信度 (conf)
KF_STATE_DIM = 8   # 状态空间维度 [x, y, a, h, vx, vy, va, vh]
KF_OBS_DIM = 4     # 观测空间维度 [x, y, a, h]
KF_HIDDEN_DIM = (KF_STATE_DIM ** 2) + (KF_OBS_DIM ** 2)  # 论文设定: m^2 + n^2 = 80



class KalmanNetNN(nn.Module):
    """
    KalmanNet 架构 #2 (已重命名为 KalmanNetNN 以适配推理代码):
    物理启发的解耦跟踪，使用三个级联的 GRU 分别隐式跟踪 Q, Sigma, 和 S 矩阵。
    """

    def __init__(self, input_dim=KF_INPUT_DIM, state_dim=KF_STATE_DIM, obs_dim=KF_OBS_DIM, hidden_dim=KF_HIDDEN_DIM):
        super(KalmanNetNN, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # --- 维度定义 ---
        self.h1_dim = state_dim * state_dim  # GRU 1 跟踪 Q (m^2 = 64)
        self.h2_dim = state_dim * state_dim  # GRU 2 跟踪 Sigma (m^2 = 64)
        self.h3_dim = obs_dim * obs_dim  # GRU 3 跟踪 S (n^2 = 16)

        # --- 特征提取层 ---
        self.fc_in1 = nn.Linear(input_dim, self.h1_dim)
        self.fc_in2 = nn.Linear(input_dim, self.h2_dim)
        self.fc_in3 = nn.Linear(input_dim, self.h3_dim)

        # --- 核心记忆网络 ---
        # 1. 跟踪 Q 的 GRU
        self.gru_q = nn.GRU(self.h1_dim, self.h1_dim, batch_first=True)

        # 2. 跟踪 Sigma 的 GRU
        self.gru_sigma = nn.GRU(self.h2_dim + self.h1_dim, self.h2_dim, batch_first=True)

        # 3. 跟踪 S 的 GRU
        self.fc_sigma_to_s = nn.Linear(self.h2_dim, self.h3_dim)
        self.gru_s = nn.GRU(self.h3_dim + self.h3_dim, self.h3_dim, batch_first=True)

        # --- 增益输出层 ---
        # hidden_dim = 80 按照论文设置为hidden_dim的整数倍
        self.fc_out = nn.Sequential(
            nn.Linear(self.h2_dim + self.h3_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, state_dim * obs_dim)
        )

    def forward(self, inputs, hidden_states=None):
        batch_size = inputs.size(0)

        # --- 1. 隐状态初始化处理 ---
        if hidden_states is None:
            device = inputs.device
            h_q_0 = torch.zeros(1, batch_size, self.h1_dim, device=device)
            h_sigma_0 = torch.zeros(1, batch_size, self.h2_dim, device=device)
            h_s_0 = torch.zeros(1, batch_size, self.h3_dim, device=device)
        else:
            h_q_0, h_sigma_0, h_s_0 = hidden_states

        # --- 2. 级联前向传播 ---
        # [第一级] 估计 Q
        x1 = torch.relu(self.fc_in1(inputs))
        out_q, h_q_n = self.gru_q(x1, h_q_0)

        # [第二级] 估计 Sigma
        x2 = torch.relu(self.fc_in2(inputs))
        gru_sigma_input = torch.cat([x2, out_q], dim=-1)
        out_sigma, h_sigma_n = self.gru_sigma(gru_sigma_input, h_sigma_0)

        # [第三级] 估计 S
        x3 = torch.relu(self.fc_in3(inputs))
        sigma_mapped = torch.relu(self.fc_sigma_to_s(out_sigma))
        gru_s_input = torch.cat([x3, sigma_mapped], dim=-1)
        out_s, h_s_n = self.gru_s(gru_s_input, h_s_0)

        # --- 3. 计算最终的卡尔曼增益 K ---
        last_out_sigma = out_sigma[:, -1, :]  # [Batch, 64]
        last_out_s = out_s[:, -1, :]  # [Batch, 16]

        k_input = torch.cat([last_out_sigma, last_out_s], dim=-1)  # [Batch, 80]
        k_flat = self.fc_out(k_input)  # [Batch, 32]

        k_gain = k_flat.view(-1, self.state_dim, self.obs_dim)

        new_hidden_states = (h_q_n, h_sigma_n, h_s_n)

        return k_gain, new_hidden_states