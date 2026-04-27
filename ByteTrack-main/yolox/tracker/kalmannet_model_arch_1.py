import torch
import torch.nn as nn

# ==========================================
# 全局维度宏定义
# F2(4维) + F4(8维) + conf(1维) = 13 维
# ==========================================
KF_INPUT_DIM  = 13
KF_STATE_DIM  = 8   # [x, y, a, h, vx, vy, va, vh]
KF_OBS_DIM    = 4   # [x, y, a, h]
KF_HIDDEN_DIM = (KF_STATE_DIM ** 2) + (KF_OBS_DIM ** 2)  # m²+n² = 80


class KalmanNetNN(nn.Module):
    """
    KalmanNet 架构一 (Architecture #1, 论文 Fig. 3)

    使用单个 GRU 联合隐式追踪所有二阶统计矩（Q、Sigma、S），
    通过 FC 输入层将特征映射到 GRU 输入空间，
    再由 FC 输出层将 GRU 隐状态映射为卡尔曼增益 K。

    输入特征 (13 维):
        F2  (4-d): 新息差值  y_t - y_hat_{t|t-1}
        F4  (8-d): 前向更新差值  x_hat_{t|t} - x_hat_{t|t-1}
        conf(1-d): 检测置信度

    fc_in  使用 Tanh: 保留残差正负符号的同时引入非线性。
    fc_out 中间层使用 ReLU: 仅在特征高度抽象后使用。
    隐状态格式: 单个张量 h_n，shape = (1, B, gru_hidden_dim)
    """

    def __init__(self,
                 input_dim:  int = KF_INPUT_DIM,
                 state_dim:  int = KF_STATE_DIM,
                 obs_dim:    int = KF_OBS_DIM,
                 hidden_dim: int = KF_HIDDEN_DIM):
        super(KalmanNetNN, self).__init__()

        self.state_dim      = state_dim
        self.obs_dim        = obs_dim
        self.gru_hidden_dim = hidden_dim  # 80 = m^2 + n^2

        # FC 输入层: 13 → gru_hidden_dim（Tanh 保留符号 + 非线性）
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, self.gru_hidden_dim),
            nn.Tanh()
        )

        # 核心单 GRU（联合追踪所有二阶矩）
        self.gru = nn.GRU(self.gru_hidden_dim, self.gru_hidden_dim,
                          batch_first=True)

        # FC 输出层: gru_hidden_dim → K (state_dim x obs_dim = 32)
        self.fc_out = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, hidden_dim * 2),  # 80 → 160
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, state_dim * obs_dim)   # 160 → 32
        )

    def forward(self, inputs, hidden_states=None):
        """
        inputs:        (B, 1, 13)
        hidden_states: None 或 (1, B, gru_hidden_dim)
        returns:
            k_gain:     (B, state_dim, obs_dim) = (B, 8, 4)
            h_n:        (1, B, gru_hidden_dim)
        """
        batch_size = inputs.size(0)

        # 隐状态初始化
        if hidden_states is None:
            h_0 = torch.zeros(1, batch_size, self.gru_hidden_dim,
                              device=inputs.device)
        else:
            h_0 = hidden_states

        # FC 输入层: 特征映射 + Tanh
        x = self.fc_in(inputs)          # (B, 1, gru_hidden_dim)

        # GRU 前向
        out, h_n = self.gru(x, h_0)    # out: (B, 1, gru_hidden_dim)

        # FC 输出层: 隐状态 → 卡尔曼增益
        last_out = out[:, -1, :]        # (B, gru_hidden_dim)
        k_flat   = self.fc_out(last_out)  # (B, state_dim * obs_dim)
        k_gain   = k_flat.view(-1, self.state_dim, self.obs_dim)  # (B, 8, 4)

        return k_gain, h_n