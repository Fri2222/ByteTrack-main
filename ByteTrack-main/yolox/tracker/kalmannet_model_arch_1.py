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
    KalmanNet 架构 #1: 单层大容量黑盒跟踪 (Single GRU)
    用一个大容量的 GRU 直接隐式拟合所有的卡尔曼滤波状态。
    """

    # hidden_dim = 80(即m ^ 2 + n ^ 2)
    def __init__(self, input_dim=KF_INPUT_DIM, state_dim=KF_STATE_DIM, obs_dim=KF_OBS_DIM, hidden_dim=KF_HIDDEN_DIM):
        super(KalmanNetNN, self).__init__()
        super(KalmanNetNN, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim


        self.hidden_dim = hidden_dim

        # 核心记忆网络：单个大容量 GRU 单元
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # 增益输出层：直接将 GRU 的隐藏特征映射为卡尔曼增益 K
        # hidden_dim的整数倍
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            # 第二层：把 160 维的中间特征，压缩成最终需要的卡尔曼增益大小
            # 这里的 state_dim = 8, self.obs_dim = 4
            # 所以这行等价于 nn.Linear(160, 32)
            nn.Linear(hidden_dim * 2, state_dim * obs_dim)
        )

    def forward(self, inputs, hidden_state=None):
        # GRU 前向传播
        # 如果 hidden_state 是 None，PyTorch 的 GRU 会极其省事地自动初始化为全零张量
        gru_out, new_hidden = self.gru(inputs, hidden_state)

        # 提取序列最后一个时间步的输出特征进行映射
        # gru_out shape: [Batch, Seq, hidden_dim] -> 取 [:, -1, :] 变成 [Batch, hidden_dim]
        last_out = gru_out[:, -1, :]

        # 计算增益 K 扁平化数据
        k_flat = self.fc(last_out)

        # 重塑为目标矩阵维度 [Batch, 8, 4]
        k_gain = k_flat.view(-1, self.state_dim, self.obs_dim)

        return k_gain, new_hidden