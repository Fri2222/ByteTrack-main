# yolox/tracker/kalman_filter_improved.py
from loguru import logger  # <--- 添加这一行
import numpy as np
import scipy.linalg
import torch
import os

# 尝试导入 KalmanNet 模型定义
# 如果你还没有创建 kalmannet_model.py，请先创建它，否则这里会报错
try:
    from .kalmannet_model import KalmanNetNN
except ImportError:
    print("[Warning] kalmannet_model.py not found. Neural Kalman features disabled.")
    KalmanNetNN = None


class ImprovedKalmanFilter(object):
    """
    改进版卡尔曼滤波 (NSA-Kalman + Neural Interface)
    1. 支持 NSA (Noise Scale Adaptive): 根据检测置信度动态调整测量噪声 R。
    2. 支持 Neural Kalman Gain: 使用训练好的 GRU/LSTM 网络预测卡尔曼增益 K。
    """

    #状态空间 (State Space)：滤波器用 8 个数字来描述一个物体：[x, y, a, h, vx, vy, va, vh]。
    # x, y 是框的中心点坐标。
    # a 是框的宽高比（宽除以高），h 是框的高度。
    # vx, vy, va, vh 是它们对应的速度（变化率）
    def __init__(self, model_path="pretrained/kalmannet_best.pth"):
        ndim, dt = 4, 1.

        # F: 状态转移矩阵 (8x8),_motion_mat（运动矩阵）用来算预测
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # H: 观测矩阵 (4x8),_update_mat（观测矩阵）用来提取坐标
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 基础权重 (经验值)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # === 集成 KalmanNet (可选) ===
        self.use_neural_k = False
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # [调试 1] 确认一下这里的状态
        # logger.info(f"DEBUG: KalmanNetNN Class is: {KalmanNetNN}")
        # logger.info(f"DEBUG: Path exists? {os.path.exists(model_path)}")

        # 只有当定义了网络类且权重文件存在时，才启用
        if KalmanNetNN is not None and os.path.exists(model_path):
            try:
                self.net = KalmanNetNN().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.net.load_state_dict(checkpoint)
                self.net.eval()
                self.use_neural_k = True

                # [修改点] 使用 logger.info 并添加颜色标记
                logger.info(f"✅ [KalmanNet] ACTIVATED! Loaded from: {model_path}")
                logger.info(f"   [KalmanNet] Device: {self.device}")

            except Exception as e:
                logger.error(f"❌ [KalmanNet] Load Failed: {e}")
                logger.info("   [KalmanNet] Fallback to Standard NSA-Kalman Filter.")
                self.use_neural_k = False
        else:
            # 打印为什么没进来的原因
            if KalmanNetNN is None:
                logger.warning("⚠️ [KalmanNet] Class import failed (is None). Using Standard KF.")
            elif not os.path.exists(model_path):
                logger.warning(f"⚠️ [KalmanNet] Weight file not found: {model_path}. Using Standard KF.")
            pass

    #当第一次在视频里看到一个新目标时，调用这个函数
    #不知道它的速度，所以速度全部设为 0 (mean_vel = np.zeros_like(mean_pos))
    #给它初始化一个“不确定性矩阵”（covariance）来表示位置误差
    def initiate(self, measurement):
        """初始化轨迹"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    #推测下一帧物体在哪里
    def predict(self, mean, covariance):
        """预测步骤 (纯物理模型)"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]

        # Q: 过程噪声矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # x' = Fx
        mean = np.dot(mean, self._motion_mat.T)
        # P' = FPF^T + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    #confidence 是目标检测器 YOLOX给的置信度
    def project(self, mean, covariance, confidence=None):
        """
        投影状态到观测空间 (论文复现版)
        Reference: Eq(5) in Paper: R_tilde = (1 - c_k) * R
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # 基础 R 矩阵
        innovation_cov = np.diag(np.square(std))

        # === [改进核心: 论文版 NSA] ===
        if confidence is not None:
            # 论文策略：置信度越低，噪声 R 越大
            # 公式变种：R_new = R_base / confidence
            # (当 confidence=1.0, R不变; confidence=0.5, R变大2倍)
            # 或者使用论文的 (1 - ck) 逻辑的逆变种，这里推荐用除法更稳健

            # 限制 confidence 范围防止除零
            conf = np.clip(confidence, 0.1, 0.99)

            # 系数 factor：置信度高(0.9) -> factor小(1.1); 置信度低(0.1) -> factor大(10.0)
            #如果置信度很低（比如 0.2），scale_factor 就会变成 5,把它乘到测量噪声 innovation_cov 上
            scale_factor = 1.0 / conf

            # 放大噪声协方差
            innovation_cov *= scale_factor

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """向量化预测 (代码保持不变，用于加速)"""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, confidence=None, hidden_state=None):
        """
        更新步骤。
        Args:
            mean: 预测的状态均值
            covariance: 预测的协方差
            measurement: 当前观测值 (x, y, a, h)
            confidence: YOLOX 检测置信度 (用于 NSA)
            hidden_state: (可选) RNN/GRU 的隐状态，用于 KalmanNet
        Returns:
            new_mean, new_covariance
        """
        # 1. 投影 (包含 NSA 噪声调整)
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        # 2. 计算残差 (Innovation),预测值与观测值之间的差值
        innovation = measurement - projected_mean

        # 3. 计算卡尔曼增益 K
        kalman_gain = None

        # === [改进分支: Neural Kalman] ===
        #Neural Kalman 分支，将残差缩小到0~1范围内，再送进神经网络 self.net，得到一个K
        #传统分支，复杂的矩阵求逆 scipy.linalg.cho_solve来计算K
        if self.use_neural_k:
            try:
                # [关键修复]: 必须进行归一化！
                # MOT17 图片大小约为 1920x1080。这里我们用一个近似值进行缩放。
                # 输入是 (x, y, a, h)，对应缩放因子 (1920, 1080, 1, 1080)
                scale_tensor = torch.tensor([[[1920.0, 1080.0, 1.0, 1080.0]]], device=self.device)

                # 准备 Tensor 输入: [Batch=1, Seq=1, Feat=4]
                inno_tensor = torch.tensor(innovation, dtype=torch.float32).view(1, 1, -1).to(self.device)

                # 将输入缩小到 0~1 范围 (匹配训练时的分布)
                inno_tensor_norm = inno_tensor / scale_tensor

                with torch.no_grad():
                    # 神经网络前向传播
                    k_tensor, _ = self.net(inno_tensor_norm, hidden_state)

                    # 转回 Numpy: [8, 4]
                    kalman_gain = k_tensor.squeeze(0).cpu().numpy()

            except Exception as e:
                # print(f"Neural K Error: {e}") # 调试时可以打开
                kalman_gain = None

        # === [标准分支: 传统计算] ===
        # 如果没开启神经网络，或者神经网络失败，或者神经网络输出 NaN，使用标准公式
        # 增加 np.isnan 检查，防止网络输出 nan 导致崩溃
        if kalman_gain is None or np.isnan(kalman_gain).any():
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T

        # 4. 修正状态，用算好的 K 乘以残差，加到预测值上，得到最精准的当前状态
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    #匹配，马氏距离匈牙利匹配
    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """计算马氏距离 (用于匈牙利匹配)"""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')