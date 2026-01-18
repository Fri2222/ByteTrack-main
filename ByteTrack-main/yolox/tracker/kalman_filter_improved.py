# yolox/tracker/kalman_filter_improved.py
import numpy as np
import scipy.linalg


class ImprovedKalmanFilter(object):
    """
    改进版卡尔曼滤波 (NSA-Kalman + Neural Interface)
    1. 支持 NSA (Noise Scale Adaptive): 根据检测置信度动态调整测量噪声。
    2. 支持 Neural Kalman Gain: 允许外部神经网络注入增益 K。
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # F: 状态转移矩阵 (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # H: 观测矩阵 (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 基础权重 (经验值)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

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

    def project(self, mean, covariance, confidence=None):
        """
        投影状态到观测空间。
        [改进点]: 增加 confidence 参数实现 NSA。
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # R: 原始测量噪声矩阵
        innovation_cov = np.diag(np.square(std))

        # === [改进核心: NSA 噪声自适应] ===
        if confidence is not None:
            # 逻辑: 置信度越低 -> scale_factor 越大 -> 噪声 R 越大 -> K 越小 (更信预测)
            # 这是一个简单的线性映射，你可以根据需要调整公式
            # 例如: confidence=0.9 -> factor=1.1; confidence=0.1 -> factor=1.9
            scale_factor = 2.0 - confidence

            # 限制范围，防止过度修正或负数
            scale_factor = np.clip(scale_factor, 1.0, 10.0)

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

    def update(self, mean, covariance, measurement, confidence=None, k_gain_override=None):
        """
        更新步骤。
        Args:
            confidence: YOLOX 检测置信度 (用于 NSA)
            k_gain_override: 外部神经网络预测的 Kalman Gain (用于 KalmanNet)
        """
        # 1. 投影 (包含 NSA 噪声调整)
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        # 2. 计算残差 (Innovation)
        innovation = measurement - projected_mean

        # 3. 计算卡尔曼增益 K
        if k_gain_override is not None:
            # === [改进接口: Neural Kalman] ===
            # 如果有神经网络算好的 K，直接用
            kalman_gain = k_gain_override
        else:
            # 标准计算: K = P H^T S^-1
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T

        # 4. 修正状态
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

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