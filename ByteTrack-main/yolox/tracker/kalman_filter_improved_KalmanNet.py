import numpy as np
import scipy.linalg
import torch
import os
from loguru import logger
import warnings

try:
    from .kalmannet_model import KalmanNetNN
except ImportError:
    KalmanNetNN = None


class ImprovedKalmanFilter(object):
    def __init__(self, model_path="pretrained/kalmannet_best.pth"):
        ndim, dt = 4, 1.

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.use_neural_k = False
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if KalmanNetNN is not None and os.path.exists(model_path):
            try:
                self.net = KalmanNetNN(input_dim=17).to(self.device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    checkpoint = torch.load(model_path, map_location=self.device)
                self.net.load_state_dict(checkpoint)
                self.net.eval()
                self.use_neural_k = True
                logger.info(f"✅ [KalmanNet] ACTIVATED! Loaded from: {model_path}")
            except Exception as e:
                logger.error(f"❌ [KalmanNet] Load Failed: {e}")
                self.use_neural_k = False

    def initiate(self, measurement):
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
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=None):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        if confidence is not None:
            conf = np.clip(confidence, 0.1, 0.99)
            scale_factor = 1.0 / conf
            innovation_cov *= scale_factor

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
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
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        innovation = measurement - projected_mean
        kalman_gain = None

        # [核心防线]：背包被抛弃时，F1和F4必须彻底斩断
        if hidden_state is None:
            gru_hidden = None
            f1 = np.zeros(4)
            f4 = np.zeros(8)
        else:
            gru_hidden, prev_measurement, prev_update_term = hidden_state
            f1 = measurement - prev_measurement
            f4 = prev_update_term

        new_gru_hidden = gru_hidden

        if self.use_neural_k:
            try:
                f2 = innovation
                scale_4d = torch.tensor([[[1920., 1080., 1., 1080.]]], device=self.device)
                scale_8d = torch.tensor([[[1920., 1080., 1., 1080., 1920., 1080., 1., 1080.]]], device=self.device)

                f1_norm = torch.tensor(f1, dtype=torch.float32).view(1, 1, -1).to(self.device) / scale_4d
                f2_norm = torch.tensor(f2, dtype=torch.float32).view(1, 1, -1).to(self.device) / scale_4d
                f4_norm = torch.tensor(f4, dtype=torch.float32).view(1, 1, -1).to(self.device) / scale_8d

                conf_val = confidence if confidence is not None else 1.0
                conf_tensor = torch.tensor([[[conf_val]]], dtype=torch.float32, device=self.device)

                net_input = torch.cat([f1_norm, f2_norm, f4_norm, conf_tensor], dim=-1)

                with torch.no_grad():
                    k_tensor, new_gru_hidden = self.net(net_input, gru_hidden)
                    kalman_gain_norm = k_tensor.squeeze(0).cpu().numpy()

                    scale_8d_np = np.array([1920., 1080., 1., 1080., 1920., 1080., 1., 1080.])
                    scale_4d_np = np.array([1920., 1080., 1., 1080.])
                    kalman_gain = kalman_gain_norm * (scale_8d_np[:, None] / scale_4d_np[None, :])

            except Exception as e:
                print(f"Neural K Error: {e}")
                kalman_gain = None

        if kalman_gain is None or np.isnan(kalman_gain).any():
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T

        update_term = np.dot(innovation, kalman_gain.T)
        new_mean = mean + update_term
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        # 打包完整的 17 维所需记忆返回
        new_hidden_state_packaged = (new_gru_hidden, measurement, update_term)
        return new_mean, new_covariance, new_hidden_state_packaged

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
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
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('invalid distance metric')