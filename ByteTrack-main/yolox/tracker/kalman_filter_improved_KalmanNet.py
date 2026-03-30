import os
import warnings

import numpy as np
import scipy.linalg
import torch
from loguru import logger

try:
    from .kalmannet_model import KalmanNetNN
except ImportError:
    KalmanNetNN = None


class ImprovedKalmanFilter(object):
    def __init__(self, model_path="pretrained/kalmannet_best.pth"):
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self.use_neural_k = False
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_scale_4d = np.array([1920.0, 1080.0, 1.0, 1080.0], dtype=np.float32)
        self.image_scale_8d = np.array(
            [1920.0, 1080.0, 1.0, 1080.0, 1920.0, 1080.0, 1.0, 1080.0],
            dtype=np.float32,
        )
        self.residual_gain_limit = 0.30

        if KalmanNetNN is not None and os.path.exists(model_path):
            try:
                self.net = KalmanNetNN(input_dim=13).to(self.device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    checkpoint = torch.load(model_path, map_location=self.device)
                self.net.load_state_dict(checkpoint)
                self.net.eval()
                self.use_neural_k = True
                logger.info(f"[KalmanNet] ACTIVATED! Loaded from: {model_path}")
            except Exception as e:
                logger.error(f"[KalmanNet] Load Failed: {e}")
                self.use_neural_k = False

    def _get_measurement_noise(self, mean, confidence=None):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        if confidence is not None:
            conf = np.clip(confidence, 0.1, 0.99)
            innovation_cov *= (1.0 / conf)

        return innovation_cov

    def _compute_classical_gain(self, covariance, projected_cov):
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        return scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

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
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)
        ) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=None):
        innovation_cov = self._get_measurement_noise(mean, confidence)
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement, confidence=None, hidden_state=None):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        innovation = measurement - projected_mean
        classical_gain = self._compute_classical_gain(covariance, projected_cov)
        kalman_gain = classical_gain

        if hidden_state is None:
            gru_hidden = None
            f4 = np.zeros(8, dtype=np.float32)
        else:
            gru_hidden, _, prev_update_term = hidden_state
            f4 = prev_update_term.astype(np.float32, copy=False)

        new_gru_hidden = gru_hidden

        if self.use_neural_k:
            try:
                f2_norm = torch.tensor(
                    innovation / self.image_scale_4d, dtype=torch.float32
                ).view(1, 1, -1).to(self.device)
                f4_norm = torch.tensor(
                    f4 / self.image_scale_8d, dtype=torch.float32
                ).view(1, 1, -1).to(self.device)
                conf_val = confidence if confidence is not None else 1.0
                conf_tensor = torch.tensor(
                    [[[conf_val]]], dtype=torch.float32, device=self.device
                )
                net_input = torch.cat([f2_norm, f4_norm, conf_tensor], dim=-1)

                with torch.no_grad():
                    delta_k_tensor, new_gru_hidden = self.net(net_input, gru_hidden)
                    delta_k_norm = delta_k_tensor.squeeze(0).cpu().numpy()

                scale_matrix = self.image_scale_8d[:, None] / self.image_scale_4d[None, :]
                delta_k = delta_k_norm * scale_matrix
                gain_span = np.maximum(np.abs(classical_gain), 1e-3)
                gain_residual = np.tanh(delta_k) * (self.residual_gain_limit * gain_span)
                kalman_gain = classical_gain + gain_residual
            except Exception as e:
                logger.warning(f"[KalmanNet] Neural gain fallback to classical KF: {e}")
                kalman_gain = classical_gain

        if np.isnan(kalman_gain).any() or np.isinf(kalman_gain).any():
            kalman_gain = classical_gain

        update_term = np.dot(innovation, kalman_gain.T)
        new_mean = mean + update_term

        measurement_cov = self._get_measurement_noise(mean, confidence)
        identity = np.eye(covariance.shape[0], dtype=np.float32)
        innovation_factor = identity - np.dot(kalman_gain, self._update_mat)
        new_covariance = np.linalg.multi_dot(
            (innovation_factor, covariance, innovation_factor.T)
        )
        new_covariance += np.linalg.multi_dot(
            (kalman_gain, measurement_cov, kalman_gain.T)
        )
        new_covariance = 0.5 * (new_covariance + new_covariance.T)

        new_hidden_state_packaged = (new_gru_hidden, measurement, update_term)
        return new_mean, new_covariance, new_hidden_state_packaged

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric="maha"):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor,
                d.T,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            return np.sum(z * z, axis=0)
        raise ValueError("invalid distance metric")
