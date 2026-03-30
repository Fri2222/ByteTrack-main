import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN.")
    raise SystemExit(1)


def build_init_cov(measurement):
    h = measurement[:, 3].clamp_min(1e-3)
    std = torch.stack(
        [
            2.0 * (1.0 / 20.0) * h,
            2.0 * (1.0 / 20.0) * h,
            torch.full_like(h, 1e-2),
            2.0 * (1.0 / 20.0) * h,
            10.0 * (1.0 / 160.0) * h,
            10.0 * (1.0 / 160.0) * h,
            torch.full_like(h, 1e-5),
            10.0 * (1.0 / 160.0) * h,
        ],
        dim=1,
    )
    return torch.diag_embed(std.pow(2))


def build_motion_cov(state):
    h = state[:, 3].clamp_min(1e-3)
    std = torch.stack(
        [
            (1.0 / 20.0) * h,
            (1.0 / 20.0) * h,
            torch.full_like(h, 1e-2),
            (1.0 / 20.0) * h,
            (1.0 / 160.0) * h,
            (1.0 / 160.0) * h,
            torch.full_like(h, 1e-5),
            (1.0 / 160.0) * h,
        ],
        dim=1,
    )
    return torch.diag_embed(std.pow(2))


def build_meas_cov(pred_state, conf):
    h = pred_state[:, 3].clamp_min(1e-3)
    std = torch.stack(
        [
            (1.0 / 20.0) * h,
            (1.0 / 20.0) * h,
            torch.full_like(h, 1e-1),
            (1.0 / 20.0) * h,
        ],
        dim=1,
    )
    cov = torch.diag_embed(std.pow(2))
    conf = conf.clamp(0.1, 0.99).view(-1, 1, 1)
    return cov * (1.0 / conf)


def classical_gain(cov_pred, h_mat, s_mat):
    cross_cov = torch.matmul(cov_pred, h_mat.t())
    s_inv = torch.linalg.inv(s_mat)
    return torch.matmul(cross_cov, s_inv)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    batch_size = 32
    epochs = 60
    lr = 1e-3
    val_split = 0.1
    residual_gain_limit = 0.30
    data_file_path = "mot_train_data.pt"

    if os.path.exists(data_file_path):
        print(f"INFO Found real data file: {data_file_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_obs, train_gt = torch.load(data_file_path)
    else:
        print("Error: Training data not found.")
        return

    scale = torch.tensor([1920, 1080, 1, 1080], dtype=torch.float32)
    train_obs[:, :, :4] /= scale
    train_gt /= scale

    total_size = len(train_obs)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    full_dataset = TensorDataset(train_obs, train_gt)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train: {train_size} samples  |  Val: {val_size} samples")

    model = KalmanNetNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    f_mat = torch.eye(8, device=device)
    for i in range(4):
        f_mat[i, 4 + i] = 1.0
    h_mat = torch.eye(4, 8, device=device)

    print("Start Training KalmanNet (Residual K on top of classical KF) ...")

    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for b_obs, b_gt in dataloader:
            b_obs = b_obs.to(device)
            b_gt = b_gt.to(device)
            optimizer.zero_grad()

            batch_n = b_obs.size(0)
            identity = torch.eye(8, device=device).unsqueeze(0).expand(batch_n, -1, -1)
            h_batch = h_mat.unsqueeze(0).expand(batch_n, -1, -1)
            f_batch = f_mat.unsqueeze(0).expand(batch_n, -1, -1)

            current_state = torch.zeros(batch_n, 8, device=device)
            current_state[:, :4] = b_obs[:, 0, :4]
            covariance = build_init_cov(b_obs[:, 0, :4]).to(device)
            hidden = None
            prev_update = torch.zeros(batch_n, 8, device=device)
            batch_loss = 0.0

            seq_len = b_obs.size(1)
            for t in range(1, seq_len):
                pred_state = torch.matmul(current_state, f_mat.t())
                motion_cov = build_motion_cov(current_state).to(device)
                cov_pred = torch.matmul(torch.matmul(f_batch, covariance), f_mat.t().unsqueeze(0))
                cov_pred = cov_pred + motion_cov

                z_meas = b_obs[:, t, :4]
                conf = b_obs[:, t, 4:5]

                if torch.rand(1).item() < 0.15:
                    conf = conf * 0.01
                    z_meas = z_meas + torch.randn_like(z_meas) * 0.02

                pred_meas = torch.matmul(pred_state, h_mat.t())
                innovation = z_meas - pred_meas
                meas_cov = build_meas_cov(pred_state, conf).to(device)
                s_mat = torch.matmul(torch.matmul(h_batch, cov_pred), h_mat.t().unsqueeze(0))
                s_mat = s_mat + meas_cov
                k_classic = classical_gain(cov_pred, h_mat, s_mat)

                net_input = torch.cat([innovation, prev_update, conf], dim=1).unsqueeze(1)
                delta_k, hidden = model(net_input, hidden)
                gain_span = torch.clamp(k_classic.abs(), min=1e-3)
                k_gain = k_classic + torch.tanh(delta_k) * (residual_gain_limit * gain_span)

                update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
                current_state = pred_state + update_term
                prev_update = update_term

                innovation_factor = identity - torch.matmul(k_gain, h_batch)
                covariance = torch.matmul(torch.matmul(innovation_factor, cov_pred), innovation_factor.transpose(1, 2))
                covariance = covariance + torch.matmul(torch.matmul(k_gain, meas_cov), k_gain.transpose(1, 2))
                covariance = 0.5 * (covariance + covariance.transpose(1, 2))

                loss_pos = criterion(current_state[:, :4], b_gt[:, t, :])
                gt_vel = b_gt[:, t, :] - b_gt[:, t - 1, :]
                loss_vel = criterion(current_state[:, 4:8], gt_vel)
                batch_loss = batch_loss + loss_pos + 2.0 * loss_vel

            batch_loss = batch_loss / (seq_len - 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for b_obs, b_gt in val_dataloader:
                b_obs = b_obs.to(device)
                b_gt = b_gt.to(device)

                batch_n = b_obs.size(0)
                identity = torch.eye(8, device=device).unsqueeze(0).expand(batch_n, -1, -1)
                h_batch = h_mat.unsqueeze(0).expand(batch_n, -1, -1)
                f_batch = f_mat.unsqueeze(0).expand(batch_n, -1, -1)

                current_state = torch.zeros(batch_n, 8, device=device)
                current_state[:, :4] = b_obs[:, 0, :4]
                covariance = build_init_cov(b_obs[:, 0, :4]).to(device)
                hidden = None
                prev_update = torch.zeros(batch_n, 8, device=device)
                seq_len = b_obs.size(1)
                val_loss = 0.0

                for t in range(1, seq_len):
                    pred_state = torch.matmul(current_state, f_mat.t())
                    motion_cov = build_motion_cov(current_state).to(device)
                    cov_pred = torch.matmul(torch.matmul(f_batch, covariance), f_mat.t().unsqueeze(0))
                    cov_pred = cov_pred + motion_cov

                    z_meas = b_obs[:, t, :4]
                    conf = b_obs[:, t, 4:5]
                    pred_meas = torch.matmul(pred_state, h_mat.t())
                    innovation = z_meas - pred_meas
                    meas_cov = build_meas_cov(pred_state, conf).to(device)
                    s_mat = torch.matmul(torch.matmul(h_batch, cov_pred), h_mat.t().unsqueeze(0))
                    s_mat = s_mat + meas_cov
                    k_classic = classical_gain(cov_pred, h_mat, s_mat)

                    net_input = torch.cat([innovation, prev_update, conf], dim=1).unsqueeze(1)
                    delta_k, hidden = model(net_input, hidden)
                    gain_span = torch.clamp(k_classic.abs(), min=1e-3)
                    k_gain = k_classic + torch.tanh(delta_k) * (residual_gain_limit * gain_span)

                    update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
                    current_state = pred_state + update_term
                    prev_update = update_term

                    innovation_factor = identity - torch.matmul(k_gain, h_batch)
                    covariance = torch.matmul(torch.matmul(innovation_factor, cov_pred), innovation_factor.transpose(1, 2))
                    covariance = covariance + torch.matmul(torch.matmul(k_gain, meas_cov), k_gain.transpose(1, 2))
                    covariance = 0.5 * (covariance + covariance.transpose(1, 2))

                    loss_pos = criterion(current_state[:, :4], b_gt[:, t, :])
                    gt_vel = b_gt[:, t, :] - b_gt[:, t - 1, :]
                    loss_vel = criterion(current_state[:, 4:8], gt_vel)
                    val_loss += (loss_pos + 2.0 * loss_vel).item()

                total_val_loss += val_loss / (seq_len - 1)

        avg_val_loss = total_val_loss / len(val_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 2 == 0:
            current_lr = scheduler.get_last_lr()[0]
            best_flag = " <- best" if avg_val_loss == best_val_loss else ""
            print(
                f"Epoch [{epoch + 1}/{epochs}]  "
                f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  "
                f"LR: {current_lr:.6f}{best_flag}"
            )

    print(f"Training Complete. Best val_loss={best_val_loss:.6f}, saved to {save_path}")


if __name__ == "__main__":
    train()
