import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN.")
    raise SystemExit(1)


CONF_BUCKETS = (
    ("low", 0.0, 0.4),
    ("mid", 0.4, 0.7),
    ("high", 0.7, 1.01),
)


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


def stable_observation_diff(curr_meas, prev_meas):
    raw_diff = curr_meas - prev_meas
    scale = torch.tensor([0.05, 0.05, 0.02, 0.05], device=curr_meas.device).view(1, -1)
    return torch.tanh(raw_diff / scale)


class FixedWindowTrackDataset(Dataset):
    def __init__(self, tracks_obs, tracks_gt, tracks_frame_ids, seq_len=20, step=2):
        self.samples = []
        for obs, gt, frame_ids in zip(tracks_obs, tracks_gt, tracks_frame_ids):
            track_len = obs.shape[0]
            if track_len < seq_len:
                continue
            last_start = track_len - seq_len
            for start in range(0, last_start + 1, step):
                end = start + seq_len
                self.samples.append((obs[start:end], gt[start:end], frame_ids[start:end]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def normalize_tracks(tracks_obs, tracks_gt, scale):
    norm_obs = []
    norm_gt = []
    for obs, gt in zip(tracks_obs, tracks_gt):
        obs_clone = obs.clone()
        gt_clone = gt.clone()
        obs_clone[:, :4] /= scale
        gt_clone /= scale
        norm_obs.append(obs_clone)
        norm_gt.append(gt_clone)
    return norm_obs, norm_gt


def split_tracks(tracks_obs, tracks_gt, tracks_frame_ids, val_split=0.1, seed=42):
    total_size = len(tracks_obs)
    val_size = max(1, int(total_size * val_split))
    if total_size - val_size < 1:
        val_size = max(0, total_size - 1)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=generator).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    if not train_indices:
        train_indices = val_indices[:1]
        val_indices = val_indices[1:]

    train_obs = [tracks_obs[i] for i in train_indices]
    train_gt = [tracks_gt[i] for i in train_indices]
    train_frame_ids = [tracks_frame_ids[i] for i in train_indices]

    val_obs = [tracks_obs[i] for i in val_indices]
    val_gt = [tracks_gt[i] for i in val_indices]
    val_frame_ids = [tracks_frame_ids[i] for i in val_indices]
    return train_obs, train_gt, train_frame_ids, val_obs, val_gt, val_frame_ids


def init_epoch_stats():
    return {
        "delta_k_abs_sum": 0.0,
        "delta_k_sq_sum": 0.0,
        "delta_k_count": 0,
        "gain_ratio_sum": 0.0,
        "gain_ratio_count": 0,
        "conf_bucket_loss": {name: 0.0 for name, _, _ in CONF_BUCKETS},
        "conf_bucket_count": {name: 0 for name, _, _ in CONF_BUCKETS},
        "occlusion_loss_sum": 0.0,
        "occlusion_count": 0,
    }


def update_epoch_stats(stats, delta_k, k_classic, loss_scalar, conf, occ_mask):
    delta_k_abs = delta_k.abs()
    stats["delta_k_abs_sum"] += float(delta_k_abs.sum().item())
    stats["delta_k_sq_sum"] += float(delta_k_abs.pow(2).sum().item())
    stats["delta_k_count"] += delta_k.numel()

    delta_k_norm = torch.linalg.norm(delta_k.reshape(delta_k.shape[0], -1), dim=1)
    k_classic_norm = torch.linalg.norm(k_classic.reshape(k_classic.shape[0], -1), dim=1).clamp_min(1e-6)
    gain_ratio = delta_k_norm / k_classic_norm
    stats["gain_ratio_sum"] += float(gain_ratio.sum().item())
    stats["gain_ratio_count"] += gain_ratio.numel()

    conf_flat = conf.view(-1)
    for name, low, high in CONF_BUCKETS:
        bucket_mask = (conf_flat >= low) & (conf_flat < high)
        bucket_count = int(bucket_mask.sum().item())
        if bucket_count > 0:
            stats["conf_bucket_loss"][name] += float(loss_scalar) * bucket_count
            stats["conf_bucket_count"][name] += bucket_count

    occ_count = int(occ_mask.sum().item())
    if occ_count > 0:
        stats["occlusion_loss_sum"] += float(loss_scalar) * occ_count
        stats["occlusion_count"] += occ_count


def format_epoch_stats(prefix, stats):
    delta_k_count = max(1, stats["delta_k_count"])
    gain_ratio_count = max(1, stats["gain_ratio_count"])
    delta_k_mean = stats["delta_k_abs_sum"] / delta_k_count
    delta_k_var = (stats["delta_k_sq_sum"] / delta_k_count) - (delta_k_mean ** 2)
    delta_k_std = max(delta_k_var, 0.0) ** 0.5
    gain_ratio_mean = stats["gain_ratio_sum"] / gain_ratio_count

    conf_parts = []
    for name, _, _ in CONF_BUCKETS:
        count = stats["conf_bucket_count"][name]
        avg_loss = stats["conf_bucket_loss"][name] / max(1, count)
        conf_parts.append(f"{name}:{avg_loss:.6f}({count})")

    occ_loss = stats["occlusion_loss_sum"] / max(1, stats["occlusion_count"])
    return (
        f"{prefix} delta_k_abs_mean={delta_k_mean:.6f} "
        f"delta_k_abs_std={delta_k_std:.6f} "
        f"gain_ratio={gain_ratio_mean:.6f} "
        f"conf_loss[{', '.join(conf_parts)}] "
        f"occlusion_loss={occ_loss:.6f}({stats['occlusion_count']})"
    )


def run_sequence_batch(
    model,
    batch_obs,
    batch_gt,
    batch_frame_ids,
    device,
    f_mat,
    h_mat,
    residual_gain_limit,
    criterion,
    epoch_stats,
    add_noise=False,
):
    b_obs = batch_obs.to(device)
    b_gt = batch_gt.to(device)
    b_frame_ids = batch_frame_ids.to(device)

    batch_n = b_obs.size(0)
    identity = torch.eye(8, device=device).unsqueeze(0).expand(batch_n, -1, -1)
    h_batch = h_mat.unsqueeze(0).expand(batch_n, -1, -1)
    f_batch = f_mat.unsqueeze(0).expand(batch_n, -1, -1)

    current_state = torch.zeros(batch_n, 8, device=device)
    current_state[:, :4] = b_obs[:, 0, :4]
    covariance = build_init_cov(b_obs[:, 0, :4]).to(device)
    hidden = None
    prev_z_meas = b_obs[:, 0, :4].clone()
    prev_update = torch.zeros(batch_n, 8, device=device)
    total_loss = 0.0

    seq_len = b_obs.size(1)
    for t in range(1, seq_len):
        pred_state = torch.matmul(current_state, f_mat.t())
        motion_cov = build_motion_cov(current_state).to(device)
        cov_pred = torch.matmul(
            torch.matmul(f_batch, covariance), f_mat.t().unsqueeze(0)
        )
        cov_pred = cov_pred + motion_cov

        z_meas = b_obs[:, t, :4]
        conf = b_obs[:, t, 4:5]

        if add_noise and torch.rand(1).item() < 0.15:
            conf = conf * 0.01
            z_meas = z_meas + torch.randn_like(z_meas) * 0.02

        pred_meas = torch.matmul(pred_state, h_mat.t())
        f1 = stable_observation_diff(z_meas, prev_z_meas)
        innovation = z_meas - pred_meas

        meas_cov = build_meas_cov(pred_state, conf).to(device)
        s_mat = torch.matmul(
            torch.matmul(h_batch, cov_pred), h_mat.t().unsqueeze(0)
        )
        s_mat = s_mat + meas_cov
        k_classic = classical_gain(cov_pred, h_mat, s_mat)

        net_input = torch.cat([f1, innovation, prev_update, conf], dim=1).unsqueeze(1)
        delta_k, hidden = model(net_input, hidden)
        gain_span = torch.clamp(k_classic.abs(), min=1e-3)

        conf_val_clamped = conf.clamp(0.1, 1.0).view(-1, 1, 1)
        dynamic_limit = residual_gain_limit - 0.20 * conf_val_clamped
        k_gain = k_classic + torch.tanh(delta_k) * (dynamic_limit * gain_span)

        update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
        current_state = pred_state + update_term
        prev_z_meas = z_meas
        prev_update = update_term

        innovation_factor = identity - torch.matmul(k_gain, h_batch)
        covariance = torch.matmul(
            torch.matmul(innovation_factor, cov_pred),
            innovation_factor.transpose(1, 2),
        )
        covariance = covariance + torch.matmul(
            torch.matmul(k_gain, meas_cov), k_gain.transpose(1, 2)
        )
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))

        loss_pos = criterion(current_state[:, :4], b_gt[:, t, :])
        gt_vel = b_gt[:, t, :] - b_gt[:, t - 1, :]
        loss_vel = criterion(current_state[:, 4:8], gt_vel)
        loss_reg = torch.mean(delta_k.pow(2))
        step_loss = loss_pos + 2.0 * loss_vel + 0.05 * loss_reg
        total_loss = total_loss + step_loss

        frame_gap = b_frame_ids[:, t] - b_frame_ids[:, t - 1]
        occ_mask = (frame_gap > 1) | (conf.view(-1) < 0.3)
        update_epoch_stats(epoch_stats, delta_k, k_classic, step_loss.item(), conf, occ_mask)

    return total_loss / (seq_len - 1)


def load_long_track_payload(data_file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = torch.load(data_file_path)

    if not isinstance(payload, dict) or payload.get("dataset_type") != "long_track":
        raise RuntimeError(
            "mot_train_data.pt is not a long-track dataset. Please rerun prepare_mot_data.py."
        )
    if payload.get("version", 0) < 3:
        raise RuntimeError(
            "mot_train_data.pt is an old long-track dataset without frame_ids. "
            "Please rerun prepare_mot_data.py to regenerate version 3."
        )
    return payload


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    short_batch_size = 32
    short_epochs = 40
    long_epochs = 20
    short_lr = 1e-3
    long_lr = 2e-4
    val_split = 0.1
    residual_gain_limit = 0.30
    data_file_path = "mot_train_data.pt"

    if not os.path.exists(data_file_path):
        print("Error: training data not found. Please run prepare_mot_data.py first.")
        return

    payload = load_long_track_payload(data_file_path)
    print(
        "INFO Found long-track dataset: "
        f"version={payload['version']} short_seq_len={payload['short_seq_len']} "
        f"short_seq_step={payload['short_seq_step']}"
    )

    scale = torch.tensor([1920, 1080, 1, 1080], dtype=torch.float32)
    tracks_obs = payload["long_tracks_input"]
    tracks_gt = payload["long_tracks_gt"]
    tracks_frame_ids = payload["long_tracks_frame_ids"]
    short_seq_len = payload["short_seq_len"]
    short_seq_step = payload["short_seq_step"]

    tracks_obs, tracks_gt = normalize_tracks(tracks_obs, tracks_gt, scale)
    (
        train_tracks_obs,
        train_tracks_gt,
        train_tracks_frame_ids,
        val_tracks_obs,
        val_tracks_gt,
        val_tracks_frame_ids,
    ) = split_tracks(tracks_obs, tracks_gt, tracks_frame_ids, val_split=val_split, seed=42)

    short_train_dataset = FixedWindowTrackDataset(
        train_tracks_obs, train_tracks_gt, train_tracks_frame_ids, seq_len=short_seq_len, step=short_seq_step
    )
    short_val_dataset = FixedWindowTrackDataset(
        val_tracks_obs, val_tracks_gt, val_tracks_frame_ids, seq_len=short_seq_len, step=short_seq_step
    )
    print(
        "Long-track split (by whole tracks): "
        f"train_tracks={len(train_tracks_obs)} val_tracks={len(val_tracks_obs)} | "
        f"train_windows={len(short_train_dataset)} val_windows={len(short_val_dataset)}"
    )

    short_train_loader = DataLoader(short_train_dataset, batch_size=short_batch_size, shuffle=True)
    short_val_loader = DataLoader(short_val_dataset, batch_size=short_batch_size, shuffle=False)

    model = KalmanNetNN(input_dim=17).to(device)
    criterion = nn.MSELoss()

    f_mat = torch.eye(8, device=device)
    for i in range(4):
        f_mat[i, 4 + i] = 1.0
    h_mat = torch.eye(4, 8, device=device)

    save_path = "pretrained/kalmannet_best.pth"
    os.makedirs("pretrained", exist_ok=True)
    best_val_loss = float("inf")

    stages = [
        ("Stage-1 Short BPTT", short_train_loader, short_val_loader, short_epochs, short_lr, True),
        ("Stage-2 Long-window Fine-tune", short_train_loader, short_val_loader, long_epochs, long_lr, False),
    ]

    for stage_name, train_loader, val_loader, epochs, lr, add_noise in stages:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        print(f"Start {stage_name} ...")

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            train_stats = init_epoch_stats()

            for b_obs, b_gt, b_frame_ids in train_loader:
                optimizer.zero_grad()
                batch_loss = run_sequence_batch(
                    model,
                    b_obs,
                    b_gt,
                    b_frame_ids,
                    device,
                    f_mat,
                    h_mat,
                    residual_gain_limit,
                    criterion,
                    train_stats,
                    add_noise=add_noise,
                )
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += batch_loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / max(1, len(train_loader))

            model.eval()
            total_val_loss = 0.0
            val_stats = init_epoch_stats()
            with torch.no_grad():
                for b_obs, b_gt, b_frame_ids in val_loader:
                    batch_loss = run_sequence_batch(
                        model,
                        b_obs,
                        b_gt,
                        b_frame_ids,
                        device,
                        f_mat,
                        h_mat,
                        residual_gain_limit,
                        criterion,
                        val_stats,
                        add_noise=False,
                    )
                    total_val_loss += batch_loss.item()

            avg_val_loss = total_val_loss / max(1, len(val_loader))
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

            current_lr = scheduler.get_last_lr()[0]
            best_flag = " <- best" if is_best else ""
            if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
                print(
                    f"{stage_name} Epoch [{epoch + 1}/{epochs}]  "
                    f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  "
                    f"LR: {current_lr:.6f}{best_flag}"
                )
                print(format_epoch_stats("  Train Monitor:", train_stats))
                print(format_epoch_stats("  Val Monitor:  ", val_stats))

    print(f"Training complete. Best val_loss={best_val_loss:.6f}, saved to {save_path}")


if __name__ == "__main__":
    train()
