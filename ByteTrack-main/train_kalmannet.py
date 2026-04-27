import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN, KF_INPUT_DIM, KF_STATE_DIM, KF_OBS_DIM
except ImportError:
    print("Error: Could not import KalmanNetNN. Make sure 'yolox/tracker/kalmannet_model.py' exists.")
    raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train KalmanNet (Exp10 modified with Exp21 Training Strategy).")
    parser.add_argument("--short-epochs", type=int, default=40)
    parser.add_argument("--long-epochs", type=int, default=20)
    parser.add_argument("--short-lr", type=float, default=1e-3)
    parser.add_argument("--long-lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--save-path", default="pretrained/kalmannet_best.pth")
    parser.add_argument("--data-file", default="mot_train_data.pt")
    # Exp21 的困难样本采样权重参数
    parser.add_argument("--hard-gap-weight", type=float, default=2.5)
    parser.add_argument("--hard-lowconf-weight", type=float, default=2.0)
    parser.add_argument("--lowconf-thresh", type=float, default=0.3)
    parser.add_argument("--lowconf-resample-boost", type=float, default=2.0)
    parser.add_argument("--gap-resample-boost", type=float, default=2.5)
    return parser.parse_args()


# ================= 数据集与策略部分 (引入自 Exp21) =================

class FixedWindowTrackDataset(Dataset):
    def __init__(
            self,
            tracks_obs,
            tracks_gt,
            tracks_frame_ids,
            seq_len=20,
            step=2,
            lowconf_thresh=0.3,
            lowconf_resample_boost=2.0,
            gap_resample_boost=2.5,
    ):
        self.samples = []
        self.sample_weights = []
        for obs, gt, frame_ids in zip(tracks_obs, tracks_gt, tracks_frame_ids):
            track_len = obs.shape[0]
            if track_len < seq_len:
                continue
            last_start = track_len - seq_len
            for start in range(0, last_start + 1, step):
                end = start + seq_len
                obs_window = obs[start:end]
                gt_window = gt[start:end]
                frame_window = frame_ids[start:end]
                self.samples.append((obs_window, gt_window, frame_window))

                # 为困难样本分配更高的采样权重
                frame_gap = frame_window[1:] - frame_window[:-1]
                has_gap = bool(torch.any(frame_gap > 1))
                lowconf_count = int(torch.sum(obs_window[:, 4] < lowconf_thresh).item())
                weight = 1.0
                if has_gap:
                    weight += gap_resample_boost
                if lowconf_count > 0:
                    weight += lowconf_resample_boost * (lowconf_count / max(1, seq_len))
                self.sample_weights.append(weight)

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


def load_long_track_payload(data_file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = torch.load(data_file_path)

    if not isinstance(payload, dict) or payload.get("dataset_type") != "long_track":
        raise RuntimeError(
            "mot_train_data.pt is not a long-track dataset. Please run Exp21's prepare_mot_data.py first."
        )
    return payload


def compute_step_weights(frame_gap, conf, lowconf_thresh, hard_gap_weight, hard_lowconf_weight):
    weights = torch.ones_like(conf.view(-1))
    weights = weights + (frame_gap > 1).float() * (hard_gap_weight - 1.0)
    weights = weights + (conf.view(-1) < lowconf_thresh).float() * (hard_lowconf_weight - 1.0)
    return weights


# ================= 核心前向传递与损失计算 (保持 Exp10 特性) =================

def run_sequence_batch(
        model,
        batch_obs,
        batch_gt,
        batch_frame_ids,
        device,
        f_mat,
        h_mat,
        lowconf_thresh,
        hard_gap_weight,
        hard_lowconf_weight,
        add_noise=False,
):
    b_obs = batch_obs.to(device)
    b_gt = batch_gt.to(device)
    b_frame_ids = batch_frame_ids.to(device)

    batch_n = b_obs.size(0)

    # 遵循 Exp10，不计算协方差矩阵 P 和 S，仅保留纯净的状态更新逻辑
    current_state = torch.zeros(batch_n, 8, device=device)
    current_state[:, :4] = b_obs[:, 0, :4]

    hidden = None
    prev_z_meas = b_obs[:, 0, :4].clone()
    prev_update = torch.zeros(batch_n, 8, device=device)
    total_loss = 0.0

    seq_len = b_obs.size(1)
    for t in range(1, seq_len):
        pred_state = torch.matmul(current_state, f_mat.t())

        z_meas = b_obs[:, t, :4]
        conf = b_obs[:, t, 4:5]

        # 数据增强 (Exp21策略)
        if add_noise and torch.rand(1).item() < 0.15:
            conf = conf * 0.01
            z_meas = z_meas + torch.randn_like(z_meas) * 0.02

        pred_meas = torch.matmul(pred_state, h_mat.t())
        innovation = z_meas - pred_meas

        # 遵循 Exp10: 17维输入构建方式，不作分支切割，直接输入到网络
        f1 = z_meas - prev_z_meas
        f2 = innovation
        f4 = prev_update

        net_input = torch.cat([f1, f2, f4, conf], dim=1).unsqueeze(1)

        # 遵循 Exp10: 模型直接输出完全体 k_gain (而非残差 delta_k)
        k_gain, hidden = model(net_input, hidden)

        update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
        current_state = pred_state + update_term

        # 更新记忆
        prev_z_meas = z_meas
        prev_update = update_term

        # 计算当前步的时间片权重 (Exp21策略)
        frame_gap = b_frame_ids[:, t] - b_frame_ids[:, t - 1]
        step_weights = compute_step_weights(
            frame_gap, conf, lowconf_thresh, hard_gap_weight, hard_lowconf_weight
        )

        # 遵循 Exp10: 仅计算位置的 MSE Loss
        loss_pos = torch.mean((current_state[:, :4] - b_gt[:, t, :]).pow(2), dim=1)
        step_loss = torch.mean(loss_pos * step_weights)
        total_loss = total_loss + step_loss

    return total_loss / (seq_len - 1)


# ================= 主训练循环 =================

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if not os.path.exists(args.data_file):
        print("Error: training data not found. Please run prepare_mot_data.py first.")
        return

    payload = load_long_track_payload(args.data_file)
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
    ) = split_tracks(tracks_obs, tracks_gt, tracks_frame_ids, val_split=args.val_split, seed=42)

    short_train_dataset = FixedWindowTrackDataset(
        train_tracks_obs, train_tracks_gt, train_tracks_frame_ids,
        seq_len=short_seq_len, step=short_seq_step,
        lowconf_thresh=args.lowconf_thresh,
        lowconf_resample_boost=args.lowconf_resample_boost,
        gap_resample_boost=args.gap_resample_boost,
    )
    short_val_dataset = FixedWindowTrackDataset(
        val_tracks_obs, val_tracks_gt, val_tracks_frame_ids,
        seq_len=short_seq_len, step=short_seq_step,
        lowconf_thresh=args.lowconf_thresh,
        lowconf_resample_boost=args.lowconf_resample_boost,
        gap_resample_boost=args.gap_resample_boost,
    )

    # 构建加权采样器
    sample_weights = torch.DoubleTensor(short_train_dataset.sample_weights)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(short_train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(short_val_dataset, batch_size=args.batch_size, shuffle=False)

    # 实例化 Exp10 模型
    model = KalmanNetNN().to(device)

    f_mat = torch.eye(8, device=device)
    for i in range(4):
        f_mat[i, 4 + i] = 1.0
    h_mat = torch.eye(4, 8, device=device)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val_loss = float("inf")

    # Exp21 的多阶段微调策略
    stages = [
        ("Stage-1 Short BPTT", train_loader, val_loader, args.short_epochs, args.short_lr, True),
        ("Stage-2 Long-window Fine-tune", train_loader, val_loader, args.long_epochs, args.long_lr, False),
    ]

    for stage_name, t_loader, v_loader, epochs, lr, add_noise in stages:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        print(f"Start {stage_name} ...")

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0

            for b_obs, b_gt, b_frame_ids in t_loader:
                optimizer.zero_grad()
                batch_loss = run_sequence_batch(
                    model, b_obs, b_gt, b_frame_ids, device, f_mat, h_mat,
                    args.lowconf_thresh, args.hard_gap_weight, args.hard_lowconf_weight,
                    add_noise=add_noise,
                )
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += batch_loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / max(1, len(t_loader))

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for b_obs, b_gt, b_frame_ids in v_loader:
                    batch_loss = run_sequence_batch(
                        model, b_obs, b_gt, b_frame_ids, device, f_mat, h_mat,
                        args.lowconf_thresh, args.hard_gap_weight, args.hard_lowconf_weight,
                        add_noise=False,
                    )
                    total_val_loss += batch_loss.item()

            avg_val_loss = total_val_loss / max(1, len(v_loader))
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.save_path)

            current_lr = scheduler.get_last_lr()[0]
            best_flag = " <- best" if is_best else ""
            if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
                print(
                    f"{stage_name} Epoch [{epoch + 1}/{epochs}]  "
                    f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  "
                    f"LR: {current_lr:.6f}{best_flag}"
                )

    print(f"✅ Training complete. Best val_loss={best_val_loss:.6f}, saved to {args.save_path}")


if __name__ == "__main__":
    train()