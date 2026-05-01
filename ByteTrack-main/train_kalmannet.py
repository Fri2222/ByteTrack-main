import argparse
import os
import glob
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN.")
    raise SystemExit(1)


# ================= 1. 命令行参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="KalmanNet Ablation Study (Sequence Length)")
    parser.add_argument("--epochs", type=int, default=60, help="总训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--seq-len", type=int, default=20, help="BPTT截断序列长度 (如 15, 20, 25...)")
    parser.add_argument("--seq-step", type=int, default=5, help="滑动窗口采样步长")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--residual-gain-limit", type=float, default=0.30)
    return parser.parse_args()


# ================= 2. 数据集解析模块 =================
def parse_mot_dataset(base_dir, min_seq_len):
    """
    直接扫描并解析 MOT 规范格式下的 gt.txt
    """
    tracks_obs = []
    tracks_gt = []

    search_pattern = os.path.join(base_dir, '**', 'gt.txt')
    gt_files = glob.glob(search_pattern, recursive=True)

    if not gt_files:
        print(f"❌ 错误: 在 {base_dir} 中未找到任何 gt.txt 文件！")
        return [], []

    for gt_file in gt_files:
        try:
            data = np.loadtxt(gt_file, delimiter=',')
        except Exception as e:
            continue

        if len(data) == 0: continue

        track_ids = np.unique(data[:, 1])
        for tid in track_ids:
            track_data = data[data[:, 1] == tid]
            track_data = track_data[track_data[:, 0].argsort()]

            frames = track_data[:, 0]
            step_diffs = np.diff(frames)
            split_indices = np.where(step_diffs > 1)[0] + 1
            segments = np.split(track_data, split_indices)

            for seg in segments:
                # 过滤掉比当前实验的 seq_len 还要短的轨迹
                if len(seg) < min_seq_len:
                    continue

                left, top, w, h = seg[:, 2], seg[:, 3], seg[:, 4], seg[:, 5]
                cx = left + w / 2
                cy = top + h / 2
                gt_boxes = np.stack([cx, cy, w, h], axis=1)

                obs_boxes = np.zeros((len(seg), 5))
                obs_boxes[:, 0] = cx + np.random.normal(0, 3.0, size=len(seg))
                obs_boxes[:, 1] = cy + np.random.normal(0, 3.0, size=len(seg))
                obs_boxes[:, 2] = w + np.random.normal(0, 1.5, size=len(seg))
                obs_boxes[:, 3] = h + np.random.normal(0, 2.5, size=len(seg))
                obs_boxes[:, 4] = np.random.uniform(0.6, 0.95, size=len(seg))

                tracks_obs.append(torch.tensor(obs_boxes, dtype=torch.float32))
                tracks_gt.append(torch.tensor(gt_boxes, dtype=torch.float32))

    return tracks_obs, tracks_gt


# ================= 3. 协方差与增益模块 =================
def build_init_cov(measurement):
    h = measurement[:, 3].clamp_min(1e-3)
    std = torch.stack([
        2.0 * (1.0 / 20.0) * h, 2.0 * (1.0 / 20.0) * h, torch.full_like(h, 1e-2), 2.0 * (1.0 / 20.0) * h,
        10.0 * (1.0 / 160.0) * h, 10.0 * (1.0 / 160.0) * h, torch.full_like(h, 1e-5), 10.0 * (1.0 / 160.0) * h,
    ], dim=1)
    return torch.diag_embed(std.pow(2))


def build_motion_cov(state):
    h = state[:, 3].clamp_min(1e-3)
    std = torch.stack([
        (1.0 / 20.0) * h, (1.0 / 20.0) * h, torch.full_like(h, 1e-2), (1.0 / 20.0) * h,
        (1.0 / 160.0) * h, (1.0 / 160.0) * h, torch.full_like(h, 1e-5), (1.0 / 160.0) * h,
    ], dim=1)
    return torch.diag_embed(std.pow(2))


def build_meas_cov(pred_state, conf):
    h = pred_state[:, 3].clamp_min(1e-3)
    std = torch.stack([
        (1.0 / 20.0) * h, (1.0 / 20.0) * h, torch.full_like(h, 1e-1), (1.0 / 20.0) * h,
    ], dim=1)
    cov = torch.diag_embed(std.pow(2))
    conf = conf.clamp(0.1, 0.99).view(-1, 1, 1)
    return cov * (1.0 / conf)


def classical_gain(cov_pred, h_mat, s_mat):
    return torch.matmul(torch.matmul(cov_pred, h_mat.t()), torch.linalg.inv(s_mat))


# ================= 4. Dataset 构建模块 =================
class FixedWindowDataset(Dataset):
    def __init__(self, tracks_obs, tracks_gt, seq_len=20, step=5):
        self.samples = []
        for obs, gt in zip(tracks_obs, tracks_gt):
            track_len = obs.shape[0]
            if track_len < seq_len:
                continue
            last_start = track_len - seq_len
            for start in range(0, last_start + 1, step):
                self.samples.append((obs[start:start + seq_len], gt[start:start + seq_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def normalize_tracks(tracks_obs, tracks_gt, scale):
    norm_obs, norm_gt = [], []
    for obs, gt in zip(tracks_obs, tracks_gt):
        obs_c, gt_c = obs.clone(), gt.clone()
        obs_c[:, :4] /= scale
        gt_c /= scale
        norm_obs.append(obs_c)
        norm_gt.append(gt_c)
    return norm_obs, norm_gt


def split_tracks(tracks_obs, tracks_gt, val_split=0.1, seed=42):
    total_size = len(tracks_obs)
    val_size = max(1, int(total_size * val_split))
    perm = torch.randperm(total_size, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    return ([tracks_obs[i] for i in train_idx], [tracks_gt[i] for i in train_idx],
            [tracks_obs[i] for i in val_idx], [tracks_gt[i] for i in val_idx])


# ================= 5. 训练核心逻辑 =================
def run_sequence_batch(model, batch_obs, batch_gt, device, f_mat, h_mat, residual_gain_limit, criterion,
                       add_noise=False):
    b_obs, b_gt = batch_obs.to(device), batch_gt.to(device)
    batch_n, seq_len = b_obs.size(0), b_obs.size(1)

    identity = torch.eye(8, device=device).unsqueeze(0).expand(batch_n, -1, -1)
    h_batch = h_mat.unsqueeze(0).expand(batch_n, -1, -1)
    f_batch = f_mat.unsqueeze(0).expand(batch_n, -1, -1)

    current_state = torch.zeros(batch_n, 8, device=device)
    current_state[:, :4] = b_obs[:, 0, :4]
    covariance = build_init_cov(b_obs[:, 0, :4]).to(device)
    hidden = None
    prev_update = torch.zeros(batch_n, 8, device=device)
    total_loss = 0.0

    for t in range(1, seq_len):
        pred_state = torch.matmul(current_state, f_mat.t())
        cov_pred = torch.matmul(torch.matmul(f_batch, covariance), f_mat.t().unsqueeze(0)) + build_motion_cov(
            current_state).to(device)

        z_meas = b_obs[:, t, :4]
        conf = b_obs[:, t, 4:5]

        if add_noise and torch.rand(1).item() < 0.15:
            conf = conf * 0.01
            z_meas = z_meas + torch.randn_like(z_meas) * 0.02

        pred_meas = torch.matmul(pred_state, h_mat.t())
        innovation = z_meas - pred_meas
        meas_cov = build_meas_cov(pred_state, conf).to(device)
        s_mat = torch.matmul(torch.matmul(h_batch, cov_pred), h_mat.t().unsqueeze(0)) + meas_cov
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
        total_loss = total_loss + loss_pos + 2.0 * loss_vel

    return total_loss / (seq_len - 1)


# ================= 6. 主程序 =================
def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"🚀 启动消融实验: 截断长度 = {args.seq_len} 帧 | 步长 = {args.seq_step}")
    print(f"==================================================")

    data_dir = r"datasets\MOT17_Missile"
    if not os.path.exists(data_dir): data_dir = r"datasets\mot"

    # 按照当前 args.seq_len 过滤数据
    tracks_obs, tracks_gt = parse_mot_dataset(data_dir, min_seq_len=args.seq_len)
    if not tracks_obs: return

    scale = torch.tensor([1920, 1080, 1920, 1080], dtype=torch.float32)
    tracks_obs, tracks_gt = normalize_tracks(tracks_obs, tracks_gt, scale)

    train_obs, train_gt, val_obs, val_gt = split_tracks(tracks_obs, tracks_gt, val_split=args.val_split)

    # 👇 根据命令行传入的 seq_len 和 seq_step 构建定长数据集
    train_dataset = FixedWindowDataset(train_obs, train_gt, seq_len=args.seq_len, step=args.seq_step)
    val_dataset = FixedWindowDataset(val_obs, val_gt, seq_len=args.seq_len, step=args.seq_step)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"📊 生成窗口数 -> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    model = KalmanNetNN().to(device)
    criterion = nn.MSELoss()

    # 👇 【关键修改】：将权重统一命名为 kalmannet_best.pth 并保存在 exp0031
    # 这样 .bat 脚本就能顺利找到它，并将其重命名为带参数的专属名字！
    save_dir = r"H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "kalmannet_best.pth")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    f_mat = torch.eye(8, device=device)
    for i in range(4): f_mat[i, 4 + i] = 1.0
    h_mat = torch.eye(4, 8, device=device)

    best_val_loss = float("inf")

    # 单阶段训练循环 (刚好跑完设定的 60 轮)
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for b_obs, b_gt in train_loader:
            optimizer.zero_grad()
            loss = run_sequence_batch(model, b_obs, b_gt, device, f_mat, h_mat, args.residual_gain_limit, criterion,
                                      add_noise=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for b_obs, b_gt in val_loader:
                loss = run_sequence_batch(model, b_obs, b_gt, device, f_mat, h_mat, args.residual_gain_limit, criterion,
                                          add_noise=False)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == args.epochs - 1:
            best_flag = " 🌟 新高" if is_best else ""
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}{best_flag}")

    print(f"✅ [Len={args.seq_len}] 训练结束! 最佳 Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()