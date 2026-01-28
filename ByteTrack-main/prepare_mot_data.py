import os
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """
    计算检测框与GT之间的IoU矩阵
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 0] + bb_test[..., 2], bb_gt[..., 0] + bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 1] + bb_test[..., 3], bb_gt[..., 1] + bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    union = (bb_test[..., 2] * bb_test[..., 3]) + (bb_gt[..., 2] * bb_gt[..., 3]) - wh
    union = np.maximum(union, 1e-6)

    o = wh / union
    return o


def load_mot_file(filepath):
    """加载 MOT 格式文件"""
    if not os.path.exists(filepath):
        return {}
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except:
        return {}

    frames = {}
    if len(data.shape) < 2: data = data.reshape(1, -1)

    for row in data:
        frame_id = int(row[0])
        if frame_id not in frames:
            frames[frame_id] = []
        frames[frame_id].append(row)
    return frames


def prepare_real_data(data_root, det_root):
    train_dir = os.path.join(data_root, 'train')
    seqs = os.listdir(train_dir)

    all_tracks_input = []  # (x, y, a, h, conf)
    all_tracks_gt = []  # (x, y, a, h)

    SEQ_LEN = 20

    print(f"Processing Real Data from {det_root}...")

    for seq in seqs:
        if 'FRCNN' not in seq: continue

        print(f"Aligning {seq}...")

        gt_path = os.path.join(train_dir, seq, 'gt', 'gt.txt')
        gt_frames = load_mot_file(gt_path)

        det_path = os.path.join(det_root, f"{seq}.txt")
        det_frames = load_mot_file(det_path)

        if not det_frames:
            print(f"Warning: No detection file for {seq}, skipping.")
            continue

        matched_tracks = {}

        common_frames = sorted(list(set(gt_frames.keys()) & set(det_frames.keys())))

        for fid in common_frames:
            gts = np.array(gt_frames[fid])
            dets = np.array(det_frames[fid])

            gt_boxes = gts[:, 2:6]
            det_boxes = dets[:, 2:6]

            iou_matrix = iou_batch(det_boxes, gt_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < 0.5: continue

                gt_id = int(gts[c, 1])

                # Det: x, y, w, h, score (长度 5)
                d = dets[r]
                det_val = [d[2] + d[4] / 2, d[3] + d[5] / 2, d[4] / d[5], d[5], d[6]]

                # GT: x, y, w, h (长度 4)
                g = gts[c]
                gt_val = [g[2] + g[4] / 2, g[3] + g[5] / 2, g[4] / g[5], g[5]]

                if gt_id not in matched_tracks:
                    matched_tracks[gt_id] = []
                # 存入列表，暂不转 numpy
                matched_tracks[gt_id].append((det_val, gt_val))

        # === [核心修复部分] ===
        for gt_id, track in matched_tracks.items():
            if len(track) < SEQ_LEN: continue

            # 分别提取 det 和 gt，避免由维度不一致导致的创建数组失败
            # track 是一个 list of tuples: [(det1, gt1), (det2, gt2), ...]

            det_seq_list = [t[0] for t in track]  # list of [5 values]
            gt_seq_list = [t[1] for t in track]  # list of [4 values]

            det_seq_full = np.array(det_seq_list)  # (Len, 5)
            gt_seq_full = np.array(gt_seq_list)  # (Len, 4)

            # 滑动窗口切片
            for i in range(0, len(track) - SEQ_LEN, 5):
                all_tracks_input.append(det_seq_full[i: i + SEQ_LEN])
                all_tracks_gt.append(gt_seq_full[i: i + SEQ_LEN])

    # 转换为 Tensor
    X = torch.tensor(np.array(all_tracks_input), dtype=torch.float32)
    Y = torch.tensor(np.array(all_tracks_gt), dtype=torch.float32)

    print(f"Total matched samples: {len(X)}")
    torch.save((X, Y), 'mot_train_data_real.pt')
    print("Saved to mot_train_data_real.pt")


if __name__ == '__main__':
    # 请确保这个路径正确
    det_root = 'YOLOX_outputs/yolox_s_mot17_half/track_results'
    prepare_real_data('datasets/mot', det_root)