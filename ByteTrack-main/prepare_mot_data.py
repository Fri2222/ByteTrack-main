import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


SHORT_SEQ_LEN = 20
SHORT_SEQ_STEP = 2


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 0] + bb_test[..., 2], bb_gt[..., 0] + bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 1] + bb_test[..., 3], bb_gt[..., 1] + bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    union = (bb_test[..., 2] * bb_test[..., 3]) + (bb_gt[..., 2] * bb_gt[..., 3]) - inter
    union = np.maximum(union, 1e-6)
    return inter / union


def load_mot_file(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        data = np.loadtxt(filepath, delimiter=",")
    except Exception:
        return {}

    if len(data.shape) < 2:
        data = data.reshape(1, -1)

    frames = {}
    for row in data:
        frame_id = int(row[0])
        frames.setdefault(frame_id, []).append(row)
    return frames


def xywh_to_xyah(row):
    x, y, w, h = row
    return [x + w / 2.0, y + h / 2.0, w / h, h]


def build_track_collections(data_root, det_root):
    train_dir = os.path.join(data_root, "train")
    seqs = os.listdir(train_dir)

    long_tracks_input = []
    long_tracks_gt = []
    long_tracks_frame_ids = []
    long_track_meta = []
    short_sample_count = 0

    print(f"Processing real MOT data from {det_root} ...")

    for seq in seqs:
        if "FRCNN" not in seq:
            continue

        gt_path = os.path.join(train_dir, seq, "gt", "gt.txt")
        det_path = os.path.join(det_root, f"{seq}.txt")

        gt_frames = load_mot_file(gt_path)
        det_frames = load_mot_file(det_path)
        if not det_frames:
            print(f"Warning: no detection file for {seq}, skipping.")
            continue

        print(f"Aligning {seq} ...")
        matched_tracks = {}
        common_frames = sorted(set(gt_frames.keys()) & set(det_frames.keys()))

        for fid in common_frames:
            gts = np.asarray(gt_frames[fid], dtype=np.float32)
            dets = np.asarray(det_frames[fid], dtype=np.float32)

            valid_mask = dets[:, 6] > 0.1
            dets = dets[valid_mask]
            if len(dets) == 0:
                continue

            gt_boxes = gts[:, 2:6]
            det_boxes = dets[:, 2:6]
            iou_matrix = iou_batch(det_boxes, gt_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < 0.5:
                    continue

                gt_id = int(gts[c, 1])
                det_box = xywh_to_xyah(dets[r, 2:6])
                gt_box = xywh_to_xyah(gts[c, 2:6])

                det_val = det_box + [float(dets[r, 6])]
                gt_val = gt_box

                matched_tracks.setdefault(gt_id, {"det": [], "gt": [], "frame_ids": []})
                matched_tracks[gt_id]["det"].append(det_val)
                matched_tracks[gt_id]["gt"].append(gt_val)
                matched_tracks[gt_id]["frame_ids"].append(fid)

        for gt_id, track_dict in matched_tracks.items():
            det_seq = np.asarray(track_dict["det"], dtype=np.float32)
            gt_seq = np.asarray(track_dict["gt"], dtype=np.float32)
            frame_ids = np.asarray(track_dict["frame_ids"], dtype=np.int64)
            if len(det_seq) < SHORT_SEQ_LEN:
                continue

            long_tracks_input.append(torch.tensor(det_seq, dtype=torch.float32))
            long_tracks_gt.append(torch.tensor(gt_seq, dtype=torch.float32))
            long_tracks_frame_ids.append(torch.tensor(frame_ids, dtype=torch.long))

            frame_gaps = np.diff(frame_ids)
            num_gaps = int(np.sum(frame_gaps > 1))
            long_track_meta.append(
                {
                    "seq": seq,
                    "gt_id": gt_id,
                    "length": int(len(det_seq)),
                    "num_gaps": num_gaps,
                    "max_gap": int(frame_gaps.max()) if len(frame_gaps) > 0 else 1,
                }
            )
            short_sample_count += max(
                0, (len(det_seq) - SHORT_SEQ_LEN + SHORT_SEQ_STEP - 1) // SHORT_SEQ_STEP
            )

    if not long_tracks_input:
        print("Error: no valid long tracks generated. Check paths and detection results.")
        return None

    payload = {
        "version": 3,
        "dataset_type": "long_track",
        "short_seq_len": SHORT_SEQ_LEN,
        "short_seq_step": SHORT_SEQ_STEP,
        "long_tracks_input": long_tracks_input,
        "long_tracks_gt": long_tracks_gt,
        "long_tracks_frame_ids": long_tracks_frame_ids,
        "long_track_meta": long_track_meta,
    }
    return payload, short_sample_count


def prepare_real_data(data_root, det_root, output_path="mot_train_data.pt"):
    result = build_track_collections(data_root, det_root)
    if result is None:
        return

    payload, short_sample_count = result
    torch.save(payload, output_path)

    track_lengths = [meta["length"] for meta in payload["long_track_meta"]]
    track_gaps = [meta["num_gaps"] for meta in payload["long_track_meta"]]
    print(f"Saved long-track dataset to {output_path}")
    print(f"Dataset version: {payload['version']} ({payload['dataset_type']})")
    print(f"Total long tracks: {len(payload['long_tracks_input'])}")
    print(f"Approx short windows ({payload['short_seq_len']} frames): {short_sample_count}")
    print(
        "Track length stats: "
        f"min={min(track_lengths)}, max={max(track_lengths)}, mean={sum(track_lengths) / len(track_lengths):.1f}"
    )
    print(
        "Temporal gap stats: "
        f"tracks_with_gaps={sum(g > 0 for g in track_gaps)}, max_gaps_per_track={max(track_gaps)}"
    )


if __name__ == "__main__":
    det_root = "YOLOX_outputs/yolox_s_mot17_half/track_results"
    prepare_real_data("datasets/mot", det_root)
