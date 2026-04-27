import argparse
import os
import subprocess
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full MOT17 validation pipeline for custom KalmanNet/ByteTrack code."
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to run sub-scripts.",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(CURRENT_DIR, "datasets", "mot"),
        help="MOT dataset root.",
    )
    parser.add_argument(
        "--det-results-dir",
        default=os.path.join(
            CURRENT_DIR, "YOLOX_outputs", "yolox_s_mot17_half", "track_results"
        ),
        help="Detection or baseline tracking results used to build KalmanNet training data.",
    )
    parser.add_argument(
        "--data-file",
        default=os.path.join(CURRENT_DIR, "mot_train_data.pt"),
        help="Prepared long-track training data path.",
    )
    parser.add_argument(
        "--feature-mode",
        default="f2_f4_conf",
        help="KalmanNet feature mode for training.",
    )
    parser.add_argument(
        "--kalmannet-ckpt",
        default=os.path.join(CURRENT_DIR, "pretrained", "kalmannet_best.pth"),
        help="Output KalmanNet checkpoint used for tracking.",
    )
    parser.add_argument(
        "--exp-file",
        default=os.path.join(
            CURRENT_DIR, "exps", "example", "mot", "yolox_s_mot17_half.py"
        ),
        help="YOLOX experiment file for MOT17 validation.",
    )
    parser.add_argument(
        "--det-ckpt",
        default=os.path.join(CURRENT_DIR, "pretrained", "bytetrack_s_mot17.pth.tar"),
        help="Detection checkpoint used by tools/track.py.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Tracking batch size.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs for tracking.")
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--method-name", default="KalmanNet-Custom")
    parser.add_argument(
        "--tracker-results-dir",
        default="",
        help="Override tracker result directory for eval. Defaults to YOLOX_outputs/<exp_name>/track_results.",
    )
    parser.add_argument("--short-epochs", type=int, default=40)
    parser.add_argument("--long-epochs", type=int, default=20)
    parser.add_argument("--short-lr", type=float, default=1e-3)
    parser.add_argument("--long-lr", type=float, default=2e-4)
    parser.add_argument("--residual-gain-limit", type=float, default=0.45)
    parser.add_argument("--delta-k-reg-weight", type=float, default=0.01)
    parser.add_argument("--hard-gap-weight", type=float, default=2.5)
    parser.add_argument("--hard-lowconf-weight", type=float, default=2.0)
    parser.add_argument("--lowconf-thresh", type=float, default=0.3)
    parser.add_argument("--lowconf-resample-boost", type=float, default=2.0)
    parser.add_argument("--gap-resample-boost", type=float, default=2.5)
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_mot_data.py.")
    parser.add_argument("--skip-train", action="store_true", help="Skip train_kalmannet.py.")
    parser.add_argument("--skip-track", action="store_true", help="Skip tools/track.py.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval_custom.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run_step(step_name, command, dry_run=False):
    print(f"\n[STEP] {step_name}")
    print(" ".join(f'"{c}"' if " " in str(c) else str(c) for c in command))
    if dry_run:
        return
    subprocess.run(command, cwd=CURRENT_DIR, check=True)


def infer_experiment_name(exp_file):
    return os.path.splitext(os.path.basename(exp_file))[0]


def main():
    args = parse_args()

    exp_name = infer_experiment_name(args.exp_file)
    tracker_results_dir = (
        args.tracker_results_dir
        if args.tracker_results_dir
        else os.path.join(CURRENT_DIR, "YOLOX_outputs", exp_name, "track_results")
    )

    if not args.skip_prepare:
        prepare_cmd = [
            args.python_exe,
            os.path.join(CURRENT_DIR, "prepare_mot_data.py"),
            "--data-root",
            args.data_root,
            "--det-root",
            args.det_results_dir,
            "--output-path",
            args.data_file,
        ]
        run_step("Prepare long-track training data", prepare_cmd, args.dry_run)

    if not args.skip_train:
        train_cmd = [
            args.python_exe,
            os.path.join(CURRENT_DIR, "train_kalmannet.py"),
            "--feature-mode",
            args.feature_mode,
            "--data-file",
            args.data_file,
            "--save-path",
            args.kalmannet_ckpt,
            "--short-epochs",
            str(args.short_epochs),
            "--long-epochs",
            str(args.long_epochs),
            "--short-lr",
            str(args.short_lr),
            "--long-lr",
            str(args.long_lr),
            "--residual-gain-limit",
            str(args.residual_gain_limit),
            "--delta-k-reg-weight",
            str(args.delta_k_reg_weight),
            "--hard-gap-weight",
            str(args.hard_gap_weight),
            "--hard-lowconf-weight",
            str(args.hard_lowconf_weight),
            "--lowconf-thresh",
            str(args.lowconf_thresh),
            "--lowconf-resample-boost",
            str(args.lowconf_resample_boost),
            "--gap-resample-boost",
            str(args.gap_resample_boost),
        ]
        run_step("Train KalmanNet", train_cmd, args.dry_run)

    if not args.skip_track:
        track_cmd = [
            args.python_exe,
            os.path.join(CURRENT_DIR, "tools", "track.py"),
            "-f",
            args.exp_file,
            "-c",
            args.det_ckpt,
            "-b",
            str(args.batch_size),
            "-d",
            str(args.devices),
            "--fp16",
            "--fuse",
            "--track_thresh",
            str(args.track_thresh),
            "--track_buffer",
            str(args.track_buffer),
            "--match_thresh",
            str(args.match_thresh),
            "--kalmannet-ckpt",
            args.kalmannet_ckpt,
        ]
        run_step("Run MOT17 tracking", track_cmd, args.dry_run)

    if not args.skip_eval:
        eval_cmd = [
            args.python_exe,
            os.path.join(CURRENT_DIR, "eval_custom.py"),
            "--tracker-results-dir",
            tracker_results_dir,
            "--gt-dir",
            os.path.join(args.data_root, "train"),
            "--method-name",
            args.method_name,
        ]
        run_step("Evaluate MOT17 metrics", eval_cmd, args.dry_run)

    print("\nPipeline complete.")
    print(f"Tracker results dir: {tracker_results_dir}")
    print(f"KalmanNet checkpoint: {args.kalmannet_ckpt}")


if __name__ == "__main__":
    main()
