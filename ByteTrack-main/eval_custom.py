import argparse
import os
import shutil
import sys
import warnings
from typing import Dict, List

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "object"):
        np.object = object

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKEVAL_DIR = os.path.join(CURRENT_DIR, "TrackEval")
if TRACKEVAL_DIR not in sys.path:
    sys.path.insert(0, TRACKEVAL_DIR)

DEFAULT_SEQS = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking results.")
    parser.add_argument(
        "--tracker-results-dir",
        default=os.path.join(CURRENT_DIR, "YOLOX_outputs", "yolox_s_mot17_half", "track_results"),
    )
    parser.add_argument(
        "--gt-dir",
        default=os.path.join(CURRENT_DIR, "datasets", "mot", "train"),
    )
    parser.add_argument("--method-name", default="KalmanNet-Ours")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    parser.add_argument(
        "--report-dir",
        default=os.path.join(CURRENT_DIR, "myExperiments", "eval_reports"),
        help="Directory to store TrackEval summary files.",
    )
    return parser.parse_args()


def build_seq_info(gt_dir: str, seqs: List[str]) -> Dict[str, int]:
    seq_info = {}
    for seq in seqs:
        gt_file = os.path.join(gt_dir, seq, "gt", "gt_val_half.txt")
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"GT file not found: {gt_file}")

        frames = np.loadtxt(gt_file, delimiter=",", usecols=[0])
        max_frame = int(frames) if np.isscalar(frames) else int(np.max(frames))
        seq_info[seq] = max_frame
    return seq_info


def to_percent(value):
    if isinstance(value, (np.ndarray, list)):
        value = np.mean(value)
    return 100.0 * float(value)


def move_trackeval_outputs(tracker_results_dir, report_dir, method_name):
    os.makedirs(report_dir, exist_ok=True)
    for name in os.listdir(tracker_results_dir):
        if not (
            name.endswith("_summary.txt")
            or name.endswith("_detailed.csv")
            or name.endswith("_plot.pdf")
            or name.endswith("_plot.png")
        ):
            continue

        src = os.path.join(tracker_results_dir, name)
        dst_name = f"{method_name}_{name}"
        dst = os.path.join(report_dir, dst_name)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)


def evaluate(args):
    try:
        from trackeval import Evaluator, datasets, metrics
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency. Please install scipy and motmetrics."
        ) from e

    tracker_results_dir = os.path.abspath(args.tracker_results_dir)
    gt_dir = os.path.abspath(args.gt_dir)
    report_dir = os.path.abspath(args.report_dir)
    tracker_parent_dir = os.path.dirname(tracker_results_dir)
    tracker_name = os.path.basename(tracker_results_dir)

    eval_config = Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False

    dataset_config = datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_dir
    dataset_config["TRACKERS_FOLDER"] = tracker_parent_dir
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["BENCHMARK"] = args.benchmark
    dataset_config["SPLIT_TO_EVAL"] = args.split
    dataset_config["INPUT_AS_ZIP"] = False
    dataset_config["PRINT_CONFIG"] = False
    dataset_config["DO_PREPROC"] = True
    dataset_config["GT_LOC_FORMAT"] = "{gt_folder}/{seq}/gt/gt_val_half.txt"
    dataset_config["TRACKER_SUB_FOLDER"] = ""
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["SEQ_INFO"] = build_seq_info(gt_dir, args.seqs)

    metrics_list = [
        metrics.HOTA({"PRINT_CONFIG": False}),
        metrics.CLEAR({"PRINT_CONFIG": False}),
        metrics.Identity({"PRINT_CONFIG": False}),
    ]

    evaluator = Evaluator(eval_config)
    dataset = datasets.MotChallenge2DBox(dataset_config)
    results, _ = evaluator.evaluate([dataset], metrics_list)
    move_trackeval_outputs(tracker_results_dir, report_dir, args.method_name)

    combined = results["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]
    hota_res = combined["HOTA"]
    clear_res = combined["CLEAR"]
    id_res = combined["Identity"]

    return {
        "Method": args.method_name,
        "HOTA": to_percent(hota_res["HOTA"]),
        "DetA": to_percent(hota_res["DetA"]),
        "AssA": to_percent(hota_res["AssA"]),
        "MOTA": to_percent(clear_res["MOTA"]),
        "IDF1": to_percent(id_res["IDF1"]),
        "IDP": to_percent(id_res["IDP"]),
        "IDR": to_percent(id_res["IDR"]),
        "Recall": to_percent(clear_res["CLR_Re"]),
        "Precision": to_percent(clear_res["CLR_Pr"]),
        "IDSWs": int(clear_res["IDSW"]),
        "IDs": int(clear_res["IDSW"]),
        "FP": int(clear_res["CLR_FP"]),
        "FN": int(clear_res["CLR_FN"]),
        "MT": int(clear_res["MT"]),
        "PT": int(clear_res["PT"]),
        "ML": int(clear_res["ML"]),
    }


def print_summary(summary):
    print("\n" + "=" * 140)
    print(
        f"{'Method':<18} {'HOTA':>7} {'DetA':>7} {'AssA':>7} "
        f"{'MOTA':>7} {'IDF1':>7} {'IDP':>7} {'IDR':>7} "
        f"{'Recall':>8} {'Prec':>8} {'IDSWs':>8} {'FP':>8} {'FN':>8} {'MT':>6} {'PT':>6} {'ML':>6}"
    )
    print("-" * 140)
    print(
        f"{summary['Method']:<18} "
        f"{summary['HOTA']:>6.2f}% "
        f"{summary['DetA']:>6.2f}% "
        f"{summary['AssA']:>6.2f}% "
        f"{summary['MOTA']:>6.2f}% "
        f"{summary['IDF1']:>6.2f}% "
        f"{summary['IDP']:>6.2f}% "
        f"{summary['IDR']:>6.2f}% "
        f"{summary['Recall']:>7.2f}% "
        f"{summary['Precision']:>7.2f}% "
        f"{summary['IDSWs']:>8d} "
        f"{summary['FP']:>8d} "
        f"{summary['FN']:>8d} "
        f"{summary['MT']:>6d} "
        f"{summary['PT']:>6d} "
        f"{summary['ML']:>6d}"
    )
    print("=" * 140 + "\n")


if __name__ == "__main__":
    args = parse_args()
    summary = evaluate(args)
    print_summary(summary)