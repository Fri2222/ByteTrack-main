import argparse
import os
import sys
from typing import Dict, List

# 👇 解决 numpy 版本代沟
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'object'):
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
    parser.add_argument("--tracker-results-dir", default=os.path.join(CURRENT_DIR, "YOLOX_outputs", "yolox_s_mot17_half", "track_results"))
    parser.add_argument("--gt-dir", default=os.path.join(CURRENT_DIR, "datasets", "mot", "train"))
    parser.add_argument("--method-name", default="KalmanNet-Ours")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    return parser.parse_args()

def build_seq_info(gt_dir: str, seqs: List[str]) -> Dict[str, int]:
    import numpy as np
    seq_info = {}
    for seq in seqs:
        # 👇 [核心修复 1]：读取 gt_val_half.txt 获取帧数，而不是全集
        gt_file = os.path.join(gt_dir, seq, "gt", "gt_val_half.txt")
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"GT file not found: {gt_file}")

        frames = np.loadtxt(gt_file, delimiter=",", usecols=[0])
        if np.isscalar(frames):
            max_frame = int(frames)
        else:
            max_frame = int(np.max(frames))
        seq_info[seq] = max_frame
    return seq_info

def to_percent(value):
    import numpy as np
    if isinstance(value, (np.ndarray, list)):
        value = np.mean(value)
    return 100.0 * float(value)

def evaluate(args):
    try:
        from trackeval import Evaluator, datasets, metrics
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency. run `pip install scipy`") from e

    tracker_results_dir = os.path.abspath(args.tracker_results_dir)
    gt_dir = os.path.abspath(args.gt_dir)
    tracker_parent_dir = os.path.dirname(tracker_results_dir)
    tracker_name = os.path.basename(tracker_results_dir)

    eval_config = Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False

    dataset_config = datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_dir
    dataset_config["TRACKERS_FOLDER"] = tracker_parent_dir
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["BENCHMARK"] = args.benchmark
    dataset_config["SPLIT_TO_EVAL"] = args.split
    dataset_config["INPUT_AS_ZIP"] = False
    dataset_config["PRINT_CONFIG"] = False
    
    # 👇 [核心修复 2]：开启预处理！过滤掉汽车、人群等不参与评估的干扰项
    dataset_config["DO_PREPROC"] = True 
    # 👇 [核心修复 3]：强制指定 TrackEval 去跟 val_half 进行对比，而不是 gt.txt
    dataset_config["GT_LOC_FORMAT"] = '{gt_folder}/{seq}/gt/gt_val_half.txt' 
    
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

    combined = results["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]

    hota_res = combined["HOTA"]
    clear_res = combined["CLEAR"]
    id_res = combined["Identity"]

    summary = {
        "Method": args.method_name,
        "HOTA": to_percent(hota_res["HOTA"]),
        "DetA": to_percent(hota_res["DetA"]),
        "AssA": to_percent(hota_res["AssA"]),
        "MOTA": to_percent(clear_res["MOTA"]),
        "IDF1": to_percent(id_res["IDF1"]),
        "Recall": to_percent(clear_res["CLR_Re"]),
        "Precision": to_percent(clear_res["CLR_Pr"]),
        "IDSWs": int(clear_res["IDSW"]),
        "IDs": int(clear_res["IDSW"]),
        "FP": int(clear_res["CLR_FP"]),
        "FN": int(clear_res["CLR_FN"]),
    }
    return summary

def print_summary(summary):
    print("\n" + "=" * 105)
    print(
        f"{'Method':<18} {'HOTA':>7} {'DetA':>7} {'AssA':>7} "
        f"{'MOTA':>7} {'IDF1':>7} {'Recall':>8} {'Prec':>8} "
        f"{'IDs':>8} {'FP':>8} {'FN':>8}"
    )
    print("-" * 105)
    print(
        f"{summary['Method']:<18} "
        f"{summary['HOTA']:>6.2f}% "
        f"{summary['DetA']:>6.2f}% "
        f"{summary['AssA']:>6.2f}% "
        f"{summary['MOTA']:>6.2f}% "
        f"{summary['IDF1']:>6.2f}% "
        f"{summary['Recall']:>7.2f}% "
        f"{summary['Precision']:>7.2f}% "
        f"{summary['IDs']:>8d} "
        f"{summary['FP']:>8d} "
        f"{summary['FN']:>8d}"
    )
    print("=" * 105 + "\n")

if __name__ == "__main__":
    args = parse_args()
    summary = evaluate(args)
    print_summary(summary)