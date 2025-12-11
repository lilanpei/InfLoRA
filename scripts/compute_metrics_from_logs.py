#!/usr/bin/env python3
"""
Compute final_avg_acc and final_avg_forgetting from existing InfLoRA logs.

Usage:
    python scripts/compute_metrics_from_logs.py logs/slurm/
"""

import os
import re
import sys
from pathlib import Path


def parse_log_file(log_path: str) -> dict:
    """Parse an InfLoRA log file and extract metrics."""
    with open(log_path, "r") as f:
        content = f.read()

    # Extract dataset info
    dataset_match = re.search(r"dataset:\s*(\w+)", content)
    dataset = dataset_match.group(1) if dataset_match else "unknown"

    # Extract init_cls and increment
    init_cls_match = re.search(r"init_cls:\s*(\d+)", content)
    increment_match = re.search(r"increment:\s*(\d+)", content)
    init_cls = int(init_cls_match.group(1)) if init_cls_match else 10
    increment = int(increment_match.group(1)) if increment_match else 10

    # Extract seed
    seed_match = re.search(r"seed:\s*(\d+)", content)
    seed = int(seed_match.group(1)) if seed_match else None

    # Find all CNN accuracy logs with grouped per-task accuracies
    # Format: CNN: {'total': np.float64(86.74), '00-09': np.float64(94.2), ...}
    cnn_pattern = r"CNN:\s*(\{[^}]+\})"
    cnn_matches = re.findall(cnn_pattern, content)

    # Find all CNN top1 curve logs
    # Format: CNN top1 curve: [np.float64(99.4), np.float64(96.3), ...]
    curve_pattern = r"CNN top1 curve:\s*\[([^\]]+)\]"
    curve_matches = re.findall(curve_pattern, content)

    if not cnn_matches or not curve_matches:
        return None

    # Parse the last (final) CNN top1 curve to get all cumulative accuracies
    last_curve = curve_matches[-1]
    # Extract float values from np.float64(X.XX) format
    acc_values = re.findall(r"np\.float64\(([0-9.]+)\)", last_curve)
    if not acc_values:
        # Try plain float format
        acc_values = re.findall(r"([0-9.]+)", last_curve)

    cnn_top1_curve = [float(v) for v in acc_values]

    if not cnn_top1_curve:
        return None

    # Parse per-task accuracies from each CNN grouped log
    # Track DC-LoRA-style "true best over time" per-task accuracies.
    num_tasks = len(cnn_top1_curve)
    best_accuracy_per_task = [0.0 for _ in range(num_tasks)]
    current_accuracy_per_task = [0.0 for _ in range(num_tasks)]

    for task_idx, cnn_log in enumerate(cnn_matches):
        # Parse the grouped dict to extract per-task accuracies
        for t in range(task_idx + 1):
            start = t * increment
            end = (t + 1) * increment - 1
            key_pattern = rf"'{start:02d}-{end:02d}':\s*np\.float64\(([0-9.]+)\)"
            match = re.search(key_pattern, cnn_log)
            if match:
                acc_t = float(match.group(1))
                current_accuracy_per_task[t] = acc_t
                if acc_t > best_accuracy_per_task[t]:
                    best_accuracy_per_task[t] = acc_t

    # Compute CNN metrics
    final_avg_acc = cnn_top1_curve[-1]
    avg_acc_over_tasks = sum(cnn_top1_curve) / len(cnn_top1_curve)

    # Compute forgetting
    forgetting_values = []
    for idx in range(num_tasks - 1):
        forgetting = max(
            best_accuracy_per_task[idx] - current_accuracy_per_task[idx], 0.0
        )
        forgetting_values.append(forgetting)

    final_avg_forgetting = (
        sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0
    )
    final_max_forgetting = max(forgetting_values) if forgetting_values else 0.0

    # Optional: parse W-NCM (whitened nearest class mean) metrics if present.
    # Per-task W-NCM average accuracies over tasks (history)
    # Lines look like: "Ave Acc (W-NCM): 76.72%"
    wncm_curve_pattern = r"Ave Acc \(W-NCM\):\s*([0-9.]+)%"
    wncm_curve_matches = re.findall(wncm_curve_pattern, content)
    wncm_top1_curve = (
        [float(v) for v in wncm_curve_matches] if wncm_curve_matches else []
    )

    # Final W-NCM summary lines, if present
    wncm_final_acc_match = re.search(
        r"W-NCM final average accuracy:\s*([0-9.]+)%", content
    )
    wncm_avg_over_tasks_match = re.search(
        r"W-NCM average accuracy over tasks:\s*([0-9.]+)%", content
    )
    wncm_final_avg_fgt_match = re.search(
        r"W-NCM final average forgetting:\s*([0-9.]+)%", content
    )
    wncm_final_max_fgt_match = re.search(
        r"W-NCM final max forgetting:\s*([0-9.]+)%", content
    )

    if wncm_final_acc_match and wncm_avg_over_tasks_match:
        final_avg_acc_wncm = float(wncm_final_acc_match.group(1))
        avg_acc_over_tasks_wncm = float(wncm_avg_over_tasks_match.group(1))
        final_avg_forgetting_wncm = (
            float(wncm_final_avg_fgt_match.group(1))
            if wncm_final_avg_fgt_match
            else 0.0
        )
        final_max_forgetting_wncm = (
            float(wncm_final_max_fgt_match.group(1))
            if wncm_final_max_fgt_match
            else 0.0
        )
    else:
        final_avg_acc_wncm = None
        avg_acc_over_tasks_wncm = None
        final_avg_forgetting_wncm = None
        final_max_forgetting_wncm = None

    return {
        "log_file": os.path.basename(log_path),
        "dataset": dataset,
        "init_cls": init_cls,
        "increment": increment,
        "num_tasks": len(cnn_top1_curve),
        "seed": seed,
        "final_avg_acc": final_avg_acc,
        "avg_acc_over_tasks": avg_acc_over_tasks,
        "final_avg_forgetting": final_avg_forgetting,
        "final_max_forgetting": final_max_forgetting,
        "cnn_top1_curve": cnn_top1_curve,
        "best_per_task": best_accuracy_per_task,
        "current_per_task": current_accuracy_per_task,
        "final_avg_acc_wncm": final_avg_acc_wncm,
        "avg_acc_over_tasks_wncm": avg_acc_over_tasks_wncm,
        "final_avg_forgetting_wncm": final_avg_forgetting_wncm,
        "final_max_forgetting_wncm": final_max_forgetting_wncm,
        "wncm_top1_curve": wncm_top1_curve,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_metrics_from_logs.py <log_dir>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    # Find all .out log files
    log_files = list(log_dir.glob("*.out"))

    if not log_files:
        print(f"No .out files found in {log_dir}")
        sys.exit(1)

    print("=" * 140)
    print(
        f"{'Log File':<45} {'Dataset':<15} {'Tasks':<6} "
        f"{'Final Acc':<12} {'Avg Acc':<12} {'Avg Fgt':<12} {'Max Fgt':<12} "
        f"{'W-NCM Acc':<12} {'W-NCM Fgt':<12}"
    )
    print("=" * 140)

    results = []
    for log_file in sorted(log_files):
        result = parse_log_file(str(log_file))
        if result:
            results.append(result)

            fa_wncm = result.get("final_avg_acc_wncm")
            ff_wncm = result.get("final_avg_forgetting_wncm")
            if fa_wncm is not None:
                fa_wncm_str = f"{fa_wncm:>10.2f}%"
            else:
                fa_wncm_str = f"{'--':>10}"
            if ff_wncm is not None:
                ff_wncm_str = f"{ff_wncm:>10.2f}%"
            else:
                ff_wncm_str = f"{'--':>10}"

            print(
                f"{result['log_file']:<45} {result['dataset']:<15} {result['num_tasks']:<6} "
                f"{result['final_avg_acc']:>10.2f}% {result['avg_acc_over_tasks']:>10.2f}% "
                f"{result['final_avg_forgetting']:>10.2f}% {result['final_max_forgetting']:>10.2f}% "
                f"{fa_wncm_str} {ff_wncm_str}"
            )
        else:
            print(f"{log_file.name:<45} FAILED TO PARSE")

    print("=" * 100)
    print(f"\nParsed {len(results)} log files successfully.")

    # Print detailed results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)

    for result in results:
        print(f"\n{result['log_file']}")
        print(
            f"  Dataset: {result['dataset']}, Tasks: {result['num_tasks']}, Seed: {result['seed']}"
        )
        print(f"  Final Average Accuracy (CNN): {result['final_avg_acc']:.2f}%")
        print(
            f"  Average Accuracy over Tasks (CNN): {result['avg_acc_over_tasks']:.2f}%"
        )
        print(
            f"  Final Average Forgetting (CNN): {result['final_avg_forgetting']:.2f}%"
        )
        print(f"  Final Max Forgetting (CNN): {result['final_max_forgetting']:.2f}%")
        print(f"  CNN Top1 Curve: {[f'{v:.2f}' for v in result['cnn_top1_curve']]}")
        if result["best_per_task"]:
            print(f"  Best per Task: {[f'{v:.2f}' for v in result['best_per_task']]}")
            print(
                f"  Current per Task: {[f'{v:.2f}' for v in result['current_per_task']]}"
            )
        fa_wncm = result.get("final_avg_acc_wncm")
        if fa_wncm is not None:
            print(
                f"  Final Average Accuracy (W-NCM): {result['final_avg_acc_wncm']:.2f}%"
            )
            print(
                "  Average Accuracy over Tasks (W-NCM): "
                f"{result['avg_acc_over_tasks_wncm']:.2f}%"
            )
            print(
                "  Final Average Forgetting (W-NCM): "
                f"{result['final_avg_forgetting_wncm']:.2f}%"
            )
            print(
                f"  Final Max Forgetting (W-NCM): {result['final_max_forgetting_wncm']:.2f}%"
            )
            if result.get("wncm_top1_curve"):
                print(
                    "  W-NCM Top1 Curve: "
                    f"{[f'{v:.2f}' for v in result['wncm_top1_curve']]}"
                )


if __name__ == "__main__":
    main()
