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

    # Compute metrics
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

    print("=" * 100)
    print(
        f"{'Log File':<45} {'Dataset':<15} {'Tasks':<6} {'Final Acc':<12} {'Avg Acc':<12} {'Avg Fgt':<12} {'Max Fgt':<12}"
    )
    print("=" * 100)

    results = []
    for log_file in sorted(log_files):
        result = parse_log_file(str(log_file))
        if result:
            results.append(result)
            print(
                f"{result['log_file']:<45} {result['dataset']:<15} {result['num_tasks']:<6} "
                f"{result['final_avg_acc']:>10.2f}% {result['avg_acc_over_tasks']:>10.2f}% "
                f"{result['final_avg_forgetting']:>10.2f}% {result['final_max_forgetting']:>10.2f}%"
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
        print(f"  Final Average Accuracy: {result['final_avg_acc']:.2f}%")
        print(f"  Average Accuracy over Tasks: {result['avg_acc_over_tasks']:.2f}%")
        print(f"  Final Average Forgetting: {result['final_avg_forgetting']:.2f}%")
        print(f"  Final Max Forgetting: {result['final_max_forgetting']:.2f}%")
        print(f"  CNN Top1 Curve: {[f'{v:.2f}' for v in result['cnn_top1_curve']]}")
        if result["best_per_task"]:
            print(f"  Best per Task: {[f'{v:.2f}' for v in result['best_per_task']]}")
            print(
                f"  Current per Task: {[f'{v:.2f}' for v in result['current_per_task']]}"
            )


if __name__ == "__main__":
    main()
