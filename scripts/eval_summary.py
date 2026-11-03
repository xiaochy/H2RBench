"""
Compute success/progress statistics from eval_results.csv.

Usage:
    python polaris/scripts/eval_summary.py <run_folder>
    python polaris/scripts/eval_summary.py runs/diffusion_policy_human=5_robot=5
"""

import argparse
import pandas as pd
from pathlib import Path


def infer_num_stages(progress_series: pd.Series) -> int:
    nonzero = progress_series[progress_series > 1e-9]
    if nonzero.empty:
        return 3
    min_prog = nonzero.min()
    n = round(1.0 / min_prog)
    return max(n, 2)


def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        print("No episodes found.")
        return

    avg_progress = df["progress"].mean()
    num_stages = infer_num_stages(df["progress"])

    print(f"Run:      {csv_path.parent}")
    print(f"Episodes: {n}")
    print(f"Stages:   {num_stages} (auto-detected)")
    print(f"Avg progress: {avg_progress:.3f}  ({100*avg_progress:.1f}%)")

    for k in range(1, num_stages):
        threshold = k / num_stages
        count = (df["progress"] >= threshold - 1e-6).sum()
        print(f"Stage {k} (progress≥{threshold:.2f}): {count:3d}/{n}  ({100*count/n:.1f}%)")

    success_count = (df["success"] == True).sum()
    print(f"Stage {num_stages} (success=True):       {success_count:3d}/{n}  ({100*success_count/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder", type=str, help="Path to run folder containing eval_results.csv")
    args = parser.parse_args()

    csv_path = Path(args.run_folder) / "eval_results.csv"
    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return

    summarize(csv_path)


if __name__ == "__main__":
    main()
