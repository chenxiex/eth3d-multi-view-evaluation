#!/usr/bin/env python3
"""Batch evaluation script for the ETH3D multi-view benchmark.

For every <scene_name>.ply found in input_dir, runs ETH3DMultiViewEvaluation
with the corresponding ground truth file at
  gt_dir/<scene_name>/dslr_scan_eval/scan_alignment.mlp

Any additional arguments are forwarded verbatim to ETH3DMultiViewEvaluation.

Example:
    python3 batch_evaluate.py \\
        --input_dir /path/to/reconstructions \\
        --gt_dir /path/to/eth3d_gt \\
        --tolerances 0.01,0.02,0.05,0.1,0.2,0.5
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch ETH3D multi-view evaluation over all scenes in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing <scene_name>.ply reconstruction files.",
    )
    parser.add_argument(
        "--gt_dir",
        required=True,
        help=(
            "Root directory of the ETH3D ground truth data. For each scene the "
            "ground truth MLP file is expected at "
            "<gt_dir>/<scene_name>/dslr_scan_eval/scan_alignment.mlp."
        ),
    )
    # Collect all remaining arguments to forward to ETH3DMultiViewEvaluation.
    args, extra = parser.parse_known_args()
    return args, extra


def find_ply_files(input_dir):
    """Return a sorted list of .ply files found directly in input_dir."""
    ply_files = []
    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.lower().endswith(".ply"):
            ply_files.append(entry.path)
    return sorted(ply_files)


def main():
    args, extra_args = parse_args()

    ply_files = find_ply_files(args.input_dir)
    if not ply_files:
        print(f"No .ply files found in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    overall_success = True
    for ply_path in ply_files:
        scene_name = os.path.splitext(os.path.basename(ply_path))[0]
        gt_mlp_path = os.path.join(
            args.gt_dir, scene_name, "dslr_scan_eval", "scan_alignment.mlp"
        )

        if not os.path.isfile(gt_mlp_path):
            print(
                f"[SKIP] Ground truth not found for scene '{scene_name}': "
                f"{gt_mlp_path}",
                file=sys.stderr,
            )
            overall_success = False
            continue

        cmd = [
            "ETH3DMultiViewEvaluation",
            f"--reconstruction_ply_path={ply_path}",
            f"--ground_truth_mlp_path={gt_mlp_path}",
        ] + extra_args

        print(f"[RUN] {scene_name}")
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[FAIL] ETH3DMultiViewEvaluation exited with code "
                f"{result.returncode} for scene '{scene_name}'.",
                file=sys.stderr,
            )
            overall_success = False

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
