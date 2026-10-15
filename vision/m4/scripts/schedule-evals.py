#!/usr/bin/env python
# [MODULE DOC] Cron-style CLI that detects crashed evaluation jobs: for every opt_step-* checkpoint under
# checkpoints_path it deletes any start_run_evals_* marker file older than 9 hours so the eval scheduler will
# relaunch the evaluation.

#
# This tool checks if evaluation is needed
#

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


repo_path = Path(__file__).parents[2]

# we have to deal with potentially overlapping slurm jobs running on different nodes, so we can't
# rely on PIDs of a running process. Will use a control file instead as the filesystem is shared.
#
# If that file is there it means:
#
# 1. either the eval is still running
# 2. the eval got aborted (e.g. gpu-oom)
#

# should fine tune - but surely 9h per checkpoint is plenty
reasonable_eval_time_in_secs = 9 * 60 * 60


def run_cmd(cmd, check=True):
    # [FUNCTION DOC] HELPER: runs a command list via subprocess capturing UTF-8 output; returns stripped
    # stdout; with check (default True) failure raises EnvironmentError with stderr. Unused here.
    try:
        response = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=check,
            encoding="utf-8",
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return response


def get_args():
    # [FUNCTION DOC] HELPER: parses positional checkpoints_path; returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_path", type=str, help="base dir with checkpoints")
    return parser.parse_args()


def exit(msg):
    # [FUNCTION DOC] HELPER: prints msg and exits with status 0 (shadows the builtin).
    print(msg)
    sys.exit()


def check_eval_crash(path):
    # [FUNCTION DOC] For checkpoint directory path, removes each of the four start_run_evals_{0_shots,4_shots,
    # perplexity_validation,0_shots_a_la_flamingo} marker files whose mtime is older than
    # reasonable_eval_time_in_secs, printing a message. Returns nothing.
    """Heuristics to decide whether to restart this opt_step-XXX checkpoint evaluation or not"""
    eval_start_paths = map(
        lambda x: path / x,
        [
            "start_run_evals_0_shots",
            "start_run_evals_4_shots",
            "start_run_evals_perplexity_validation",
            "start_run_evals_0_shots_a_la_flamingo",
        ],
    )
    # complicated checks - has another job already started processing? or did it crash?
    for eval_start_path in eval_start_paths:
        if eval_start_path.exists():
            if eval_start_path.stat().st_mtime < time.time() - reasonable_eval_time_in_secs:
                print(f"[Y] {path} looks stale - Probably crashed - Restart evals")
                os.remove(eval_start_path)


def main():
    # [FUNCTION DOC] Entry point: validates checkpoints_path (FileNotFoundError), exits if no checkpoints,
    # sorts them by step number and calls check_eval_crash on each.
    args = get_args()

    checkpoints_path = Path(args.checkpoints_path)
    if not (checkpoints_path.exists() and checkpoints_path.is_dir()):
        raise FileNotFoundError(f"can't find a directory '{checkpoints_path}'")

    checkpoint_dirs = list(checkpoints_path.glob("opt_step-*"))
    if len(checkpoint_dirs) == 0:
        exit("No checkpoints found, exiting")

    # Check each checkpoint folder in real time to allow for overlapping jobs starting at different times
    # Additionally do not delete the last 2 checkpoints
    #
    # sort numerically to sort correctly different number of digits: opt_step-10, opt_step-100
    checkpoint_dirs_sorted = sorted(checkpoint_dirs, key=lambda x: int(str(x).split("-")[-1]))
    for i, checkpoint_dir in enumerate(checkpoint_dirs_sorted):
        print(f"\n*** Checking {checkpoint_dir} for evals")
        check_eval_crash(checkpoint_dir)


if __name__ == "__main__":
    main()
