#!/usr/bin/env python
# [MODULE DOC] Cron-style CLI converting every DeepSpeed opt_step-* checkpoint under checkpoints_path into HF
# format by running m4/models/zero_checkpoint_to_hf.py on it. -f/--force reconverts even when a control file
# says a conversion is in progress. A control file unwrapped_model/started-convert-checkpoint guards against
# concurrent jobs on a shared filesystem; files older than 2 hours are considered aborted runs.

#
# This tool converts any deepspeed checkpoints found at given path to hf format
#
# Example:
#
# ./convert-checkpoints.py checkpoints-path
#

import argparse
import subprocess
import sys
import time
from pathlib import Path


repo_path = Path(__file__).parents[2]
zero_checkpoint_to_hf_path = repo_path / "m4/models/zero_checkpoint_to_hf.py"

# we have to deal with potentially overlapping slurm jobs running on different nodes, so we can't
# rely on PIDs of a running process. Will use a control file instead as the filesystem is shared.
#
# If that file is there it means:
#
# 1. either the conversion is still running
# 2. the conversion got aborted (e.g. cpu-oom)
#
# to detect aborted conversions we will check if the control file is older than a reasonable time to perform such a conversion
control_file_name = "started-convert-checkpoint"
# should fine tune - but surely 2h per checkpoint is plenty
reasonable_conversion_time_in_secs = 2 * 60 * 60


def run_cmd(cmd, check=True):
    # [FUNCTION DOC] HELPER: runs a command list via subprocess capturing output as UTF-8 and returns stripped
    # stdout; with check (default True) a non-zero exit raises EnvironmentError containing stderr.
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
    # [FUNCTION DOC] HELPER: parses positional checkpoints_path and -f/--force; returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_path", type=str, help="base dir with checkpoints")
    parser.add_argument("-f", "--force", action="store_true", help="force rebuilding of all checkpoints")
    return parser.parse_args()


def exit(msg):
    # [FUNCTION DOC] HELPER: prints msg and exits the process with status 0 (shadows the builtin exit).
    print(msg)
    sys.exit()


def should_process(path, force, control_file_path):
    # [FUNCTION DOC] Decides whether to convert checkpoint dir path. Returns False when unwrapped_model or its
    # config.json is missing, or when any converted weights file (pytorch_model.bin[.index.json] /
    # model.safetensors[.index.json]) already exists. Then returns True if force; otherwise True if the
    # control file is absent or stale (older than reasonable_conversion_time_in_secs), False if it is fresh.
    # Prints a [Y]/[N] reason each time.
    # KNOWN ISSUE: the control_file_path argument is ignored and recomputed; the force message lacks the
    # f-prefix so it prints the literal "{checkpoint_dir}".
    """Heuristics to decide whether to convert this opt_step-XXX checkpoint or not"""

    target_dir = path / "unwrapped_model"
    config_file = target_dir / "config.json"
    
    # check if target directory exists
    if not target_dir.exists():
        print(f"[N] {path} target directory 'unwrapped_model' doesn't exist. Skipping")
        return False

    # check if config.json exists
    if not config_file.exists():
        print(f"[N] {path} config.json doesn't exist. Skipping")
        return False

    # easy checks - the conversion is clearly completed
    checklist = [
        target_dir / "pytorch_model.bin.index.json",
        target_dir / "pytorch_model.bin",
        target_dir / "model.safetensors.index.json",
        target_dir / "model.safetensors",
    ]
    checklist = [model_path.exists() for model_path in checklist]
    if any(checklist):
        print(f"[N] {path} appears to be already converted. Skipping")
        return False

    if force:
        print("[Y] Forced to re-convert {checkpoint_dir}")
        return True

    # complicated checks - has another job already started processing? or did it crash?
    control_file_path = target_dir / control_file_name
    if control_file_path.exists():
        if control_file_path.stat().st_mtime < time.time() - reasonable_conversion_time_in_secs:
            print(f"[Y] {path} looks stale - probably aborted job. Re-converting")
            return True
        else:
            print(
                f"[N] {path} either another job is converting it or less than"
                f" {reasonable_conversion_time_in_secs} secs has passed since it was launched. Skipping"
            )
            return False
    else:
        print(f"[Y] {path} is a new checkpoint. Converting")
        return True


def main():
    # [FUNCTION DOC] Entry point: validates checkpoints_path (FileNotFoundError if not a dir), exits if no
    # opt_step-* dirs, and for each approved checkpoint touches the control file, runs the converter script
    # synchronously (EnvironmentError on failure, leaving the control file behind), removes the control file
    # and prints the converter output.
    args = get_args()

    checkpoints_path = Path(args.checkpoints_path)
    if not (checkpoints_path.exists() and checkpoints_path.is_dir()):
        raise FileNotFoundError(f"can't find a directory '{checkpoints_path}'")

    checkpoint_dirs = list(checkpoints_path.glob("opt_step-*"))
    if len(checkpoint_dirs) == 0:
        exit("No checkpoints found, exiting")

    # Check each folder in real time to allow for overlapping jobs starting at different times
    for checkpoint_dir in checkpoint_dirs:
        print(f"\n*** Checking {checkpoint_dir}")

        control_file_path = checkpoint_dir / "unwrapped_model" / control_file_name

        if not should_process(checkpoint_dir, args.force, control_file_path):
            continue

        print(f"Launching conversion for {checkpoint_dir} - it could take a long time")
        cmd = [zero_checkpoint_to_hf_path, checkpoint_dir]
        # we could use flock here, to avoid a race condition, but it'd be pointless since each
        # cronjob is likely to run on a different node and flock only works within a single node
        control_file_path.touch()
        response = run_cmd(cmd)
        control_file_path.unlink()
        print(response)


if __name__ == "__main__":
    main()
