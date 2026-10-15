# [MODULE DOC] CLI script that merges a training run's W&B history with evaluation results from JSONL files and logs the
# combined per-opt-step rows into a run of the same name in the evaluation W&B project (creating or resuming it).
# Args: --wandb_entity (huggingfacem4), --wandb_training_project (VLOOM), --wandb_eval_project (VLOOM-evals),
# --evaluation_jsonl_files (one or more paths, required), --run_name_to_log (required). Network side effects.
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from time import sleep

import numpy as np
import wandb


logger = logging.getLogger(__name__)

OPT_STEP_LOG = "num_opt_steps"
WANDB_TIMEOUT = 30


def get_args():
    # [FUNCTION DOC] HELPER: build and parse the argparse options described in the module doc; returns the Namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str, default="huggingfacem4")
    parser.add_argument("--wandb_training_project", type=str, default="VLOOM")
    parser.add_argument("--wandb_eval_project", type=str, default="VLOOM-evals")
    parser.add_argument(
        "--evaluation_jsonl_files",
        nargs="+",
        type=Path,
        help="Path to the json files containing evaluation results",
        required=True,
    )
    parser.add_argument("--run_name_to_log", type=str, required=True)

    args = parser.parse_args()
    return args


def main():
    # [FUNCTION DOC] Fetch the unique (tag-filtered) training run and the optional existing eval run by name; read already logged
    # eval rows by opt step; parse all JSONL evaluations (opt step from "opt_step-N" in model_name_or_path, metrics
    # from eval(score), keys "<task>-<metric>", "_distribution" lists as wandb.Histogram, numbers as is, ValueError
    # otherwise); merge them into the training history rows (filtered of gradients/parameters/private columns),
    # adding rows for opt steps without training logs; check consistency with already logged values (ValueError on
    # mismatch); then wandb.init the eval run and log every row.
    # KNOWN ISSUE: scores are parsed with eval on file contents (arbitrary code execution).
    logging.basicConfig(
        level=logging.INFO,
        format=" - %(process)d - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    args = get_args()

    logger.info(f"args: {args}")

    api = wandb.Api(timeout=WANDB_TIMEOUT)

    def filter_out_tags(tags):
        # [FUNCTION DOC] HELPER: False if tags contain debug/failed/killed or any tag containing "job_id", else True.
        if "debug" in tags or "failed" in tags or "killed" in tags:
            return False
        for t in tags:
            if "job_id" in t:
                return False
        return True

    def fetch_training_run(training_run_name):
        # [FUNCTION DOC] HELPER: return the single run named training_run_name in the training project after tag filtering; asserts
        # exactly one match.
        """
        Fetch training run. There can only be one corresponding training run.
        If not, double check the tags (killed, failed, etc.)
        """
        matching_runs = []

        runs = api.runs(f"{args.wandb_entity}/{args.wandb_training_project}")
        for run in runs:
            if run.name == training_run_name:
                matching_runs.append(run)

        matching_runs = [r for r in matching_runs if filter_out_tags(r.tags)]
        assert len(matching_runs) == 1, f"There are 0 or more than 1 matching runs: {matching_runs}"

        return matching_runs[0]

    def fetch_evaluation_run(evaluation_run_name):
        # [FUNCTION DOC] HELPER: return the run named evaluation_run_name in the eval project after tag filtering, or None; asserts at
        # most one match.
        """
        Fetch evaluation run. There can only be one corresponding evaluation run at most.
        If not, double check the tags (killed, failed, etc.)
        """
        matching_runs = []

        runs = api.runs(f"{args.wandb_entity}/{args.wandb_eval_project}")
        for run in runs:
            if run.name == evaluation_run_name:
                matching_runs.append(run)

        matching_runs = [r for r in matching_runs if filter_out_tags(r.tags)]
        assert len(matching_runs) <= 1, f"There are more than 2 matching runs: {matching_runs}"

        if len(matching_runs) == 0:
            return None
        else:
            return matching_runs[0]

    training_run = fetch_training_run(args.run_name_to_log)
    logger.info("Successfully fetched the training run.")
    evaluation_run = fetch_evaluation_run(args.run_name_to_log)
    logger.info("Successfully fetched the (potentially `None`) evaluation run.")

    def get_logged_eval_values(evaluation_run):
        # [FUNCTION DOC] HELPER: mapping opt step -> full history row of the existing eval run (empty if evaluation_run is None).
        """
        If `evaluation_run` already exists, get the already logged values into a dictionary.
        """
        logged_evaluation_values = defaultdict()

        if evaluation_run is not None:
            for row in evaluation_run.scan_history():
                opt_step = row[OPT_STEP_LOG]
                logged_evaluation_values[opt_step] = row
        return logged_evaluation_values

    already_logged_eval_values = get_logged_eval_values(evaluation_run)
    logger.info(f"LOGGED_VALUES: {already_logged_eval_values}")

    def get_evaluations_values_from_json():
        # [FUNCTION DOC] HELPER: parse all evaluation JSONL files into a nested mapping opt step -> {"<task>-<metric>": value} (see main).
        """
        Load all values from the json file
        """
        evaluation_values = defaultdict(lambda: defaultdict())
        for evaluation_jsonl_file in args.evaluation_jsonl_files:
            with open(evaluation_jsonl_file, "r") as f:
                for line in f.readlines():
                    evaluation = json.loads(line)
                    opt_step = int(evaluation["model_name_or_path"].split("/opt_step-")[1].split("/")[0])
                    task = evaluation["task"]

                    for metric, value in eval(evaluation["score"]).items():
                        metric_name = f"{task}-{metric}"
                        if "_distribution" in metric_name:
                            assert isinstance(
                                value, list
                            ), f"Don't know how to handle metric {metric_name} of type {type(value)} | {value}"
                            evaluation_values[opt_step][metric_name] = wandb.Histogram(value)
                        elif isinstance(value, float) or isinstance(value, int):
                            evaluation_values[opt_step][metric_name] = value
                        else:
                            raise ValueError(
                                f"Don't know how to handle metric {metric_name} of type {type(value)} | {value}"
                            )
        return evaluation_values

    evaluation_metrics = get_evaluations_values_from_json()
    logger.info(f"Evaluation values: {evaluation_metrics}")

    def filter_out_columns(row):
        # [FUNCTION DOC] HELPER: drop keys containing "gradients/" or "parameters/" or starting with "_" from a history row.
        return {
            k: v
            for k, v in row.items()
            if ("gradients/" not in k and "parameters/" not in k and not k.startswith("_"))
        }

    def convert_training_run_to_dict(training_run):
        # [FUNCTION DOC] HELPER: mapping opt step -> filtered history row for rows having num_opt_steps; asserts one row per opt step.
        """
        Get all the logged values from the training into a dictionary.
        """
        training_history = training_run.scan_history()
        d = defaultdict(dict)
        for row in training_history:
            if "num_opt_steps" not in row:
                continue
            row = filter_out_columns(row)
            opt_step = row[OPT_STEP_LOG]
            assert opt_step not in d, (
                f"The current code does not support having multiple entries for a single `opt_step` ({opt_step})."
                " Please double check what's happening, and if necessary, support this case (for instance by only"
                " considering the entry with the last timestamp.)"
            )
            d[opt_step] = row
        return d

    training_dict = convert_training_run_to_dict(training_run)
    # Add values from json file to the `training_dict`
    for opt_step, eval_metrics_for_opt_step in evaluation_metrics.items():
        if opt_step in training_dict:
            for k, v in eval_metrics_for_opt_step.items():
                assert k not in training_dict[opt_step]
                training_dict[opt_step][k] = v
        else:
            # This case only happens when we are saving a checkpoint without logging metrics on wandb.
            # If `train_saving_opt_steps` is a multiple of `wandb_log_freq`, this will happens when we enter the
            # manual exit conditions, and then evaluate this checkpoint.
            training_dict[opt_step] = eval_metrics_for_opt_step
            training_dict[opt_step][OPT_STEP_LOG] = opt_step

    # Go through the `training_dict` and check for compatibilities with already logged runs
    if evaluation_run is not None:
        for opt_step, training_metrics_for_opt_step in training_dict.items():
            if opt_step not in already_logged_eval_values:
                continue
            for metric_name, metric_value in training_metrics_for_opt_step.items():
                if metric_name in already_logged_eval_values[opt_step]:
                    print("already logged")
                    if isinstance(metric_value, wandb.Histogram):
                        if already_logged_eval_values[opt_step][metric_name]["_type"] != "histogram":
                            msg = (
                                "You are trying to log a histogram but the metric logged previously is not a"
                                " histogram: YOU SHOULD CHECK!"
                            )
                            raise ValueError(msg)
                        elif (
                            metric_value.to_json()["values"]
                            != already_logged_eval_values[opt_step][metric_name]["values"]
                        ):
                            msg = (
                                "values already logged are different from the new ones \nBef:"
                                f" {already_logged_eval_values[opt_step][metric_name]['values']}\nAft:"
                                f" {metric_value.to_json()['values']}"
                            )
                            raise ValueError(msg)
                    elif (
                        already_logged_eval_values[opt_step][metric_name] != metric_value
                        and metric_value is not None
                        and not np.isnan(metric_value)
                    ):
                        raise ValueError("YOU SHOULD CHECK!!")

    def get_wandb_logger(evaluation_run):
        # [FUNCTION DOC] HELPER: wandb.init in the eval project with the run name, reusing evaluation_run.id if it exists or a freshly
        # generated id otherwise (resume=None, allow_val_change). Returns the run.
        """
        Init the wandb logger.
        """
        if evaluation_run is not None:
            print("Resuming wandb run")
            wandb_logger = wandb.init(
                resume=None,
                project=args.wandb_eval_project,
                entity=args.wandb_entity,
                name=args.run_name_to_log,
                allow_val_change=True,
                id=evaluation_run.id,
            )
        else:
            wandb_id = wandb.util.generate_id()
            print(f"Creating wandb run with id {wandb_id}")
            wandb_logger = wandb.init(
                resume=None,
                project=args.wandb_eval_project,
                entity=args.wandb_entity,
                name=args.run_name_to_log,
                allow_val_change=True,
                id=wandb_id,
            )
        return wandb_logger

    wandb_logger = get_wandb_logger(evaluation_run)
    for v in training_dict.values():
        assert OPT_STEP_LOG in v
        wandb_logger.log(v)
    sleep(1)
    wandb.finish(quiet=True)

    logger.info("Finished wandb sync")


if __name__ == "__main__":
    main()
