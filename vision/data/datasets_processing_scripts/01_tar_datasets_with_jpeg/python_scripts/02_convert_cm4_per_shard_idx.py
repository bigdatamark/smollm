# [MODULE DOC] Converts CM4 (web documents) HF dataset shards into webdataset tar files.
# Configures INFO logging and disables datasets caching at import time. As main: parses args and runs main.
import argparse
import logging
from pathlib import Path

from datasets import set_caching_enabled

from m4.training.types import DatasetTypes
from m4.utils.datasets.create_webdataset_tar import export_dataset_shard_idx_to_tar


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

set_caching_enabled(False)


def get_args():
    # [FUNCTION DOC] Parses CLI args: --shard_dir_path (dir of saved dataset dirs), --saving_dir,
    # --num_examples_per_shard, --num_proc, --shard_idx (all required), --min_num_shards (optional). Returns
    # the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir_path", type=Path, required=True)
    parser.add_argument("--saving_dir", type=Path, required=True)
    parser.add_argument("--num_examples_per_shard", type=int, required=True)
    parser.add_argument("--num_proc", type=int, required=True)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--min_num_shards", type=int)
    args = parser.parse_args()
    return args


def main(args):
    # [FUNCTION DOC] Uses every subdirectory of shard_dir_path as a dataset path and calls
    # export_dataset_shard_idx_to_tar (deprecated wrapper, exports only shard number shard_idx) with type
    # WEB_DOCUMENTS.
    shard_1_dirs = [shard_dir for shard_dir in args.shard_dir_path.iterdir()]
    ds_type = DatasetTypes.WEB_DOCUMENTS

    export_dataset_shard_idx_to_tar(
        hf_datasets_paths=shard_1_dirs,
        saving_dir=args.saving_dir,
        ds_type=ds_type,
        num_examples_per_shard=args.num_examples_per_shard,
        num_proc=args.num_proc,
        shard_idx=args.shard_idx,
        min_num_shards=args.min_num_shards,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
