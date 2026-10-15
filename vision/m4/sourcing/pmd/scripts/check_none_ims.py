# [MODULE DOC] CLI script: count rows whose "image" column is None in a sharded on-disk dataset. Expects
# `--dataset_folder` to contain one sub-folder per split, each containing shards saved with
# `save_to_disk`; `--num_proc` (default 1) sets filter parallelism. For every split it loads each shard,
# filters to the None-image rows, and prints the split's total size, the count without image and the last
# 10 such rows. Sets PIL `ImageFile.LOAD_TRUNCATED_IMAGES = True` globally at import.
# KNOWN ISSUE: the "last 10 instances" print uses only the last shard's filtered rows, not the whole split.
import argparse
import os
from typing import Dict

from datasets import load_from_disk
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_example_with_none_image(batch: Dict) -> Dict:
    # [FUNCTION DOC] HELPER: batched filter predicate; given a batch mapping with an "image" list, return a list of booleans that
    # are True where the image is None. (Annotated return type says Dict but a list is returned.)
    result_bools = [im is None for im in batch["image"]]
    return result_bools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count the number of rows with `image==None`")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to folder containing the dataset. Example: `$cnw_ALL_CCFRSCRATCH/general_pmd/image/wit`",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processed for multiprocessing (in particular calls to `map`s).",
    )
    args = parser.parse_args()

    for split_name in os.listdir(args.dataset_folder):
        split_size = 0
        split_filtered_size = 0
        for shard_name in sorted(os.listdir(os.path.join(args.dataset_folder, split_name))):
            print(f"Doing: {os.path.join(args.dataset_folder, split_name, shard_name)}")
            shard = load_from_disk(os.path.join(args.dataset_folder, split_name, shard_name))
            shard_no_ims = shard.filter(
                get_example_with_none_image,
                batched=True,
                batch_size=250,
                num_proc=args.num_proc,
            )
            split_size += len(shard)
            split_filtered_size += len(shard_no_ims)
        print(f"Dataset: {args.dataset_folder}")
        print(f"---Split name: {split_name}")
        print(f"------Initial size: {split_size}")
        print(f"------Nb instances without image: {split_filtered_size}")
        print(f"------Last 10 instances with no images: {shard_no_ims[-10:]}\n")
