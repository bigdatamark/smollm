# [MODULE DOC] Merges the 5,000 small raw LAION "1/4" shards (s3://m4-datasets/general_pmd/image/train/<idx>/,
# HF arrow
# datasets) into NUM_BIG_SHARDS_TO_CREATE=50 big shards without re-encoding. argv[1] = ID_JOB selects one
# contiguous chunk of ~100 small shard indices. As __main__: for each small shard, s5cmd-syncs it into a temp
# dir,
# renames each "data-*" arrow file to data-<random 8 chars>.arrow and moves it into the big-shard folder
# (failures
# of a shard are logged and skipped); then writes a hand-made state.json (listing all files in the folder,
# fixed
# fingerprint, split "train") and an empty dataset_info.json so load_from_disk can open it, syncs the folder
# to
# s3://m4-datasets/LAION_data/laion_raw_hf_dataset_1_4/ID_JOB/ and deletes /scratch/storage_hugo/ID_JOB/.
# KNOWN ISSUE: os.system failures are not exceptions, so a failed sync is silently skipped rather than logged;
# leftover non-arrow files in the temp dir from a previous shard can be reused.
import json
import logging
import os
import random
import string
import sys

import numpy as np
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NUM_SMALL_SHARDS_LAION_1_4 = 5_000
NUM_BIG_SHARDS_TO_CREATE = 50
ID_JOB = int(sys.argv[1])  # One job per big shard to create
IDX = [el.tolist() for el in np.array_split(list(range(NUM_SMALL_SHARDS_LAION_1_4)), NUM_BIG_SHARDS_TO_CREATE)][ID_JOB]

PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo/{ID_JOB}/"
PATH_S3_RAW_LAION_SHARDS_1_4 = "s3://m4-datasets/general_pmd/image/train/"
PATH_SAVE_DISK_TMP_DIR = os.path.join(PATH_SAVE_DISK_TMP_FILES, "tpm_dir/")
PATH_SAVE_DISK_RAW_LAION_BIG_SHARD = os.path.join(PATH_SAVE_DISK_TMP_FILES, f"raw_laion_big_shard_{ID_JOB}/")
PATH_SAVE_S3_RAW_LAION_BIG_SHARD = f"s3://m4-datasets/LAION_data/laion_raw_hf_dataset_1_4/{ID_JOB}/"


def generate_random_string(length=8):
    # [FUNCTION DOC] HELPER: returns a random string of `length` (default 8) ASCII letters/digits (uses the
    # global random module).
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


if __name__ == "__main__":
    os.system(f"mkdir -p {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir -p {PATH_SAVE_DISK_RAW_LAION_BIG_SHARD}")

    logger.info("Starting downloading the small shards and renaming them")
    for idx in tqdm(IDX):
        try:
            path_s3_raw_shard_to_download = os.path.join(PATH_S3_RAW_LAION_SHARDS_1_4, str(idx), "*")
            command_sync_s3 = f"s5cmd sync {path_s3_raw_shard_to_download} {PATH_SAVE_DISK_TMP_DIR}"
            os.system(command_sync_s3)
            for filename in os.listdir(PATH_SAVE_DISK_TMP_DIR):
                if filename.startswith("data-"):
                    new_name = f"data-{generate_random_string()}.arrow"
                    old_file = os.path.join(PATH_SAVE_DISK_TMP_DIR, filename)
                    new_file = os.path.join(PATH_SAVE_DISK_TMP_DIR, new_name)
                    os.rename(old_file, new_file)
                    move_cmd = f"mv '{new_file}' '{PATH_SAVE_DISK_RAW_LAION_BIG_SHARD}'"
                    os.system(move_cmd)
        except Exception:
            logger.info(f"Failed for the small shard {idx}")

    data_files = [{"filename": filename} for filename in os.listdir(PATH_SAVE_DISK_RAW_LAION_BIG_SHARD)]
    state_dict = {
        "_data_files": data_files,
        "_fingerprint": "ca6fc44a72ef9720",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
    }
    with open(os.path.join(PATH_SAVE_DISK_RAW_LAION_BIG_SHARD, "state.json"), "w") as f:
        json.dump(state_dict, f)

    data_info = {}
    with open(os.path.join(PATH_SAVE_DISK_RAW_LAION_BIG_SHARD, "dataset_info.json"), "w") as f:
        json.dump(data_info, f)
    logger.info("Finished downloading the small shards and renaming them")

    logger.info("Starting uploading the big shard to S3")
    command_sync_s3 = f"s5cmd sync {PATH_SAVE_DISK_RAW_LAION_BIG_SHARD} {PATH_SAVE_S3_RAW_LAION_BIG_SHARD}"
    os.system(command_sync_s3)
    logger.info("Finished uploading the big shard to S3")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
