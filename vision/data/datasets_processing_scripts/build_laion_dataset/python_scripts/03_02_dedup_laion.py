# [MODULE DOC] LAION step 3.2: removes duplicates from one filtered LAION shard
# (laion_dataset_filtered/IDX_JOB, argv[1]) using
# the pickled set of duplicate MD5s (set_dup_md5.pkl from S3, built by 03_01). Recreates the temp dir
# /scratch/storage_hugo_IDX_JOB/, syncs input (3 attempts), filters with NUM_PROC=10, saves, syncs to
# laion_dataset_filtered_dedup/IDX_JOB, logs counts before/after and deletes the temp dir.
import json
import logging
import os
import pickle
import sys

from datasets import load_from_disk
from PIL import Image, ImageFile


# Useful to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None
# Load even truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_LAION_DATASET_FILTERED_S3 = f"s3://m4-datasets/LAION_data/laion_dataset_filtered/{IDX_JOB}"
PATH_LAION_DATASET_FILTERED_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "laion_dataset_filtered")

PATH_SET_DUP_MD5_S3 = "s3://m4-datasets/trash/set_dup_md5.pkl"
PATH_SET_DUP_MD5_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "set_dup_md5.pkl")
NUM_PROC = 10

PATH_SAVE_DISK_LAION_DATASET_FILTERED_DEDUP = os.path.join(PATH_SAVE_DISK_TMP_FILES, "laion_dataset_filtered_dedup")
PATH_SAVE_S3_LAION_DATASET_FILTERED_DEDUP = os.path.join(
    "s3://m4-datasets/LAION_data/laion_dataset_filtered_dedup", str(IDX_JOB)
)


class LAIONDeduplication:
    # [CLASS DOC] Picklable Dataset.filter predicate holding the set of duplicate MD5s.
    def __init__(self, path_set_dup_md5):
        # [FUNCTION DOC] Stores path_set_dup_md5 and unpickles the set from it.
        self.path_set_dup_md5 = path_set_dup_md5
        with open(path_set_dup_md5, "rb") as f:
            self.set_dup_md5 = pickle.load(f)

    def __call__(self, example):
        # [FUNCTION DOC] Parses example["meta"] JSON and returns False if its "md5" is in the duplicate set,
        # True otherwise.
        md5 = json.loads(example["meta"])["md5"]
        if md5 in self.set_dup_md5:
            return False
        return True

    def __reduce__(self):
        # [FUNCTION DOC] Pickles as (class, (path,)) so worker processes reload the set from disk.
        return self.__class__, (self.path_set_dup_md5,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading and loading the LAION dataset")
    command_sync_s3 = f"aws s3 sync {PATH_LAION_DATASET_FILTERED_S3} {PATH_LAION_DATASET_FILTERED_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    laion_dataset = load_from_disk(PATH_LAION_DATASET_FILTERED_LOCAL)
    num_pairs_before_dedup = laion_dataset.num_rows
    logger.info("Finished downloading and loading the LAION dataset")

    logger.info("Starting deduplicating the LAION dataset")
    command_sync_s3 = f"aws s3 cp {PATH_SET_DUP_MD5_S3} {PATH_SET_DUP_MD5_LOCAL}"
    os.system(command_sync_s3)

    laion_deduplication = LAIONDeduplication(path_set_dup_md5=PATH_SET_DUP_MD5_LOCAL)
    laion_dataset = laion_dataset.filter(laion_deduplication, num_proc=NUM_PROC)
    logger.info("Finished deduplicating the LAION dataset")

    logger.info("Starting saving the LAION dataset filtered deduplicated")
    laion_dataset.save_to_disk(PATH_SAVE_DISK_LAION_DATASET_FILTERED_DEDUP, num_proc=NUM_PROC)

    command_sync_s3 = (
        f"aws s3 sync {PATH_SAVE_DISK_LAION_DATASET_FILTERED_DEDUP} {PATH_SAVE_S3_LAION_DATASET_FILTERED_DEDUP}"
    )
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the LAION dataset filtered deduplicated")

    logger.info(f"Number of pairs in the LAION dataset before the deduplication: {num_pairs_before_dedup}")
    logger.info(f"Number of pairs in the LAION dataset after the deduplication: {laion_dataset.num_rows}")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
