# [MODULE DOC] Builds one shard of the LAION-COCO dataset from a raw LAION HF-dataset shard. argv[1] =
# ID_SHARD (string).
# As __main__: s5cmd-syncs s3://m4-datasets/LAION_data/laion_raw_hf_dataset/ID_SHARD into
# /scratch/storage_hugo/, keeps only rows whose URL is in the pickled LAION-COCO URL->caption mapping
# (PATH_LAION_COCO_IMAGEURLS_TO_CAPTIONS), replaces captions with the LAION-COCO ones, saves (40 procs), syncs
# to
# s3://m4-datasets/LAION_data/laion_coco_dataset/ID_SHARD/ and deletes the temp dir. Logs each step; PIL size
# and truncation checks disabled.
# KNOWN ISSUE: the temp dir /scratch/storage_hugo/ is not shard-specific, so concurrent jobs on one node would
# clobber each other (the final rm -rf deletes all of them).
import json
import logging
import os
import pickle
import sys

from datasets import load_from_disk
from PIL import Image, ImageFile


# Useful to avoid DecompressionBombError and truncated image error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ID_SHARD = sys.argv[1]

PATH_S3_RAW_LAION_SHARD = f"s3://m4-datasets/LAION_data/laion_raw_hf_dataset/{ID_SHARD}/*"
PATH_SAVE_DISK_TMP_FILES = "/scratch/storage_hugo/"
PATH_SAVE_DISK_RAW_LAION_SHARD = os.path.join(PATH_SAVE_DISK_TMP_FILES, f"raw_laion_shard_{ID_SHARD}/")

PATH_LAION_COCO_IMAGEURLS_TO_CAPTIONS = "/fsx/hugo/prepare_laion_coco/laion_coco_imageurls_to_captions.pkl"

NUM_PROC_MODIF = 1
NUM_PROC_SAVE = 40

PATH_SAVE_DISK_LAION_COCO_SHARD = os.path.join(PATH_SAVE_DISK_TMP_FILES, f"laion_coco_shard_{ID_SHARD}/")
PATH_SAVE_S3_LAION_COCO_SHARD = f"s3://m4-datasets/LAION_data/laion_coco_dataset/{ID_SHARD}/"


class LAIONCOCOFiltering:
    # [CLASS DOC] Picklable Dataset.filter predicate keeping only LAION rows whose image URL is in LAION-COCO.
    def __init__(self, path_laion_coco_imageurls_to_captions):
        # [FUNCTION DOC] Stores the path and unpickles the mapping image URL -> LAION-COCO caption from it.
        self.path_laion_coco_imageurls_to_captions = path_laion_coco_imageurls_to_captions
        with open(path_laion_coco_imageurls_to_captions, "rb") as f:
            self.laion_coco_imageurls_to_captions = pickle.load(f)

    def __call__(self, example):
        # [FUNCTION DOC] Parses example["meta"] (JSON string) and returns True iff its "url" is a key of the
        # mapping.
        image_url = json.loads(example["meta"])["url"]
        if image_url not in self.laion_coco_imageurls_to_captions:
            return False
        return True

    def __reduce__(self):
        # [FUNCTION DOC] Pickles as (class, (path,)) so workers reload the mapping from disk.
        return self.__class__, (self.path_laion_coco_imageurls_to_captions,)


class LAIONCOCOReplacement:
    # [CLASS DOC] Picklable Dataset.map function replacing LAION alt-texts with LAION-COCO synthetic captions
    # (uses __slots__).
    __slots__ = (
        "path_laion_coco_imageurls_to_captions",
        "laion_coco_imageurls_to_captions",
    )

    def __init__(self, path_laion_coco_imageurls_to_captions):
        # [FUNCTION DOC] Stores the path and unpickles the mapping image URL -> LAION-COCO caption from it.
        self.path_laion_coco_imageurls_to_captions = path_laion_coco_imageurls_to_captions
        with open(path_laion_coco_imageurls_to_captions, "rb") as f:
            self.laion_coco_imageurls_to_captions = pickle.load(f)

    def __call__(self, example):
        # [FUNCTION DOC] Sets example["source"] = "laion_coco" and example["text"] = caption for the URL in
        # example["meta"]; returns the
        # example. Raises KeyError for URLs not in the mapping (rows are expected to be pre-filtered).
        example["source"] = "laion_coco"
        image_url = json.loads(example["meta"])["url"]
        laion_coco_caption = self.laion_coco_imageurls_to_captions[image_url]
        example["text"] = laion_coco_caption
        return example

    def __reduce__(self):
        # [FUNCTION DOC] Pickles as (class, (path,)) so workers reload the mapping from disk.
        return self.__class__, (self.path_laion_coco_imageurls_to_captions,)


if __name__ == "__main__":
    logger.info("Starting downloading the raw laion shard")
    command_sync_s3 = f"s5cmd sync {PATH_S3_RAW_LAION_SHARD} {PATH_SAVE_DISK_RAW_LAION_SHARD}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the raw laion shard")

    logger.info("Starting loading the raw laion shard")
    ds = load_from_disk(PATH_SAVE_DISK_RAW_LAION_SHARD)
    logger.info("Finished loading the raw laion shard")

    logger.info("Starting filtering the raw laion shard to obtain only examples in laion coco")
    laion_coco_filtering = LAIONCOCOFiltering(
        path_laion_coco_imageurls_to_captions=PATH_LAION_COCO_IMAGEURLS_TO_CAPTIONS
    )
    ds = ds.filter(laion_coco_filtering, num_proc=NUM_PROC_MODIF)
    del laion_coco_filtering
    logger.info("Finished filtering the raw laion shard to obtain only examples in laion coco")

    logger.info("Starting modifying the raw laion shard to replace the captions by the ones of laion coco")
    laion_coco_replacement = LAIONCOCOReplacement(
        path_laion_coco_imageurls_to_captions=PATH_LAION_COCO_IMAGEURLS_TO_CAPTIONS
    )
    ds = ds.map(laion_coco_replacement, num_proc=NUM_PROC_MODIF)
    logger.info("Finished modifying the raw laion shard to replace the captions by the ones of laion coco")

    logger.info("Starting saving the laion coco shard")
    ds.save_to_disk(PATH_SAVE_DISK_LAION_COCO_SHARD, num_proc=NUM_PROC_SAVE)
    logger.info("Finished saving the laion coco shard")

    logger.info("Starting uploading the laion coco shard to S3")
    command_sync_s3 = f"s5cmd sync {PATH_SAVE_DISK_LAION_COCO_SHARD} {PATH_SAVE_S3_LAION_COCO_SHARD}"
    os.system(command_sync_s3)
    logger.info("Finished uploading the laion coco shard to S3")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
