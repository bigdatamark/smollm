# [MODULE DOC] Web-docs step 6.1: counts image URLs in one filtered web-document shard. As __main__: recreates
# /scratch/storage_hugo_<idx_job>/, syncs the shard from S3 (3 attempts), keeps only the "metadata" column
# (JSON
# list aligned with the document nodes, with a dict containing "src" for image nodes and null for text),
# extracts
# every image "src", builds a Counter URL -> number of occurrences, pickles it (highest protocol) and copies
# it to
# <path_save_image_urls_in_web_document_dataset_filtered>/<idx_job>/image_urls_in_web_document_dataset_filtere
# d.pickle,
# then deletes the temp dir.
import argparse
import json
import logging
import os
import pickle
from collections import Counter
from multiprocessing import cpu_count

from datasets import load_from_disk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    # [FUNCTION DOC] HELPER: parses idx_job (int, positional), --path_web_document_dataset_filtered (S3 input
    # dir),
    # --path_save_image_urls_in_web_document_dataset_filtered (S3 output dir) and --num_proc (cpu_count,
    # unused).
    parser = argparse.ArgumentParser(description="Create the set of image urls in the web document dataset.")
    parser.add_argument(
        "idx_job",
        type=int,
        help="Index of the job (between 0 and 199).",
    )
    parser.add_argument(
        "--path_web_document_dataset_filtered",
        type=str,
        default="s3://m4-datasets/webdocs/web_document_dataset_filtered/",
        help="Path of the web document dataset filtered.",
    )
    parser.add_argument(
        "--path_save_image_urls_in_web_document_dataset_filtered",
        type=str,
        default="s3://m4-datasets/webdocs/image_urls_in_web_document_dataset_filtered/",
        help="Path to save the image URLs in the web document dataset filtered.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    path_save_disk_tmp_files = f"/scratch/storage_hugo_{args.idx_job}/"
    if os.path.exists(path_save_disk_tmp_files):
        os.system(f"rm -r {path_save_disk_tmp_files}")
    os.system(f"mkdir {path_save_disk_tmp_files}")

    logger.info("Starting loading the web document dataset filtered")
    path_sync_s3 = os.path.join(args.path_web_document_dataset_filtered, str(args.idx_job))
    path_save_disk_web_document_dataset_filtered = os.path.join(
        path_save_disk_tmp_files, "web_document_dataset_filtered"
    )
    os.system(f"mkdir {path_save_disk_web_document_dataset_filtered}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_web_document_dataset_filtered}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_document_dataset_filtered = load_from_disk(path_save_disk_web_document_dataset_filtered)
    logger.info("Finished loading the web document dataset filtered")

    logger.info("Starting making the set of image URLs in the web document dataset filtered")
    web_document_dataset_filtered = web_document_dataset_filtered.remove_columns(
        [c_n for c_n in web_document_dataset_filtered.column_names if c_n != "metadata"]
    )
    metadata = web_document_dataset_filtered["metadata"]
    logger.info("Step 1 done")
    metadata = [[el["src"] for el in json.loads(md) if el] for md in metadata]
    logger.info("Step 2 done")
    metadata = [sub_el for el in metadata for sub_el in el]
    metadata = Counter(metadata)
    logger.info("Finished making the set of image URLs in the web document dataset filtered")

    logger.info("Starting saving the set of image URLs in the web document dataset filtered")
    path_save_disk_image_urls_in_web_document_dataset_filtered = os.path.join(
        path_save_disk_tmp_files, "image_urls_in_web_document_dataset_filtered.pickle"
    )
    with open(path_save_disk_image_urls_in_web_document_dataset_filtered, "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

    path_sync_s3 = os.path.join(
        args.path_save_image_urls_in_web_document_dataset_filtered,
        str(args.idx_job),
        "image_urls_in_web_document_dataset_filtered.pickle",
    )
    command_sync_s3 = f"aws s3 cp {path_save_disk_image_urls_in_web_document_dataset_filtered} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the set of image URLs in the web document dataset filtered")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {path_save_disk_tmp_files}")
    logger.info("Finished deleting the tmp files")
