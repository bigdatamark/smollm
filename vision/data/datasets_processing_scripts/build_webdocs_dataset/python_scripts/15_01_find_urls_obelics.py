# [MODULE DOC] Script (step 15_01 of the OBELICS web-document pipeline): collects every document URL and
# image URL of
# OBELICS so they can be checked against the Spawning opt-out API (step 15_02).
# When run as main: syncs the image-less OBELICS dataset from S3 to local disk (s5cmd then aws s3 sync),
# loads it with datasets.load_from_disk, maps func_map_find_urls (NUM_PROC processes), then builds three
# deduplicated single-column datasets (document_url / image_url / url = docs + images concatenated), saves
# each to /scratch and uploads it to S3 via `aws s3 sync` shell commands. Paths are hard-coded constants.
"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
"""


import json
import os

from datasets import Dataset, load_from_disk


PATH_S3_OBELICS_WITHOUT_IMAGES = "s3://m4-datasets-us-east-1/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv_finalprocessing_replaceimgbyurl_concatenated/*"
PATH_SAVE_DISK_OBELICS_WITHOUT_IMAGES = "/fsx/hugo/obelics_without_images/"
NUM_PROC = 48
PATH_SAVE_DISK_DOC_URLS_OBELICS = "/scratch/ds_doc_urls_obelics"
PATH_SAVE_S3_DOC_URLS_OBELICS = "s3://m4-datasets-us-east-1/webdocs/ds_doc_urls_obelics"
PATH_SAVE_DISK_IMAGE_URLS_OBELICS = "/scratch/ds_image_urls_obelics"
PATH_SAVE_S3_IMAGE_URLS_OBELICS = "s3://m4-datasets-us-east-1/webdocs/ds_image_urls_obelics"
PATH_SAVE_DISK_DOC_AND_IMAGE_URLS_OBELICS = "/scratch/ds_doc_and_image_urls_obelics"
PATH_SAVE_S3_DOC_AND_IMAGE_URLS_OBELICS = "s3://m4-datasets-us-east-1/webdocs/ds_doc_and_image_urls_obelics"


def func_map_find_urls(example):
    # [FUNCTION DOC] datasets.map function. Input: an OBELICS example whose "general_metadata" is a JSON
    # string with
    # a "url" key and whose "metadata" is a JSON list (one entry per text/image slot; None for texts, mapping
    # with "src" for images). Adds "document_url" (str) and "image_urls" (list of src strings, None entries
    # skipped) to the example and returns it (the example dict is mutated in place).
    document_url = json.loads(example["general_metadata"])["url"]
    example["document_url"] = document_url
    #
    metadata = json.loads(example["metadata"])
    image_urls = [meta["src"] for meta in metadata if meta is not None]
    example["image_urls"] = image_urls
    return example


if __name__ == "__main__":
    # Download Obelics without the images
    os.system(f"s5cmd sync {PATH_S3_OBELICS_WITHOUT_IMAGES} {PATH_SAVE_DISK_OBELICS_WITHOUT_IMAGES}")
    # Check that everything was downloaded
    os.system(f"aws s3 sync {PATH_S3_OBELICS_WITHOUT_IMAGES} {PATH_SAVE_DISK_OBELICS_WITHOUT_IMAGES}")

    # Load Obelics without the images
    ds = load_from_disk(PATH_SAVE_DISK_OBELICS_WITHOUT_IMAGES)

    # Find document and image urls for each example of Obelics (done in only 5 min)
    ds = ds.map(func_map_find_urls, remove_columns=ds.column_names, num_proc=NUM_PROC)

    # Creating the set of urls, saving and uploading
    document_urls = ds["document_url"]
    document_urls = list(set(document_urls))
    ds_doc_urls = Dataset.from_dict({"document_url": document_urls})
    ds_doc_urls.save_to_disk(PATH_SAVE_DISK_DOC_URLS_OBELICS, num_proc=NUM_PROC)
    os.system(f"aws s3 sync {PATH_SAVE_DISK_DOC_URLS_OBELICS} {PATH_SAVE_S3_DOC_URLS_OBELICS}")

    image_urls = ds["image_urls"]
    image_urls = [sub_el for el in image_urls for sub_el in el]
    image_urls = list(set(image_urls))
    ds_image_urls = Dataset.from_dict({"image_url": image_urls})
    ds_image_urls.save_to_disk(PATH_SAVE_DISK_IMAGE_URLS_OBELICS, num_proc=NUM_PROC)
    os.system(f"aws s3 sync {PATH_SAVE_DISK_IMAGE_URLS_OBELICS} {PATH_SAVE_S3_IMAGE_URLS_OBELICS}")

    doc_and_image_urls = document_urls + image_urls
    ds_doc_and_image_urls = Dataset.from_dict({"url": doc_and_image_urls})
    ds_doc_and_image_urls.save_to_disk(PATH_SAVE_DISK_DOC_AND_IMAGE_URLS_OBELICS, num_proc=NUM_PROC)
    os.system(f"aws s3 sync {PATH_SAVE_DISK_DOC_AND_IMAGE_URLS_OBELICS} {PATH_SAVE_S3_DOC_AND_IMAGE_URLS_OBELICS}")
