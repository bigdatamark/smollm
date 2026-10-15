# [MODULE DOC] One-off script that builds the COCO validation set and uploads it to the Hugging Face Hub.
# Side effects at import time: `s5cmd sync` of s3://m4-datasets/general_pmd/image/coco/validation/00000-00001/*
# into /scratch/general_pmd/image/coco/validation, loads it with load_from_disk, prints it, and pushes it as
# split "valid" of private Hub repo HuggingFaceM4/coco_valid. No functions; no CLI arguments.
import os

from datasets import load_from_disk


SUBSET_DIR_PATH = "/scratch/general_pmd/image/coco/"
valid_path = f"{SUBSET_DIR_PATH}/validation"
sync_cmd = f"s5cmd sync s3://m4-datasets/general_pmd/image/coco/validation/00000-00001/* {valid_path}"

os.system(sync_cmd)

ds = load_from_disk(valid_path)
print(ds)
repo_id = "HuggingFaceM4/coco_valid"
ds.push_to_hub(repo_id, "valid", private=True)
