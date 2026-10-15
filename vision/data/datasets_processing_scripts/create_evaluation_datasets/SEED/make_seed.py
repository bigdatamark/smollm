# [MODULE DOC] Script (all top-level code): builds HuggingFaceM4/SEED from AILab-CVC/SEED-Bench (test split):
# keeps only
# image questions, loads images from the local unzipped SEED-Bench-image folder (PATH_IMAGES), makes "answer"
# a ClassLabel A-D, drops data_id/data_type/question_id/segment and pushes the full set, then one config per
# question type (names from QUESTION_TYPE_ID_TO_TASK) to the Hub. Disables the PIL pixel limit.
"""
srun --pty --cpus-per-task=48 --mem-per-cpu=11G bash -i
conda activate /fsx/m4/conda/shared-m4-2023-03-10

cd fsx/hugo/seed_bench
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip
unzip SEED-Bench-image.zip
"""


import os
from copy import deepcopy
from functools import partial

import datasets
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

ORIGINAL_DS_NAME = "AILab-CVC/SEED-Bench"
REPO_ID = "HuggingFaceM4/SEED"

PATH_IMAGES = "/fsx/hugo/seed_bench/SEED-Bench-image"
NUM_PROC = 10

QUESTION_TYPE_ID_TO_TASK = {
    "1": "Scene_Understanding",
    "2": "Instance_Identity",
    "3": "Instance_Attributes",
    "4": "Instance_Location",
    "5": "Instances_Counting",
    "6": "Spatial_Relation",
    "7": "Instance_Interaction",
    "8": "Visual_Reasoning",
    "9": "Text_Understanding",
    "10": "Action_Recognition",
    "11": "Action_Prediction",
    "12": "Procedure_Understanding",
}


def filter_data_type(example):
    # [FUNCTION DOC] Filter predicate: True only when example["data_type"] == "image" (drops video questions).
    if example["data_type"] != "image":
        return False
    return True


def map_replace_images(example):
    # [FUNCTION DOC] Map function: sets example["image"] to the PIL image opened from PATH_IMAGES/<data_id>;
    # returns example.
    path_image = os.path.join(PATH_IMAGES, example["data_id"])
    image = Image.open(path_image)
    example["image"] = image
    return example


def filter_question_type_id(example, question_type_id):
    # [FUNCTION DOC] Filter predicate: True when example["question_type_id"] equals question_type_id (string
    # key of
    # QUESTION_TYPE_ID_TO_TASK).
    if example["question_type_id"] != question_type_id:
        return False
    return True


ds = load_dataset(ORIGINAL_DS_NAME, split="test")

ds = ds.filter(filter_data_type, num_proc=NUM_PROC)

new_features = deepcopy(ds.features)
new_features["image"] = datasets.Image()
new_features["answer"] = datasets.features.ClassLabel(names=["A", "B", "C", "D"])
ds = ds.map(map_replace_images, features=new_features, num_proc=NUM_PROC)

ds = ds.remove_columns(column_names=["data_id", "data_type", "question_id", "segment"])

ds.push_to_hub(REPO_ID, split="test")


for question_type_id in tqdm(QUESTION_TYPE_ID_TO_TASK):
    sub_ds = ds.filter(partial(filter_question_type_id, question_type_id=question_type_id), num_proc=NUM_PROC)
    sub_ds.push_to_hub(REPO_ID, QUESTION_TYPE_ID_TO_TASK[question_type_id], split="test")
    print("ok", question_type_id)
