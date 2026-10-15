# [MODULE DOC] Builds a chatbot-style GQA evaluation dataset and pushes it (private) to HuggingFaceM4/GQA on the Hub.
# For each split (train/validation/test) of Graphcore/gqa, images are read from local /fsx/hugo/gqa/images,
# re-encoded as JPEG, the question gets a random "answer shortly" suffix from PROMPTS_ANSWER_SHORTLY (imported
# from the SFT builder), and the label is wrapped into a one-element "answers" list. Uses 96 processes.
# Runs entirely at import time (network, disk, Hub upload).
import os
import random

import datasets
from datasets import DatasetDict, load_dataset

from datasets_processing_scripts.build_concatenation_datasets_sft.build_ds_sft import (
    PROMPTS_ANSWER_SHORTLY,
    convert_img_to_bytes,
)


NUM_PROC = 96

FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "question_id": datasets.Value("int64"),
    }
)


def map_transform_gqa(example):
    # [FUNCTION DOC] dataset.map function: loads the image file named by basename(example["image_id"]) from /fsx/hugo/gqa/images
    # as JPEG bytes into example["image"] ({"path": None, "bytes": ...}), appends a random short-answer prompt to
    # example["question"], and sets example["answers"] = [example["label"]]. Returns the mutated example.
    # Non-deterministic (random.choice without a seed).
    path_image = os.path.join("/fsx/hugo/gqa/images", os.path.basename(example["image_id"]))
    image_bytes = convert_img_to_bytes(img_path=path_image, format="JPEG")
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    example["image"] = {"path": None, "bytes": image_bytes}
    example["question"] = question
    example["answers"] = [example["label"]]
    return example


def load_gqa(split):
    # [FUNCTION DOC] Load the given split of Graphcore/gqa, map map_transform_gqa over it (removing all columns except image,
    # question, answers, question_id, casting to FEATURES) and return the resulting Dataset.
    ds_gqa = load_dataset("Graphcore/gqa", split=split)
    columns_to_keep = ["image", "question", "answers", "question_id"]
    columns_to_remove = [c_n for c_n in ds_gqa.column_names if c_n not in columns_to_keep]
    ds_gqa = ds_gqa.map(map_transform_gqa, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC)
    return ds_gqa


ds_gqa_all_splits = DatasetDict(
    {"train": load_gqa("train"), "validation": load_gqa("validation"), "test": load_gqa("test")}
)

ds_gqa_all_splits.push_to_hub("HuggingFaceM4/GQA", private=True)
