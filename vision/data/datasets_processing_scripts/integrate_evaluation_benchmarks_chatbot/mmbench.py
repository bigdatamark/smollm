# [MODULE DOC] Builds two MMBench (English, 2023-10-03 TSV release, downloaded locally) datasets and pushes them to the Hub:
# 1) HuggingFaceM4/MMBench_modif_chatbot: base64 images decoded to PIL, a multiple-choice prompt in "context"
#    (Question / optional Hint / Choices A-D / "Answer with a letter."), "label" as ClassLabel A-D (None for test),
#    keeping only index, image, context, label.
# 2) HuggingFaceM4/MMBench: the raw TSV rows as a Dataset (test split gets an empty "answer" column).
# Both have "validation" (dev TSV) and "test" splits. The module redefines PATH_MMBENCH_DATA, NUM_PROC, REPO_ID and
# create_ds_mmbench halfway through; everything runs at import time.
"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""

import base64
from copy import deepcopy
from io import BytesIO

import datasets
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image


PATH_MMBENCH_DATA = {
    "validation": "/fsx/hugo/mmbench/mmbench/mmbench_dev_en_20231003.tsv",
    "test": "/fsx/hugo/mmbench/mmbench/mmbench_test_en_20231003.tsv",
}  # DL from https://mmbench.opencompass.org.cn/home

NUM_PROC = 48

REPO_ID = "HuggingFaceM4/MMBench_modif_chatbot"


def create_ds_mmbench(split):
    # [FUNCTION DOC] Build the chatbot-format MMBench split. Input: split, "validation" or "test" (key of PATH_MMBENCH_DATA).
    # Reads the TSV with pandas, converts to a Dataset, maps map_transform_mmbench with 48 processes and new features
    # (image as Image, context string, label ClassLabel A-D), drops other columns and returns the Dataset.
    """`split` is "validation" or "test"."""

    data_frame_mmbench = pd.read_csv(PATH_MMBENCH_DATA[split], sep="\t", header=0)

    ds_mmbench = Dataset.from_pandas(data_frame_mmbench)

    def map_transform_mmbench(example):
        # [FUNCTION DOC] dataset.map function: decodes the base64 "image" into a PIL image, builds the prompt string from question,
        # non-empty hint and non-empty choices A-D, stores it in "context", and sets "label" to the answer letter for
        # the validation split or None otherwise (closure over split). Returns the mutated example.
        example["image"] = Image.open(BytesIO(base64.b64decode(example["image"])))
        prompt = ""
        question = example["question"]
        prompt += f"Question: {question}\n"
        hint = example["hint"]
        if hint:
            prompt += f"Hint: {hint}\n"
        prompt += "Choices:\n"
        for key in ["A", "B", "C", "D"]:
            ans = example[key]
            if ans:
                prompt += f"{key}. {ans}\n"
        prompt += "Answer with a letter."
        example["context"] = prompt
        if split == "validation":
            example["label"] = example["answer"]
        else:
            example["label"] = None
        return example

    new_features = deepcopy(ds_mmbench.features)
    new_features["image"] = datasets.Image(decode=True)
    new_features["context"] = datasets.Value("string")
    new_features["label"] = datasets.features.ClassLabel(names=["A", "B", "C", "D"])

    ds_mmbench = ds_mmbench.map(map_transform_mmbench, features=new_features, num_proc=NUM_PROC)
    column_to_keep = ["index", "image", "context", "label"]
    ds_mmbench = ds_mmbench.remove_columns(
        column_names=[c_n for c_n in ds_mmbench.column_names if c_n not in column_to_keep]
    )
    return ds_mmbench


ds_mmbench_validation = create_ds_mmbench(split="validation")
ds_mmbench_test = create_ds_mmbench(split="test")
ds_mmbench_all_splits = DatasetDict({"validation": ds_mmbench_validation, "test": ds_mmbench_test})

ds_mmbench_all_splits.push_to_hub(REPO_ID)


# MAKE Raw MMBench
PATH_MMBENCH_DATA = {
    "validation": "/fsx/hugo/mmbench/mmbench/mmbench_dev_en_20231003.tsv",
    "test": "/fsx/hugo/mmbench/mmbench/mmbench_test_en_20231003.tsv",
}  # DL from https://mmbench.opencompass.org.cn/home

NUM_PROC = 48

REPO_ID = "HuggingFaceM4/MMBench"


def create_ds_mmbench(split):
    # [FUNCTION DOC] Raw variant (second definition, shadows the first): read the split's TSV with pandas, add an empty "answer"
    # column for the test split (which has no answers), and return Dataset.from_pandas of it unchanged otherwise.
    """`split` is "validation" or "test"."""

    data_frame_mmbench = pd.read_csv(PATH_MMBENCH_DATA[split], sep="\t", header=0)
    if split == "test":
        data_frame_mmbench["answer"] = ""

    ds_mmbench = Dataset.from_pandas(data_frame_mmbench)
    return ds_mmbench


ds_mmbench_validation = create_ds_mmbench(split="validation")
ds_mmbench_test = create_ds_mmbench(split="test")
ds_mmbench_all_splits = DatasetDict({"validation": ds_mmbench_validation, "test": ds_mmbench_test})

ds_mmbench_all_splits.push_to_hub(REPO_ID)
