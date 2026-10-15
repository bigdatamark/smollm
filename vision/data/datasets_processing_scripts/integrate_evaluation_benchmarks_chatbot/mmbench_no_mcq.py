# [MODULE DOC] Builds a "no multiple-choice-letters" MMBench variant and pushes it to HuggingFaceM4/MMBench_modif_chatbot_NoMCQ.
# Instead of letters, the model is scored on the answer texts: each example has "context" (question plus
# optional hint), "tested_labels" (the choice texts, padded to 4 with the dummy WRONG_LABEL string) and "label"
# (the correct choice text, ClassLabel over every choice text seen in both splits plus WRONG_LABEL; None for
# test). Splits: validation (dev TSV) and test. Re-loads the pushed repo at the end as a smoke check.
# Runs at import time.
"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""

import base64
from copy import deepcopy
from io import BytesIO

import datasets
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image


PATH_MMBENCH_DATA = {
    "validation": "/fsx/hugo/mmbench/mmbench/mmbench_dev_en_20231003.tsv",
    "test": "/fsx/hugo/mmbench/mmbench/mmbench_test_en_20231003.tsv",
}  # DL from https://mmbench.opencompass.org.cn/home

NUM_PROC = 48

WRONG_LABEL = "cfghjvkljhgjffxghckjvlbknlmjghfdfgfghggdfxsdgfg"

REPO_ID = "HuggingFaceM4/MMBench_modif_chatbot_NoMCQ"


def find_max_num_choices_and_all_tested_labels(all_splits):
    # [FUNCTION DOC] HELPER: read the TSV of each split in all_splits and collect the unique set of non-empty choice texts from
    # columns A-D plus WRONG_LABEL. Returns (max_num_choices, all_tested_labels) where max_num_choices is hard-coded
    # to 4 and all_tested_labels is an unordered de-duplicated list (order varies between runs).
    all_tested_labels = []
    for split in all_splits:
        data_frame_mmbench = pd.read_csv(PATH_MMBENCH_DATA[split], sep="\t", header=0)
        ds_mmbench = Dataset.from_pandas(data_frame_mmbench)
        all_choices = ds_mmbench["A"] + ds_mmbench["B"] + ds_mmbench["C"] + ds_mmbench["D"] + [WRONG_LABEL]
        all_choices = [choice for choice in all_choices if choice]
        all_tested_labels.extend(all_choices)
    all_tested_labels = list(set(all_tested_labels))
    max_num_choices = 4
    return max_num_choices, all_tested_labels


def create_ds_mmbench(split, max_num_choices, all_tested_labels):
    # [FUNCTION DOC] Build one split. Inputs: split ("validation"/"test"), max_num_choices (int, pad length), all_tested_labels
    # (class names for the label ClassLabel). Reads the TSV, maps map_transform_mmbench (48 processes) with features
    # image/context/label/tested_labels, keeps only those columns and returns the Dataset.
    """`split` is "validation" or "test"."""
    #
    data_frame_mmbench = pd.read_csv(PATH_MMBENCH_DATA[split], sep="\t", header=0)
    ds_mmbench = Dataset.from_pandas(data_frame_mmbench)

    #
    def map_transform_mmbench(example):
        # [FUNCTION DOC] dataset.map function: decode the base64 image; context = "Question: q\nHint: h" when a hint exists else the
        # bare question; tested_labels = non-empty choices A-D padded with WRONG_LABEL up to max_num_choices; label =
        # text of the choice named by the "answer" letter on validation, None on test. Returns the mutated example.
        example["image"] = Image.open(BytesIO(base64.b64decode(example["image"])))
        #
        question = example["question"]
        hint = example["hint"]
        if hint:
            example["context"] = f"Question: {question}\nHint: {hint}"
        else:
            example["context"] = question
        #
        all_choices = [example["A"], example["B"], example["C"], example["D"]]
        all_choices = [choice for choice in all_choices if choice]
        all_choices = all_choices + (max_num_choices - len(all_choices)) * [WRONG_LABEL]
        example["tested_labels"] = all_choices
        #
        if split == "validation":
            example["label"] = example[example["answer"]]
        else:
            example["label"] = None
        return example

    #
    new_features = deepcopy(ds_mmbench.features)
    new_features["image"] = datasets.Image(decode=True)
    new_features["context"] = datasets.Value("string")
    new_features["label"] = datasets.features.ClassLabel(names=all_tested_labels)
    new_features["tested_labels"] = datasets.Sequence(datasets.Value("string"))
    #
    ds_mmbench = ds_mmbench.map(map_transform_mmbench, features=new_features, num_proc=NUM_PROC)
    column_to_keep = ["image", "context", "label", "tested_labels"]
    ds_mmbench = ds_mmbench.remove_columns(
        column_names=[c_n for c_n in ds_mmbench.column_names if c_n not in column_to_keep]
    )
    return ds_mmbench


all_splits = ["validation", "test"]
max_num_choices, all_tested_labels = find_max_num_choices_and_all_tested_labels(all_splits=all_splits)

ds_mmbench_validation = create_ds_mmbench(
    split="validation", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_mmbench_test = create_ds_mmbench(split="test", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels)
ds_mmbench_all_splits = DatasetDict({"validation": ds_mmbench_validation, "test": ds_mmbench_test})

ds_mmbench_all_splits.push_to_hub(REPO_ID)

test_ds_mmbench_all_splits = load_dataset(REPO_ID)
