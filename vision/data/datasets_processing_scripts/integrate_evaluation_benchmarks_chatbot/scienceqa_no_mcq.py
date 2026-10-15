# [MODULE DOC] Builds the image subset of ScienceQA in "no MCQ" format (answers scored as texts instead of letters) and
# pushes train/validation/test to HuggingFaceM4/ScienceQAImg_Modif_NoMCQ. Columns: image, context (lecture,
# question and hint, stripped), label (correct choice text as ClassLabel over all choice texts plus WRONG_LABEL),
# tested_labels (choices padded with WRONG_LABEL to the global max number of choices). Rows without image are
# dropped. Runs at import time.
from copy import deepcopy

import datasets
from datasets import DatasetDict, load_dataset


DS_NAME = "derek-thomas/ScienceQA"

NUM_PROC = 48

REPO_ID = "HuggingFaceM4/ScienceQAImg_Modif_NoMCQ"

WRONG_LABEL = "cfghjvkljhgjffxghckjvlbknlmjghfdfgfghggdfxsdgfg"


def find_max_num_choices_and_all_tested_labels(all_splits):
    # [FUNCTION DOC] HELPER: loads every split in all_splits and returns (max number of choices over all questions, unordered
    # de-duplicated list of all choice texts plus WRONG_LABEL). Downloads the dataset if needed.
    max_num_choices = []
    all_tested_labels = []
    for split in all_splits:
        ds_scienceqa = load_dataset(DS_NAME, split=split)
        all_choices = ds_scienceqa["choices"]
        max_num_choices.append(max([len(choices) for choices in all_choices]))
        all_tested_labels.extend([choice for choices in all_choices for choice in choices] + [WRONG_LABEL])
    all_tested_labels = list(set(all_tested_labels))
    max_num_choices = max(max_num_choices)
    return max_num_choices, all_tested_labels


def create_ds_scienceqa(split, max_num_choices, all_tested_labels):
    # [FUNCTION DOC] Build one split. Inputs: split name, max_num_choices (padding length), all_tested_labels (ClassLabel names).
    # Maps map_transform_scienceqa with the new features, keeps image/context/label/tested_labels, filters out rows
    # without image and returns the Dataset.
    """`split` is "train", "validation" or "test"."""
    ds_scienceqa = load_dataset(DS_NAME, split=split)

    #
    def map_transform_scienceqa(example):
        # [FUNCTION DOC] dataset.map function: context = optional "Lecture: ...\n" + "Question: ...\n" + optional "Hint: ...", stripped;
        # tested_labels = choices padded with WRONG_LABEL to max_num_choices; label = choice text at the answer index.
        # Returns the mutated example.
        question = example["question"]
        lecture = example["lecture"]
        hint = example["hint"]
        context = ""
        if lecture != "":
            context += f"Lecture: {lecture}\n"
        context += f"Question: {question}\n"
        if hint != "":
            context += f"Hint: {hint}\n"
        example["context"] = context.strip()
        #
        all_choices = example["choices"]
        all_choices = all_choices + (max_num_choices - len(all_choices)) * [WRONG_LABEL]
        example["label"] = all_choices[example["answer"]]
        example["tested_labels"] = all_choices
        return example

    #
    def filter_no_image_scienceqa(example):
        # [FUNCTION DOC] HELPER: dataset.filter predicate returning False when example["image"] is None, True otherwise.
        if example["image"] is None:
            return False
        return True

    #
    new_features = deepcopy(ds_scienceqa.features)
    new_features["image"] = datasets.Image(decode=True)
    new_features["context"] = datasets.Value("string")
    new_features["label"] = datasets.features.ClassLabel(names=all_tested_labels)
    new_features["tested_labels"] = datasets.Sequence(datasets.Value("string"))
    #
    ds_scienceqa = ds_scienceqa.map(map_transform_scienceqa, features=new_features, num_proc=NUM_PROC)
    #
    columns_to_keep = ["image", "context", "label", "tested_labels"]
    ds_scienceqa = ds_scienceqa.remove_columns(
        column_names=[c_n for c_n in ds_scienceqa.column_names if c_n not in columns_to_keep]
    )
    #
    ds_scienceqa = ds_scienceqa.filter(filter_no_image_scienceqa, num_proc=NUM_PROC)
    return ds_scienceqa


all_splits = ["train", "validation", "test"]
max_num_choices, all_tested_labels = find_max_num_choices_and_all_tested_labels(all_splits=all_splits)

ds_scienceqa_train = create_ds_scienceqa(
    split="train", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_scienceqa_validation = create_ds_scienceqa(
    split="validation", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_scienceqa_test = create_ds_scienceqa(
    split="test", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_scienceqa_all_splits = DatasetDict(
    {"train": ds_scienceqa_train, "validation": ds_scienceqa_validation, "test": ds_scienceqa_test}
)

ds_scienceqa_all_splits.push_to_hub(REPO_ID)
