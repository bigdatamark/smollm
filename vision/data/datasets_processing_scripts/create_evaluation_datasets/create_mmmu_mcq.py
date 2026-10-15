# [MODULE DOC] Script (top-level code): builds HuggingFaceM4/MMMU_MCQ (private DatasetDict
# dev/validation/test) from the
# multiple-choice questions of HuggingFaceM4/MMMU, then reloads it to cache. Features: id, question (images
# header + lettered choices + "Answer with the letter."), label (ClassLabel A-I, None for test), images.
"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""


import ast

import datasets
from datasets import DatasetDict, load_dataset


NAME_DS = "HuggingFaceM4/MMMU"
ALL_SPLITS = ["dev", "validation", "test"]

NUM_PROC = 96

POSSIBLE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

IMAGE_NUMBERS = ["1", "2", "3", "4", "5", "6", "7"]

FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "images": datasets.Sequence(datasets.Image(decode=True)),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/MMMU_MCQ"


def filter_func_mmmu_mcq_ds(example):
    # [FUNCTION DOC] Filter predicate: True for question_type "multiple-choice", False for "open", ValueError
    # otherwise.
    if example["question_type"] == "open":
        return False
    elif example["question_type"] == "multiple-choice":
        return True
    else:
        raise ValueError("Different `question_type`")


def format_value_to_list(value):
    # [FUNCTION DOC] HELPER: parses a string that may hold a Python list literal (e.g. MMMU
    # "options"/"answer"). Returns the
    # parsed list if ast.literal_eval yields a list, otherwise (non-list literal, SyntaxError or ValueError)
    # returns [value].
    try:
        evaluated_value = ast.literal_eval(value)
        if isinstance(evaluated_value, list):
            return evaluated_value
        else:
            return [value]
    except (SyntaxError, ValueError):
        return [value]


def map_func_transform_mmmu_mcq_ds(example):
    # [FUNCTION DOC] Map function: the non-None image_1..image_7 columns are gathered into "images" and a
    # header line
    # "<image i>:<image>" per image (numbered from 1) is prepended to the question; question = images header
    # + "\nQuestion: <q>\nChoices:\n" +
    # "<letter>. <choice>" lines + "Answer with the letter."; label = answer letter, None when answer is "?"
    # (test). IndexError if more than 9 options. Returns the example.
    example["images"] = [
        example[f"image_{image_number}"]
        for image_number in IMAGE_NUMBERS
        if example[f"image_{image_number}"] is not None
    ]
    image_string = ""
    for idx_image in range(len(example["images"])):
        image_string += f"<image {IMAGE_NUMBERS[idx_image]}>:<image>\n"
    image_string = image_string.strip()
    #
    choices = format_value_to_list(example["options"])
    question = example["question"]
    question = f"Question: {question}\n"
    question += "Choices:\n"
    for idx_choice, choice in enumerate(choices):
        letter_choice = POSSIBLE_LABELS[idx_choice]
        question += f"{letter_choice}. {choice}\n"
    question += "Answer with the letter."
    question = f"{image_string}\n{question}"
    example["question"] = question
    #
    if example["answer"] == "?":  # split "test"
        example["label"] = None
    else:
        example["label"] = example["answer"]
    return example


def prepare_mmmu_mcq(split):
    # [FUNCTION DOC] Loads HuggingFaceM4/MMMU split `split`, keeps MCQ rows, transforms them keeping only
    # FEATURES columns
    # (NUM_PROC processes) and returns the dataset.
    ds = load_dataset(NAME_DS, split=split)
    ds = ds.filter(filter_func_mmmu_mcq_ds, num_proc=NUM_PROC)
    columns_to_remove = [c_n for c_n in ds.column_names if c_n not in list(FEATURES.keys())]
    ds = ds.map(map_func_transform_mmmu_mcq_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC)
    return ds


ds_all_splits = DatasetDict({split: prepare_mmmu_mcq(split=split) for split in ALL_SPLITS})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)

# Cache dataset
test_loading = load_dataset(NAME_DS_PUSH_HUB)
