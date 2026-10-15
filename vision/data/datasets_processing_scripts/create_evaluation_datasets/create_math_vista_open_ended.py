# [MODULE DOC] Script (top-level code): builds HuggingFaceM4/MathVista_OpenEnded (private) with the free-form
# questions of
# MathVista "testmini" and "test". Features: pid (int), question (with " (Unit: <unit>)" appended when a unit
# exists), instruction (the hint part of MathVista's "query" prompt), answer (list with one str, None for
# test), image.
"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""


import datasets
from datasets import DatasetDict, load_dataset


NAME_DS = "AI4Math/MathVista"
ALL_SPLITS = ["testmini", "test"]

NUM_PROC = 96

FEATURES = datasets.Features(
    {
        "pid": datasets.Value("int64"),
        "question": datasets.Value("string"),
        "instruction": datasets.Value("string"),
        "answer": datasets.Sequence(datasets.Value("string")),
        "image": datasets.Image(decode=True),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/MathVista_OpenEnded"


def filter_func_math_vista_openended_ds(example):
    # [FUNCTION DOC] Filter predicate: True for "free_form", False for "multi_choice"; ValueError for other
    # question_type values.
    if example["question_type"] == "free_form":
        return True
    elif example["question_type"] == "multi_choice":
        return False
    else:
        raise ValueError("Different `question_type`")


def map_func_transform_math_vista_openended_ds(example):
    # [FUNCTION DOC] Map function: pid -> int; image = decoded_image; answer -> [answer] (None if empty, i.e.
    # test split);
    # appends the unit to the question if not None; instruction = text of "query" before "\nQuestion: " with
    # "Hint: " removed (if the marker is missing, find returns -1 and the last character is dropped).
    # Returns the example.
    example["pid"] = int(example["pid"])
    example["image"] = example["decoded_image"]
    #
    if example["answer"] == "":  # split "test"
        example["answer"] = None
    else:
        example["answer"] = [example["answer"]]
    #
    unit = example["unit"]
    if unit is not None:
        example["question"] = example["question"] + f" (Unit: {unit})"
    #
    query = example["query"]
    example["instruction"] = query[: query.find("\nQuestion: ")].replace("Hint: ", "")
    #
    return example


def prepare_math_vista_openended(split):
    # [FUNCTION DOC] Loads MathVista split `split`, keeps free-form rows, applies the transform keeping only
    # FEATURES columns
    # (NUM_PROC processes) and returns the dataset.
    ds = load_dataset(NAME_DS, split=split)
    ds = ds.filter(filter_func_math_vista_openended_ds, num_proc=NUM_PROC)
    columns_to_remove = [c_n for c_n in ds.column_names if c_n not in list(FEATURES.keys())]
    ds = ds.map(
        map_func_transform_math_vista_openended_ds,
        remove_columns=columns_to_remove,
        features=FEATURES,
        num_proc=NUM_PROC,
    )
    return ds


ds_all_splits = DatasetDict({split: prepare_math_vista_openended(split=split) for split in ALL_SPLITS})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)
