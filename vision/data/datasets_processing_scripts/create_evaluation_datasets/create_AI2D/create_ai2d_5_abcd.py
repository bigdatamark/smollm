# [MODULE DOC] Script (top-level code): builds HuggingFaceM4/AI2D_5 from the AI2D questions whose options ARE
# single
# letters (591 examples, diagram labels). Labels are the (uppercased) option texts; the ClassLabel names are
# the set of all such options (order from set(), so label ids are not deterministic across runs). Adds
# tested_labels (the 4 options). Question: instruction + "Question: <q>\n" + "The choices are: <sorted
# options comma-separated>". Pushed privately and reloaded.
"""
srun --pty --cpus-per-task=8 --partition=hopper-cpu --qos high bash -i
conda activate shared-m4
"""


import datasets
from datasets import DatasetDict, load_dataset


ORIGINAL_NAME_DS = "lmms-lab/ai2d"
ORIGINAL_SPLIT_DS = "test"

NUM_PROC = 8


ds_test = load_dataset(ORIGINAL_NAME_DS, split=ORIGINAL_SPLIT_DS)  # 3088 examples


def filter_ds(example):
    # [FUNCTION DOC] Predicate over an lmms-lab/ai2d example: returns False if any option in
    # example["options"] is a single
    # alphabetic character (letter-labelled diagrams), True otherwise. Used negated at module level, so only
    # the
    # letter-option questions are kept.
    options = example["options"]
    if any([option.isalpha() and (len(option) == 1) for option in options]):
        return False
    return True


ds_test = ds_test.filter(lambda ex: not filter_ds(ex), num_proc=NUM_PROC)  # 591 examples


possible_labels = [option for options in ds_test["options"] for option in options]
POSSIBLE_LABELS = list(set([option.upper() if (len(option) == 1) else option for option in possible_labels]))

FEATURES = datasets.Features(
    {
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "image": datasets.Image(decode=True),
        # For these questions with a letter as the answer, it's always 4 tested labels
        "tested_labels": datasets.Sequence(datasets.Value("string")),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/AI2D_5"


def map_func_transform_ai2d_ds(example):
    # [FUNCTION DOC] Map function: uppercases single-char options, sets tested_labels to them and label to
    # the option at the
    # answer index, then builds the question with the comma-separated sorted options. Returns the example.
    choices = example["options"]
    choices = [choice.upper() if (len(choice) == 1) else choice for choice in choices]
    example["tested_labels"] = choices
    example["label"] = choices[int(example["answer"])]
    choices = sorted(choices)
    question = example["question"].strip()
    question = f"Using the image, answer exactly as one of the choices.\nQuestion: {question}\n"
    choices_string = ", ".join(choices)
    question += f"The choices are: {choices_string}"
    example["question"] = question
    return example


columns_to_remove = [c_n for c_n in ds_test.column_names if c_n not in list(FEATURES.keys())]
ds_test = ds_test.map(
    map_func_transform_ai2d_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC
)
print(ds_test[0]["question"])


ds_all_splits = DatasetDict({"test": ds_test})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)

# Cache dataset
test_loading = load_dataset(NAME_DS_PUSH_HUB)
