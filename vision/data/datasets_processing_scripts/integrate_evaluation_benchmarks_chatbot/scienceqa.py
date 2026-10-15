# [MODULE DOC] Builds the image subset of ScienceQA (derek-thomas/ScienceQA) in multiple-choice-letter chatbot format and
# pushes train/validation/test to HuggingFaceM4/ScienceQAImg_Modif. Columns kept: image, context (prompt),
# label (ClassLabel A-E). Examples without image are filtered out. Runs at import time (48 processes).
from copy import deepcopy

import datasets
from datasets import DatasetDict, load_dataset


DS_NAME = "derek-thomas/ScienceQA"

NUM_PROC = 48

REPO_ID = "HuggingFaceM4/ScienceQAImg_Modif"


def create_ds_scienceqa(split):
    # [FUNCTION DOC] Load one split ("train"/"validation"/"test") of ScienceQA, map map_transform_scienceqa with new features
    # (image, context string, label ClassLabel A-E), keep only image/context/label, drop rows without image, and
    # return the Dataset.
    # KNOWN ISSUE: prompts support up to 12 letters but label names are only A-E, so a question with more than 5
    # choices whose answer is past E would fail the ClassLabel cast.
    """`split` is "train", "validation" or "test"."""
    ds_scienceqa = load_dataset(DS_NAME, split=split)

    #
    def map_transform_scienceqa(example):
        # [FUNCTION DOC] dataset.map function: builds context = optional "Lecture: ...", "Question: ...", optional "Hint: ...",
        # "Choices:" then one "<Letter>. <choice>" line per choice and "Answer with the letter."; label = letter of the
        # answer index. Returns the mutated example.
        question = example["question"]
        #
        all_choices = example["choices"]
        index_answer = example["answer"]
        #
        lecture = example["lecture"]
        hint = example["hint"]
        #
        prompt = ""
        if lecture != "":
            prompt += f"Lecture: {lecture}\n"
        prompt += f"Question: {question}\n"
        if hint != "":
            prompt += f"Hint: {hint}\n"
        prompt += "Choices:\n"
        #
        letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        for idx, choice in enumerate(all_choices):
            letter = letters_cap[idx]
            prompt += f"{letter}. {choice}\n"
        prompt += "Answer with the letter."
        #
        example["context"] = prompt
        #
        letter_answer = letters_cap[index_answer]
        example["label"] = letter_answer
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
    new_features["label"] = datasets.features.ClassLabel(names=["A", "B", "C", "D", "E"])
    #
    ds_scienceqa = ds_scienceqa.map(map_transform_scienceqa, features=new_features, num_proc=NUM_PROC)
    #
    columns_to_keep = ["image", "context", "label"]
    ds_scienceqa = ds_scienceqa.remove_columns(
        column_names=[c_n for c_n in ds_scienceqa.column_names if c_n not in columns_to_keep]
    )
    #
    ds_scienceqa = ds_scienceqa.filter(filter_no_image_scienceqa, num_proc=NUM_PROC)
    return ds_scienceqa


ds_scienceqa_train = create_ds_scienceqa(split="train")
ds_scienceqa_validation = create_ds_scienceqa(split="validation")
ds_scienceqa_test = create_ds_scienceqa(split="test")
ds_scienceqa_all_splits = DatasetDict(
    {"train": ds_scienceqa_train, "validation": ds_scienceqa_validation, "test": ds_scienceqa_test}
)

ds_scienceqa_all_splits.push_to_hub(REPO_ID)
