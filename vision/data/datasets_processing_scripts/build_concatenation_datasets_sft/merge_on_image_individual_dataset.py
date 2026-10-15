# [MODULE DOC] Script (one SLURM array job per dataset) that merges SFT examples sharing identical images into
# one multi-turn
# example. argv[1] = integer index into ALL_NAMES_DS selecting NAME_DS. Reads from COMMON_FOLDER_DS/NAME_DS,
# writes to COMMON_FOLDER_SAVE_DS/NAME_DS. Output schema FEATURES: "images" (sequence of decoded images) and
# "texts" (list of dicts with string fields user/assistant/source). TOKENIZER names the Hub tokenizer used
# only
# for statistics. Running as __main__ builds MergeOnSameImage(NAME_DS) and calls it.
import hashlib
import os
import sys
from typing import List

import datasets
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


IDX_JOB = int(sys.argv[1])
ALL_NAMES_DS = [
    "ai2d",
    "aokvqa",
    "chartqa",
    "clevr",
    "cocoqa",
    "datikz",
    "diagram_image_to_text",
    "docvqa",
    "dvqa",
    "geomverse",
    "hateful_memes",
    "iam",
    "iconqa",
    "infographic_vqa",
    "intergps",
    "mimic_cgd",
    "nlvr2",
    "ocrvqa",
    "okvqa",
    "scienceqa",
    "screen2words",
    "spot_the_diff",
    "st_vqa",
    "tallyqa",
    "textcaps",
    "textvqa",
    "visual7w",
    "visualmrc",
    "vqarad",
    "vqav2",
    "vsr",
]
NAME_DS = ALL_NAMES_DS[IDX_JOB]

COMMON_FOLDER_DS = "/fsx/hugo/fine_tuning_datasets_to_be_merged"
COMMON_FOLDER_SAVE_DS = "/fsx/hugo/fine_tuning_datasets_merge_image_individual"

NUM_PROC = 48

FEATURES = datasets.Features(
    {
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "texts": [
            {
                "user": datasets.Value("string"),
                "assistant": datasets.Value("string"),
                "source": datasets.Value("string"),
            }
        ],
    }
)

TOKENIZER = "HuggingFaceM4/Mistral-7B-v0.1-tokenizer-pad-is-unk"


class MergeOnSameImage:
    # [CLASS DOC] Pipeline object for one dataset. Attributes set along the way: name_ds, ds (loaded, then
    # with an added
    # "image_hash" column), image_hash_to_ds_index, all_answers, new_ds.
    def __init__(self, name_ds: str) -> None:
        # [FUNCTION DOC] Stores name_ds (sub-folder name of the dataset to process). No other work.
        self.name_ds = name_ds

    def __call__(self) -> None:
        # [FUNCTION DOC] Runs the full pipeline in order: load_dataset, compute_hash_images,
        # create_image_hash_to_ds_index,
        # create_new_ds, save_new_ds, print_stats. Returns None.
        self.load_dataset()
        self.compute_hash_images()
        self.create_image_hash_to_ds_index()
        self.create_new_ds()
        self.save_new_ds()
        self.print_stats()

    def load_dataset(self):
        # [FUNCTION DOC] Loads the dataset from COMMON_FOLDER_DS/name_ds with load_from_disk into self.ds and
        # prints a message.
        path_ds = os.path.join(COMMON_FOLDER_DS, self.name_ds)
        self.ds = load_from_disk(path_ds)
        print("Dataset loaded")

    def compute_hash_images(self):
        # [FUNCTION DOC] Adds an "image_hash" string column to self.ds (map with NUM_PROC processes): the
        # concatenation of the MD5 hex
        # digests of the raw pixel bytes of all images of the example, in order. Prints when done.
        def map_compute_hash_images(example):
            # [FUNCTION DOC] HELPER: map function; takes an example mapping, sets example["image_hash"] as
            # described above, returns it.
            image_hash = ""
            images = example["images"]
            for image in images:
                md5hash = hashlib.md5(image.tobytes()).hexdigest()
                image_hash += md5hash
            example["image_hash"] = image_hash
            return example

        self.ds = self.ds.map(map_compute_hash_images, num_proc=NUM_PROC)
        print("Finished computing the hash of the images")

    def create_image_hash_to_ds_index(self):
        # [FUNCTION DOC] Builds self.image_hash_to_ds_index: mapping image_hash -> list of row indices (in
        # dataset order) having that
        # hash. Prints when done.
        all_image_hashes = self.ds["image_hash"]
        self.image_hash_to_ds_index = {}
        for idx_image_hash, image_hash in enumerate(all_image_hashes):
            self.image_hash_to_ds_index[image_hash] = self.image_hash_to_ds_index.get(image_hash, []) + [
                idx_image_hash
            ]
        print("Finished creating the mapping to go from an image hash to the associate indices in the dataset")

    def create_new_ds(self):
        # [FUNCTION DOC] Builds the merged dataset: sets self.all_answers to the flat list of all "assistant"
        # strings over all rows; then,
        # for each hash group (insertion order), keeps the images of the first row of the group and
        # concatenates the
        # "texts" turn lists of all rows of the group. Materializes self.new_ds with Dataset.from_generator
        # using FEATURES
        # and writer_batch_size=100 (writes to the HF cache). Prints when done.
        dict_new_ds = {"images": [], "texts": []}
        all_texts = self.ds["texts"]
        self.all_answers = [sub_el["assistant"] for el in all_texts for sub_el in el]

        for list_indices in tqdm(list(self.image_hash_to_ds_index.values())):
            common_image = self.ds[list_indices[0]]["images"]
            dict_new_ds["images"].append(common_image)

            texts = [all_texts[idx] for idx in list_indices]
            texts = [sub_el for el in texts for sub_el in el]
            dict_new_ds["texts"].append(texts)

        # self.new_ds = datasets.Dataset.from_dict(dict_new_ds, features=FEATURES)

        def data_generator():
            # [FUNCTION DOC] HELPER: generator yielding one mapping {"images", "texts"} per merged group from
            # the accumulated lists.
            for images, texts in zip(dict_new_ds["images"], dict_new_ds["texts"]):
                yield {"images": images, "texts": texts}

        self.new_ds = datasets.Dataset.from_generator(data_generator, features=FEATURES, writer_batch_size=100)
        print("Finished creating the new dataset")

    def save_new_ds(self):
        # [FUNCTION DOC] Saves self.new_ds to COMMON_FOLDER_SAVE_DS/name_ds with save_to_disk (NUM_PROC
        # processes) and prints.
        path_save_new_ds = os.path.join(COMMON_FOLDER_SAVE_DS, self.name_ds)
        self.new_ds.save_to_disk(path_save_new_ds, num_proc=NUM_PROC)
        print("Finished saving the new dataset")

    def print_stats(self):
        # [FUNCTION DOC] Prints dataset name, number of distinct images (rows of new_ds), number of QA pairs
        # (len of all_answers),
        # average QA pairs per image, total token count of all answers (via tokenize_and_count_tokens) and
        # average
        # tokens per answer. Raises ZeroDivisionError on an empty dataset.
        number_tokens_answers = self.tokenize_and_count_tokens(list_texts=self.all_answers)
        print(f"Dataset: {self.name_ds}")
        print(f"Number of different images: {self.new_ds.num_rows}")
        print(f"Number of question/answer pairs: {len(self.all_answers)}")
        frac_qa_pairs_num_images = len(self.all_answers) / self.new_ds.num_rows
        print(f"Average number of question/answer pairs per image: {frac_qa_pairs_num_images}")
        print(f"Total number of tokens for the answers: {number_tokens_answers}")
        mean_num_tok_per_answer = number_tokens_answers / len(self.all_answers)
        print(f"Average number of tokens per answer: {mean_num_tok_per_answer}")

    def tokenize_and_count_tokens(self, list_texts: List[str]) -> int:
        # [FUNCTION DOC] HELPER: loads the TOKENIZER from the Hub, tokenizes each string of list_texts (with
        # special tokens, default
        # settings) and returns the total number of input_ids as an int.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        list_texts_tokenized = [tokenizer(text)["input_ids"] for text in list_texts]
        number_tokens = sum([len(tokens) for tokens in list_texts_tokenized])
        return number_tokens


if __name__ == "__main__":
    merge_on_same_image = MergeOnSameImage(name_ds=NAME_DS)
    merge_on_same_image()
