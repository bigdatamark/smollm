# [MODULE DOC] CLI script: extract 1- to 4-gram ids (GPT-2 tokenization) for every document of a text
# sub-shard and print
# them as CSV lines "<SHARD_NAME>,<document id>,<ngram id>" to stdout. Arguments are parsed at import:
# `--filepath` (sub-shard file, one document per line, whose name ends in ".<subshard index>") and
# `--nb_docs_per_subshard` (default 1000), used to compute globally unique 1-based document ids. SHARD_NAME is
# hard-coded to 0. Loads the GPT-2 tokenizer at import.
import argparse
import logging
import sys
from typing import List

from transformers import AutoTokenizer

from utils import get_ngrams


parser = argparse.ArgumentParser(description="Extract the ngrams")
parser.add_argument("--filepath", type=str, required=True, help="The filepath to the subshard.")
parser.add_argument(
    "--nb_docs_per_subshard",
    type=int,
    default=1_000,
    help="The number of documents per subshard.",
)
args = parser.parse_args()

SHARD_NAME = 0

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size


def process_document(text: str):
    # [FUNCTION DOC] Strip the text, tokenize it with the GPT-2 tokenizer and return the concatenated lists of
    # unique 1-, 2-, 3-
    # and 4-gram ids (see utils.get_ngrams). Raises ValueError if the document has fewer than 4 tokens.
    text = text.strip()
    input_ids = tokenizer(text, return_attention_mask=False, return_tensors="np")["input_ids"][0]

    # Dirty implem. TODO: update
    ngrams = (
        get_ngrams(ids=input_ids, voc_size=vocab_size, n=1)
        + get_ngrams(ids=input_ids, voc_size=vocab_size, n=2)
        + get_ngrams(ids=input_ids, voc_size=vocab_size, n=3)
        + get_ngrams(ids=input_ids, voc_size=vocab_size, n=4)
    )
    return ngrams


def process_docs(documents: List[str], subshard_idx: int):
    # [FUNCTION DOC] Iterate over documents (any iterable of strings, e.g. an open file), assigning ids
    # starting at
    # subshard_idx * nb_docs_per_subshard + 1, and print one CSV line per n-gram id. Documents that raise
    # ValueError (too short) are skipped with a logged warning; the counter still advances.
    doc_counter = subshard_idx * args.nb_docs_per_subshard + 1
    for doc in documents:
        try:
            ngrams = process_document(doc)
            for id in ngrams:
                print(f"{SHARD_NAME},{doc_counter},{id}")
        except ValueError:
            logging.warning(f"Document nb {doc_counter} is invalid. Skipping.\nDoc = {doc}.")
            # TODO: get some way of logging number of skipped documents (not at the process level tough...)
        doc_counter += 1


if __name__ == "__main__":
    input_filepath = args.filepath
    subshard_idx = int(input_filepath.split(".")[-1])
    with open(input_filepath, "r") as file:
        process_docs(file, subshard_idx)
    sys.stdout.flush()
