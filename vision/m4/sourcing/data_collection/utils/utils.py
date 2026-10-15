# [MODULE DOC] Small helpers to stream the HTML dataset and parse HTML with selectolax.
from datasets import load_dataset
from selectolax.parser import HTMLParser


def load_dataset_html(shuffle=False, buffer_size=10000, seed=42):
    # [FUNCTION DOC] Return an iterator over the streamed train split of the HF hub dataset
    # `bs-modeling-metadata/c4-en-html-with-metadata` (requires an auth token). If shuffle, applies a buffered
    # streaming shuffle (`buffer_size`, `seed`).
    dataset = load_dataset(
        "bs-modeling-metadata/c4-en-html-with-metadata",
        streaming=True,
        split="train",
        use_auth_token=True,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = iter(dataset)
    return dataset


def make_selectolax_tree(html_str):
    # [FUNCTION DOC] Parse an HTML string and return the selectolax HTMLParser tree.
    selectolax_tree = HTMLParser(html_str)
    return selectolax_tree
