# [MODULE DOC] Script: lists the URLs of documents of one HTML shard that contain a "more-link" element (CSS
# "[class~='more-link']"; truncated blog-index style pages), to be banned later by 05_filter_cm4.py.
# Configures INFO logging. When run as main: get_args then main.
import argparse
import logging
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_from_disk
from selectolax.parser import HTMLParser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    # [FUNCTION DOC] Parses CLI args: --arrow_shard_dir (Path), --shard_id (int), --extraction_dir (Path),
    # --shard_name (str),
    # --num_proc (int, default cpu_count(), unused by main). Returns the namespace.
    parser = argparse.ArgumentParser(description="Extract bannel urls.")
    # ---------------- Path args ----------------
    parser.add_argument(
        "--arrow_shard_dir",
        type=Path,
        help="The directory containing the wikipedia html enterprise dataset.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        help="The shard to process.",
    )
    parser.add_argument(
        "--extraction_dir",
        type=Path,
        help="The directory to save all the extracted artifacts",
    )
    parser.add_argument(
        "--shard_name",
        type=str,
        help="The shard name to process.",
    )
    # ---------------- Extraction args ----------------
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


def main(args):
    # [FUNCTION DOC] Loads the "train" split of <arrow_shard_dir>/<shard_name>, filters (32 processes,
    # hard-coded) to documents
    # whose "html" matches the more-link selector and writes their "url" values, one per line, to
    # <extraction_dir>/shard_<shard_id>_banned_urls.txt (overwritten).
    html_dataset = load_from_disk(args.arrow_shard_dir / args.shard_name)["train"]

    css_rule = "[class~='more-link']"

    def filter_doc_with_matching_class(example, css_rule):
        # [FUNCTION DOC] HELPER: filter predicate; True if example["html"] contains a node matching css_rule
        # (selectolax).
        current_html = example["html"]
        tree = HTMLParser(current_html)
        match = tree.css_first(css_rule)
        if match:
            return True
        return False

    html_dataset = html_dataset.filter(partial(filter_doc_with_matching_class, css_rule=css_rule), num_proc=32)

    with open(args.extraction_dir / f"shard_{args.shard_id}_banned_urls.txt", "w") as f:
        f.write("\n".join(html_dataset["url"]))


if __name__ == "__main__":
    args = get_args()
    main(args)
