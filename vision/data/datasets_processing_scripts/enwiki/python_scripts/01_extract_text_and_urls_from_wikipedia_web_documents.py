# [MODULE DOC] Step 01 of the enwiki pipeline: converts one shard of the Wikipedia HTML Enterprise dump into "web documents"
# (interleaved lists of texts and image URLs) and saves them with save_to_disk.
# Uses the m4 data_collection processors: DOMTreeSimplificator (Wikipedia-specific cleaning, drops everything
# after Notes/References/See_also/Further_reading/External_links and nodes matching some CSS rules),
# PreExtractionSimplificator, and html_to_web_documents. Configures INFO-level logging at import.
# Run as a script: parses CLI args then calls main.
import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_dataset

from m4.sourcing.data_collection.processors import DOMTreeSimplificator, PreExtractionSimplificator
from m4.sourcing.data_collection.processors.web_document_extractor import html_to_web_documents
from m4.sourcing.data_collection.utils import InterestingAttributesSetCategory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    # [FUNCTION DOC] HELPER: Build and parse the CLI arguments.
    # Options: --wikipedia_html_enterprise_dir (Path, dataset data dir), --shard_id (int, shard to process),
    # --extraction_dir (Path, output root), --num_proc (int, default cpu_count()). All others have no default (None).
    # Returns the argparse Namespace.
    parser = argparse.ArgumentParser(description="Extract web documents.")
    # ---------------- Path args ----------------
    parser.add_argument(
        "--wikipedia_html_enterprise_dir",
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
    # [FUNCTION DOC] Run the extraction for one shard.
    # Input: args Namespace from get_args. Loads dataset script "SaulLu/wikipedia_html_enterprise" (split "train",
    # shard=args.shard_id, data_dir=args.wikipedia_html_enterprise_dir), builds the two simplificators with
    # Wikipedia settings, calls html_to_web_documents with args.num_proc processes, and saves the resulting dataset
    # to args.extraction_dir / args.dataset_name, logging start/end. Returns None.
    # KNOWN ISSUE: args.dataset_name is never defined by get_args, so this raises AttributeError before saving.
    html_dataset = load_dataset(
        "SaulLu/wikipedia_html_enterprise",
        shard=args.shard_id,
        data_dir=args.wikipedia_html_enterprise_dir,
        split="train",
    )

    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=True,
        strip_multiple_spaces=True,
        remove_html_comments=True,
        replace_line_break_tags=True,
        unwrap_tags=True,
        strip_tags=False,
        strip_special_divs=True,
        remove_dates=True,
        remove_empty_leaves=True,
        unnest_nodes=True,
        remake_tree=True,
        preserve_img_children=True,
        remove_everything_after_node_id=["Notes", "References", "See_also", "Further_reading", "External_links"],
        css_rules=[
            "[class~='locmap']",
            "[class~='reference']",
            "[role='presentation']",
            "[role~='note']",
        ],
        interesting_attributes_set_cat=InterestingAttributesSetCategory.WIKIPEDIA,
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=True,
        format_texts=True,
        merge_consecutive_text_nodes=True,
        interesting_attributes_set_cat=InterestingAttributesSetCategory.WIKIPEDIA,
    )

    dataset = html_to_web_documents(html_dataset, dom_tree_simplificator, pre_extraction_simplificator, args.num_proc)

    path_save_dir_dataset = args.extraction_dir / args.dataset_name
    logger.info("Starting saving the dataset")
    dataset.save_to_disk(path_save_dir_dataset)
    logger.info("Finished saving the dataset")


if __name__ == "__main__":
    args = get_args()
    main(args)
