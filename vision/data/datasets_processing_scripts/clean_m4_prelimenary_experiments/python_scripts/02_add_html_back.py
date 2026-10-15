# [MODULE DOC] Script: re-attaches raw HTML to one processed web-document shard by matching document URLs
# against the C4
# HTML shards. CLI args (all required): --html_dir_path (dir of "c4-*" HTML datasets with "url"/"html"
# columns), --processed_dir_path, --output_dir_path, --shard_idx, --num_proc. Output: the processed shard
# with added "document_url" and "html" columns saved to <output_dir_path>/shard_<shard_idx>.
import json
from pathlib import Path

import tqdm
from datasets import concatenate_datasets, load_from_disk


# from datasets.utils.logging import set_verbosity_info


# set_verbosity_info()


def load_html_ds(dir_path: Path):
    # [FUNCTION DOC] HELPER: loads and concatenates the "train" split of every sub-directory of dir_path
    # (pathlib.Path) whose
    # name starts with "c4-" (others are skipped with a print); tqdm progress. Returns one datasets.Dataset.
    # Raises ValueError (from concatenate_datasets) if no shard is found.
    # Unused by main.
    html_ds_list = []
    for arrow_shard_dir in tqdm.tqdm(dir_path.iterdir()):
        if not arrow_shard_dir.is_dir() or not arrow_shard_dir.name.startswith("c4-"):
            print(f"Skipping {arrow_shard_dir}")
            continue
        print(f"Loading {arrow_shard_dir}")
        html_ds = load_from_disk(arrow_shard_dir)["train"]
        html_ds_list.append(html_ds)
    html_ds = concatenate_datasets(html_ds_list)
    return html_ds


def load_processed_shard(dir_path: Path, shard_idx: int):
    # [FUNCTION DOC] HELPER: loads and returns the dataset saved at dir_path / "shard_<shard_idx>"
    # (load_from_disk).
    shard_dir = dir_path / f"shard_{shard_idx}"
    processed_shard = load_from_disk(shard_dir)
    return processed_shard


def add_doc_url(example):
    # [FUNCTION DOC] datasets.map function: parses the example's JSON "metadata" list and sets
    # example["document_url"] to the
    # "document_url" field of the first non-None entry (None if all entries are None). Returns the mutated
    # example.
    metadata = json.loads(example["metadata"])
    document_url = None
    for meta in metadata:
        if meta is not None:
            document_url = meta["document_url"]
            break
    example["document_url"] = document_url
    return example


def get_args():
    # [FUNCTION DOC] Parses and returns the CLI arguments (paths as pathlib.Path, shard_idx and num_proc as
    # int, all required).
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--html_dir_path", type=Path, required=True)
    parser.add_argument("--processed_dir_path", type=Path, required=True)
    parser.add_argument("--output_dir_path", type=Path, required=True)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--num_proc", type=int, required=True)
    return parser.parse_args()


def main():
    # [FUNCTION DOC] Skips (prints and returns) if the output already exists. Loads the processed shard, adds
    # document_url,
    # then iterates over the entries of html_dir_path (filesystem order), skipping non "c4-" dirs and those
    # whose
    # enumeration index is below shard_idx; for each C4 shard builds a url -> row index mapping and fills the
    # "html" column of rows still lacking it (None when not found). Stops early once every row has HTML, then
    # saves the dataset and prints the path.
    # KNOWN ISSUE: skipping uses the enumeration index of iterdir() (unordered, counting non-c4 entries) as
    # if it
    # were a shard number, so which C4 shards are searched is arbitrary and some matches can be missed.
    args = get_args()

    save_path = args.output_dir_path / f"shard_{args.shard_idx}"
    if save_path.exists():
        print(f"==================== Dataset already exists at {save_path} ====================")
        return

    processed_shard_ds = load_processed_shard(args.processed_dir_path, args.shard_idx)
    print(f"Loaded {len(processed_shard_ds)} processed examples")

    processed_shard_ds = processed_shard_ds.map(add_doc_url, num_proc=args.num_proc)

    for html_shard_idx, arrow_shard_dir in enumerate(tqdm.tqdm(args.html_dir_path.iterdir())):
        if not arrow_shard_dir.is_dir() or not arrow_shard_dir.name.startswith("c4-"):
            print(f"Skipping {arrow_shard_dir}")
            continue
        if html_shard_idx < args.shard_idx:
            print(f"Skipping {arrow_shard_dir}")
            continue

        print(f"Loading {arrow_shard_dir}")
        html_ds = load_from_disk(arrow_shard_dir)["train"]
        print("building url_to_index dict")
        url_to_index = {url: idx for (idx, url) in enumerate(html_ds["url"])}
        print("built url_to_index dict")

        def get_html_from_c4(example):
            # [FUNCTION DOC] HELPER: map function; leaves examples that already have non-None "html"
            # untouched, otherwise looks up
            # example["document_url"] in the closure's url_to_index and sets example["html"] to the matching
            # C4 html
            # (or None). Returns the example.
            if example.get("html", None) is not None:
                return example

            document_url = example["document_url"]
            html_idx = url_to_index.get(document_url, None)
            if html_idx is not None:
                html = html_ds[html_idx]["html"]
            else:
                html = None
            example["html"] = html
            return example

        processed_shard_ds = processed_shard_ds.map(get_html_from_c4, num_proc=args.num_proc)

        not_none = len([1 for html in processed_shard_ds["html"] if html is not None])

        print(f"Found html for {not_none} examples out of {len(processed_shard_ds)}")

        if not_none == len(processed_shard_ds):
            print("Found html for all examples, breaking")
            break

    processed_shard_ds.save_to_disk(save_path)
    print(f"==================== Saved to {save_path} ====================")


if __name__ == "__main__":
    main()
