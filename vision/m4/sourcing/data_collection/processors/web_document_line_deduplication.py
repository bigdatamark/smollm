# [MODULE DOC] Per-domain paragraph ("line") deduplication of a sharded interleaved web-document dataset.
# Paragraphs
# (texts split on blank lines) that occur more than once among all documents of the same domain (netloc of
# `document_url` in metadata) are treated as boilerplate and removed. Three steps: domain -> positions,
# domain -> duplicated paragraphs, then rewrite one shard. Configures root logging at INFO on import.
import json
import logging
import os
from glob import glob
from urllib.parse import urlparse

from datasets import load_from_disk
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WebDocumentLineDeduplication:
    # [CLASS DOC] Driver for the three-step line deduplication; intermediate results are JSON files on disk.
    def __init__(
        self,
        path_sharded_dataset,
        path_save_domain_to_positions,
        path_save_domain_to_duplicated_texts,
        id_shard_to_line_deduplicate,
        num_proc,
        path_save_line_deduplicated_sharded_dataset,
    ):
        # [FUNCTION DOC] Store paths and params: sharded dataset dir (subdirs `shard_<id>`), JSON output
        # paths, the shard id to
        # rewrite in the last step, `num_proc`, and the output dir for deduplicated shards. No I/O.
        self.path_sharded_dataset = path_sharded_dataset
        self.path_save_domain_to_positions = path_save_domain_to_positions
        self.path_save_domain_to_duplicated_texts = path_save_domain_to_duplicated_texts
        self.id_shard_to_line_deduplicate = id_shard_to_line_deduplicate
        self.num_proc = num_proc
        self.path_save_line_deduplicated_sharded_dataset = path_save_line_deduplicated_sharded_dataset

    def get_paths_subdatasets(self):
        # [FUNCTION DOC] Set `self.paths_subdatasets` to all subdirectories of the sharded dataset dir (glob,
        # trailing slash).
        self.paths_subdatasets = glob(f"{self.path_sharded_dataset}/*/")

    def remove_empty_els_in_list(self, list_):
        # [FUNCTION DOC] HELPER: return the list with None elements removed.
        return [el for el in list_ if el is not None]

    def get_domain_to_positions(self):
        # [FUNCTION DOC] For every shard (needs `get_paths_subdatasets` first), read each doc's metadata,
        # take the first non-null
        # node's `document_url` domain, and build `self.domain_to_positions`: domain -> {shard path -> [row
        # indices]}.
        # Saves it as JSON to `path_save_domain_to_positions`; logs; loads shards with only the metadata
        # column.
        logger.info(
            "Starting creating the dictionary to go from a domain name to positions in the web document dataset"
        )
        self.domain_to_positions = {}

        for path_subdatasets in tqdm(self.paths_subdatasets):
            sub_ds = load_from_disk(path_subdatasets)
            sub_ds = sub_ds.remove_columns([c_n for c_n in sub_ds.column_names if c_n != "metadata"])
            metadata_sub_ds = sub_ds["metadata"]
            metadata_sub_ds = [json.loads(meta) for meta in metadata_sub_ds]
            metadata_sub_ds = [self.remove_empty_els_in_list(meta)[0] for meta in metadata_sub_ds]
            domains = [urlparse(meta["document_url"]).netloc for meta in metadata_sub_ds]

            new_domain_to_pos = {}
            for idx, domain in enumerate(domains):
                new_domain_to_pos[domain] = new_domain_to_pos.get(domain, []) + [idx]
            for domain in new_domain_to_pos:
                if domain not in self.domain_to_positions:
                    self.domain_to_positions[domain] = {}
                self.domain_to_positions[domain][path_subdatasets] = new_domain_to_pos[domain]

        with open(self.path_save_domain_to_positions, "w") as f:
            json.dump(self.domain_to_positions, f)
        logger.info(
            "Finished creating and saving the dictionary to go from a domain name to positions in the web document"
            " dataset"
        )

    def get_domain_to_duplicated_texts(self):
        # [FUNCTION DOC] Load domain->positions JSON; for each domain, count occurrences of each paragraph
        # over all its docs' non-null
        # texts (split on "\n\n") and keep those with count > 1 in `self.domain_to_duplicated_texts`
        # (domain -> {paragraph: count}); save as JSON. Loads each shard once per domain (slow).
        # KNOWN ISSUE: a paragraph repeated twice inside a single document also counts as "duplicated" for
        # the domain.
        logger.info("Starting finding the duplicated texts for each domain")
        with open(self.path_save_domain_to_positions) as f:
            self.domain_to_positions = json.load(f)

        self.domain_to_duplicated_texts = {}

        for domain in tqdm(self.domain_to_positions):
            duplicated_texts = {}
            positions = self.domain_to_positions[domain]

            for path_subdatasets in positions:
                sub_ds = load_from_disk(path_subdatasets)
                sub_ds = sub_ds.remove_columns([c_n for c_n in sub_ds.column_names if c_n != "texts"])
                idx_pos = positions[path_subdatasets]

                for idx in idx_pos:
                    tot_texts = self.remove_empty_els_in_list(sub_ds[idx]["texts"])
                    tot_texts = [text.split("\n\n") for text in tot_texts]
                    tot_texts = [paragraph for text in tot_texts for paragraph in text]
                    for text in tot_texts:
                        duplicated_texts[text] = duplicated_texts.get(text, 0) + 1

            duplicated_texts = {k: v for k, v in duplicated_texts.items() if v > 1}
            self.domain_to_duplicated_texts[domain] = duplicated_texts

        with open(self.path_save_domain_to_duplicated_texts, "w") as f:
            json.dump(self.domain_to_duplicated_texts, f)
        logger.info("Finished finding and saving the duplicated texts for each domain")

    def line_deduplicate_web_documents(self):
        # [FUNCTION DOC] Load domain->duplicated paragraphs JSON, rewrite shard
        # `shard_<id_shard_to_line_deduplicate>` by removing
        # duplicated paragraphs from every text node, drop nodes whose text became empty, and save it under
        # the
        # output dir with the same shard name (creates the dir via a `mkdir -p` shell call).
        logger.info(
            f"Starting line deduplicating the web document dataset for shard {self.id_shard_to_line_deduplicate}"
        )
        with open(self.path_save_domain_to_duplicated_texts) as f:
            self.domain_to_duplicated_texts = json.load(f)

        def func_mac_line_deduplicate_web_documents(example):
            # [FUNCTION DOC] HELPER: map function for one doc: determine domain from first non-null metadata,
            # filter paragraphs, and
            # drop emptied text nodes from texts/images/metadata. Raises KeyError if the domain is not in the
            # mapping.
            metadata = json.loads(example["metadata"])
            domain = urlparse(self.remove_empty_els_in_list(metadata)[0]["document_url"]).netloc

            indices_to_remove = set()
            for idx in range(len(example["texts"])):
                if example["texts"][idx] is not None:
                    example["texts"][idx] = "\n\n".join(
                        [
                            paragraph
                            for paragraph in example["texts"][idx].split("\n\n")
                            if paragraph not in self.domain_to_duplicated_texts[domain]
                        ]
                    )
                    if not example["texts"][idx]:
                        indices_to_remove.add(idx)

            if indices_to_remove:
                example["texts"] = [el for ind, el in enumerate(example["texts"]) if ind not in indices_to_remove]
                example["images"] = [el for ind, el in enumerate(example["images"]) if ind not in indices_to_remove]
                example["metadata"] = json.dumps(
                    [el for ind, el in enumerate(metadata) if ind not in indices_to_remove]
                )

            return example

        os.system(f"mkdir -p {self.path_save_line_deduplicated_sharded_dataset}")

        path_subdataset = os.path.join(self.path_sharded_dataset, f"shard_{self.id_shard_to_line_deduplicate}")
        sub_ds = load_from_disk(path_subdataset)
        sub_ds_line_deduplicated = sub_ds.map(func_mac_line_deduplicate_web_documents, num_proc=self.num_proc)
        name_shard = os.path.basename(os.path.normpath(path_subdataset))
        sub_ds_line_deduplicated.save_to_disk(
            os.path.join(self.path_save_line_deduplicated_sharded_dataset, name_shard)
        )

        logger.info(
            f"Finished line deduplicating the web document dataset for shard {self.id_shard_to_line_deduplicate}"
        )
