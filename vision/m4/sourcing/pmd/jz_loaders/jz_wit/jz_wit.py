# coding=utf-8
# [MODULE DOC] `datasets` builder script for WIT (Wikipedia-based Image Text) on the Jean Zay cluster,
# English rows only.
# Reads the 10 WIT train TSV shards and an image index TSV ($DSDIR/WIT/train_images.tsv with url, downloaded,
# path) and outputs the PMD schema (image, text, source="google/wit", meta JSON with the WIT row fields).
# Images are opened/verified in a multiprocessing pool per chunk of CHUNK_SIZE rows. Sets datasets logging to
# INFO.
# KNOWN ISSUE: `_URLs` reads os.environ['DSDIR'] at import time, so importing outside JZ raises KeyError.
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wikipedia-based Image Text (WIT) Dataset is a large multimodal multilingual dataset. JZ Version"""
import csv
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import datasets
import PIL.Image
import pyarrow as pa
from datasets.utils import logging

from m4.sourcing.pmd import _FEATURES, get_jz_dataset_dir
from m4.sourcing.pmd.helpers import json_serializer


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{srinivasan2021wit,
  title={WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning},
  author={Srinivasan, Krishna and Raman, Karthik and Chen, Jiecao and Bendersky, Michael and Najork, Marc},
  journal={arXiv preprint arXiv:2103.01913},
  year={2021}
}
"""

# You can copy an official description
_DESCRIPTION = """\
Wikipedia-based Image Text (WIT) Dataset is a large multimodal multilingual dataset.
WIT is composed of a curated set of 37.6 million entity rich image-text examples with 11.5 million unique images across 108 Wikipedia languages.
Its size enables WIT to be used as a pretraining dataset for multimodal machine learning models.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/wit"

_LICENSE = "Data is available under the Creative Commons Attribution-ShareAlike 3.0 Unported license."

# _URLs = [f"https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-{i:05}-of-00010.tsv.gz" for i in range(0, 10)]
_URLs = [f"{os.environ['DSDIR']}/WIT/wit_v1.train.all-{i:05}-of-00010.tsv" for i in range(0, 10)]


class JZWIT(datasets.ArrowBasedBuilder):
    # [CLASS DOC] ArrowBasedBuilder with the default config, all CPUs (num_proc) and chunks of 1000 rows;
    # train split only.
    """Builder for WIT."""

    num_proc = cpu_count()
    CHUNK_SIZE = 1_000

    def _info(self):
        # [FUNCTION DOC] Return DatasetInfo with the PMD `_FEATURES` schema, description, homepage, license
        # and citation.
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # [FUNCTION DOC] Return a single TRAIN split generator with files=_URLs and paths=<JZ dataset
        # dir>/WIT/train_images.tsv.
        # files = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": _URLs, "paths": get_jz_dataset_dir() / "WIT" / "train_images.tsv"},
            ),
        ]

    def _generate_examples(self, files, paths):
        # [FUNCTION DOC] HELPER: build url -> (downloaded flag, absolute local path) from the image index
        # TSV, then stream every WIT
        # shard and yield chunks (lists of up to CHUNK_SIZE dicts {image_path or None, text, source, meta})
        # for
        # language "en" rows. Caption choice: reference description if >= 2 words; else the English part of
        # the
        # attribution description (text after "English:"; if another language follows, its last word --
        # assumed to be
        # the next language name -- is dropped) if >= 3 words; else alt text if >= 3 words; else row skipped.
        # Non-ASCII
        # characters are removed and empty captions skipped. KeyError if an image url is missing from the
        # index.
        # KNOWN ISSUE: `bool(instance["downloaded"])` is True for any non-empty string (including
        # "False"/"0").
        # KNOWN ISSUE: CSV fields are never None (empty string instead), so the `is not None` checks are
        # always true.
        image_paths = {}
        with open(paths, encoding="utf-8") as f:
            for instance in csv.DictReader(f, delimiter="\t"):
                image_paths[instance["url"]] = (
                    bool(instance["downloaded"]),
                    os.path.join(os.environ["DSDIR"], "WIT", instance["path"]),
                )
        logger.info("Loaded paths.")

        buffer = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                examples = csv.DictReader(f, delimiter="\t")
                for example in examples:
                    caption = None
                    if example["language"] != "en":
                        continue

                    if example["caption_reference_description"] is not None:
                        caption = example["caption_reference_description"]
                        if len(caption.split(" ")) < 2:
                            caption = None

                    if caption is None and example["caption_attribution_description"] is not None:
                        if "english:" in example["caption_attribution_description"].lower():
                            attribution = example["caption_attribution_description"]
                            # Splits usually occurs as "English: [Text] Italian: [Text]""
                            # so we split and take relevant section
                            splits = attribution.split(": ")
                            for idx, split in enumerate(splits):
                                clean = split.strip().lower()
                                if clean.endswith("english") or clean.startswith("english"):
                                    if idx + 1 < len(splits):
                                        caption = splits[idx + 1].strip()
                                    # Case of extra languages at the end: English: [Text] Italian: [Text]
                                    if idx + 2 < len(splits):
                                        caption = " ".join(caption.split(" ")[:-1]).strip()

                        if caption is not None and len(caption.split(" ")) < 3:
                            caption = None

                    if caption is None and example["caption_alt_text_description"] is not None:
                        caption = example["caption_alt_text_description"]
                        if len(caption.split(" ")) < 3:
                            caption = None

                    if caption is None:
                        continue

                    caption = "".join(c for c in caption if ord(c) < 128)
                    if len(caption) == 0:
                        continue

                    metadata_dict = {
                        "caption_reference_description": example["caption_reference_description"],
                        "caption_attribution_description": example["caption_attribution_description"],
                        "caption_alt_text_description": example["caption_alt_text_description"],
                        "page_url": example["page_url"],
                        "page_title": example["page_title"],
                        "section_title": example["section_title"],
                        "hierarchical_section_title": example["hierarchical_section_title"],
                        "mime_type": example["mime_type"],
                        "original_height": example["original_height"],
                        "original_width": example["original_width"],
                        "attribution_passes_lang_id": example["attribution_passes_lang_id"],
                        "page_changed_recently": example["page_changed_recently"],
                        "context_page_description": example["context_page_description"],
                        "context_section_description": example["context_section_description"],
                        "image_url": example["image_url"],
                        "language": example["language"],
                        "is_main_image": example["is_main_image"],
                    }
                    image_downloaded, image_path = image_paths[example["image_url"]]
                    if image_downloaded:
                        buffer.append(
                            {
                                "image_path": image_path,
                                "text": caption,
                                "source": "google/wit",
                                "meta": json.dumps(metadata_dict, default=json_serializer, indent=2),
                            }
                        )
                    else:
                        buffer.append(
                            {
                                "image_path": None,
                                "text": caption,
                                "source": "google/wit",
                                "meta": json.dumps(metadata_dict, default=json_serializer, indent=2),
                            }
                        )
                    if len(buffer) == self.CHUNK_SIZE:
                        yield buffer
                        buffer = []
        if len(buffer) > 0:
            yield buffer

    def _generate_tables(self, files, paths):
        # [FUNCTION DOC] Yield (index, pyarrow Table) for each chunk, built in parallel with a Pool(num_proc)
        # `imap` (order kept)
        # over _create_table, then log warnings with the totals of missing / unopenable / unverifiable images.
        rows_iterator = self._generate_examples(files, paths)

        counter_images_not_found = 0
        count_images_opening_error = 0
        counter_images_not_verified = 0
        with Pool(self.num_proc) as pool:
            tables_iterator = pool.imap(
                partial(self._create_table),
                rows_iterator,
                chunksize=1,
            )
            for idx, (not_found, opening_error, not_verified, table) in enumerate(tables_iterator):
                yield idx, table
                counter_images_not_found += not_found
                count_images_opening_error += opening_error
                counter_images_not_verified += not_verified
        logger.warning(f"Skipped {counter_images_not_found} images (not found).")
        logger.warning(f"Skipped {count_images_opening_error} images (opening error).")
        logger.warning(f"Skipped {counter_images_not_verified} images (not verified).")

    def _create_table(self, examples):
        # [FUNCTION DOC] HELPER: for each example open the image with PIL if the path exists and verify it;
        # failures set the image to
        # None and increment the not-found / opening-error / not-verified counters. Replaces image_path by
        # image
        # (closed after verify) and returns (not_found, opening_error, not_verified, Arrow table encoded with
        # _FEATURES).
        # KNOWN ISSUE: when opening fails the image is None and verify() then also fails, so the image is
        # counted in
        # both the opening-error and not-verified counters.
        counter_images_not_found = 0
        count_images_opening_error = 0
        counter_images_not_verified = 0
        result = []

        for ex in examples:
            image_path = ex["image_path"]
            if image_path is not None and os.path.exists(image_path):
                try:
                    image = PIL.Image.open(image_path)
                except PIL.UnidentifiedImageError:
                    count_images_opening_error += 1
                    # logger.warning(f"Loading image with path {image_path} returned an `PIL.UnidentifiedImageError`. Skipping this image.")
                    image = None
                except PIL.Image.DecompressionBombError:
                    count_images_opening_error += 1
                    # logger.warning(f"Loading image with path {image_path} returned an `PIL.Image.DecompressionBombError`. Skipping this image")
                    image = None
                except Exception:
                    count_images_opening_error += 1
                    # logger.warning(f"Loading image with path {image_path} returned an unexpected error: {e}. Skipping this image")
                    image = None

                try:
                    image.verify()
                except Exception:
                    counter_images_not_verified += 1
                    # logger.warning(f"Verifying image with path {image_path} raised an error {e}. Skipping this image.")
                    image = None
            else:
                counter_images_not_found += 1
                # logger.warning(f"Image with path {image_path} could not be found. Skipping this image.")
                image = None
            del ex["image_path"]
            ex["image"] = image
            if image is not None:
                image.close()
            result.append(ex)

        return (
            counter_images_not_found,
            count_images_opening_error,
            counter_images_not_verified,
            pa.table(_FEATURES.encode_batch(self._recast(result))),
        )

    def _recast(self, examples):
        # [FUNCTION DOC] HELPER: convert a list of example dicts into a column mapping {image, text, source,
        # meta}.
        return {
            "image": [ex["image"] for ex in examples],
            "text": [ex["text"] for ex in examples],
            "source": [ex["source"] for ex in examples],
            "meta": [ex["meta"] for ex in examples],
        }
