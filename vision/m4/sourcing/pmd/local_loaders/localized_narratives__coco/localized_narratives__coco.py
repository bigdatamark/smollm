# coding=utf-8
# [MODULE DOC] `datasets` builder script for the COCO subset of Localized Narratives (as packaged for PMD):
# downloads the
# caption JSONL files (facebook/pmd hub repo), the COCO 2017 train/val image zips and the Karpathy split file,
# and yields one example per narrative, skipping images in the Karpathy val/test splits (evaluation leakage).
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Localized narratives - COCO subset"""
import json
import os
from pathlib import Path

import datasets


# TODO: @thomasw21
_CITATION = """"""

# TODO: @thomasw21
_DESCRIPTION = """"""

# TODO: @thomasw21
_HOMEPAGE = ""

# TODO: @thomasw21
_LICENSE = ""

_ANNOTATION_URLs = {
    "train": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_coco_train_captions_multi.jsonl",
    "validation": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_coco_val_captions_multi.jsonl",
}

_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "validation": "http://images.cocodataset.org/zips/val2017.zip",
}

_KARPATHY_FILES_URL = "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/caption_datasets.zip"

_SPLIT_MAP = {"train": "train2017", "validation": "val2017"}

_FEATURES = datasets.Features(
    {
        "dataset_id": datasets.Value("string"),
        "image_id": datasets.Value("string"),
        "annotator_id": datasets.Value("int32"),
        "caption": datasets.Value("string"),
        "original_caption": datasets.Value("string"),
        "image": datasets.Image(),
    }
)


class LocalizedNarrativesCOCO(datasets.GeneratorBasedBuilder):
    # [CLASS DOC] GeneratorBasedBuilder with train and validation splits.
    """Builder for COCO subset of Localized Narratives."""

    def _info(self):
        # [FUNCTION DOC] Return DatasetInfo with this module's `_FEATURES` (citation/description/license are
        # empty TODOs).
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # [FUNCTION DOC] Download annotations, image zips and the Karpathy file; build invalid_images (COCO
        # id -> 1 for Karpathy val
        # and test images); return one split generator per annotation split with the annotation file, image
        # dir
        # <extracted>/<train2017|val2017> and invalid_images.
        annotation_files = dl_manager.download(_ANNOTATION_URLs)
        image_folder = dl_manager.download_and_extract(_IMAGES_URLS)
        karpathy_coco_file = os.path.join(dl_manager.download_and_extract(_KARPATHY_FILES_URL), "dataset_coco.json")

        with open(karpathy_coco_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
            invalid_images = {}
            for annotation in annotations["images"]:
                if annotation["split"] == "val" or annotation["split"] == "test":
                    invalid_images[int(annotation["cocoid"])] = 1

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "base_image_path": Path(image_folder[split_name]) / _SPLIT_MAP[split_name],
                    "invalid_images": invalid_images,
                },
            )
            for split_name, annotation_file in annotation_files.items()
        ]

    def _generate_examples(self, annotation_file, base_image_path, invalid_images):
        # [FUNCTION DOC] Yield (counter, example) for each JSONL line whose int image_id is not in
        # invalid_images; image = absolute path
        # <base_image_path>/<image_id zero-padded to 12 digits>.jpg (existence not checked).
        # KNOWN ISSUE: the Karpathy val/test exclusion is applied to the validation split too, removing most
        # of it.
        counter = 0
        with open(annotation_file, "r", encoding="utf-8") as fi:
            for idx, line in enumerate(fi):
                annotation = json.loads(line)
                if int(annotation["image_id"]) in invalid_images:
                    continue
                yield counter, {
                    "dataset_id": annotation["dataset_id"],
                    "image_id": annotation["image_id"],
                    "annotator_id": annotation["annotator_id"],
                    "caption": annotation["caption"],
                    "original_caption": annotation["original_caption"],
                    "image": str((base_image_path / f"{annotation['image_id'].zfill(12)}.jpg").absolute()),
                }
                counter += 1
