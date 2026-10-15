# coding=utf-8
# [MODULE DOC] HuggingFace `datasets` loading script for the Open Images subset of Localized Narratives, as
# pre-processed for PMD (Public Multimodal Dataset). Downloads per-split JSONL annotation files from the
# facebook/pmd hub repo and yields one example per line, with the image referenced by its public S3 URL
# (images themselves are not downloaded). Citation/description/homepage/license are empty placeholders.
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
"""Localized narratives - Open Images subset"""
import json
from copy import deepcopy

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
    "train": (
        "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_open_images_train_v6_captions_multi.jsonl"
    ),
    "validation": (
        "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_open_images_validation_captions_multi.jsonl"
    ),
    "test": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_open_images_test_captions_multi.jsonl",
}

# # TODO @thomasw21: We download ALL Open-images from a tar file (that also has bounding boxes but oh well), and we filter only those we have annotations for
# _IMAGES = {}

_FEATURES = datasets.Features(
    {
        "dataset_id": datasets.Value("string"),
        "image_id": datasets.Value("string"),
        "annotator_id": datasets.Value("int32"),
        "caption": datasets.Value("string"),
        "original_caption": datasets.Value("string"),
        "image_url": datasets.Value("string"),
    }
)


def _split_to_single_caption(annotations):
    # [FUNCTION DOC] HELPER: split each annotation's multi-sentence "caption" paragraph on "." into one annotation per sentence.
    # Input: iterable of annotation mappings with a "caption" key. Output: new list of deep copies, one per
    # non-empty sentence, with "caption" set to the stripped sentence plus a trailing "." and "original_caption"
    # set to the full paragraph. Not used by the builder in this file (annotations are already split upstream).
    """This function is mainly used in Localized Narratives where a paragraph can contain
    multiple relevant captions to a single image. We split the paragraph into multiple
    captions and then return each as an individual sample.
    """
    extended = []
    for annotation in annotations:
        captions = annotation["caption"].split(".")
        for caption in captions:
            if len(caption.strip()) == 0 or caption.strip() == ".":
                continue
            current = deepcopy(annotation)
            current["caption"] = caption.strip() + "."
            current["original_caption"] = annotation["caption"]
            extended.append(current)
    return extended


class LocalizedNarrativesOpenImages(datasets.GeneratorBasedBuilder):
    # [CLASS DOC] `datasets.GeneratorBasedBuilder` for Localized Narratives / Open Images with features dataset_id,
    # image_id, annotator_id (int32), caption, original_caption, image_url (all strings except annotator_id).
    """Builder for Open Images subset of Localized Narratives."""

    def _info(self):
        # [FUNCTION DOC] Return a `datasets.DatasetInfo` built from the module-level description, features, homepage, license and
        # citation constants.
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # [FUNCTION DOC] Download the train/validation/test annotation JSONL files via `dl_manager` (network/cache side effect) and
        # return one `SplitGenerator` per split, passing the local annotation file path and the split name.
        annotation_files = dl_manager.download(_ANNOTATION_URLs)
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"annotation_file": annotation_file, "split": split_name},
            )
            for split_name, annotation_file in annotation_files.items()
        ]

    def _generate_examples(self, annotation_file: str, split: str):
        # [FUNCTION DOC] Read the JSONL annotation file line by line and yield (running integer key, example) pairs. Each example
        # copies dataset_id, image_id, annotator_id, caption and original_caption from the JSON record and builds
        # image_url as the open-images-dataset S3 URL "<split>/<image_id>.jpg".
        counter = 0
        with open(annotation_file, "r", encoding="utf-8") as fi:
            for idx, line in enumerate(fi):
                annotation = json.loads(line)
                yield counter, {
                    "dataset_id": annotation["dataset_id"],
                    "image_id": annotation["image_id"],
                    "annotator_id": annotation["annotator_id"],
                    "caption": annotation["caption"],
                    "original_caption": annotation["original_caption"],
                    "image_url": f"https://s3.amazonaws.com/open-images-dataset/{split}/{annotation['image_id']}.jpg",
                }
                counter += 1
