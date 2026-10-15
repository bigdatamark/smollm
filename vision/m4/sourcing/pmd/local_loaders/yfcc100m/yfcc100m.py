# coding=utf-8
# [MODULE DOC] HuggingFace `datasets` loading script for the PMD pre-processed YFCC100M subset. Downloads a single JSONL
# annotation file from the facebook/pmd hub repo and yields one example per line (TRAIN split only);
# images are referenced by URL only. Citation/description/homepage/license are empty placeholders.
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
"""YFCC100M - PMD pre-processed version"""
import json

import datasets


# TODO: @thomasw21
_CITATION = """"""

# TODO: @thomasw21
_DESCRIPTION = """"""

# TODO: @thomasw21
_HOMEPAGE = ""

# TODO: @thomasw21
_LICENSE = ""

_ANNOTATION_URL = "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/yfcc100m_subset.jsonl"

_FEATURES = datasets.Features(
    {
        "image_url": datasets.Value("string"),
        "text": datasets.Value("string"),
        "source": datasets.Value("string"),
        "meta": datasets.Value("string"),
    }
)


class YFCC100M(datasets.GeneratorBasedBuilder):
    # [CLASS DOC] `datasets.GeneratorBasedBuilder` with string features image_url, text, source and meta.
    """YFCC100M - PMD pre-processed version"""

    # @property
    # def manual_download_instructions(self):
    #     return """\
    # You need to go to https://huggingface.co/datasets/facebook/pmd/tree/main,
    # and manually download the file ("yfcc100m_subset.jsonl") created by Aman
    # and place it under <path/to/folder>.
    # The <path/to/folder> can e.g. be "~/manual_data".
    # dataset can then be loaded using the following command `datasets.load_dataset("yfcc100m.py", data_dir="<path/to/folder>")`.
    # """

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
        # [FUNCTION DOC] Download the JSONL annotation file via `dl_manager` (network/cache side effect) and return a single TRAIN
        # `SplitGenerator` whose kwargs carry the local annotation file path.
        annotation_file = dl_manager.download(_ANNOTATION_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file,
                },
            )
        ]

    def _generate_examples(self, annotation_file):
        # [FUNCTION DOC] Read the JSONL file line by line and yield (line index, example) where the example has image_url, the
        # first element of the record's "texts" list as text, source and meta copied from the record.
        with open(annotation_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                yield idx, {
                    "image_url": data["image_url"],
                    "text": data["texts"][0],
                    "source": data["source"],
                    "meta": data["meta"],
                }
