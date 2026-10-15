# [MODULE DOC] Script (runs at import) creating "-Sample" versions of several evaluation datasets without class balancing:
# for each entry of name_datasets, load all splits (optional config/data_dir), shuffle with seed 42 and keep the
# first num_examples rows of every split, push to HuggingFaceM4/<path>[-<config>]-Sample (private) and save to
# the Jean Zay local datasets folder. For NoCaps (no train split) a train split is synthesised from COCO 2014
# captions and Localized Narratives OpenImages captions mapped to the NoCaps schema.
# KNOWN ISSUE: select(range(n)) fails on splits with fewer than n rows; the NoCaps train shuffle result is
# discarded (so the first rows of COCO are kept, not a random sample).
"""
Simplified as it does not care about class balanceness (as opposed to `create_sample_evaluation_datasets.py`)
"""
import os
from itertools import chain

from datasets import concatenate_datasets, load_dataset


# NoCaps utils


def _split_to_single_caption(caption):
    # [FUNCTION DOC] HELPER: split a paragraph on "." into non-empty stripped sentences, each re-terminated with "."; returns list.
    """This function is mainly used in Localized Narratives where a paragraph can contain
    multiple relevant captions to a single image. We split the paragraph into multiple
    captions and then return each as an individual sample.
    """
    extended = []
    captions = caption.split(".")
    for c in captions:
        if len(c.strip()) == 0 or c.strip() == ".":
            continue
        extended.append(c.strip() + ".")
    return extended


def map_localized_narratives_to_nocaps(ex):
    # [FUNCTION DOC] HELPER: map a Localized Narratives example to the NoCaps columns: keep image, split its captions into single
    # sentences as annotations_captions with ids 0..n-1, fill all other NoCaps metadata with empty/zero values.
    annotations_captions = list(chain.from_iterable([_split_to_single_caption(caption) for caption in ex["captions"]]))
    return {
        "image": ex["image"],
        "image_coco_url": "",
        "image_date_captured": "",
        "image_file_name": "",
        "image_height": 0,
        "image_width": 0,
        "image_id": 0,
        "image_license": 0,
        "image_open_images_id": "",
        "annotations_ids": list(range(len(annotations_captions))),
        "annotations_captions": annotations_captions,
    }


def map_coco_to_nocaps(ex):
    # [FUNCTION DOC] HELPER: map a COCO captions example to the NoCaps columns: image, sentences_raw as annotations_captions with
    # ids 0..n-1, other metadata empty/zero.
    annotations_captions = ex["sentences_raw"]
    return {
        "image": ex["image"],
        "image_coco_url": "",
        "image_date_captured": "",
        "image_file_name": "",
        "image_height": 0,
        "image_width": 0,
        "image_id": 0,
        "image_license": 0,
        "image_open_images_id": "",
        "annotations_ids": list(range(len(annotations_captions))),
        "annotations_captions": annotations_captions,
    }


# Real extraction

default_num_examples_kept = 1000
path_local_datasets_jz = "/gpfsscratch/rech/cnw/commun/local_datasets/"
seed = 42

name_datasets = [
    {"path": "HuggingFaceM4/VQAv2_modif", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/OK-VQA_modif", "num_examples": default_num_examples_kept},
    {"path": "textvqa", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/FairFace", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/TextCaps", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/common_gen", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/clevr", "num_examples": default_num_examples_kept, "config": "classification"},
    # data_dir depends on where you launch the job from. For SNLI-VE The directory needs to contain the following file: flickr30k-images.tar.gz
    # In JZ, this is in "$cnw_ALL_CCFRSCRATCH/local_datasets". In GCP, it is in "hf-science-m4/raw_datasets".
    {
        "path": "HuggingFaceM4/SNLI-VE",
        "num_examples": default_num_examples_kept,
        "data_dir": "/Users/leotronchon/Downloads",
    },
    {
        "path": "HuggingFaceM4/SNLI-VE_modif_premise_hypothesis",
        "num_examples": default_num_examples_kept,
        "data_dir": "/Users/leotronchon/Downloads",
    },
    {"path": "HuggingFaceM4/NoCaps", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/COCO", "num_examples": default_num_examples_kept, "config": "2014_captions"},
    {"path": "HuggingFaceM4/IIIT-5K-classif", "num_examples": 500},
    {"path": "HuggingFaceM4/IIIT-5K", "num_examples": default_num_examples_kept},
    {"path": "HuggingFaceM4/NLVR2", "num_examples": default_num_examples_kept},
]


for config in name_datasets:
    path_ds = config["path"]
    config_name = config.get("config", None)
    data_dir = config.get("data_dir", None)

    ds_sample_name = path_ds if config_name is None else path_ds + "-" + config_name
    ds_sample_name = ds_sample_name + "-Sample"
    if not ds_sample_name.startswith("HuggingFaceM4/"):  # To make textvqa work
        ds_sample_name = "HuggingFaceM4/" + ds_sample_name

    num_examples_kept = config.get("num_examples", default_num_examples_kept)
    ds = load_dataset(path_ds, name=config_name, data_dir=data_dir, use_auth_token=True)
    for split in ds:
        ds[split] = ds[split].shuffle(seed=seed)
        ds[split] = ds[split].select(range(num_examples_kept))

    if path_ds == "HuggingFaceM4/NoCaps":
        assert "train" not in ds
        coco_train = load_dataset("HuggingFaceM4/COCO", "2014_captions", split="train", use_auth_token=True)
        mapped_coco_train = coco_train.map(
            map_coco_to_nocaps,
            remove_columns=list(set(coco_train.column_names) - set(ds["validation"].column_names)),
            num_proc=os.cpu_count(),
        )
        ln_openimages_train = load_dataset(
            "HuggingFaceM4/LocalizedNarratives", "OpenImages_captions", split="train", use_auth_token=True
        )
        mapped_ln_openimages_train = ln_openimages_train.map(
            map_localized_narratives_to_nocaps,
            remove_columns=list(set(ln_openimages_train.column_names) - set(ds["validation"].column_names)),
            num_proc=os.cpu_count(),
        )
        ds["train"] = concatenate_datasets([mapped_coco_train, mapped_ln_openimages_train])
        ds["train"].shuffle(seed=seed)
        ds["train"] = ds["train"].select(range(num_examples_kept))

    ds.push_to_hub(ds_sample_name, private=True)
    ds.save_to_disk(path_local_datasets_jz + ds_sample_name)
