# [MODULE DOC] Web-docs step 7.1: finds NSFW image URLs of one shard (argv[1] = IDX_JOB) with the NSFW
# MobileNetV2 Keras
# classifier (224x224, TensorFlow + tensorflow_hub). Forces the "spawn" start method for multiprocess (map
# hangs
# otherwise). As __main__: recreates /scratch/storage_hugo_IDX_JOB/; copies the shard's image-URL Counter
# (6.1)
# and the global too-duplicated URL pickle (6.2); writes the JSON list of URLs to consider (shard URLs minus
# too-duplicated ones); syncs the two image datasets (image_dataset, image_dataset_2, columns "url" and raw
# "image" bytes); keeps only rows with considered URLs; then keeps only rows the classifier flags as NSFW
# (batched, 1000 per batch, NUM_PROC=24); writes the flagged URLs as JSON to
# s3://m4-datasets/webdocs/nsfw_image_urls/IDX_JOB/nsfw_image_urls.json and deletes the temp dir.
import json
import logging
import os
import pickle
import sys
from io import BytesIO

import multiprocess.context as ctx
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datasets import load_from_disk
from PIL import Image, ImageFile
from tensorflow import keras


# Useful otherwise the `map` hangs in multiprocessing
ctx._force_start_method("spawn")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = sys.argv[1]
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_IMAGE_URLS_IN_WEBDOCS_S3 = f"s3://m4-datasets/webdocs/image_urls_in_web_document_dataset_filtered/{IDX_JOB}/image_urls_in_web_document_dataset_filtered.pickle"
PATH_IMAGE_URLS_IN_WEBDOCS_LOCAL = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "image_urls_in_web_document_dataset_filtered.pickle"
)
PATH_IMAGE_URLS_TO_REMOVE_S3 = (
    "s3://m4-datasets/webdocs/tot_image_urls_in_web_document_dataset_filtered_too_duplicated.pickle"
)
PATH_IMAGE_URLS_TO_REMOVE_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "image_urls_to_remove.pickle")
PATH_IMAGE_URLS_TO_KEEP_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "image_urls_to_keep.json")

PATH_IMAGE_DATASET_1_S3 = f"s3://m4-datasets/webdocs/image_dataset/{IDX_JOB}/"
PATH_IMAGE_DATASET_1_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "image_dataset_1")

PATH_IMAGE_DATASET_2_S3 = f"s3://m4-datasets/webdocs/image_dataset_2/{IDX_JOB}/"
PATH_IMAGE_DATASET_2_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "image_dataset_2")

IMAGE_DIM = 224
PATH_NSFW_CLASSIFIER = (  # Download at https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5
    "/fsx/hugo/nsfw_image/models/nsfw_mobilenet2.224x224.h5"
)

NUM_PROC = 24
BATCH_SIZE = 1_000

NSFW_THRESHOLDS = {
    "hentai": 0.9,
    "porn": 0.85,
    "sexy": 1.0,
}

PATH_SAVE_DISK_NSFW_IMAGES_URLS = os.path.join(PATH_SAVE_DISK_TMP_FILES, "nsfw_image_urls.json")
PATH_SAVE_S3_NSFW_IMAGES_URLS = os.path.join(
    "s3://m4-datasets/webdocs/nsfw_image_urls", str(IDX_JOB), "nsfw_image_urls.json"
)


class ImageDatasetURLFiltering:
    # [CLASS DOC] Picklable Dataset.filter predicate keeping rows whose "url" is in a JSON list of URLs to
    # keep.
    def __init__(self, path_image_urls_to_keep):
        # [FUNCTION DOC] Stores the path and loads the JSON list into a set.
        self.path_image_urls_to_keep = path_image_urls_to_keep
        with open(path_image_urls_to_keep) as f:
            self.image_urls_to_keep = set(json.load(f))

    def __call__(self, example):
        # [FUNCTION DOC] Returns True iff example["url"] is in the keep set.
        if example["url"] not in self.image_urls_to_keep:
            return False
        return True

    def __reduce__(self):
        # [FUNCTION DOC] Pickles as (class, (path,)) so workers reload the set from disk.
        return self.__class__, (self.path_image_urls_to_keep,)


class ImageDatasetNSFWFiltering:
    # [CLASS DOC] Picklable batched filter predicate that keeps the NSFW images (used to collect their URLs),
    # wrapping the Keras
    # NSFW classifier (uses __slots__).
    __slots__ = (
        "path_nsfw_classifier",
        "nsfw_classifier",
    )

    def __init__(
        self,
        path_nsfw_classifier,
    ):
        # [FUNCTION DOC] Stores the model path and loads the Keras model (KerasLayer custom object, not
        # compiled).
        self.path_nsfw_classifier = path_nsfw_classifier
        self.nsfw_classifier = tf.keras.models.load_model(
            path_nsfw_classifier, custom_objects={"KerasLayer": hub.KerasLayer}, compile=False
        )

    def __call__(self, batch):
        # [FUNCTION DOC] Input: batch mapping with "image" (list of raw bytes). Returns one boolean per image:
        # True (keep = NSFW) if
        # hentai > 0.9 or porn > 0.85 or sexy > 1.0 (NSFW_THRESHOLDS).
        # KNOWN ISSUE: the "sexy" threshold of 1.0 can never be exceeded by a probability, so that class is
        # effectively
        # disabled; the variable name bool_keep_examples means "is NSFW" here.
        images_bytes = batch["image"]
        nsfw_scores = self.compute_nsfw_scores(images_bytes=images_bytes, nsfw_classifier=self.nsfw_classifier)
        bool_keep_examples = [
            any(
                [
                    nsfw_scores_["hentai"] > NSFW_THRESHOLDS["hentai"],
                    nsfw_scores_["porn"] > NSFW_THRESHOLDS["porn"],
                    nsfw_scores_["sexy"] > NSFW_THRESHOLDS["sexy"],
                ]
            )
            for nsfw_scores_ in nsfw_scores
        ]
        return bool_keep_examples

    def process_image(self, image_bytes):
        # [FUNCTION DOC] HELPER: decodes bytes with PIL, converts to RGB, resizes to IMAGE_DIM x IMAGE_DIM,
        # returns a float array
        # scaled to [0, 1]; on any error returns an all-black image array instead.
        try:
            image = Image.open(BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.size != (IMAGE_DIM, IMAGE_DIM):
                image = image.resize((IMAGE_DIM, IMAGE_DIM))
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            return image
        except Exception:
            return keras.preprocessing.image.img_to_array(Image.new("RGB", (IMAGE_DIM, IMAGE_DIM))) / 255

    def compute_nsfw_scores(self, images_bytes, nsfw_classifier):
        # [FUNCTION DOC] HELPER: preprocesses all images into one array, runs nsfw_classifier.predict, and
        # returns a list of mappings
        # with float scores "hentai" (output 1), "porn" (output 3) and "sexy" (output 4).
        images = np.asarray([self.process_image(image_bytes) for image_bytes in images_bytes])
        predictions = nsfw_classifier.predict(images)
        predictions = [
            {"hentai": float(pred[1]), "porn": float(pred[3]), "sexy": float(pred[4])} for pred in predictions
        ]
        return predictions

    def __reduce__(self):
        # [FUNCTION DOC] Pickles as (class, (model path,)) so workers reload the model.
        return (self.__class__, (self.path_nsfw_classifier,))


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting preparing the set of image urls to consider")
    command_sync_s3 = f"aws s3 cp {PATH_IMAGE_URLS_IN_WEBDOCS_S3} {PATH_IMAGE_URLS_IN_WEBDOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    command_sync_s3 = f"aws s3 cp {PATH_IMAGE_URLS_TO_REMOVE_S3} {PATH_IMAGE_URLS_TO_REMOVE_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    with open(PATH_IMAGE_URLS_IN_WEBDOCS_LOCAL, "rb") as f:
        image_urls_to_keep = pickle.load(f)
    with open(PATH_IMAGE_URLS_TO_REMOVE_LOCAL, "rb") as f:
        image_urls_to_remove = set(pickle.load(f))
    image_urls_to_keep = [url for url in image_urls_to_keep if url not in image_urls_to_remove]
    with open(PATH_IMAGE_URLS_TO_KEEP_LOCAL, "w") as f:
        json.dump(image_urls_to_keep, f)
    logger.info("Finished preparing the set of image urls to consider")

    logger.info("Starting loading the image dataset")
    command_sync_s3 = f"aws s3 sync {PATH_IMAGE_DATASET_1_S3} {PATH_IMAGE_DATASET_1_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    command_sync_s3 = f"aws s3 sync {PATH_IMAGE_DATASET_2_S3} {PATH_IMAGE_DATASET_2_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    image_dataset_1 = load_from_disk(PATH_IMAGE_DATASET_1_LOCAL)
    image_dataset_2 = load_from_disk(PATH_IMAGE_DATASET_2_LOCAL)
    logger.info("Finished loading the image dataset")

    logger.info("Starting keeping only the interesting images in the image datasets")
    image_dataset_url_filtering = ImageDatasetURLFiltering(path_image_urls_to_keep=PATH_IMAGE_URLS_TO_KEEP_LOCAL)
    image_dataset_1 = image_dataset_1.filter(image_dataset_url_filtering, num_proc=NUM_PROC)

    image_dataset_url_filtering = ImageDatasetURLFiltering(path_image_urls_to_keep=PATH_IMAGE_URLS_TO_KEEP_LOCAL)
    image_dataset_2 = image_dataset_2.filter(image_dataset_url_filtering, num_proc=NUM_PROC)
    logger.info("Finished keeping only the interesting images in the image datasets")

    logger.info("Starting computing the NSFW scores")
    image_dataset_nsfw_filtering = ImageDatasetNSFWFiltering(path_nsfw_classifier=PATH_NSFW_CLASSIFIER)
    image_dataset_1 = image_dataset_1.filter(
        image_dataset_nsfw_filtering, num_proc=NUM_PROC, batched=True, batch_size=BATCH_SIZE
    )

    image_dataset_nsfw_filtering = ImageDatasetNSFWFiltering(path_nsfw_classifier=PATH_NSFW_CLASSIFIER)
    image_dataset_2 = image_dataset_2.filter(
        image_dataset_nsfw_filtering, num_proc=NUM_PROC, batched=True, batch_size=BATCH_SIZE
    )
    logger.info("Finished computing the NSFW scores")

    logger.info("Starting gathering and saving the NSFW image URLs to be discarded")
    image_dataset_1 = image_dataset_1.remove_columns([c_n for c_n in image_dataset_1.column_names if c_n != "url"])
    image_dataset_2 = image_dataset_2.remove_columns([c_n for c_n in image_dataset_2.column_names if c_n != "url"])
    nsfw_image_urls = image_dataset_1["url"] + image_dataset_2["url"]
    with open(PATH_SAVE_DISK_NSFW_IMAGES_URLS, "w") as f:
        json.dump(nsfw_image_urls, f)

    command_sync_s3 = f"aws s3 cp {PATH_SAVE_DISK_NSFW_IMAGES_URLS} {PATH_SAVE_S3_NSFW_IMAGES_URLS}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished gathering and saving the NSFW image URLs to be discarded")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
