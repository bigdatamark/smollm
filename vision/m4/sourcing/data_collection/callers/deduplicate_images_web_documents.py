# [MODULE DOC] CLI pipeline step for image deduplication in the web-document dataset. Parses paths/parameters,
# seeds Python's random module (unless --seed is negative), builds a WebDocumentImageDeduplication processor
# with all paths and runs only overlap_train_eval_deduplication(reload_files=True) (the other dedup steps are
# commented out and are expected to have been run in earlier invocations). Sets PIL MAX_IMAGE_PIXELS to None
# to avoid decompression bomb errors.
import argparse
from multiprocessing import cpu_count
from random import seed as random_seed

from PIL import Image

from m4.sourcing.data_collection.processors import WebDocumentImageDeduplication


# Useful to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


def get_args():
    # [FUNCTION DOC] HELPER: defines and parses the CLI options (train/valid web-document dataset paths,
    # num_proc default cpu_count, output paths for the url->position map, the train image dataset, the
    # to-be-deduplicated JSON, the evaluation-task image dataset and the deduplicated dataset, the extraction
    # image dataset path, seed default 0, hamming_distance_threshold default 3, type_dedup "remove_image" or
    # "remove_all_doc"); returns the namespace.
    parser = argparse.ArgumentParser(description="Deduplicate images in web documents.")
    parser.add_argument(
        "--path_web_document_dataset_train",
        type=str,
        default="./large_files/output_extraction/web_document_dataset_train_100",
        help="Path of the dataset containing the web documents (train split).",
    )
    parser.add_argument(
        "--path_web_document_dataset_valid",
        type=str,
        default="./large_files/output_extraction/web_document_dataset_valid_100",
        help="Path of the dataset containing the web documents (valid split).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    parser.add_argument(
        "--path_save_file_map_image_url_to_pos",
        type=str,
        default="./large_files/output_deduplication/map_image_url_to_pos.json",
        help="The path to save the map to go from image urls to their positions in the web document dataset.",
    )
    parser.add_argument(
        "--path_images_web_document_dataset_extraction",
        type=str,
        default="./large_files/output_extraction/dataset_images",
        help="The dataset containing all the images created during the web document extraction.",
    )
    parser.add_argument(
        "--path_save_dir_images_web_document_dataset_train",
        type=str,
        default="./large_files/output_deduplication/images_web_document_dataset_train",
        help=(
            "The path of the directory to save the dataset containing all the images of the web document dataset"
            " (after a potential filtering)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to generate random numbers. -1 for no seed.",
    )
    parser.add_argument(
        "--path_save_file_to_be_deduplicated",
        type=str,
        default="./large_files/output_deduplication/to_be_deduplicated.json",
        help="The path to save the json containing the positions of the images to be deduplicated.",
    )
    parser.add_argument(
        "--path_save_dir_images_evaluation_tasks_dataset",
        type=str,
        default="./large_files/output_deduplication/images_evaluation_tasks_dataset",
        help="The path of the directory to save the dataset containing all the images of the evaluation tasks.",
    )
    parser.add_argument(
        "--hamming_distance_threshold",
        type=int,
        default=3,
        help="The hamming distance threshold to consider two images as near duplicates.",
    )
    parser.add_argument(
        "--type_dedup",
        type=str,
        default="remove_image",
        help="The type of deduplication to perform. Choose between 'remove_all_doc' and 'remove_image'.",
    )
    parser.add_argument(
        "--path_save_dir_web_document_dataset_train_deduplicated",
        type=str,
        default="./large_files/output_deduplication/web_document_dataset_train_deduplicated",
        help="The directory to save the deduplicated web document dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    path_web_document_dataset_train = args.path_web_document_dataset_train
    path_web_document_dataset_valid = args.path_web_document_dataset_valid
    num_proc = args.num_proc
    path_save_file_map_image_url_to_pos = args.path_save_file_map_image_url_to_pos
    path_images_web_document_dataset_extraction = args.path_images_web_document_dataset_extraction
    path_save_dir_images_web_document_dataset_train = args.path_save_dir_images_web_document_dataset_train
    seed = args.seed
    path_save_file_to_be_deduplicated = args.path_save_file_to_be_deduplicated
    path_save_dir_images_evaluation_tasks_dataset = args.path_save_dir_images_evaluation_tasks_dataset
    hamming_distance_threshold = args.hamming_distance_threshold
    type_dedup = args.type_dedup
    path_save_dir_web_document_dataset_train_deduplicated = args.path_save_dir_web_document_dataset_train_deduplicated

    if seed >= 0:
        random_seed(seed)

    web_document_image_deduplication = WebDocumentImageDeduplication(
        path_web_document_dataset_train=path_web_document_dataset_train,
        path_web_document_dataset_valid=path_web_document_dataset_valid,
        num_proc=num_proc,
        path_save_file_map_image_url_to_pos=path_save_file_map_image_url_to_pos,
        path_images_web_document_dataset_extraction=path_images_web_document_dataset_extraction,
        path_save_dir_images_web_document_dataset_train=path_save_dir_images_web_document_dataset_train,
        path_save_file_to_be_deduplicated=path_save_file_to_be_deduplicated,
        path_save_dir_images_evaluation_tasks_dataset=path_save_dir_images_evaluation_tasks_dataset,
        hamming_distance_threshold=hamming_distance_threshold,
        type_dedup=type_dedup,
        path_save_dir_web_document_dataset_train_deduplicated=path_save_dir_web_document_dataset_train_deduplicated,
    )

    # web_document_image_deduplication.create_map_image_url_to_pos()

    # web_document_image_deduplication.build_images_web_document_dataset_train(reload_files=True)

    # web_document_image_deduplication.exact_image_deduplication(reload_files=True)

    # web_document_image_deduplication.build_images_evaluation_tasks_dataset()

    web_document_image_deduplication.overlap_train_eval_deduplication(reload_files=True)

    # web_document_image_deduplication.remove_duplicates(reload_files=True)
