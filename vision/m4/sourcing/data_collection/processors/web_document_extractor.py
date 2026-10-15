# [MODULE DOC] Web document extraction pipeline for Common Crawl HTML: turns HTML pages into interleaved
# documents (parallel lists "texts" and "images" where each position holds either a text or an image),
# collects image URLs, downloads them with img2dataset (webdataset tars), builds a URL-indexed image dataset,
# replaces URLs by image bytes and saves/shards the result. Module-level functions do the work;
# CommonCrawlWebDocumentExtractor chains them with stored config. Configures INFO logging at import.
import glob
import json
import logging
import math
import os
import tarfile
from copy import deepcopy

import git
from datasets import Dataset, Image, Sequence, Value, concatenate_datasets, load_from_disk
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_file(path_file, to_write):
    # [FUNCTION DOC] HELPER: writes the string to_write to path_file, overwriting it.
    f = open(path_file, "w")
    f.truncate(0)
    f.write(to_write)
    f.close()


def html_to_web_documents(
    dataset,
    dom_tree_simplificator,
    pre_extraction_simplificator,
    num_proc,
    html_column_name="html",
    url_column_name="url",
    extraction_suffix="",
    remove_columns=True,
):
    # [FUNCTION DOC] Maps func_html_to_web_documents over dataset (num_proc processes).
    # html_column_name/url_column_name (defaults "html" /"url") select inputs; extraction_suffix is appended
    # to output column names; remove_columns (default True) drops all input columns. Returns the new dataset
    # with texts, images, metadata (JSON string) and general_metadata (JSON).
    def func_html_to_web_documents(example):
        # [FUNCTION DOC] HELPER: simplifies one page and builds aligned lists: a text node gives (text, ""
        # image, None metadata), an img node gives (None text, src URL, media_info). Other nodes (e.g. figure)
        # are ignored. general_metadata holds url and WARC location fields when all are present. On any
        # simplification exception prints "EXCEPTION" and writes empty lists (general_metadata becomes an
        # empty JSON list). Returns the mutated example.
        html_str = example[html_column_name]
        page_url = example[url_column_name]
        general_metadata = {}
        if all(
            [
                column_name in example
                for column_name in ["url", "warc_filename", "warc_record_offset", "warc_record_length"]
            ]
        ):
            general_metadata = {
                "url": example["url"],
                "warc_filename": example["warc_filename"],
                "warc_record_offset": example["warc_record_offset"],
                "warc_record_length": example["warc_record_length"],
            }

        try:
            selectolax_tree = dom_tree_simplificator(html_str, type_return="selectolax_tree")
            list_nodes = pre_extraction_simplificator(selectolax_tree, page_url=page_url)
        except Exception:
            print("EXCEPTION")
            example[f"texts{extraction_suffix}"] = []
            example[f"images{extraction_suffix}"] = []
            example[f"metadata{extraction_suffix}"] = json.dumps([])
            example[f"general_metadata{extraction_suffix}"] = json.dumps([])
            return example
        texts = []
        images = []
        metadata = []
        for node in list_nodes:
            if node.tag == "-text":
                texts.append(node.text)
                images.append("")
                metadata.append(None)
            elif node.tag == "img":
                texts.append(None)
                images.append(node.media_info["src"])
                metadata.append(node.media_info)

        example[f"texts{extraction_suffix}"] = texts
        example[f"images{extraction_suffix}"] = images
        example[f"metadata{extraction_suffix}"] = json.dumps(metadata)
        example[f"general_metadata{extraction_suffix}"] = json.dumps(general_metadata)

        return example

    kwargs = {"remove_columns": dataset.column_names} if remove_columns else {}
    logger.info("Starting extracting the documents")
    dataset = dataset.map(func_html_to_web_documents, num_proc=num_proc, **kwargs)
    logger.info("Finished extracting the documents")
    return dataset


def get_image_urls(dataset, num_proc, path_save_file_image_urls):
    # [FUNCTION DOC] Collects all non-empty image URLs of the dataset's "images" column (num_proc processes),
    # deduplicates them (order not preserved) and writes them one per line to path_save_file_image_urls.
    def func_get_image_urls(example):
        # [FUNCTION DOC] HELPER: sets example["urls"] to the non-empty entries of example["images"].
        example["urls"] = [el for el in example["images"] if el]
        return example

    logger.info("Starting getting the urls of all images")
    image_urls = dataset.map(func_get_image_urls, remove_columns=dataset.column_names, num_proc=num_proc)
    image_urls = [sub_el for el in image_urls["urls"] for sub_el in el if sub_el]
    image_urls = list(set(image_urls))

    write_file(path_file=path_save_file_image_urls, to_write="\n".join(image_urls))
    logger.info("Finished getting the urls of all images")


def download_images(
    path_save_file_image_urls,
    path_save_dir_downloaded_images,
    number_sample_per_shard,
    image_size,
    resize_mode,
    num_proc,
    thread_count,
):
    # [FUNCTION DOC] Runs the img2dataset command line via os.system on the URL file, writing webdataset
    # shards to path_save_dir_downloaded_images with the given processes, threads, samples per shard,
    # image_size and resize_mode. Exit status is ignored. A DNS resolver should be configured beforehand.
    # Before calling this method, set up a DNS solver
    # https://github.com/rom1504/img2dataset#setting-up-a-bind9-resolver
    logger.info("Starting downloading the images")
    os.system(
        "img2dataset"
        f" --url_list={path_save_file_image_urls} --output_folder={path_save_dir_downloaded_images}"
        f" --processes_count={num_proc} --thread_count={thread_count}"
        f" --number_sample_per_shard={number_sample_per_shard} --image_size={image_size}"
        f" --resize_mode={resize_mode} --output_format=webdataset"
    )
    logger.info("Finished downloading the images")


def create_dataset_images_from_tar(
    tar_paths,
    path_save_dir_tmp_datasets_images,
    num_proc,
    path_save_file_map_url_idx,
    path_save_dir_dataset_images,
):
    # [FUNCTION DOC] Processes each tar in tar_paths in a pathos process pool (num_proc): each becomes a
    # temporary dataset of (url, image bytes) saved under path_save_dir_tmp_datasets_images/<tar index>. Then
    # writes a JSON mapping url -> global row index (concatenation order) to path_save_file_map_url_idx,
    # concatenates the temporary datasets, saves to path_save_dir_dataset_images and returns it.
    def process_one_tar(args):
        # [FUNCTION DOC] HELPER: args is (tar_path, idx_tar). Reads .jpg members (bytes) and .json members
        # (status, url); for successful downloads maps url -> image bytes, saves them as a Dataset to the temp
        # dir and returns the list of URLs in saved order.
        (tar_path, idx_tar) = args
        with tarfile.open(tar_path) as tar_file:
            tar_members = tar_file.getmembers()
            name_to_url = {}
            name_to_img = {}
            url_to_img = {}
            for tar_member in tar_members:
                if tar_member.name.endswith(".jpg"):
                    name = tar_member.name.replace(".jpg", "")
                    tar_member_file = tar_file.extractfile(tar_member)
                    img = tar_member_file.read()
                    tar_member_file.close()
                    name_to_img[name] = img
                elif tar_member.name.endswith(".json"):
                    name = tar_member.name.replace(".json", "")
                    tar_member_file = tar_file.extractfile(tar_member)
                    json_val = json.loads(tar_member_file.read())
                    status = json_val["status"]
                    url = json_val["url"]
                    tar_member_file.close()
                    if status == "success":  # Should always happend with webdataset format, not with parquet
                        name_to_url[name] = url
            for name in name_to_url:
                url_to_img[name_to_url[name]] = name_to_img[name]
            new_urls_indexed = list(url_to_img.keys())
            new_datasets_images = Dataset.from_dict(
                {"url": list(url_to_img.keys()), "image": list(url_to_img.values())}
            )
            # We need to save the new datasets and then reload them, since `from_dict` store the dataset
            # in the RAM and does not use the disk space
            new_datasets_images.save_to_disk(os.path.join(path_save_dir_tmp_datasets_images, str(idx_tar)))
            return new_urls_indexed

    logger.info("Starting creating the dataset of all images")
    args_pool = [(tar_path, idx_tar) for idx_tar, tar_path in enumerate(tar_paths)]
    pool = Pool(num_proc)
    urls_indexed = pool.map(process_one_tar, args_pool)
    urls_indexed = [sub_el for el in urls_indexed for sub_el in el]

    map_url_idx = {url: idx for idx, url in enumerate(urls_indexed)}
    with open(path_save_file_map_url_idx, "w") as f:
        json.dump(map_url_idx, f)
    datasets_images = [
        load_from_disk(os.path.join(path_save_dir_tmp_datasets_images, str(idx_tar)))
        for idx_tar in range(len(tar_paths))
    ]
    dataset_images = concatenate_datasets(datasets_images)
    dataset_images.save_to_disk(path_save_dir_dataset_images)
    logger.info("Finished creating the dataset of all images")
    return dataset_images


def create_dataset_images(
    path_save_dir_downloaded_images,
    path_save_dir_tmp_datasets_images,
    num_proc,
    path_save_file_map_url_idx,
    path_save_dir_dataset_images,
):
    # [FUNCTION DOC] Finds all *.tar files in path_save_dir_downloaded_images and delegates to
    # create_dataset_images_from_tar; returns the image dataset.
    tar_paths = glob.glob(os.path.join(path_save_dir_downloaded_images, "*.tar"))
    dataset_images = create_dataset_images_from_tar(
        tar_paths=tar_paths,
        path_save_dir_tmp_datasets_images=path_save_dir_tmp_datasets_images,
        num_proc=num_proc,
        path_save_file_map_url_idx=path_save_file_map_url_idx,
        path_save_dir_dataset_images=path_save_dir_dataset_images,
    )
    return dataset_images


def urls_to_images(dataset, dataset_images, map_url_idx, num_proc, some_urls_are_already_retrieved=False):
    # [FUNCTION DOC] Replaces image URLs with image bytes in dataset using dataset_images and map_url_idx (url
    # -> row). Without some_urls_are_already_retrieved (default False) the "images" column holds URLs: they
    # are copied to "images_urls" and replaced by images; with it, the dataset must already have "images_urls"
    # and "images" (ValueError otherwise) and only missing images are filled. Adds num_found/num_not_found
    # counts; casts images to Sequence(Image()). Maps with num_proc, cache disabled. Returns the new dataset.
    if some_urls_are_already_retrieved:
        if "images_urls" not in dataset.features or "images" not in dataset.features:
            raise ValueError(
                "If some urls are already retrieved, the dataset must contain the features 'images_urls' and 'images'"
            )

    def retrieve_image(url):
        # [FUNCTION DOC] HELPER: returns {"path": None, "bytes": ...} for url from dataset_images, or None if
        # the URL was not downloaded.
        if url not in map_url_idx:
            return None
        image = {"path": None, "bytes": dataset_images[map_url_idx[url]]["image"]}
        return image

    def func_urls_to_images_urls_in_images_col(example):
        # [FUNCTION DOC] HELPER: copies images (URLs) to images_urls, replaces each non-empty URL by
        # retrieve_image (None for text slots), and sets num_found and num_not_found (URLs not resolved).
        example["images_urls"] = deepcopy(example["images"])
        num_urls = sum([(url is not None and url != "") for url in example["images_urls"]])

        example["images"] = [retrieve_image(url) if url else None for url in example["images"]]

        num_found = sum([img is not None for img in example["images"]])
        num_not_found = num_urls - num_found

        example["num_found"] = num_found
        example["num_not_found"] = num_not_found
        return example

    def func_urls_to_images_urls_in_images_urls_col(example):
        # [FUNCTION DOC] HELPER: keeps existing images and fills None entries from images_urls via
        # retrieve_image; sets num_found and num_not_found.
        num_urls = sum([(url is not None and url != "") for url in example["images_urls"]])

        example["images"] = [
            img if img is not None else retrieve_image(url) if url else None
            for img, url in zip(example["images"], example["images_urls"])
        ]

        num_found = sum([img is not None for img in example["images"]])
        num_not_found = num_urls - num_found

        example["num_found"] = num_found
        example["num_not_found"] = num_not_found
        return example

    func_urls_to_images = (
        func_urls_to_images_urls_in_images_urls_col
        if some_urls_are_already_retrieved
        else func_urls_to_images_urls_in_images_col
    )

    logger.info("Starting replacing urls by images")

    new_features = deepcopy(dataset.features)
    new_features["images"] = Sequence(Image())
    new_features["images_urls"] = Sequence(Value("string"))
    new_features["num_found"] = Value("int32")
    new_features["num_not_found"] = Value("int32")

    dataset = dataset.map(
        func_urls_to_images,
        features=new_features,
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    logger.info("Finished replacing urls by images")
    return dataset


def save_split_sharded_already_splitted_dataset(dataset, path_save_dir_sharded_dataset, shard_size):
    # [FUNCTION DOC] Saves a DatasetDict with "train" and "valid" splits as contiguous shards of about
    # shard_size rows under path_save_dir_sharded_dataset/<split>/shard_<i>, plus a dataset_dict.json listing
    # the splits.
    def save_split_ds(split_dataset, split_name):
        # [FUNCTION DOC] HELPER: shards split_dataset into ceil(len / shard_size) contiguous pieces and saves
        # each to <dir>/<split_name>/shard_<idx>.
        num_shards = math.ceil(len(split_dataset) / shard_size)
        for idx in tqdm(range(num_shards)):
            shard = split_dataset.shard(num_shards=num_shards, index=idx, contiguous=True)
            shard.save_to_disk(os.path.join(path_save_dir_sharded_dataset, split_name, f"shard_{idx}"))

    os.makedirs(path_save_dir_sharded_dataset, exist_ok=True)

    f = open(os.path.join(path_save_dir_sharded_dataset, "dataset_dict.json"), "w")
    f.write('{"splits": ["train", "valid"]}')
    f.close()

    os.makedirs(os.path.join(path_save_dir_sharded_dataset, "train"), exist_ok=True)
    os.makedirs(os.path.join(path_save_dir_sharded_dataset, "valid"), exist_ok=True)

    logger.info("Starting sharding the dataset")
    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]

    save_split_ds(train_dataset, "train")
    save_split_ds(valid_dataset, "valid")
    logger.info("Finished sharding the dataset")


def save_split_sharded_dataset(dataset, path_save_dir_sharded_dataset, shard_size):
    # [FUNCTION DOC] Shards a single dataset into ceil(len / shard_size) contiguous shards; the first two
    # shards go to the "valid" directory and the rest to "train" under path_save_dir_sharded_dataset; writes
    # dataset_dict.json.
    os.makedirs(path_save_dir_sharded_dataset, exist_ok=True)

    f = open(os.path.join(path_save_dir_sharded_dataset, "dataset_dict.json"), "w")
    f.write('{"splits": ["train", "valid"]}')
    f.close()

    os.makedirs(os.path.join(path_save_dir_sharded_dataset, "train"), exist_ok=True)
    os.makedirs(os.path.join(path_save_dir_sharded_dataset, "valid"), exist_ok=True)

    logger.info("Starting sharding the dataset")

    num_shards = math.ceil(len(dataset) / shard_size)
    for idx in tqdm(range(num_shards)):
        shard = dataset.shard(num_shards=num_shards, index=idx, contiguous=True)
        if idx < 2:
            shard.save_to_disk(os.path.join(path_save_dir_sharded_dataset, "valid", f"shard_{idx}"))
        else:
            shard.save_to_disk(os.path.join(path_save_dir_sharded_dataset, "train", f"shard_{idx}"))

    logger.info("Finished sharding the dataset")


class CommonCrawlWebDocumentExtractor:
    # [CLASS DOC] Stateful orchestrator holding the dataset and all paths/parameters; each method runs one
    # pipeline step and updates self.dataset (and self.dataset_images / self.map_url_idx).
    def __init__(
        self,
        html_dataset,
        dom_tree_simplificator,
        pre_extraction_simplificator,
        path_save_dir_dataset,
        num_proc,
        path_save_file_image_urls,
        path_save_dir_downloaded_images,
        thread_count,
        number_sample_per_shard,
        image_size,
        resize_mode,
        path_save_dir_tmp_datasets_images,
        path_save_dir_dataset_images,
        path_save_file_map_url_idx,
        num_proc_urls_to_images,
        path_save_dir_sharded_dataset,
        shard_size,
    ):
        # [FUNCTION DOC] Stores html_dataset as self.dataset plus the simplificators and every path/parameter
        # (see the module functions for their meaning).
        self.dataset = html_dataset

        self.dom_tree_simplificator = dom_tree_simplificator
        self.pre_extraction_simplificator = pre_extraction_simplificator

        self.path_save_dir_dataset = path_save_dir_dataset
        self.num_proc = num_proc
        self.path_save_file_image_urls = path_save_file_image_urls
        self.path_save_dir_downloaded_images = path_save_dir_downloaded_images
        self.thread_count = thread_count
        self.number_sample_per_shard = number_sample_per_shard
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.path_save_dir_tmp_datasets_images = path_save_dir_tmp_datasets_images
        self.path_save_dir_dataset_images = path_save_dir_dataset_images
        self.path_save_file_map_url_idx = path_save_file_map_url_idx
        self.num_proc_urls_to_images = num_proc_urls_to_images
        self.path_save_dir_sharded_dataset = path_save_dir_sharded_dataset
        self.shard_size = shard_size

    def html_to_web_documents(self):
        # [FUNCTION DOC] Replaces self.dataset by its web-document extraction.
        self.dataset = html_to_web_documents(
            dataset=self.dataset,
            dom_tree_simplificator=self.dom_tree_simplificator,
            pre_extraction_simplificator=self.pre_extraction_simplificator,
            num_proc=self.num_proc,
        )

    def get_image_urls(self):
        # [FUNCTION DOC] Writes the unique image URLs of self.dataset to the configured file.
        get_image_urls(
            dataset=self.dataset, num_proc=self.num_proc, path_save_file_image_urls=self.path_save_file_image_urls
        )

    def download_images(self):
        # [FUNCTION DOC] Runs img2dataset with the configured parameters.
        download_images(
            path_save_file_image_urls=self.path_save_file_image_urls,
            path_save_dir_downloaded_images=self.path_save_dir_downloaded_images,
            number_sample_per_shard=self.number_sample_per_shard,
            image_size=self.image_size,
            resize_mode=self.resize_mode,
            num_proc=self.num_proc,
            thread_count=self.thread_count,
        )

    def create_dataset_images(self):
        # [FUNCTION DOC] Builds the image dataset from downloaded tars and stores it in self.dataset_images.
        self.dataset_images = create_dataset_images(
            path_save_dir_downloaded_images=self.path_save_dir_downloaded_images,
            path_save_dir_tmp_datasets_images=self.path_save_dir_tmp_datasets_images,
            num_proc=self.num_proc,
            path_save_file_map_url_idx=self.path_save_file_map_url_idx,
            path_save_dir_dataset_images=self.path_save_dir_dataset_images,
        )

    def urls_to_images(self, reload_files=False):
        # [FUNCTION DOC] Loads map_url_idx from its JSON file; if reload_files (default False) reloads
        # self.dataset and self.dataset_images from disk; then replaces URLs by images using
        # num_proc_urls_to_images processes.
        # KNOWN ISSUE: when not reloading, missing attributes only trigger a printed hint, then the following
        # call raises AttributeError anyway.
        with open(self.path_save_file_map_url_idx) as f:
            self.map_url_idx = json.load(f)
        # Useful when this method is called independently without
        # the previous ones, so we need to load some files
        if reload_files:
            logger.info("Starting reloading variables for the step urls_to_images")
            self.dataset = load_from_disk(self.path_save_dir_dataset)
            self.dataset_images = load_from_disk(self.path_save_dir_dataset_images)
            logger.info("Finished reloading variables for the step urls_to_images")

        else:
            try:
                _ = self.dataset
                _ = self.dataset_images
                _ = self.map_url_idx
            except Exception:
                print("Set `reload_files=True` if you're calling this method alone to define the missing variables")

        self.dataset = urls_to_images(
            dataset=self.dataset,
            dataset_images=self.dataset_images,
            map_url_idx=self.map_url_idx,
            num_proc=self.num_proc_urls_to_images,
        )

    def save_dataset(self):
        # [FUNCTION DOC] Saves self.dataset to path_save_dir_dataset with num_proc shards.
        logger.info("Starting saving the dataset")
        self.dataset.save_to_disk(self.path_save_dir_dataset, num_proc=self.num_proc)
        logger.info("Finished saving the dataset")

    def save_commit_hash(self):
        # [FUNCTION DOC] Writes the current git HEAD sha (repo found from the CWD upward) to
        # <path_save_dir_dataset>/commit_hash.txt.
        logger.info("Starting writing the commit hash")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        write_file(os.path.join(self.path_save_dir_dataset, "commit_hash.txt"), sha)
        logger.info("Finished writing the commit hash")

    def save_split_sharded_dataset(self, reload_files=False):
        # [FUNCTION DOC] Optionally reloads self.dataset from disk (reload_files, default False), then shards
        # it into valid (first two shards) and train directories.
        # Useful when this method is called independently without
        # the previous ones, so we need to load some files
        if reload_files:
            self.dataset = load_from_disk(self.path_save_dir_dataset)

        else:
            try:
                _ = self.dataset

            except Exception:
                print("Set `reload_files=True` if you're calling this method alone to define the missing variables")

        save_split_sharded_dataset(
            dataset=self.dataset,
            path_save_dir_sharded_dataset=self.path_save_dir_sharded_dataset,
            shard_size=self.shard_size,
        )
