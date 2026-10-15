# [MODULE DOC] Helper to rewrite image file paths stored in `datasets` Image columns (stored with
# decode=False as
# {path, bytes}) so that paths created on another machine point to the local cache, then decode the images.
from typing import Callable, Dict, List

import datasets


"""
Images can be stored in `datasets` using bytes or path to an actual file. If the a path is given, one needs to make the
path work with local setup. The way to do so we remove a prefix and replace with a environment dependent one.
`home/thomas_wang_hugginface_co/.cache/m4/...` -> f"{get_m4_cache_dir()}/..."
"""


def get_image_paths_fixer(image_column_name: str, image_path_fixer: Callable[[str], str]):
    # [FUNCTION DOC] Return a batched map function for column `image_column_name`. `image_path_fixer` maps an
    # old path to the
    # new path (must not return None).
    image_feature = datasets.Image(decode=True)

    def image_paths_fixer(batch: Dict[str, List]) -> Dict[str, List]:
        # [FUNCTION DOC] For each image dict of the batch: skip those with bytes; otherwise replace (in
        # place) `path` with
        # image_path_fixer(path) (asserts path and new path are not None). Then replaces the column with
        # decoded PIL
        # images (datasets.Image decode) and returns the batch. Opens the image files.
        # Image(decode=False) which allows the images to be `{'path': str, 'bytes': str}`
        image_dicts = batch[image_column_name]

        for image_dict in image_dicts:
            # We ignore Images that store bytes directly
            if image_dict["bytes"] is not None:
                continue

            path = image_dict["path"]
            assert path is not None
            new_path = image_path_fixer(path)
            assert new_path is not None
            # Careful that's an in-place operation, which updates the dict stored in `batch`
            image_dict["path"] = new_path

        batch[image_column_name] = [image_feature.decode_example(image_dict) for image_dict in image_dicts]
        return batch

    return image_paths_fixer
