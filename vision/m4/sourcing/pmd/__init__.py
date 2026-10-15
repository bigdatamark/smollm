# [PACKAGE DOC] PMD (Public Multimodal Dataset) sourcing package: loaders building image/text datasets for
# PMD sub-sources.
# Defines the shared schema `_FEATURES` (image, text, source name, meta as JSON string) that every loader must
# output, and cache/dataset directory resolution helpers.
import os
from functools import lru_cache
from pathlib import Path

import datasets


DEFAULT_M4_CACHE_HOME = Path("~/.cache/m4")


@lru_cache(maxsize=1)
def get_m4_cache_dir() -> Path:
    # [FUNCTION DOC] Return the m4 cache dir: $M4_MANUAL_DIR if set, else ~/.cache/m4, user-expanded. Cached
    # (computed once).
    return (Path(os.environ["M4_MANUAL_DIR"]) if "M4_MANUAL_DIR" in os.environ else DEFAULT_M4_CACHE_HOME).expanduser()


@lru_cache(maxsize=1)
def get_jz_dataset_dir() -> Path:
    # [FUNCTION DOC] Return the Jean Zay (JZ) cluster dataset dir from $DSDIR (user-expanded); raises
    # ValueError if DSDIR is unset.
    # Cached once successful.
    if "DSDIR" in os.environ:
        return Path(os.environ["DSDIR"]).expanduser()
    raise ValueError("We're not in JZ. This method should only be called when running in JZ.")


# All PMD datasets should following a single feature API.
_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "text": datasets.Value("string"),
        # Define where the sample comes from, this is necessary when we start to use aggregated versions like PMD.
        "source": datasets.Value("string"),
        # We commit any kind of additional information in json format in `meta`
        "meta": datasets.Value("string"),
    }
)
