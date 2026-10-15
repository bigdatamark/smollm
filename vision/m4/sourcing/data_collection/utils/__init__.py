# [PACKAGE DOC] Utilities for the data_collection pipeline. Re-exports: CLIP scoring (`compute_clip_score`),
# cached image
# fetching (`fetch_single_image`), text-filtering constants (regexes, punctuation, stopwords, flagged words),
# KL divergence (`kl_div`, `NB_BINS`), HTML simplification helpers and tag/attribute sets, and HTML dataset
# loading/parsing helpers. Importing it loads a CLIP model (via clip_utils) and creates the image cache dir.
from m4.sourcing.data_collection.utils.clip_utils import compute_clip_score
from m4.sourcing.data_collection.utils.fetching_utils import fetch_single_image
from m4.sourcing.data_collection.utils.filtering_utils import (
    DIGITS_RE,
    FLAGGED_WORDS,
    NON_PRINTING_CHARACTERS_RE,
    PUNCTUATION,
    SPECIAL_CHARACTERS,
    STOPWORDS,
    UNICODE_PUNCTUATION,
)
from m4.sourcing.data_collection.utils.kl_utils import NB_BINS, kl_div
from m4.sourcing.data_collection.utils.simplification_utils import (
    TAG_TO_SEP,
    format_filename,
    format_image_size,
    format_relative_to_absolute_path,
    get_media_src,
    is_url_valid,
    simplify_media_node,
)
from m4.sourcing.data_collection.utils.tags_attributes import (
    INTERESTING_TAGS_SET,
    MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET,
    UNWRAP_TAGS,
    InterestingAttributesSetCategory,
)
from m4.sourcing.data_collection.utils.utils import load_dataset_html, make_selectolax_tree
