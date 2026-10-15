# [MODULE DOC] VMistral evaluation task classes for image-caption matching (e.g. Winoground), scoring every image/caption
# pair by log-likelihood. Logic is inherited from Vgpt2ImageCaptionMatching; these classes only set
# configuration attributes.
from typing import List

from m4.evaluation.custom_metrics.image_caption_matching_metrics import MetricsImageCaptionMatching
from m4.models.vgpt2.evaluation_image_caption_matching_vgpt2 import Vgpt2ImageCaptionMatching


class VMistralImageCaptionMatching(Vgpt2ImageCaptionMatching):
    # [CLASS DOC] Base image-caption matching task for VMistral: model_class "VMistralForCausalLM", tokenizer_max_seq_len
    # 4096.
    model_class: str = "VMistralForCausalLM"
    tokenizer_max_seq_len = 4096


class WinogroundVMistralImageCaptionMatchingAccWithKLAndEntropy(VMistralImageCaptionMatching):
    # [CLASS DOC] Winoground (facebook/winoground) task: every split name is "test" (support splits unused); two image
    # columns (image_0, image_1) and two caption columns (caption_0, caption_1), id column "id"; metrics text,
    # image and group score; length-normalized log-likelihoods; prompt template 0 ("<image> Description of the
    # image: {caption}") is selected out of 7 templates in prompt_templates_dict (templates differ in bos/eos
    # tokens and wording).
    # KNOWN ISSUE: the class name mentions "AccWithKLAndEntropy" but metric_kwargs only requests
    # text/image/group scores.
    dataset_name: str = "facebook/winoground"
    metric_name: str = "ImageCaptionMatchingMetrics"
    metric_kwargs = {
        "metrics": [
            MetricsImageCaptionMatching.TEXT_SCORE,
            MetricsImageCaptionMatching.IMAGE_SCORE,
            MetricsImageCaptionMatching.GROUP_SCORE,
        ]
    }
    # support split names are never used for this dataset
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_names: List[str] = ["image_0", "image_1"]
    caption_column_names: List[str] = ["caption_0", "caption_1"]
    id_column_name: str = "id"
    length_normalize: bool = True
    selected_prompt_template_id = 0
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}Description of the image: {caption}",
        },
        1: {
            "prefix": None,
            "example": "{bos_token}{token_around_image}{image_token}{token_around_image}{caption}{eos_token}",
        },
        2: {
            "prefix": None,
            "example": (
                "{bos_token}{token_around_image}{image_token}{token_around_image}In this image, we can see:"
                " {caption}{eos_token}"
            ),
        },
        3: {
            "prefix": None,
            "example": (
                "{bos_token}{token_around_image}{image_token}{token_around_image}Description of the image:"
                " {caption}{eos_token}"
            ),
        },
        4: {
            "prefix": None,
            "example": (
                "{token_around_image}{image_token}{token_around_image}Accurate description of the image: {caption}"
            ),
        },
        5: {
            "prefix": None,
            "example": (
                "{token_around_image}{image_token}{token_around_image}{caption}. Does this caption accurately describe"
                " the image? Yes or no? Yes"
            ),
        },
        6: {
            "prefix": None,
            "example": (
                "{token_around_image}{image_token}{token_around_image}caption: {caption}.\n Question: Does this"
                " caption accurately describe the image? Yes"
            ),
        },
    }
