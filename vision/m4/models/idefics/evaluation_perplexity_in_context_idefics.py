# [MODULE DOC] IDEFICS registrations of in-context perplexity evaluation tasks (log-likelihood of reference
# texts given images). Logic is inherited from Vgpt2PerplexityInContext; classes set dataset, columns,
# sequence length and max image count, and two override the image-caption prompt format. Every task uses the
# single PERPLEXITY metric.
from m4.evaluation.custom_metrics.perplexity_metrics import MetricsPerplexity
from m4.models.vgpt2.evaluation_perplexity_in_context_vgpt2 import Vgpt2PerplexityInContext
from m4.training.types import DatasetTypes


class IdeficsPerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Base perplexity task for IDEFICS: model_class "IdeficsForCausalLM", adds begin- and end-of-
    # document tokens, tokenizer_max_seq_len 2048.
    model_class: str = "IdeficsForCausalLM"
    add_end_of_doc_token: bool = True
    add_begin_of_doc_token: bool = True
    tokenizer_max_seq_len = 2048


class Cm4IdeficsPerplexityInContextMaxSeLen1024(IdeficsPerplexityInContext):
    # [CLASS DOC] CM4 web documents (HuggingFaceM4/cm4_valid, split "valid", columns images/texts, ds_type
    # WEB_DOCUMENTS) truncated at 1024 tokens with at most 70 images.
    dataset_name: str = "HuggingFaceM4/cm4_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70  # To be computed once the dataset is created


class Cm4IdeficsPerplexityInContextMaxSeLen512(Cm4IdeficsPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] CM4 task at 512 tokens / max 35 images.
    tokenizer_max_seq_len = 512
    max_num_images = 35  # To be computed once the dataset is created


class Cm4SampleIdeficsPerplexityInContextMaxSeLen1024(Cm4IdeficsPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] CM4 1024-token task on the cm4_valid-Sample dataset (max 53 images).
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class Cm4SampleIdeficsPerplexityInContextMaxSeLen512(Cm4IdeficsPerplexityInContextMaxSeLen512):
    # [CLASS DOC] CM4 512-token task on the cm4_valid-Sample dataset (max 53 images, larger than the 35 of the
    # full task).
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class EnWikiIdeficsPerplexityInContextMaxSeLen1024(IdeficsPerplexityInContext):
    # [CLASS DOC] English Wikipedia web documents (HuggingFaceM4/enwiki-v2_valid, split "valid") at 1024
    # tokens / max 70 images.
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70


class EnWikiIdeficsPerplexityInContextMaxSeLen512(EnWikiIdeficsPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] EnWiki task at 512 tokens / max 35 images.
    tokenizer_max_seq_len = 512
    max_num_images = 35


class EnWikiSampleIdeficsPerplexityInContextMaxSeLen1024(EnWikiIdeficsPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] EnWiki 1024-token task on enwiki-v2_valid-Sample.
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class EnWikiSampleIdeficsPerplexityInContextMaxSeLen512(EnWikiIdeficsPerplexityInContextMaxSeLen512):
    # [CLASS DOC] EnWiki 512-token task on enwiki-v2_valid-Sample.
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class TextCapsIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    # [CLASS DOC] Caption perplexity on TextCaps validation (image / reference_strs).
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"


class TextCapsSampleIdeficsPerplexityInContext(TextCapsIdeficsPerplexityInContext):
    # [CLASS DOC] TextCaps perplexity on HuggingFaceM4/TextCaps-Sample.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    # [CLASS DOC] CommonGen perplexity (validation; image, context "concepts", text "target") with a custom
    # prompt.
    dataset_name: str = "HuggingFaceM4/common_gen"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    context_column_name: str = "concepts"
    text_column_name: str = "target"

    def _create_image_caption_pair_prompt(self, caption="", context=""):
        # [FUNCTION DOC] Return the prompt string: image tokens (token_around_image, image_token,
        # token_around_image) followed by "Input: {context}. Output: {caption}". caption and context default
        # to empty strings.
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}Input: {context}. Output: {caption}"
        )


class NoCapsIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    # [CLASS DOC] Caption perplexity on NoCaps validation (annotations_captions).
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "annotations_captions"


class NoCapsSampleIdeficsPerplexityInContext(NoCapsIdeficsPerplexityInContext):
    # [CLASS DOC] NoCaps perplexity on HuggingFaceM4/NoCaps-Sample.
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    # [CLASS DOC] Caption perplexity on COCO 2014 captions validation (sentences_raw).
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "sentences_raw"


class CocoSampleIdeficsPerplexityInContext(CocoIdeficsPerplexityInContext):
    # [CLASS DOC] COCO perplexity on the COCO-2014_captions-Sample dataset (no config).
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class IIIT5KIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    # [CLASS DOC] Perplexity of the written word (label) on IIIT-5K test images with a custom prompt.
    dataset_name: str = "HuggingFaceM4/IIIT-5K"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "test"
    image_column_name: str = "image"
    text_column_name: str = "label"

    def _create_image_caption_pair_prompt(self, caption="", context=None):
        # [FUNCTION DOC] Return image tokens followed by "A photo where it is written {caption}". Raises
        # NotImplementedError when context is not None (default None).
        if context is not None:
            raise NotImplementedError("Context not implemented for this task")
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}A photo where"
            f" it is written {caption}"
        )


class IIIT5KSampleIdeficsPerplexityInContext(IIIT5KIdeficsPerplexityInContext):
    # [CLASS DOC] IIIT-5K perplexity on HuggingFaceM4/IIIT-5K-Sample.
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"


class MiniGPTCaptionsIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    # [CLASS DOC] Caption perplexity on HuggingFaceM4/mini-GPT-captions (train split, reference_strs).
    dataset_name: str = "HuggingFaceM4/mini-GPT-captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "train"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"
