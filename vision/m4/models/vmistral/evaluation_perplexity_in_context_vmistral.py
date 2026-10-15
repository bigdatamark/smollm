# [MODULE DOC] In-context perplexity evaluation task definitions for VMistral models. Every class is a
# declarative
# configuration subclass of Vgpt2PerplexityInContext (the evaluation logic lives there); subclasses only set
# class attributes such as dataset_name, metric (PerplexityMetrics with PERPLEXITY), split names, image/text
# column names, dataset type, tokenizer_max_seq_len and max_num_images. "Sample" variants point to small
# sample copies of the same Hub datasets.
from m4.evaluation.custom_metrics.perplexity_metrics import MetricsPerplexity
from m4.models.vgpt2.evaluation_perplexity_in_context_vgpt2 import Vgpt2PerplexityInContext
from m4.training.types import DatasetTypes


class VMistralPerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Base config for VMistral perplexity tasks: model_class "VMistralForCausalLM", adds begin-
    # and end-of-document
    # tokens, tokenizer_max_seq_len 4096.
    model_class: str = "VMistralForCausalLM"
    add_end_of_doc_token: bool = True
    add_begin_of_doc_token: bool = True
    tokenizer_max_seq_len = 4096


class Cm4VMistralPerplexityInContextMaxSeLen1024(VMistralPerplexityInContext):
    # [CLASS DOC] Perplexity on HuggingFaceM4/cm4_valid (web documents, "images"/"texts" columns, "valid"
    # split), sequences
    # capped at 1024 tokens and at most 70 images per example.
    dataset_name: str = "HuggingFaceM4/cm4_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70  # To be computed once the dataset is created


class Cm4VMistralPerplexityInContextMaxSeLen512(Cm4VMistralPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] Same as the 1024 CM4 task but with 512-token sequences and at most 35 images.
    tokenizer_max_seq_len = 512
    max_num_images = 35  # To be computed once the dataset is created


class Cm4SampleVMistralPerplexityInContextMaxSeLen1024(Cm4VMistralPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] CM4 1024-token task on the cm4_valid-Sample dataset, max 53 images.
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class Cm4SampleVMistralPerplexityInContextMaxSeLen512(Cm4VMistralPerplexityInContextMaxSeLen512):
    # [CLASS DOC] CM4 512-token task on the cm4_valid-Sample dataset, max 53 images.
    # KNOWN ISSUE: max_num_images is 53 here although the 512-token parent uses 35; looks copy-pasted from the
    # 1024 sample variant.
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class EnWikiVMistralPerplexityInContextMaxSeLen1024(VMistralPerplexityInContext):
    # [CLASS DOC] Perplexity on HuggingFaceM4/enwiki-v2_valid (web documents, "valid" split), 1024-token
    # sequences, max 70 images.
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70


class EnWikiVMistralPerplexityInContextMaxSeLen512(EnWikiVMistralPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] EnWiki task with 512-token sequences and at most 35 images.
    tokenizer_max_seq_len = 512
    max_num_images = 35


class EnWikiSampleVMistralPerplexityInContextMaxSeLen1024(EnWikiVMistralPerplexityInContextMaxSeLen1024):
    # [CLASS DOC] EnWiki 1024-token task on the enwiki-v2_valid-Sample dataset.
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class EnWikiSampleVMistralPerplexityInContextMaxSeLen512(EnWikiVMistralPerplexityInContextMaxSeLen512):
    # [CLASS DOC] EnWiki 512-token task on the enwiki-v2_valid-Sample dataset.
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class TextCapsVMistralPerplexityInContext(VMistralPerplexityInContext):
    # [CLASS DOC] Caption perplexity on HuggingFaceM4/TextCaps ("validation" split, "image" column,
    # "reference_strs" captions).
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"


class TextCapsSampleVMistralPerplexityInContext(TextCapsVMistralPerplexityInContext):
    # [CLASS DOC] TextCaps task on the TextCaps-Sample dataset.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVMistralPerplexityInContext(VMistralPerplexityInContext):
    # [CLASS DOC] Perplexity on HuggingFaceM4/common_gen: "concepts" column is used as context and "target" as
    # the text to score.
    dataset_name: str = "HuggingFaceM4/common_gen"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    context_column_name: str = "concepts"
    text_column_name: str = "target"

    def _create_image_caption_pair_prompt(self, caption="", context=""):
        # [FUNCTION DOC] Builds one prompt: image placeholder wrapped in token_around_image, then "Input:
        # <context>. Output: <caption>".
        # caption and context default to empty strings. Returns the string.
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}Input: {context}. Output: {caption}"
        )


class NoCapsVMistralPerplexityInContext(VMistralPerplexityInContext):
    # [CLASS DOC] Caption perplexity on HuggingFaceM4/NoCaps ("validation" split, "annotations_captions"
    # column).
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "annotations_captions"


class NoCapsSampleVMistralPerplexityInContext(NoCapsVMistralPerplexityInContext):
    # [CLASS DOC] NoCaps task on the NoCaps-Sample dataset.
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVMistralPerplexityInContext(VMistralPerplexityInContext):
    # [CLASS DOC] Caption perplexity on HuggingFaceM4/COCO config "2014_captions" ("validation" split,
    # "sentences_raw" column).
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "sentences_raw"


class CocoSampleVMistralPerplexityInContext(CocoVMistralPerplexityInContext):
    # [CLASS DOC] COCO task on the COCO-2014_captions-Sample dataset (no dataset config).
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class IIIT5KVMistralPerplexityInContext(VMistralPerplexityInContext):
    # [CLASS DOC] OCR-style perplexity on HuggingFaceM4/IIIT-5K ("test" split, "label" column holds the
    # written word).
    dataset_name: str = "HuggingFaceM4/IIIT-5K"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "test"
    image_column_name: str = "image"
    text_column_name: str = "label"

    def _create_image_caption_pair_prompt(self, caption="", context=None):
        # [FUNCTION DOC] Builds the prompt: wrapped image placeholder followed by "A photo where it is written
        # <caption>".
        # Raises NotImplementedError if context is not None.
        if context is not None:
            raise NotImplementedError("Context not implemented for this task")
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}A photo where"
            f" it is written {caption}"
        )


class IIIT5KSampleVMistralPerplexityInContext(IIIT5KVMistralPerplexityInContext):
    # [CLASS DOC] IIIT-5K task on the IIIT-5K-Sample dataset.
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"


class MiniGPTCaptionsVMistralPerplexityInContext(VMistralPerplexityInContext):
    # [CLASS DOC] Caption perplexity on HuggingFaceM4/mini-GPT-captions; both query and support examples come
    # from "train".
    dataset_name: str = "HuggingFaceM4/mini-GPT-captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "train"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"
