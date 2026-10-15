# [MODULE DOC] IDEFICS registrations of in-context image captioning (generation) evaluation tasks. All logic
# (prompt building with few-shot support examples, generation, metric computation) is inherited from
# Vgpt2ImageCaptioningInContext; each class only sets dataset, split, column, metric and prompt attributes and
# is looked up by class name by the evaluation registry. "Sample" variants point to small sample datasets;
# "NewSplits" variants use *_support_query_sets datasets with explicit validation/test(/server_check) query
# and support splits and prompt template 7 from the base class.
from m4.evaluation.custom_metrics.unfolded_image_captioning_metrics import ImageCaptioningMetrics
from m4.models.vgpt2.evaluation_captioning_in_context_vgpt2 import Vgpt2ImageCaptioningInContext


class IdeficsImageCaptioningInContext(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Base captioning task for IDEFICS: only overrides model_class to "IdeficsForCausalLM" (keeps
    # the base tokenizer_max_seq_len of 1024).
    model_class: str = "IdeficsForCausalLM"


class TextCapsIdeficsImageCaptioningInContextTextGenMetrics(IdeficsImageCaptioningInContext):
    # [CLASS DOC] TextCaps (HuggingFaceM4/TextCaps): validation queries, train support, columns
    # image/reference_strs; metrics BLEU-4, CIDEr, METEOR, ROUGE-L, SPICE and DEFAULT_TO_SERVER_RESULTS.
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.SPICE,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    reference_captions_column_name: str = "reference_strs"


class TextCapsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsIdeficsImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] TextCaps task without SPICE (BLEU-4, CIDEr, METEOR, ROUGE-L, server results).
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }


class TextCapsNewSplitsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] TextCaps support/query-set version (HuggingFaceM4/TextCaps_support_query_sets) with
    # validation, test and server_check splits, id column image_id, prompt template 7.
    dataset_name: str = "HuggingFaceM4/TextCaps_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    server_check_support_split_name: str = "server_check_support_set"
    server_check_query_split_name: str = "server_check_query_set"
    id_column_name: str = "image_id"
    selected_prompt_template_id = 7


class TextCapsSampleIdeficsImageCaptioningInContextTextGenMetrics(
    TextCapsIdeficsImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] TextCaps all-metrics task on HuggingFaceM4/TextCaps-Sample.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class TextCapsSampleIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] TextCaps no-SPICE task on HuggingFaceM4/TextCaps-Sample.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenIdeficsImageCaptioningInContextTextGenMetrics(IdeficsImageCaptioningInContext):
    # [CLASS DOC] CommonGen (HuggingFaceM4/common_gen): generate a sentence from image + concepts (context
    # column "concepts", reference "target"); BLEU-4/CIDEr/METEOR/ROUGE-L/SPICE; stop words Input/Output; a
    # single prompt template "Input: {context}. Output: {caption}" after the image tokens.
    dataset_name: str = "HuggingFaceM4/common_gen"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.SPICE,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    context_column_name: str = "concepts"
    reference_captions_column_name: str = "target"
    stop_words = ["Input", "Output"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}Input: {context}. Output: {caption}",
        }
    }


class CommonGenIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    CommonGenIdeficsImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] CommonGen task without SPICE.
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }


class NoCapsIdeficsImageCaptioningInContextTextGenMetrics(IdeficsImageCaptioningInContext):
    # [CLASS DOC] NoCaps (HuggingFaceM4/NoCaps): reference column annotations_captions;
    # BLEU-4/CIDEr/METEOR/ROUGE-L plus server results.
    # KNOWN ISSUE: default support split "train" does not exist in the dataset (see inline comment), so few-
    # shot runs on the default splits fail.
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
            # ImageCaptioningMetrics.SPICE,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    # This does not exist yet... it would require adding a training split to the dataset (see `create_sample_evaluation_datasets_simplified.py`)
    image_column_name: str = "image"
    reference_captions_column_name: str = "annotations_captions"


class NoCapsNewSplitsIdeficsImageCaptioningInContextTextGenMetrics(
    NoCapsIdeficsImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] NoCaps support/query-set version with validation, test and server_check splits, id image_id,
    # template 7.
    dataset_name: str = "HuggingFaceM4/NoCaps_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    server_check_support_split_name: str = "server_check_support_set"
    server_check_query_split_name: str = "server_check_query_set"
    id_column_name: str = "image_id"
    selected_prompt_template_id = 7


class NoCapsSampleIdeficsImageCaptioningInContextTextGenMetrics(NoCapsIdeficsImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] NoCaps task on HuggingFaceM4/NoCaps-Sample.
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoIdeficsImageCaptioningInContextBleuCiderMeteorRouge(IdeficsImageCaptioningInContext):
    # [CLASS DOC] COCO 2014 captions (HuggingFaceM4/COCO, config 2014_captions): reference column
    # sentences_raw, BLEU-4/CIDEr/METEOR/ROUGE-L.
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    reference_captions_column_name: str = "sentences_raw"


class CocoNewSplitsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    CocoIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] COCO support/query-set version (validation and test splits), prompt template 7.
    dataset_name: str = "HuggingFaceM4/coco_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 7


class CocoSampleIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    CocoIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] COCO task on HuggingFaceM4/COCO-2014_captions-Sample (no dataset config).
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class Flickr30kNewSplitsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(IdeficsImageCaptioningInContext):
    # [CLASS DOC] Flickr30k support/query-set task (HuggingFaceM4/flickr30k_support_query_sets): reference
    # column sentences, BLEU-4/CIDEr/METEOR/ROUGE-L, prompt template 7.
    dataset_name: str = "HuggingFaceM4/flickr30k_support_query_sets"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    image_column_name: str = "image"
    reference_captions_column_name: str = "sentences"
    selected_prompt_template_id = 7


class IIIT5KIdeficsImageCaptioningInContextExactMatch(IdeficsImageCaptioningInContext):
    # [CLASS DOC] IIIT-5K scene-text recognition framed as captioning: test queries, train support, reference
    # "label", exact-match metric, prompt "A photo where it is written {caption}", stop word "A photo".
    dataset_name: str = "HuggingFaceM4/IIIT-5K"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {"metrics": [ImageCaptioningMetrics.EXACT_MATCH]}
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    reference_captions_column_name: str = "label"
    stop_words = ["A photo"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}A photo where it is written {caption}",
        }
    }


class IIIT5KSampleIdeficsImageCaptioningInContextExactMatch(IIIT5KIdeficsImageCaptioningInContextExactMatch):
    # [CLASS DOC] IIIT-5K task on HuggingFaceM4/IIIT-5K-Sample.
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"
