# [MODULE DOC] VMistral in-context (few-shot) image captioning evaluation tasks. Generation, prompt building and
# post-processing are inherited from Vgpt2ImageCaptioningInContext; each class only sets configuration
# attributes: dataset, metric list, split names, column names, and optionally prompt templates / stop words.
# "Sample" variants point to small sample copies of a dataset for quick runs.
from m4.evaluation.custom_metrics.unfolded_image_captioning_metrics import ImageCaptioningMetrics
from m4.models.vgpt2.evaluation_captioning_in_context_vgpt2 import Vgpt2ImageCaptioningInContext


class VMistralImageCaptioningInContext(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Base captioning task for VMistral: model_class "VMistralForCausalLM", tokenizer_max_seq_len 4096.
    model_class: str = "VMistralForCausalLM"
    tokenizer_max_seq_len = 4096


class TextCapsVMistralImageCaptioningInContextTextGenMetrics(VMistralImageCaptioningInContext):
    # [CLASS DOC] TextCaps (HuggingFaceM4/TextCaps) captioning: query split "validation", support "train"; columns image /
    # reference_strs; metrics BLEU-4, CIDEr, METEOR, ROUGE-L, SPICE plus DEFAULT_TO_SERVER_RESULTS (dump
    # predictions for server-side scoring).
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


class TextCapsVMistralImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVMistralImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] TextCaps task without SPICE: BLEU-4, CIDEr, METEOR, ROUGE-L, DEFAULT_TO_SERVER_RESULTS.
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }


class TextCapsNewSplitsVMistralImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVMistralImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] TextCaps with the new support/query splits. Uses the "*_support_query_sets" variant of the dataset whose
    # splits are named <split>_query_set / <split>_support_set (validation, test and, where set,
    # server_check), with id column "image_id" and prompt template 7 of the base class.
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


class TextCapsSampleVMistralImageCaptioningInContextTextGenMetrics(
    TextCapsVMistralImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] TextCaps full-metric task on "HuggingFaceM4/TextCaps-Sample".
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class TextCapsSampleVMistralImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVMistralImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] TextCaps no-SPICE task on "HuggingFaceM4/TextCaps-Sample".
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVMistralImageCaptioningInContextTextGenMetrics(VMistralImageCaptioningInContext):
    # [CLASS DOC] CommonGen (HuggingFaceM4/common_gen): generate a sentence from a list of concepts (context column
    # "concepts", reference column "target") with the single template "<image>Input: {context}. Output:
    # {caption}"; generation is cut at stop words "Input"/"Output"; splits validation/train; metrics BLEU-4,
    # CIDEr, METEOR, ROUGE-L, SPICE.
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
            "example": "<image>Input: {context}. Output: {caption}",
        }
    }


class CommonGenVMistralImageCaptioningInContextBleuCiderMeteorRouge(
    CommonGenVMistralImageCaptioningInContextTextGenMetrics
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


class NoCapsVMistralImageCaptioningInContextTextGenMetrics(VMistralImageCaptioningInContext):
    # [CLASS DOC] NoCaps (HuggingFaceM4/NoCaps): columns image / annotations_captions; metrics BLEU-4, CIDEr, METEOR,
    # ROUGE-L, DEFAULT_TO_SERVER_RESULTS (SPICE disabled).
    # KNOWN ISSUE: default_support_split_name is "train" but, per the inline comment, NoCaps has no train
    # split, so few-shot support sampling would fail.
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


class NoCapsNewSplitsVMistralImageCaptioningInContextTextGenMetrics(
    NoCapsVMistralImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] NoCaps with the new support/query splits. Uses the "*_support_query_sets" variant of the dataset whose
    # splits are named <split>_query_set / <split>_support_set (validation, test and, where set,
    # server_check), with id column "image_id" and prompt template 7 of the base class.
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


class NoCapsSampleVMistralImageCaptioningInContextTextGenMetrics(NoCapsVMistralImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] NoCaps task on "HuggingFaceM4/NoCaps-Sample".
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVMistralImageCaptioningInContextBleuCiderMeteorRouge(VMistralImageCaptioningInContext):
    # [CLASS DOC] COCO 2014 captions (HuggingFaceM4/COCO, config "2014_captions"): columns image / sentences_raw; splits
    # validation/train; metrics BLEU-4, CIDEr, METEOR, ROUGE-L.
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


class CocoNewSplitsVMistralImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVMistralImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] COCO with the new support/query splits ("HuggingFaceM4/coco_support_query_sets", validation and test
    # query/support sets, template 7). Inherits dataset_config "2014_captions" from the parent.
    dataset_name: str = "HuggingFaceM4/coco_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 7


class CocoSampleVMistralImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVMistralImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] COCO task on "HuggingFaceM4/COCO-2014_captions-Sample" with no dataset config.
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class Flickr30kNewSplitsVMistralImageCaptioningInContextBleuCiderMeteorRouge(VMistralImageCaptioningInContext):
    # [CLASS DOC] Flickr30k ("HuggingFaceM4/flickr30k_support_query_sets"): validation/test query and support sets,
    # columns image / sentences, prompt template 7; metrics BLEU-4, CIDEr, METEOR, ROUGE-L.
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


class IIIT5KVMistralImageCaptioningInContextExactMatch(VMistralImageCaptioningInContext):
    # [CLASS DOC] IIIT-5K scene-text recognition framed as captioning: template "<image>A photo where it is written
    # {caption}", reference column "label", stop word "A photo", splits test/train, metric exact match.
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
            "example": "<image>A photo where it is written {caption}",
        }
    }


class IIIT5KSampleVMistralImageCaptioningInContextExactMatch(IIIT5KVMistralImageCaptioningInContextExactMatch):
    # [CLASS DOC] IIIT-5K task on "HuggingFaceM4/IIIT-5K-Sample".
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"
