# [MODULE DOC] In-context captioning task configurations for VLlama3: reuse Vgpt2ImageCaptioningInContext
# logic with model_class "VLlama3ForCausalLM" and a 10240-token limit, and declare dataset
# subclasses (TextCaps, CommonGen, NoCaps, COCO, Flickr30k, IIIT5K, incl. "NewSplits"
# support/query-set versions and -Sample variants).
from m4.evaluation.custom_metrics.unfolded_image_captioning_metrics import ImageCaptioningMetrics
from m4.models.vgpt2.evaluation_captioning_in_context_vgpt2 import Vgpt2ImageCaptioningInContext


class VLlama3ImageCaptioningInContext(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Base VLlama3 captioning task: Vgpt2ImageCaptioningInContext with model_class
    # "VLlama3ForCausalLM" and tokenizer_max_seq_len 10240.
    model_class: str = "VLlama3ForCausalLM"
    tokenizer_max_seq_len = 10240


class TextCapsVLlama3ImageCaptioningInContextTextGenMetrics(VLlama3ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of VLlama3ImageCaptioningInContext; declares only class
    # attributes (no new behavior). Dataset: 'HuggingFaceM4/TextCaps'. Metric
    # 'UnfoldedImageCaptioningMetrics' computing: BLEU_4, CIDER, METEOR, ROUGE_L, SPICE,
    # DEFAULT_TO_SERVER_RESULTS. Splits: default_query_split_name='validation',
    # default_support_split_name='train'. Columns: image_column_name='image',
    # reference_captions_column_name='reference_strs'.
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


class TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVLlama3ImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] Task configuration subclass of TextCapsVLlama3ImageCaptioningInContextTextGenMetrics;
    # declares only class attributes (no new behavior). Metric computing: BLEU_4, CIDER, METEOR,
    # ROUGE_L, DEFAULT_TO_SERVER_RESULTS.
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }


class TextCapsNewSplitsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] Task configuration subclass of TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge;
    # declares only class attributes (no new behavior). Dataset:
    # 'HuggingFaceM4/TextCaps_support_query_sets'. Splits:
    # default_query_split_name='validation_query_set',
    # default_support_split_name='validation_support_set',
    # validation_query_split_name='validation_query_set',
    # validation_support_split_name='validation_support_set',
    # test_query_split_name='test_query_set', test_support_split_name='test_support_set',
    # server_check_support_split_name='server_check_support_set',
    # server_check_query_split_name='server_check_query_set'. Columns: id_column_name='image_id'.
    # selected_prompt_template_id=7.
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


class TextCapsSampleVLlama3ImageCaptioningInContextTextGenMetrics(
    TextCapsVLlama3ImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] Same task configuration as TextCapsVLlama3ImageCaptioningInContextTextGenMetrics, but loads
    # dataset 'HuggingFaceM4/TextCaps-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class TextCapsSampleVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] Same task configuration as TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge, but
    # loads dataset 'HuggingFaceM4/TextCaps-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVLlama3ImageCaptioningInContextTextGenMetrics(VLlama3ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of VLlama3ImageCaptioningInContext; declares only class
    # attributes (no new behavior). Dataset: 'HuggingFaceM4/common_gen'. Metric
    # 'UnfoldedImageCaptioningMetrics' computing: BLEU_4, CIDER, METEOR, ROUGE_L, SPICE. Splits:
    # default_query_split_name='validation', default_support_split_name='train'. Columns:
    # image_column_name='image', context_column_name='concepts',
    # reference_captions_column_name='target'. stop_words=['Input', 'Output'].
    # prompt_templates_dict overrides templates with ids [0] (prefix + per-example pattern).
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


class CommonGenVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    CommonGenVLlama3ImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] Task configuration subclass of CommonGenVLlama3ImageCaptioningInContextTextGenMetrics;
    # declares only class attributes (no new behavior). Metric computing: BLEU_4, CIDER, METEOR,
    # ROUGE_L.
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }


class NoCapsVLlama3ImageCaptioningInContextTextGenMetrics(VLlama3ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of VLlama3ImageCaptioningInContext; declares only class
    # attributes (no new behavior). Dataset: 'HuggingFaceM4/NoCaps'. Metric
    # 'UnfoldedImageCaptioningMetrics' computing: BLEU_4, CIDER, METEOR, ROUGE_L,
    # DEFAULT_TO_SERVER_RESULTS. Splits: default_query_split_name='validation',
    # default_support_split_name='train'. Columns: image_column_name='image',
    # reference_captions_column_name='annotations_captions'.
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


class NoCapsNewSplitsVLlama3ImageCaptioningInContextTextGenMetrics(
    NoCapsVLlama3ImageCaptioningInContextTextGenMetrics
):
    # [CLASS DOC] Task configuration subclass of NoCapsVLlama3ImageCaptioningInContextTextGenMetrics; declares
    # only class attributes (no new behavior). Dataset: 'HuggingFaceM4/NoCaps_support_query_sets'.
    # Splits: default_query_split_name='validation_query_set',
    # default_support_split_name='validation_support_set',
    # validation_query_split_name='validation_query_set',
    # validation_support_split_name='validation_support_set',
    # test_query_split_name='test_query_set', test_support_split_name='test_support_set',
    # server_check_support_split_name='server_check_support_set',
    # server_check_query_split_name='server_check_query_set'. Columns: id_column_name='image_id'.
    # selected_prompt_template_id=7.
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


class NoCapsSampleVLlama3ImageCaptioningInContextTextGenMetrics(NoCapsVLlama3ImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] Same task configuration as NoCapsVLlama3ImageCaptioningInContextTextGenMetrics, but loads
    # dataset 'HuggingFaceM4/NoCaps-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(VLlama3ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of VLlama3ImageCaptioningInContext; declares only class
    # attributes (no new behavior). Dataset: 'HuggingFaceM4/COCO' (config '2014_captions'). Metric
    # 'UnfoldedImageCaptioningMetrics' computing: BLEU_4, CIDER, METEOR, ROUGE_L. Splits:
    # default_query_split_name='validation', default_support_split_name='train'. Columns:
    # image_column_name='image', reference_captions_column_name='sentences_raw'.
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


class CocoNewSplitsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] Task configuration subclass of CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge;
    # declares only class attributes (no new behavior). Dataset:
    # 'HuggingFaceM4/coco_support_query_sets' (config None). Splits:
    # default_query_split_name='validation_query_set',
    # default_support_split_name='validation_support_set',
    # validation_query_split_name='validation_query_set',
    # validation_support_split_name='validation_support_set',
    # test_query_split_name='test_query_set', test_support_split_name='test_support_set'.
    # selected_prompt_template_id=7.
    dataset_name: str = "HuggingFaceM4/coco_support_query_sets"
    dataset_config = None
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 7


class CocoSampleVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] Same task configuration as CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge, but
    # loads dataset 'HuggingFaceM4/COCO-2014_captions-Sample' (dataset_config=None) - a small
    # sample used for quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class Flickr30kNewSplitsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(VLlama3ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of VLlama3ImageCaptioningInContext; declares only class
    # attributes (no new behavior). Dataset: 'HuggingFaceM4/flickr30k_support_query_sets'. Metric
    # 'UnfoldedImageCaptioningMetrics' computing: BLEU_4, CIDER, METEOR, ROUGE_L. Splits:
    # default_query_split_name='validation_query_set',
    # default_support_split_name='validation_support_set',
    # validation_query_split_name='validation_query_set',
    # validation_support_split_name='validation_support_set',
    # test_query_split_name='test_query_set', test_support_split_name='test_support_set'. Columns:
    # image_column_name='image', reference_captions_column_name='sentences'.
    # selected_prompt_template_id=7.
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


class IIIT5KVLlama3ImageCaptioningInContextExactMatch(VLlama3ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of VLlama3ImageCaptioningInContext; declares only class
    # attributes (no new behavior). Dataset: 'HuggingFaceM4/IIIT-5K'. Metric
    # 'UnfoldedImageCaptioningMetrics' computing: EXACT_MATCH. Splits:
    # default_query_split_name='test', default_support_split_name='train'. Columns:
    # image_column_name='image', reference_captions_column_name='label'. stop_words=['A photo'].
    # prompt_templates_dict overrides templates with ids [0] (prefix + per-example pattern).
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


class IIIT5KSampleVLlama3ImageCaptioningInContextExactMatch(IIIT5KVLlama3ImageCaptioningInContextExactMatch):
    # [CLASS DOC] Same task configuration as IIIT5KVLlama3ImageCaptioningInContextExactMatch, but loads
    # dataset 'HuggingFaceM4/IIIT-5K-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"
