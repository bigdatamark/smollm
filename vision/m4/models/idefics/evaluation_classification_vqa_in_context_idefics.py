# [MODULE DOC] IDEFICS registrations of the in-context classification-VQA evaluation task (answer chosen by
# scoring a fixed set of candidate answers). All logic is inherited from the VGPT2 implementation; these
# classes only set model/dataset/metric attributes and are discovered by name by the m4 evaluation registry.
from m4.evaluation.custom_metrics.classification_vqa_metrics import ClassifVQAMetrics
from m4.models.vgpt2.evaluation_classification_vqa_in_context_vgpt2 import Vgpt2ClassificationVQAInContext


class IdeficsClassificationVQAInContext(Vgpt2ClassificationVQAInContext):
    # [CLASS DOC] Base task for IDEFICS: Vgpt2ClassificationVQAInContext with model_class "IdeficsForCausalLM"
    # and tokenizer_max_seq_len 2048.
    model_class: str = "IdeficsForCausalLM"
    tokenizer_max_seq_len = 2048


class VQAv2IdeficsClassificationVQAInContextAcc(IdeficsClassificationVQAInContext):
    # [CLASS DOC] VQAv2 (HuggingFaceM4/VQAv2_modif) task: query split validation, support split train, columns
    # image/question/answers, metrics VQA accuracy + entropy distribution/mean, no length normalization.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif"
    metric_name: str = "ClassificationVQAMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifVQAMetrics.VQA_ACCURACY,
            ClassifVQAMetrics.ENTROPY_DISTRIBUTION,
            ClassifVQAMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    length_normalize: bool = False


class VQAv2SampleIdeficsClassificationVQAInContextAcc(VQAv2IdeficsClassificationVQAInContextAcc):
    # [CLASS DOC] Same as the VQAv2 task but on the small HuggingFaceM4/VQAv2_modif-Sample dataset.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"
