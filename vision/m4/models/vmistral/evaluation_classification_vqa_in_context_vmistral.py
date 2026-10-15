# [MODULE DOC] VMistral evaluation task classes for in-context classification-style VQA (scoring candidate answers by
# log-likelihood). Behavior is inherited entirely from Vgpt2ClassificationVQAInContext; these classes only set
# class-level configuration attributes (model class, max length, dataset, metric, split and column names).
from m4.evaluation.custom_metrics.classification_vqa_metrics import ClassifVQAMetrics
from m4.models.vgpt2.evaluation_classification_vqa_in_context_vgpt2 import Vgpt2ClassificationVQAInContext


class VMistralClassificationVQAInContext(Vgpt2ClassificationVQAInContext):
    # [CLASS DOC] Base classification-VQA task for VMistral: reuses the VGPT2 implementation with model_class
    # "VMistralForCausalLM" and tokenizer_max_seq_len 4096.
    model_class: str = "VMistralForCausalLM"
    tokenizer_max_seq_len = 4096


class VQAv2VMistralClassificationVQAInContextAcc(VMistralClassificationVQAInContext):
    # [CLASS DOC] VQAv2 (HuggingFaceM4/VQAv2_modif) classification-VQA task: query split "validation", support split
    # "train"; columns image/question/answers; metrics VQA accuracy, entropy distribution and entropy mean; no
    # length normalization of answer log-likelihoods.
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


class VQAv2SampleVMistralClassificationVQAInContextAcc(VQAv2VMistralClassificationVQAInContextAcc):
    # [CLASS DOC] Same as the VQAv2 task but on the small "HuggingFaceM4/VQAv2_modif-Sample" dataset (for quick runs).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"
