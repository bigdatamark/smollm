# [MODULE DOC] IDEFICS registrations of in-context open-ended VQA evaluation tasks (answers are generated and
# scored with VQA accuracy variants). Logic is inherited from Vgpt2OpenEndedVQAInContext; classes only set
# dataset, split, column, metric and prompt attributes and are looked up by class name by the evaluation
# registry. Metrics are first-word and full open-ended VQA accuracy, each also with GT-vision-lab
# normalization, plus DEFAULT_TO_SERVER_RESULTS for datasets with server-evaluated test sets (VQAv2, VizWiz).
from m4.evaluation.custom_metrics.open_ended_vqa_metrics import OEVQAMetrics
from m4.models.vgpt2.evaluation_open_ended_vqa_in_context_vgpt2 import Vgpt2OpenEndedVQAInContext


class IdeficsOpenEndedVQAInContext(Vgpt2OpenEndedVQAInContext):
    # [CLASS DOC] Base open-ended VQA task for IDEFICS: model_class "IdeficsForCausalLM",
    # tokenizer_max_seq_len 2048.
    model_class: str = "IdeficsForCausalLM"
    tokenizer_max_seq_len = 2048


class VQAv2IdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
    # [CLASS DOC] VQAv2 (HuggingFaceM4/VQAv2_modif): validation queries, train support, columns
    # image/question/answers, id question_id, the four accuracy metrics plus server results.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "question_id"


class VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2IdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 support/query-set version (HuggingFaceM4/VQAv2_modif_support_query_sets) with
    # validation and test query/support splits and base prompt template 9.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class VQAv2Part0NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard dataset ..._support_query_sets_part_0.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_0"


class VQAv2Part1NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard dataset ..._support_query_sets_part_1.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_1"


class VQAv2Part2NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard dataset ..._support_query_sets_part_2.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_2"


class VQAv2Part3NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard dataset ..._support_query_sets_part_3.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_3"


class VQAv2Part4NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard dataset ..._support_query_sets_part_4.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_4"


class VQAv2Part5NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard dataset ..._support_query_sets_part_5.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_5"


class VQAv2SampleIdeficsOpenEndedVQAInContextAcc(VQAv2IdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 task on HuggingFaceM4/VQAv2_modif-Sample.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"


class VQAv2DummyIdeficsOpenEndedVQAInContextAcc(VQAv2IdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 task on the tiny HuggingFaceM4/VQAv2_modif-Dummy dataset (debugging).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Dummy"


class OKVQAIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
    # [CLASS DOC] OK-VQA (HuggingFaceM4/OK-VQA_modif): raw answers (no stemming, unlike the paper), the four
    # accuracy metrics, validation queries / train support.
    # We are considering the raw answers. In the original paper,
    # they are doing a step of stemming (standardize pluralization and conjugation).
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class OKVQANewSplitsIdeficsOpenEndedVQAInContextAcc(OKVQAIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] OK-VQA support/query-set version (HuggingFaceM4/OK-VQA_modif_support_query_sets) with
    # validation and test query/support splits and base prompt template 9.
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class OKVQASampleIdeficsOpenEndedVQAInContextAcc(OKVQAIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] OK-VQA task on HuggingFaceM4/OK-VQA_modif-Sample.
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif-Sample"


class TextVQAIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
    # [CLASS DOC] TextVQA (hub dataset "textvqa"; answers column is a list of strings), four accuracy metrics,
    # validation queries / train support.
    dataset_name: str = "textvqa"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"  # List of strings


class TextVQANewSplitsIdeficsOpenEndedVQAInContextAcc(TextVQAIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] TextVQA support/query-set version (HuggingFaceM4/textvqa_support_query_sets) with validation
    # and test query/support splits and base prompt template 9.
    dataset_name: str = "HuggingFaceM4/textvqa_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class TextVQASampleIdeficsOpenEndedVQAInContextAcc(TextVQAIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] TextVQA task on HuggingFaceM4/textvqa-Sample.
    dataset_name: str = "HuggingFaceM4/textvqa-Sample"


class AdVQAIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
    # [CLASS DOC] AdVQA (HuggingFaceM4/AdVQA_modif) with the four accuracy metrics.
    dataset_name: str = "HuggingFaceM4/AdVQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class AdVQASampleIdeficsOpenEndedVQAInContextAcc(AdVQAIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] AdVQA task on HuggingFaceM4/AdVQA_modif-Sample.
    dataset_name: str = "HuggingFaceM4/AdVQA_modif-Sample"


class VizWizIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
    # [CLASS DOC] VizWiz (HuggingFaceM4/VizWiz): id column filename, four accuracy metrics plus server
    # results.
    dataset_name: str = "HuggingFaceM4/VizWiz"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "filename"


class VizWizSampleIdeficsOpenEndedVQAInContextAcc(VizWizIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VizWiz task on HuggingFaceM4/VizWiz-Sample.
    dataset_name: str = "HuggingFaceM4/VizWiz-Sample"


class VizWizNewSplitsIdeficsOpenEndedVQAInContextAcc(VizWizIdeficsOpenEndedVQAInContextAcc):
    # [CLASS DOC] VizWiz support/query-set version (HuggingFaceM4/VizWiz_support_query_sets) with its own
    # eight prompt templates (0-7): each has an instruction prefix telling the model to answer from the image
    # or say "unanswerable", and an "Image:<image tokens>Question: {question} Answer: {answer}" example line.
    # Template 3 is selected.
    dataset_name: str = "HuggingFaceM4/VizWiz_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    prompt_templates_dict = {
        0: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question. Use the image to answer. Answer"
                " 'unanswerable' if the question is ambiguous or impossible to answer.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        1: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question. Use the image to answer. Answer"
                " 'unanswerable' if the question is impossible to answer.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        2: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question when possible, otherwise say unanswerable."
                " Use the image to answer.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        3: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image when possible, otherwise say unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        4: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image. When it's ambiguous, answer unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        5: {
            "prefix": (
                "{bos_token}Homework:\nExercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        6: {
            "prefix": (
                "{bos_token}Exercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        7: {
            "prefix": (
                "{bos_token}Homework:\nExercise:When possible, answer the questions based on the image, otherwise say"
                " unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
    }
    selected_prompt_template_id = 3
