# [MODULE DOC] VLlama3 open-ended (generative) VQA evaluation tasks, used in-context (few-shot) or zero-shot with chat
# templates. All behavior (prompt construction, generation, answer post-processing, metric updates) is
# inherited from Vgpt2OpenEndedVQAInContext; every class here only sets class-level configuration: dataset
# name/config, metric name and metric list, split names (default/validation/test x query/support), column
# names, prompt templates and which template is selected. "Sample"/"Dummy" variants point to tiny copies of a
# dataset; "NewSplits"/"PartN" variants use pre-made support/query splits; DEFAULT_TO_SERVER_RESULTS metrics
# dump predictions for external scoring.
from typing import List

from m4.evaluation.custom_metrics.doc_vqa_metrics import DVQAMetrics
from m4.evaluation.custom_metrics.open_ended_vqa_metrics import OEVQAMetrics
from m4.models.vgpt2.evaluation_open_ended_vqa_in_context_vgpt2 import Vgpt2OpenEndedVQAInContext


class VLlama3OpenEndedVQAInContext(Vgpt2OpenEndedVQAInContext):
    # [CLASS DOC] Base open-ended VQA task for VLlama3: model_class "VLlama3ForCausalLM", tokenizer_max_seq_len 10240.
    model_class: str = "VLlama3ForCausalLM"
    tokenizer_max_seq_len = 10240


class VQAv2VLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] VQAv2 (HuggingFaceM4/VQAv2_modif): query split validation, support train; columns image / question /
    # answers, id "question_id"; metrics first-word and full open-ended VQA accuracy, plus the same two with
    # GT Vision Lab normalization, plus DEFAULT_TO_SERVER_RESULTS.
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


class VQAv2Modif10kVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] VQAv2 10k-question validation subset (HuggingFaceM4/VQAv2_modif_10k): query split "validation_10k",
    # validation support split "train"; same metrics as VQAv2. Column names come from the base class defaults.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_10k"
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
    default_query_split_name: str = "validation_10k"
    validation_query_split_name: str = "validation_10k"
    validation_support_split_name: str = "train"


class VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 on HuggingFaceM4/VQAv2_modif_support_query_sets. Uses the "*_support_query_sets" dataset variant
    # whose splits are <split>_query_set / <split>_support_set (validation and test); prompt template 9 of the
    # base class.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class VQAv2Part0NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard 0 of the dataset (HuggingFaceM4/VQAv2_modif_support_query_sets_part_0).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_0"


class VQAv2Part1NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard 1 of the dataset (HuggingFaceM4/VQAv2_modif_support_query_sets_part_1).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_1"


class VQAv2Part2NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard 2 of the dataset (HuggingFaceM4/VQAv2_modif_support_query_sets_part_2).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_2"


class VQAv2Part3NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard 3 of the dataset (HuggingFaceM4/VQAv2_modif_support_query_sets_part_3).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_3"


class VQAv2Part4NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard 4 of the dataset (HuggingFaceM4/VQAv2_modif_support_query_sets_part_4).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_4"


class VQAv2Part5NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 new-splits task on shard 5 of the dataset (HuggingFaceM4/VQAv2_modif_support_query_sets_part_5).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_5"


class VQAv2SampleVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 task on the small HuggingFaceM4/VQAv2_modif-Sample dataset.
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"


class VQAv2DummyVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 task on the tiny HuggingFaceM4/VQAv2_modif-Dummy dataset (debugging).
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Dummy"


class VQAv2ChatbotVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VQAv2 chatbot evaluation with dataset config "validation"; query validation/test, support train;
    # template asks the question followed by "Give a very brief answer." Chatbot (instruct) variant:
    # bool_instruct_templates is True so prompts use prompt_templates_dict_instruct, formatted as
    # "User:<image>... <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos
    # token.
    dataset_config: str = "validation"
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n"
                "Give a very brief answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class OKVQAVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # We are considering the raw answers. In the original paper,
    # they are doing a step of stemming (standardize pluralization and conjugation).
    # [CLASS DOC] OK-VQA (HuggingFaceM4/OK-VQA_modif), raw answers (no stemming as in the paper); splits validation/train;
    # columns image / question / answers; metrics first-word and full open-ended VQA accuracy, plus the same
    # two with GT Vision Lab normalization.
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


class OKVQANewSplitsVLlama3OpenEndedVQAInContextAcc(OKVQAVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] OK-VQA on HuggingFaceM4/OK-VQA_modif_support_query_sets. Uses the "*_support_query_sets" dataset variant
    # whose splits are <split>_query_set / <split>_support_set (validation and test); prompt template 9 of the
    # base class.
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class OKVQASampleVLlama3OpenEndedVQAInContextAcc(OKVQAVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] OK-VQA task on HuggingFaceM4/OK-VQA_modif-Sample.
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif-Sample"


class OKVQAChatbotVLlama3OpenEndedVQAInContextAcc(OKVQAVLlama3OpenEndedVQAInContextAcc):
    # We are considering the raw answers. In the original paper,
    # they are doing a step of stemming (standardize pluralization and conjugation).
    # [CLASS DOC] OK-VQA chatbot evaluation (re-declares the parent's dataset, metrics and columns); query validation,
    # support train (test splits inherited from the base class). Template: question + "Give a very brief
    # answer." Chatbot (instruct) variant: bool_instruct_templates is True so prompts use
    # prompt_templates_dict_instruct, formatted as "User:<image>... <end_of_utterance>\nAssistant:
    # {answer}<end_of_utterance>" turns prefixed by the bos token.
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
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n"
                "Give a very brief answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class TextVQAVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] TextVQA (Hub dataset "textvqa"): splits validation/train; columns image / question / answers (list of
    # strings); metrics first-word and full open-ended VQA accuracy, plus the same two with GT Vision Lab
    # normalization.
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


class TextVQANewSplitsVLlama3OpenEndedVQAInContextAcc(TextVQAVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] TextVQA on HuggingFaceM4/textvqa_support_query_sets. Uses the "*_support_query_sets" dataset variant
    # whose splits are <split>_query_set / <split>_support_set (validation and test); prompt template 9 of the
    # base class.
    dataset_name: str = "HuggingFaceM4/textvqa_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class TextVQASampleVLlama3OpenEndedVQAInContextAcc(TextVQAVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] TextVQA task on HuggingFaceM4/textvqa-Sample.
    dataset_name: str = "HuggingFaceM4/textvqa-Sample"


class TextVQAChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] TextVQA chatbot evaluation: query validation (test query "test"), support train; VQA metrics plus
    # DEFAULT_TO_SERVER_RESULTS. Two instruct templates: 0 = question + "Give a very brief answer."; 1
    # (selected) = a long instruction list (answer briefly, Yes/No for binary questions, no full stops or
    # quotes, etc.). Chatbot (instruct) variant: bool_instruct_templates is True so prompts use
    # prompt_templates_dict_instruct, formatted as "User:<image>... <end_of_utterance>\nAssistant:
    # {answer}<end_of_utterance>" turns prefixed by the bos token.
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
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"  # List of strings
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n"
                "Give a very brief answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        1: {
            "prefix": "{bos_token}",
            "example": """<image>Answer the following question about the image using as few words as possible. Follow these additional instructions:
-Always answer a binary question with Yes or No.
-When asked what time it is, reply with the time seen in the image.
-Do not put any full stops at the end of the answer.
-Do not put quotation marks around the answer.
-An answer with one or two words is favorable.
-Do not apply common sense knowledge. The answer can be found in the image.
Question: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n""",
        },
    }
    selected_prompt_template_id = 1
    bool_instruct_templates = True


class AdVQAVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] AdVQA (HuggingFaceM4/AdVQA_modif): splits validation/train; columns image / question / answers; metrics
    # first-word and full open-ended VQA accuracy, plus the same two with GT Vision Lab normalization.
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


class AdVQASampleVLlama3OpenEndedVQAInContextAcc(AdVQAVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] AdVQA task on HuggingFaceM4/AdVQA_modif-Sample.
    dataset_name: str = "HuggingFaceM4/AdVQA_modif-Sample"


class VizWizVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] VizWiz (HuggingFaceM4/VizWiz): splits validation/train; columns image / question / answers, id
    # "filename"; metrics first-word and full open-ended VQA accuracy, plus the same two with GT Vision Lab
    # normalization, plus DEFAULT_TO_SERVER_RESULTS.
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


class VizWizSampleVLlama3OpenEndedVQAInContextAcc(VizWizVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VizWiz task on HuggingFaceM4/VizWiz-Sample.
    dataset_name: str = "HuggingFaceM4/VizWiz-Sample"


class VizWizNewSplitsVLlama3OpenEndedVQAInContextAcc(VizWizVLlama3OpenEndedVQAInContextAcc):
    # [CLASS DOC] VizWiz on HuggingFaceM4/VizWiz_support_query_sets (validation/test query and support sets) with 8 custom
    # few-shot templates whose prefix instructs the model to answer "unanswerable" when the question cannot be
    # answered and whose example is "Image:<image>Question: {question} Answer: {answer}"; template 3 is
    # selected.
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
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        1: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question. Use the image to answer. Answer"
                " 'unanswerable' if the question is impossible to answer.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        2: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question when possible, otherwise say unanswerable."
                " Use the image to answer.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        3: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image when possible, otherwise say unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        4: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image. When it's ambiguous, answer unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        5: {
            "prefix": (
                "{bos_token}Homework:\nExercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        6: {
            "prefix": (
                "{bos_token}Exercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        7: {
            "prefix": (
                "{bos_token}Homework:\nExercise:When possible, answer the questions based on the image, otherwise say"
                " unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
    }
    selected_prompt_template_id = 3


class DocVQAVLlama3OpenEndedVQAInContextAnls(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] DocVQA (HuggingFaceM4/DocumentVQA): query validation/test, support train; columns image / question /
    # answers, id "questionId"; metrics ANLS plus DEFAULT_TO_SERVER_RESULTS; prompt template 9 of the base
    # class.
    dataset_name: str = "HuggingFaceM4/DocumentVQA"
    metric_name: str = "DocVQAMetrics"
    metric_kwargs = {
        "metrics": [
            DVQAMetrics.ANLS,
            DVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "questionId"
    selected_prompt_template_id = 9


class DocVQASampleVLlama3OpenEndedVQAInContextAnls(DocVQAVLlama3OpenEndedVQAInContextAnls):
    # [CLASS DOC] DocVQA task on HuggingFaceM4/DocumentVQA-Sample, ANLS only.
    dataset_name: str = "HuggingFaceM4/DocumentVQA-Sample"
    metric_kwargs = {
        "metrics": [
            DVQAMetrics.ANLS,
        ]
    }


class DocVQAChatbotVLlama3OpenEndedVQAInContextAnls(DocVQAVLlama3OpenEndedVQAInContextAnls):
    # [CLASS DOC] DocVQA chatbot evaluation. Instruct templates: 0 = question + "Give a very brief answer."; 1 (selected)
    # = an instruction to give a short, terse answer without paraphrasing or full stops. Chatbot (instruct)
    # variant: bool_instruct_templates is True so prompts use prompt_templates_dict_instruct, formatted as
    # "User:<image>... <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos
    # token.
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\nGive a very brief"
                " answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        1: {
            "prefix": "{bos_token}",
            "example": """User:<image>Give a short and terse answer to the following question. Do not paraphrase or reformat the text you see in the image. Do not include any full stops. Just give the answer without additional explanation.
Question: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n""",
        },
    }
    selected_prompt_template_id = 1
    bool_instruct_templates = True


class MMMUVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] MMMU (HuggingFaceM4/MMMU-modif-with-categories): query validation/test, support "dev"; images column
    # "images" (multiple images); id "id"; metrics MMMU-style accuracy overall and per bucket (buckets
    # "broad_category", "narrow_category") plus DEFAULT_TO_SERVER_RESULTS; prompt template 14 of the base
    # class.
    dataset_name: str = "HuggingFaceM4/MMMU-modif-with-categories"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_MMMU_STYLE_PER_BUCKET_ACCURACY,
            OEVQAMetrics.OE_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "dev"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "dev"
    test_query_split_name: str = "test"
    test_support_split_name: str = "dev"
    image_column_name: str = "images"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    buckets_keys: List[str] = ["broad_category", "narrow_category"]
    id_column_name: str = "id"
    selected_prompt_template_id = 14


class MathVistaVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] MathVista (HuggingFaceM4/MathVista-modif): query validation/test, support validation; id "pid"; metrics
    # MMMU-style accuracy plus DEFAULT_TO_SERVER_RESULTS; prompt template 15 of the base class.
    dataset_name: str = "HuggingFaceM4/MathVista-modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "validation"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "validation"
    test_query_split_name: str = "test"
    test_support_split_name: str = "validation"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "pid"
    selected_prompt_template_id = 15


class MMVETChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] MM-Vet (HuggingFaceM4/MM_VET_modif), test split only; answer column "answer", id "question_id"; only
    # dumps results for external (GPT-based) MM-Vet scoring. Template: plain question turn. Chatbot (instruct)
    # variant: bool_instruct_templates is True so prompts use prompt_templates_dict_instruct, formatted as
    # "User:<image>... <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos
    # token.
    dataset_name: str = "HuggingFaceM4/MM_VET_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS_MMVET,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    id_column_name: str = "question_id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class LLaVAWildChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] LLaVA-in-the-Wild (HuggingFaceM4/LLaVA_Wild_Modif), test split only; answer column "answer", id
    # "question_id"; only dumps results for external LLaVA-Wild scoring. Template: plain question turn.
    # Chatbot (instruct) variant: bool_instruct_templates is True so prompts use
    # prompt_templates_dict_instruct, formatted as "User:<image>... <end_of_utterance>\nAssistant:
    # {answer}<end_of_utterance>" turns prefixed by the bos token.
    dataset_name: str = "HuggingFaceM4/LLaVA_Wild_Modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS_LLAVA_WILD,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    id_column_name: str = "question_id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class ChartQAChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] ChartQA (HuggingFaceM4/ChartQA): splits "val"/"test" (support = same split); question column "query",
    # answer column "label"; metric relaxed VQA accuracy. Instruct templates: 0 = question + "Give a very
    # brief answer."; 1 (selected) = a detailed instruction list (few words, numbers as digits, lists as [a,
    # b], decimals for ratios, no units/full stops). Chatbot (instruct) variant: bool_instruct_templates is
    # True so prompts use prompt_templates_dict_instruct, formatted as "User:<image>...
    # <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos token.
    dataset_name: str = "HuggingFaceM4/ChartQA"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_RELAXED_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "val"
    default_support_split_name: str = "val"
    validation_query_split_name: str = "val"
    validation_support_split_name: str = "val"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_name: str = "image"
    question_column_name: str = "query"
    answers_column_name: str = "label"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\nGive a very brief"
                " answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        1: {
            "prefix": "{bos_token}",
            "example": """User:<image>For the question below, follow the following instructions:
-The answer should contain as few words as possible.
-Don’t paraphrase or reformat the text you see in the image.
-Answer a binary question with Yes or No.
-When asked to give a numerical value, provide a number like 2 instead of Two.
-If the final answer has two or more items, provide it in the list format like [1, 2].
-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.
-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.
-Don’t include any units in the answer.
-Do not include any full stops at the end of the answer.
-Try to include the full label from the graph when asked about an entity.
Question: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n""",
        },
    }
    selected_prompt_template_id = 1
    bool_instruct_templates = True


class GQAChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] GQA (HuggingFaceM4/GQA): query validation/test, support train; id "question_id"; metrics first-word and
    # full open-ended VQA accuracy, plus the same two with GT Vision Lab normalization, plus
    # DEFAULT_TO_SERVER_RESULTS. Template: plain question turn. Chatbot (instruct) variant:
    # bool_instruct_templates is True so prompts use prompt_templates_dict_instruct, formatted as
    # "User:<image>... <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos
    # token.
    dataset_name: str = "HuggingFaceM4/GQA"
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
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "question_id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class MathVistaOpenEndedChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] MathVista open-ended subset (HuggingFaceM4/MathVista_OpenEnded): query testmini/test, support testmini;
    # answer column "answer", extra context column "instruction", id "pid"; metrics open-ended-only MMMU-style
    # accuracy plus DEFAULT_TO_SERVER_RESULTS. Three templates (question only; question then instruction;
    # instruction then question), template 2 selected. Chatbot (instruct) variant: bool_instruct_templates is
    # True so prompts use prompt_templates_dict_instruct, formatted as "User:<image>...
    # <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos token.
    dataset_name: str = "HuggingFaceM4/MathVista_OpenEnded"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_ONLY_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "testmini"
    default_support_split_name: str = "testmini"
    validation_query_split_name: str = "testmini"
    validation_support_split_name: str = "testmini"
    test_query_split_name: str = "test"
    test_support_split_name: str = "testmini"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    context_column_names: List[str] = ["instruction"]
    id_column_name: str = "pid"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
        1: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n{instruction}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        2: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{instruction}\n{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 2
    bool_instruct_templates = True


class MMMUOpenEndedChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # [CLASS DOC] MMMU open-ended subset (HuggingFaceM4/MMMU_OpenEnded): query validation/test, support dev; images column
    # "images", answer "answer", id "id"; metrics open-ended-only MMMU-style accuracy plus
    # DEFAULT_TO_SERVER_RESULTS. Template "User: {question}..." contains no <image> placeholder (images are
    # expected to be referenced inside the question text). Chatbot (instruct) variant: bool_instruct_templates
    # is True so prompts use prompt_templates_dict_instruct, formatted as "User:<image>...
    # <end_of_utterance>\nAssistant: {answer}<end_of_utterance>" turns prefixed by the bos token.
    dataset_name: str = "HuggingFaceM4/MMMU_OpenEnded"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_ONLY_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "dev"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "dev"
    test_query_split_name: str = "test"
    test_support_split_name: str = "dev"
    image_column_name: str = "images"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    id_column_name: str = "id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class MMMUOpenEndedSingleImageChatbotVLlama3OpenEndedVQAInContextAcc(
    MMMUOpenEndedChatbotVLlama3OpenEndedVQAInContextAcc
):
    # [CLASS DOC] MMMU open-ended restricted to single-image questions (HuggingFaceM4/MMMU_OpenEnded_single_image); every
    # split (query/support, default/validation/test) is "validation".
    dataset_name: str = "HuggingFaceM4/MMMU_OpenEnded_single_image"
    default_query_split_name: str = "validation"
    default_support_split_name: str = "validation"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "validation"
    test_query_split_name: str = "validation"
    test_support_split_name: str = "validation"
