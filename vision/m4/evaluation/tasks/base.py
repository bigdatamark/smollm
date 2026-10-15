# [MODULE DOC] Base task descriptors for the evaluation framework. Tasks are declared as classes whose class attributes
# describe the dataset, metric, model class and column names; the evaluator reads these attributes.
from enum import Enum
from typing import List, Optional


class Predictor(Enum):
    # [CLASS DOC] Enum of evaluator kinds a task can use: in_contexter or linear_prober (values equal the names).
    in_contexter = "in_contexter"
    linear_prober = "linear_prober"


class BaseTask:
    # [CLASS DOC] Root task description. Class attributes: dataset_name (Hub dataset id), dataset_config, default_support_split_name
    # (split for in-context shots), default_query_split_name (split evaluated), metric_name (metric class name),
    # metrics_kwargs (dict passed to the metric), extra_metrics, model_class (name of the model class to load),
    # predictor_class (Predictor), id_column_name. Annotated attributes without default must be set by subclasses.
    # KNOWN ISSUE: metrics_kwargs is never read; the evaluator and concrete tasks use metric_kwargs (no "s").
    dataset_name: str  # Dataset (example: birdsnap)
    dataset_config: Optional[str] = None  # Dataset config (example: partition_1)
    default_support_split_name: Optional[str] = None
    default_query_split_name: str
    metric_name: str  # the metric to use (example: accuracy) - use evaluate
    metrics_kwargs: Optional[dict] = {}
    extra_metrics: Optional[list] = None
    model_class: str  # The model
    predictor_class: Predictor
    id_column_name: Optional[str] = None

    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] No-op constructor accepting and ignoring arbitrary keyword arguments.
        pass


class BaseTaskClassification(BaseTask):
    # [CLASS DOC] Classification task: image_column_names, label_column_name, optional context_column_names,
    # tested_ex_excluded_context_columns, tested_labels_column_name (per-example candidate labels) and
    # relevance_scores_column_name.
    image_column_names: List[str]
    label_column_name: str
    context_column_names: Optional[List[str]] = None
    tested_ex_excluded_context_columns: Optional[List[str]] = None
    tested_labels_column_name: Optional[str] = None
    relevance_scores_column_name: Optional[str] = None


class BaseTaskOpenEndedVQA(BaseTask):
    # [CLASS DOC] Open-ended VQA task: image_column_name, question_column_name, answers_column_name, optional
    # context_column_names.
    image_column_name: str
    question_column_name: str
    answers_column_name: str
    context_column_names: Optional[List[str]] = None


class BaseTaskImageCaptioning(BaseTask):
    # [CLASS DOC] Image-captioning task: image_column_name, reference_captions_column_name, optional context_column_name.
    image_column_name: str
    reference_captions_column_name: str
    context_column_name: Optional[str] = None


class BaseTaskImageCaptionMatching(BaseTask):
    # [CLASS DOC] Image-caption matching (e.g. Winoground-style) task: image_column_names and caption_column_names lists.
    image_column_names: List[str]
    caption_column_names: List[str]
