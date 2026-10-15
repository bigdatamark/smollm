# [MODULE DOC] evaluate.Metric for VQA evaluated as classification: the model scores each candidate answer (tested label) of
# a question, the top-scoring candidate is the prediction, and the VQA soft accuracy is computed against the list
# of human answers. Also offers prediction-entropy metrics.
import logging
from enum import Enum
from typing import List

import datasets
import evaluate
import numpy as np
from scipy.special import softmax

from m4.evaluation.custom_metrics.utils import vqa_normalize_text


_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""
logger = logging.getLogger(__name__)


class ClassifVQAMetrics(Enum):
    # [CLASS DOC] Enum of metrics this class can compute: ENTROPY_DISTRIBUTION (per-example entropies), ENTROPY_MEAN,
    # VQA_ACCURACY. Values are the result keys.
    # The kl is ill defined, so not adding it
    ENTROPY_DISTRIBUTION = "entropy_distribution"
    ENTROPY_MEAN = "entropy_mean"
    VQA_ACCURACY = "vqa_accuracy"


class ClassificationVQAMetrics(evaluate.Metric):
    # [CLASS DOC] Metric object; constructed with metrics (list of ClassifVQAMetrics) plus evaluate.Metric kwargs. One input row
    # per (example, tested label) pair.
    """
    Adapted from `UnfoldedClassificationMetrics`.
    """

    def __init__(self, metrics: List[ClassifVQAMetrics], **kwargs):
        # [FUNCTION DOC] Store metrics after evaluate.Metric initialisation; raises ValueError if metrics is None.
        super().__init__(**kwargs)
        self.metrics = metrics

        if metrics is None:
            raise ValueError("`metrics` must be specified")

    def _info(self):
        # [FUNCTION DOC] HELPER: evaluate.Metric hook returning MetricInfo (empty description/citation) with the input features: predictions (float score), example_ids (str), true_labels (list of str answers), tested_labels (str candidate).
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float64"),
                    "example_ids": datasets.Value("string"),
                    "true_labels": datasets.Sequence(datasets.Value("string")),
                    "tested_labels": datasets.Value("string"),
                }
            ),
        )

    def _compute(self, predictions, example_ids, true_labels, tested_labels, tol=0.001):
        # [FUNCTION DOC] Group rows by example_id into lists of (score, tested label) plus the answer list; a tested label already seen
        # for an example is treated as a distributed duplicate (warning logged, asserts the answers are identical).
        # Top-1 prediction = tested label with highest score.
        # VQA_ACCURACY: for each example, answers are normalised with vqa_normalize_text; for each held-out answer i the
        # accuracy is min(1, #other answers equal to the prediction / 3), averaged over i, then averaged over examples ->
        # results["vqa_accuracy"] (float).
        # ENTROPY_*: softmax over each example's scores, entropy -sum(q log q); returned as list
        # ("entropy_distribution") and/or mean ("entropy_mean"). Returns the results dict. tol is unused.
        # KNOWN ISSUE: the answers are normalised but the predicted tested label is not, so candidates differing only in
        # case/punctuation/articles from the answers never match.
        data_per_id = {}
        for example_id, prediction, true_label_l, tested_label in zip(
            example_ids, predictions, true_labels, tested_labels
        ):
            if example_id not in data_per_id:
                data_per_id[example_id] = {
                    "predictions": [],
                    "tested_labels": [],
                }
            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            if tested_label in data_per_id[example_id]["tested_labels"]:
                idx_already_present = data_per_id[example_id]["tested_labels"].index(tested_label)
                # It happens in practice that different predictions for the same `example_id` differ by
                # a tiny bit, hence the use of a tolerance to validate the `assert`

                difference = abs(data_per_id[example_id]["predictions"][idx_already_present] - prediction)
                logger.warning(
                    f"prediction already present: {data_per_id[example_id]['predictions'][idx_already_present]} | new"
                    f" prediction: {prediction} | difference: {difference}"
                )
                assert data_per_id[example_id]["tested_labels"][idx_already_present] == tested_label
                assert data_per_id[example_id]["true_label_l"] == true_label_l
            else:
                data_per_id[example_id]["predictions"].append(prediction)
                data_per_id[example_id]["true_label_l"] = true_label_l
                data_per_id[example_id]["tested_labels"].append(tested_label)
        # assert list(range(len(data_per_id))) == sorted(data_per_id.keys())

        results = {}

        references = []
        top1_predictions = []
        for example_id in data_per_id.keys():
            idx = np.argmax(data_per_id[example_id]["predictions"])
            references.append(data_per_id[example_id]["true_label_l"])
            top1_predictions.append(data_per_id[example_id]["tested_labels"][idx])

        # VQA Accuracy
        if ClassifVQAMetrics.VQA_ACCURACY in self.metrics:
            vqa_accuracy_scores = []
            for prediction, answers_ in zip(top1_predictions, references):
                answers_ = [vqa_normalize_text(answer_) for answer_ in answers_]
                gt_acc = []
                for idx_ref in range(len(answers_)):
                    other_answers_ = [other_answer for idx, other_answer in enumerate(answers_) if idx != idx_ref]
                    matched = [other_answer for other_answer in other_answers_ if other_answer == prediction]
                    acc = min(1, len(matched) / 3)
                    gt_acc.append(acc)
                vqa_accuracy_scores.append(sum(gt_acc) / len(gt_acc))
            results["vqa_accuracy"] = float(sum(vqa_accuracy_scores) / len(vqa_accuracy_scores))

        # Entropy
        if ClassifVQAMetrics.ENTROPY_DISTRIBUTION in self.metrics or ClassifVQAMetrics.ENTROPY_MEAN in self.metrics:
            entropy_scores = []
            for example_id in data_per_id.keys():
                q = softmax(np.array(data_per_id[example_id]["predictions"]))
                # Source https://en.wikipedia.org/wiki/Entropy_(information_theory)
                # Given a discrete random variable X, which takes values in the alphabet M and is distributed according
                # to p : X → [ 0 , 1 ]
                # H(X):=-\sum_{x \in M} p(x) \log p(x)
                entropy = -np.sum(np.log(q) * q)
                entropy_scores.append(entropy)

        if ClassifVQAMetrics.ENTROPY_DISTRIBUTION in self.metrics:
            results["entropy_distribution"] = entropy_scores

        if ClassifVQAMetrics.ENTROPY_MEAN in self.metrics:
            results["entropy_mean"] = float(np.mean(entropy_scores))

        return results
