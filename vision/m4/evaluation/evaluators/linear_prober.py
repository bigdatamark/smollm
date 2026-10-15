# [MODULE DOC] Linear-probe evaluator: projects images to features with a transformers model, fits the task's classifier
# (scikit-learn based, implemented by the task) on the support split and scores predictions on the query split.
import logging
from functools import partial

import evaluate
import numpy as np
import transformers
from datasets import load_dataset

from m4.evaluation import custom_metrics
from m4.evaluation.config import DatasetSplit


logger = logging.getLogger(__name__)


# NOTE: Accelerator is not used because linear probing is using scikit learn's logistic regression
def linear_prober(task, accelerator, args):
    # [FUNCTION DOC] Inputs: task, accelerator (unused), args. Loads transformers.<task.model_class> from args.tasks.model_name onto
    # args.hparams.device; metric = custom_metrics.<metric_name>() if it exists else evaluate.load(metric_name). Only
    # DatasetSplit.default is supported (ValueError otherwise). Loads the support split (optionally a random subset
    # of select_n_examples, unseeded), maps task.project to add "features", and calls task.fit unless
    # only_load_datasets (then returns None after loading the query split). Projects the query split (first
    # select_n_examples), predicts with task.predict and returns metric.compute() on the labels.
    # KNOWN ISSUE: launch.main calls every evaluator as evaluator(task, accelerator, model, args), which does not match
    # this 3-argument signature (TypeError). It also maps with batch_size=hparams.mini_batch_size, a deprecated option
    # that defaults to None (whole dataset as one batch).
    model_name = args.tasks.model_name

    model_class = getattr(transformers, task.model_class)
    model = model_class.from_pretrained(model_name)
    model.eval()
    model = model.to(args.hparams.device)

    metric_class = getattr(custom_metrics, task.metric_name, None)
    if metric_class is not None:
        metric = metric_class()
    else:
        metric = evaluate.load(task.metric_name)

    if args.tasks.dataset_split == DatasetSplit.default:
        support_split_name = task.default_support_split_name
        query_split_name = task.default_query_split_name
    else:
        raise ValueError(f"Dataset split {args.tasks.dataset_split} is not supported for linear prober.")

    assert task.default_support_split_name is not None
    train_dataset = load_dataset(
        task.dataset_name, name=task.dataset_config, split=support_split_name, use_auth_token=True
    )
    if args.hparams.select_n_examples is not None:
        train_dataset = train_dataset.select(
            np.random.choice(
                range(len(train_dataset)), min(args.hparams.select_n_examples, len(train_dataset)), replace=False
            )
        )
    train_dataset = train_dataset.map(
        partial(task.project, model=model), batched=True, batch_size=args.hparams.mini_batch_size
    )

    if not args.hparams.only_load_datasets:
        logger.info(f"Info train dataset {train_dataset}")
        train_dataset = train_dataset.with_format("np")
        task.fit(features=train_dataset["features"], labels=train_dataset[task.label_column_name])

    assert query_split_name is not None
    test_dataset = load_dataset(
        task.dataset_name, name=task.dataset_config, split=query_split_name, use_auth_token=True
    )
    logger.info(f"Info test dataset {test_dataset}")

    if args.hparams.only_load_datasets:
        return

    if args.hparams.select_n_examples is not None:
        test_dataset = test_dataset.select(range(min(args.hparams.select_n_examples, len(test_dataset))))
    test_dataset = test_dataset.map(
        partial(task.project, model=model), batched=True, batch_size=args.hparams.mini_batch_size
    )
    test_dataset = test_dataset.with_format("np")
    predictions = task.predict(features=test_dataset["features"])
    metric.add_batch(
        predictions=predictions,
        references=test_dataset[task.label_column_name],
    )
    score = metric.compute()
    return score
