# [MODULE DOC] In-context (zero/few-shot) evaluator: loads a task's support (shots) and query datasets, optionally embeds
# images with a CLIP encoder for RICES shot retrieval, builds a distributed DataLoader with the task's collator
# (which assembles prompts with shots), runs the task's add_batch_metric over mini-batches and computes the
# task metric across processes. Sets PIL ImageFile.LOAD_TRUNCATED_IMAGES = True at import.
import logging
import os
import uuid
from datetime import timedelta
from typing import Dict

import GPUtil
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list
from datasets import Value, load_dataset, load_from_disk
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from m4.evaluation import custom_metrics
from m4.evaluation.config import ShotSelectionMode
from m4.evaluation.utils import get_prompt_template_id, split_batch
from m4.training.utils import _convert_to_rgb


ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger(__name__)


def show_gpu_mem_util(args):
    # [FUNCTION DOC] HELPER: print GPU utilisation with GPUtil when args.hparams.show_gpu_mem_util is set.
    if args.hparams.show_gpu_mem_util:
        GPUtil.showUtilization(all=True)


def load_vision_encoder(args):
    # [FUNCTION DOC] HELPER: load the CLIPModel (in the requested precision, eval mode, moved to args.hparams.device) and its
    # CLIPProcessor named by args.tasks.in_context_params.vision_encoder_name. Returns (model, processor).
    vision_encoder = CLIPModel.from_pretrained(
        args.tasks.in_context_params.vision_encoder_name, torch_dtype=args.tasks.model_precision.value
    )
    vision_encoder.eval()
    vision_encoder = vision_encoder.to(args.hparams.device)
    vision_encoder_processor = CLIPProcessor.from_pretrained(args.tasks.in_context_params.vision_encoder_name)
    return vision_encoder, vision_encoder_processor


def load_query_and_support_datasets(task, args):
    # [FUNCTION DOC] Resolve split names from task attributes "<split>_support_split_name" / "<split>_query_split_name" where split
    # is args.tasks.dataset_split.value; load both from args.hparams.dir_path_load_from_disk/<dataset_name>/<split>
    # with load_from_disk if set, else from the Hub (task.dataset_name, config task.dataset_config). Calls
    # task.get_info_from_dataset(query_dataset), optionally truncates both to the first select_n_examples rows, and
    # returns (support_dataset, query_dataset).
    support_split_name = getattr(task, f"{args.tasks.dataset_split.value}_support_split_name")
    query_split_name = getattr(task, f"{args.tasks.dataset_split.value}_query_split_name")

    if args.hparams.dir_path_load_from_disk is not None:
        support_path = args.hparams.dir_path_load_from_disk / task.dataset_name / support_split_name
        query_path = args.hparams.dir_path_load_from_disk / task.dataset_name / query_split_name
        support_dataset = load_from_disk(support_path)
        query_dataset = load_from_disk(query_path)
    else:
        print(f"Loading dataset {task.dataset_name} with name {task.dataset_config} and     split {support_split_name}")
        support_dataset = load_dataset(
            task.dataset_name,
            name=task.dataset_config,
            split=support_split_name,
        )
        query_dataset = load_dataset(
            task.dataset_name,
            name=task.dataset_config,
            split=query_split_name,
        )
    task.get_info_from_dataset(query_dataset)
    if args.hparams.select_n_examples is not None:
        support_dataset = support_dataset.select(range(min(args.hparams.select_n_examples, len(support_dataset))))
        query_dataset = query_dataset.select(range(min(args.hparams.select_n_examples, len(query_dataset))))
    logger.info(f"Info test dataset {query_dataset}")
    return support_dataset, query_dataset


def add_vision_encoder_embeddings_query_and_support_datasets(
    task, args, support_dataset, query_dataset, vision_encoder, vision_encoder_processor
):
    # [FUNCTION DOC] When shots are used (num_shots != 0) with a non-random selection mode, add a "vision_encoder_embeddings"
    # column (L2-normalised CLIP image features, float32) to both datasets via batched map (cached through a
    # fingerprint including the encoder name) and also return the support embeddings as a numpy array.
    # Returns (support_dataset, query_dataset, support_embeddings or None).
    rng = np.random.default_rng(seed=args.hparams.seed)

    def compute_vision_embds(examples: Dict) -> Dict:
        # [FUNCTION DOC] HELPER: batched map function. Converts images to RGB (for multi-image tasks, one image column is picked at
        # random per example with a seeded numpy generator), runs the CLIP processor and get_image_features without
        # gradients, L2-normalises and stores the embeddings in examples["vision_encoder_embeddings"]. Returns the batch.
        with torch.no_grad():
            if hasattr(task, "image_column_names"):
                images = []
                image_col_probas = np.full(
                    (len(task.image_column_names)), 1 / len(task.image_column_names), dtype=float
                )
                for i in range(len(examples[task.image_column_names[0]])):
                    col = rng.choice(np.array(task.image_column_names), p=image_col_probas)
                    img = _convert_to_rgb(examples[col][i])
                    images.append(img)
            else:
                images = [_convert_to_rgb(img) for img in examples[task.image_column_name]]
            pixel_values = vision_encoder_processor(images=images, return_tensors="pt")["pixel_values"].to(
                vision_encoder.device
            )
            pixel_values = pixel_values.to(args.tasks.model_precision.value)
            image_embeddings = vision_encoder.get_image_features(pixel_values=pixel_values)
        image_embeddings = image_embeddings.cpu().to(torch.float32).numpy()
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=1)[:, None]
        examples["vision_encoder_embeddings"] = image_embeddings
        return examples

    support_dataset_vision_encoder_embeddings = None
    if (args.tasks.in_context_params.shot_selection_mode != ShotSelectionMode.random) and (
        args.tasks.in_context_params.num_shots != 0
    ):
        logger.info("Start adding the vision encoder embeddings")
        support_dataset = support_dataset.map(
            compute_vision_embds,
            batched=True,
            batch_size=args.hparams.batch_size_per_gpu,
            new_fingerprint=(
                f"{support_dataset._fingerprint}_{args.tasks.in_context_params.vision_encoder_name}".replace("/", "_")
            ),
        )
        query_dataset = query_dataset.map(
            compute_vision_embds,
            batched=True,
            batch_size=args.hparams.batch_size_per_gpu,
            new_fingerprint=(
                f"{query_dataset._fingerprint}_{args.tasks.in_context_params.vision_encoder_name}".replace("/", "_")
            ),
        )
        support_dataset_vision_encoder_embeddings = np.array(support_dataset["vision_encoder_embeddings"])
        logger.info("Finished adding the vision encoder embeddings")
    return support_dataset, query_dataset, support_dataset_vision_encoder_embeddings


def build_dataloader(
    task, model, args, support_dataset, query_dataset, support_dataset_vision_encoder_embeddings, accelerator
):
    # [FUNCTION DOC] Create the task's data collator (support set, support embeddings, num_shots, shot selection mode, prompt
    # template id, model) and a DataLoader over the query dataset with a non-shuffling DistributedSampler
    # (per-process shard; padded with repeats so all ranks get equal counts) and batch size batch_size_per_gpu_dl.
    prompt_template_id = get_prompt_template_id(args, task)
    data_collator = task.get_data_collator(
        support_dataset=support_dataset,
        support_dataset_vision_encoder_embeddings=support_dataset_vision_encoder_embeddings,
        num_shots=args.tasks.in_context_params.num_shots,
        shot_selection_mode=args.tasks.in_context_params.shot_selection_mode,
        prompt_template_id=prompt_template_id,
        model=model,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=query_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )
    data_loader = DataLoader(
        query_dataset,
        shuffle=False,
        batch_size=args.hparams.batch_size_per_gpu_dl,
        collate_fn=data_collator,
        sampler=sampler,
    )

    return data_loader


def _get_datasets(task, args, vision_encoder, vision_encoder_processor):
    # [FUNCTION DOC] HELPER: load datasets, add vision-encoder embeddings, then make sure the query dataset has a string "id"
    # column (rename task.id_column_name to "id", or add a 0..N-1 range column when no id column is declared).
    # Returns (support_dataset, query_dataset, support_embeddings).
    support_dataset, query_dataset = load_query_and_support_datasets(task, args)
    (
        support_dataset,
        query_dataset,
        support_dataset_vision_encoder_embeddings,
    ) = add_vision_encoder_embeddings_query_and_support_datasets(
        task, args, support_dataset, query_dataset, vision_encoder, vision_encoder_processor
    )
    if task.id_column_name is not None and task.id_column_name != "id":
        query_dataset = query_dataset.rename_column(task.id_column_name, "id")
    elif (task.id_column_name is None) and (task.id_column_name != "id"):
        query_dataset = query_dataset.add_column(name="id", column=range(len(query_dataset)))
    query_dataset = query_dataset.cast_column("id", Value("string"))
    return support_dataset, query_dataset, support_dataset_vision_encoder_embeddings


def in_contexter(task, accelerator, model, args):
    # [FUNCTION DOC] Evaluate one task with the in-context protocol. Inputs: task instance, accelerator, model (prepared) and the
    # Parameters args. Loads the CLIP encoder (through a throw-away Accelerator) only when RICES-like shots are
    # needed; the main process prepares datasets first (filling the datasets cache), then the other ranks.
    # Returns None early if args.hparams.only_load_datasets. Otherwise instantiates the metric class
    # custom_metrics.<task.metric_name> with a shared random experiment_id, process info, save_generations and
    # task.metric_kwargs; for each DataLoader batch, split into mini-batches of batch_size_per_gpu and call
    # task.add_batch_metric with the model (plus generation params for open-ended VQA/captioning tasks). Raises
    # ValueError for unsupported task class names. Non-main ranks compute first, then the main rank; returns the
    # main process's score dict (None on other ranks).
    vision_encoder, vision_encoder_processor, dummy_accelerator = None, None, None
    if (args.tasks.in_context_params.shot_selection_mode != ShotSelectionMode.random) and (
        args.tasks.in_context_params.num_shots != 0
    ):
        vision_encoder, vision_encoder_processor = load_vision_encoder(args)

        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=args.hparams.timeout))]
        rngs_types = ["torch", "cuda", "generator"] if torch.cuda.is_available() else ["torch", "generator"]
        dummy_accelerator = Accelerator(rng_types=rngs_types, kwargs_handlers=kwargs_handlers)
        dummy_dataloader = torch.utils.data.DataLoader(
            [0 for _ in range(10)], batch_size=args.hparams.batch_size_per_gpu
        )
        vision_encoder, dummy_dataloader = dummy_accelerator.prepare(vision_encoder, dummy_dataloader)
        vision_encoder = dummy_accelerator.unwrap_model(vision_encoder)

    # If, in few-shot, with deepspeed, with several processes, the compute of the embeddings are hanging forever,
    # either remove `if accelerator.is_main_process`, or compute first the embeddings with a setting that works
    # (with 1 process or with pure accelerate for example)
    if accelerator.is_main_process:
        support_dataset, query_dataset, support_dataset_vision_encoder_embeddings = _get_datasets(
            task, args, vision_encoder, vision_encoder_processor
        )
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        support_dataset, query_dataset, support_dataset_vision_encoder_embeddings = _get_datasets(
            task, args, vision_encoder, vision_encoder_processor
        )
    accelerator.wait_for_everyone()
    logger.warning(f"support_dataset: {support_dataset}")
    logger.warning(f"query_dataset: {query_dataset}")

    del vision_encoder
    del vision_encoder_processor
    del dummy_accelerator

    show_gpu_mem_util(args)

    data_loader = build_dataloader(
        task, model, args, support_dataset, query_dataset, support_dataset_vision_encoder_embeddings, accelerator
    )

    if args.hparams.only_load_datasets:
        return

    metric_class = getattr(custom_metrics, task.metric_name)
    metric_kwargs = task.metric_kwargs if task.metric_kwargs is not None else {}
    save_generations = args.tasks.save_generations
    experiment_id = str(uuid.uuid4())
    experiment_id = broadcast_object_list([experiment_id])[0]
    metric = metric_class(
        experiment_id=experiment_id,
        num_process=accelerator.num_processes,
        process_id=accelerator.process_index,
        save_generations=save_generations,
        **metric_kwargs,
    )
    for batch in tqdm(data_loader, desc="Compute scores:"):
        # Splits batches that get augmented by data_collator. Mostly usefull for classification tasks
        mini_batches = split_batch(batch, chunk_size=args.hparams.batch_size_per_gpu)
        show_gpu_mem_util(args)
        for mini_batch in mini_batches:
            if (
                "ClassificationInContext" in task.__class__.__name__
                or "ClassificationVQAInContext" in task.__class__.__name__
                or "PerplexityInContext" in task.__class__.__name__
                or "ImageCaptionMatching" in task.__class__.__name__
            ):
                kwargs = {"model": model, **mini_batch}
            elif (
                "OpenEndedVQAInContext" in task.__class__.__name__
                or "ImageCaptioningInContext" in task.__class__.__name__
            ):
                kwargs = {
                    "model": model,
                    "num_beams": args.tasks.text_generation_params.num_beams,
                    "no_repeat_ngram_size": args.tasks.text_generation_params.no_repeat_ngram_size,
                    "max_new_tokens": args.tasks.text_generation_params.max_new_tokens,
                    **mini_batch,
                }
            else:
                raise ValueError(
                    f"Task class ({task.__class__.__name__}) is not supported. Expected it to be among"
                    " ['ClassificationInContext', 'OpenEndedVQAInContext', 'ImageCaptioningInContext',"
                    " 'PerplexityInContext', ImageCaptionMatching]."
                )
            accelerator.wait_for_everyone()
            metric = task.add_batch_metric(metric, **kwargs)

    # Trick suggested here: https://huggingface.slack.com/archives/C02UAKD75L7/p1664475037694469?thread_ts=1664461500.952079&cid=C02UAKD75L7
    if not accelerator.is_main_process:
        score = metric.compute()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        score = metric.compute()
    return score
