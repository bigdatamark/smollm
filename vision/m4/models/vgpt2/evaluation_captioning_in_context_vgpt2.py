# [MODULE DOC] In-context (few-shot) image captioning evaluation tasks for VGPT2-style vision-language models.
# Defines Vgpt2ImageCaptioningInContext, which builds few-shot prompts (support shots + query
# image), generates captions with the model and feeds them to an image-captioning metric, plus
# many dataset-specific subclasses (TextCaps, CommonGen, NoCaps, COCO, IIIT5K and their "-Sample"
# variants) that only set class attributes. Also used as base class by the VLlama3 captioning
# tasks.
import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from accelerate.utils import extract_model_from_parallel
from datasets import Dataset
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers import AutoTokenizer

from m4.evaluation.config import ShotSelectionMode
from m4.evaluation.custom_metrics.unfolded_image_captioning_metrics import ImageCaptioningMetrics
from m4.evaluation.tasks import BaseTaskImageCaptioning, Predictor
from m4.evaluation.utils import EvaluationVersion
from m4.training.packing import get_splitted_images_and_corresponding_text
from m4.training.utils import (
    FAKE_TOKEN_AROUND_IMAGE_V1,
    FAKE_TOKEN_AROUND_IMAGE_V2,
    IMAGE_TOKEN,
    build_image_transform,
)


class Vgpt2ImageCaptioningInContext(BaseTaskImageCaptioning):
    # [CLASS DOC] Base task for in-context captioning. Class attributes: model_class name,
    # Predictor.in_contexter, target_keys passed to the metric (reference_captions, example_ids),
    # stop_words used to cut generations, tokenizer_max_seq_len (1024), prompt_templates_dict (id
    # -> {prefix, example}; example uses {caption}, {bos_token}, {eos_token} and an <image>
    # placeholder), prompt_templates_dict_instruct (chat-style templates) and
    # bool_instruct_templates selecting which dict is used. Subclasses must set dataset_name,
    # metric_name, metric_kwargs, split names, image_column_name, reference_captions_column_name
    # (optional context_column_name).
    model_class: str = "VGPT2LMHeadModel"
    predictor_class: Predictor = Predictor.in_contexter
    target_keys: List[str] = ["reference_captions", "example_ids"]
    stop_words = ["Caption", "Description", "User", "Image", "task", "<end_of_utterance>", "<row_", "apiro", "\u2500lrow_", "row_1"]
    tokenizer_max_seq_len = 1024
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>Output: {caption}\n",
        },
        1: {
            "prefix": "{bos_token}",
            "example": "<image>Output: {caption}\n",
        },
        2: {
            "prefix": (
                "{bos_token}This is a conversation between a human, User and an intelligent visual AI, Bot. The"
                " user sends images, and Bot describes the images sent by the user.\n"
            ),
            "example": "User:<image>\nBot: {caption}\n",
        },
        3: {
            "prefix": (
                "{bos_token}This is a conversation between a human, User, and an intelligent visual AI, Bot. The"
                " user sends images, and Bot describes them. The bot"
                " should reply as accurately as possible.\n"
            ),
            "example": "User:<image>\nBot: {caption}\n",
        },
        4: {
            "prefix": "{bos_token}",
            "example": "Image to describe:<image>Description: {caption}\n",
        },
        5: {
            "prefix": None,
            "example": "{bos_token}<image>{caption}{eos_token}",
        },
        6: {
            "prefix": "{bos_token}Instruction: provide a short caption of the input image.\n",
            "example": "Image:<image>Image description: {caption}\n",
        },
        7: {
            "prefix": "{bos_token}",
            "example": "Image:<image>Caption: {caption}\n",
        },
        8: {
            "prefix": "{bos_token}Instruction: caption the image in details.\n",
            "example": "Image to caption:<image>Image caption: {caption}\n",
        },
    }
    prompt_templates_dict_instruct = {
        7: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>Describe the image briefly.<end_of_utterance>\nAssistant: {caption}<end_of_utterance>\n"
            ),
        },
    }
    bool_instruct_templates = False

    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] Instantiate the task: parse required evaluation kwargs and build tokenizer and image
        # transform. Required kwargs: tokenizer_name (HF hub id/path), evaluation_version
        # (EvaluationVersion v1/v2), vision_encoder_max_image_size (int), vision_encoder_type,
        # image_seq_len (int, image tokens per image), scale_up_images (bool),
        # image_size_after_scaling (int); optional tokenizer_use_fast (default False). Side
        # effects: downloads/loads an AutoTokenizer (truncation_side="left", auth token from
        # env HF_TOKEN or True), sets padding_side="left"; builds an eval image transform via
        # build_image_transform; stores image_token (IMAGE_TOKEN) and token_around_image
        # (FAKE_TOKEN_AROUND_IMAGE_V1 for v1, V2 for v2). Raises ValueError for an unknown
        # evaluation_version; KeyError if a required kwarg is missing.
        super().__init__(**kwargs)
        self.tokenizer_name = kwargs.pop("tokenizer_name")
        evaluation_version = kwargs.pop("evaluation_version")
        tokenizer_use_fast = kwargs.pop("tokenizer_use_fast", False)
        self.vision_encoder_max_image_size = kwargs.pop("vision_encoder_max_image_size")
        vision_encoder_type = kwargs.pop("vision_encoder_type")
        self.image_seq_len = kwargs.pop("image_seq_len")
        self.image_transform = build_image_transform(
            max_image_size=self.vision_encoder_max_image_size,
            image_size=None,
            eval=True,
            vision_encoder_type=vision_encoder_type,
        )
        self.scale_up_images = kwargs.pop("scale_up_images")
        self.image_size_after_scaling = kwargs.pop("image_size_after_scaling")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            truncation_side="left",
            use_fast=tokenizer_use_fast,
            token=os.getenv("HF_TOKEN", True),
        )
        self.tokenizer.padding_side = "left"
        self.image_token = IMAGE_TOKEN
        if evaluation_version == EvaluationVersion.v1:
            self.token_around_image = FAKE_TOKEN_AROUND_IMAGE_V1
        elif evaluation_version == EvaluationVersion.v2:
            self.token_around_image = FAKE_TOKEN_AROUND_IMAGE_V2
        else:
            raise ValueError(f"Invalid evaluation version: {evaluation_version}")

    def simpler_get_splitted_images_and_corresponding_text(self, image):
        # [FUNCTION DOC] HELPER: Split one PIL image into sub-images and get the matching prompt text.
        # Delegates to get_splitted_images_and_corresponding_text with the instance's
        # vision_encoder_max_image_size, max_image_size=image_size_after_scaling,
        # image_seq_len, no pre-split scale-up, and scale_up_factor=100 when scale_up_images
        # (forces resizing up to the max size) else 1. Returns (list of sub-images, text string
        # containing the image/fake tokens to put in place of <image>).
        splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
            image=image,
            vision_encoder_max_image_size=self.vision_encoder_max_image_size,
            max_image_size=self.image_size_after_scaling,
            pre_split_scale_up_max=None,
            pre_split_scale_up_frequency=None,
            image_seq_len=self.image_seq_len,
            # Any value sufficiently high such that the image will always be resized to max_image_size
            scale_up_factor=100 if self.scale_up_images else 1,
        )
        return splitted_images_array, text_splitted_images

    def get_info_from_dataset(self, dataset):
        # [FUNCTION DOC] No-op hook (captioning needs no dataset-level info such as class names).
        pass

    def get_data_collator(self, **kwargs):
        # [FUNCTION DOC] Return a DataLoader collate function bound to the given kwargs (support dataset,
        # num_shots, shot_selection_mode, prompt_template_id, ...) that are forwarded to
        # prepare_dataset.
        def data_collator(batch):
            # [FUNCTION DOC] HELPER: Convert a list of example dicts into a dict of column lists (keys from
            # the first example) and return prepare_dataset(columns, **kwargs) for the batch.
            exs = {key: [ex[key] for ex in batch] for key in batch[0].keys()}
            batch = self.prepare_dataset(exs, **kwargs)
            return batch

        return data_collator

    def prepare_dataset(self, exs: Dict, **kwargs) -> Dict:
        # [FUNCTION DOC] Build a model-ready batch from a dict of columns `exs` (must contain "id", the image
        # column, reference captions, optional context column, and "vision_encoder_embeddings"
        # for similarity selection). kwargs: support_dataset (datasets.Dataset),
        # support_dataset_vision_encoder_embeddings (array, optional), num_shots,
        # shot_selection_mode (ShotSelectionMode), prompt_template_id. Shot choice per query:
        # random sample when mode is random or num_shots==0; the first num_shots support rows
        # for first_without_image; otherwise the most similar support images by embedding.
        # Text: prefix + concatenated shot prompts (random reference caption, support image,
        # eos token) + query prompt (empty caption, no eos); with instruct templates the
        # trailing "<end_of_utterance>\n" of the query is cut; stripped and tokenized (left
        # truncation/padding to tokenizer_max_seq_len, no added special tokens). Images: each
        # shot and query image is split and transformed; per example all images are zero-padded
        # to a common H/W into (n_images, 3, H, W) with a boolean pixel mask (no shot images
        # for first_without_image). Returns dict with example_ids, reference_captions (strings
        # wrapped into one-element lists), input_ids, attention_mask (lists of 1-D tensors),
        # pixel_values, pixel_attention_masks. Side effects: uses the global `random` module
        # state.
        """
        Prepare batch of examples.
        """
        support_dataset: Dataset = kwargs["support_dataset"]
        support_dataset_vision_encoder_embeddings: Optional[np.ndarray] = kwargs.get(
            "support_dataset_vision_encoder_embeddings", None
        )
        num_shots: int = kwargs["num_shots"]
        shot_selection_mode: ShotSelectionMode = kwargs["shot_selection_mode"]
        prompt_template_id: int = kwargs["prompt_template_id"]

        nb_exs = len(exs["id"])

        def retrieve_idx_closest_examples(ref_embedding, embeddings_to_compare, num_examples):
            # [FUNCTION DOC] HELPER: Similarity-based shot retrieval. Computes dot products between
            # ref_embedding (1-D array) and each row of embeddings_to_compare (2-D array),
            # selects the num_examples largest with an argpartition (linear time), then orders
            # those by increasing similarity (most similar last, i.e. closest to the query in
            # the prompt). Returns a list of int row indices into the support dataset.
            "Returns the indices of the `num_examples` closest embeddings in ascending order"
            sim = np.dot(embeddings_to_compare, ref_embedding)
            # We can achieve linear complexity because we don't need to sort all the numbers,
            # but only find the `num_examples` largest ones
            idx_closest_ex = np.argpartition(sim, -num_examples)[-num_examples:]
            idx_closest_ex = idx_closest_ex[np.argsort(sim[idx_closest_ex])].tolist()
            return idx_closest_ex

        if (shot_selection_mode == ShotSelectionMode.random) or (num_shots == 0):
            idx_shots = [random.sample(range(len(support_dataset)), num_shots) for _ in range(nb_exs)]
        elif shot_selection_mode == ShotSelectionMode.first_without_image:
            idx_shots = [list(range(num_shots)) for _ in range(nb_exs)]
        else:
            idx_shots = [
                retrieve_idx_closest_examples(ref_embedding, support_dataset_vision_encoder_embeddings, num_shots)
                for ref_embedding in exs["vision_encoder_embeddings"]
            ]

        # Prepare text shots
        # These are the priming text shots - size: batch_size
        texts_shots = [
            "".join(
                [
                    self._create_example_prompt(
                        prompt_template_id=prompt_template_id,
                        caption=random.choice(support_dataset[idx_shot][self.reference_captions_column_name]),
                        image=support_dataset[idx_shot][self.image_column_name],
                        context=(
                            support_dataset[idx_shot][self.context_column_name] if self.context_column_name else None
                        ),
                        without_image=shot_selection_mode == ShotSelectionMode.first_without_image,
                        eos_token=self.tokenizer.eos_token,
                    )
                    for idx_shot in idx_shots_ex
                ]
            )
            for idx_shots_ex in idx_shots
        ]

        # These are the tested example - size: batch_size
        tested_exs = [
            self._create_example_prompt(
                prompt_template_id=prompt_template_id,
                image=exs[self.image_column_name][idx],
                context=exs[self.context_column_name][idx] if self.context_column_name else None,
                eos_token="",
            )
            for idx in range(nb_exs)
        ]
        if self.bool_instruct_templates:
            tested_exs = [ex[: -len("<end_of_utterance>\n")].strip() for ex in tested_exs]

        # These are the concatenation of the priming text shots and tested example - size: batch_siz
        tot_texts = [
            self._create_prefix_prompt(prompt_template_id=prompt_template_id) + text_shot + tested_ex
            for text_shot, tested_ex in zip(texts_shots, tested_exs)
        ]

        tot_texts = [text.strip() for text in tot_texts]
        # Tokenize and masks
        tokens = self.tokenizer(
            tot_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer_max_seq_len,
            padding=True,
            add_special_tokens=False,
        )
        input_ids = [tokens.input_ids[idx] for idx in range(len(tot_texts))]
        attention_mask = [tokens.attention_mask[idx] for idx in range(len(tot_texts))]

        # Prepare image shots
        # These are the priming image shots - size: batch_size
        if shot_selection_mode == ShotSelectionMode.first_without_image:
            pixel_values_shots = [[] for _ in range(nb_exs)]
        else:
            pixel_values_shots = [
                [
                    self.image_transform(sub_image)
                    for idx_shot in idx_shots_ex
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(
                        image=support_dataset[idx_shot][self.image_column_name],
                    )[0]
                ]
                for idx_shots_ex in idx_shots
            ]

        # These are the tested images - size: batch_size
        tested_pixel_values = [
            [
                self.image_transform(sub_image)
                for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=image)[0]
            ]
            for image in exs[self.image_column_name]
        ]

        # These are the concatenation of the priming image shots and tested images - size: batch_size
        pixel_values = []
        pixel_attention_masks = []
        for pv_shots, pv in zip(pixel_values_shots, tested_pixel_values):
            num_images = len(pv_shots) + len(pv)
            max_height = max([im.size(1) for im in pv_shots] + [im.size(1) for im in pv])
            max_width = max([im.size(2) for im in pv_shots] + [im.size(2) for im in pv])
            padded_image_tensor = torch.zeros(num_images, 3, max_height, max_width)
            padded_pixel_attention_masks = torch.zeros(num_images, max_height, max_width, dtype=torch.bool)
            for idx, im in enumerate(pv_shots + pv):
                im_height, im_width = im.size(1), im.size(2)
                padded_image_tensor[idx, :, :im_height, :im_width] = im
                padded_pixel_attention_masks[idx, :im_height, :im_width] = True
            pixel_values.append(padded_image_tensor)
            pixel_attention_masks.append(padded_pixel_attention_masks)

        example_ids: List[int] = exs["id"]
        reference_captions = exs[self.reference_captions_column_name]
        if isinstance(reference_captions[0], str):
            reference_captions = [[ref_cap] for ref_cap in reference_captions]
        return {
            "example_ids": example_ids,
            "reference_captions": reference_captions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_masks": pixel_attention_masks,
        }

    def _create_example_prompt(self, prompt_template_id, image, eos_token, caption="", context=None, without_image=False):
        # [FUNCTION DOC] HELPER: Format one example prompt from the selected template's "example" pattern with
        # bos_token, the given eos_token, caption (default "") and context, then replace the
        # single <image> placeholder with the split-image text for `image`. Returns the prompt
        # string.
        # KNOWN ISSUE: `without_image` is accepted but ignored, so first_without_image mode still inserts
        # image tokens for shots while prepare_dataset supplies no shot pixel values (token/image count
        # mismatch).
        if self.bool_instruct_templates:
            prompt_templates_dict = self.prompt_templates_dict_instruct
        else:
            prompt_templates_dict = self.prompt_templates_dict
        prompt_template = prompt_templates_dict[prompt_template_id]["example"]
        prompt_kwargs = {}
        prompt = prompt_template.format(
            bos_token=self.tokenizer.bos_token,
            eos_token=eos_token,
            # For the `eos_token`, the case is different than `bos_token`: when we include bos/eos in the shots,
            # both of them are always here (thus the usage of tokenizer.bos_token), but for the qeury example,
            # we add a `bos_token`, but not an `eos_token` to let the model continue
            context=context,
            caption=caption,
            **prompt_kwargs,
        )
        prompt = prompt.replace("<image>", "<IMAGE>")
        _, text_splitted_images = self.simpler_get_splitted_images_and_corresponding_text(image=image)
        prompt = prompt.replace("<IMAGE>", text_splitted_images, 1)
        return prompt

    def _create_prefix_prompt(self, prompt_template_id):
        # [FUNCTION DOC] HELPER: Build the prompt prefix for prompt_template_id from
        # prompt_templates_dict_instruct when bool_instruct_templates else
        # prompt_templates_dict. Returns "" when the template's prefix is None, otherwise the
        # prefix formatted with the tokenizer's bos_token/eos_token. Raises KeyError for an
        # unknown template id.
        if self.bool_instruct_templates:
            prompt_templates_dict = self.prompt_templates_dict_instruct
        else:
            prompt_templates_dict = self.prompt_templates_dict
        prompt_template = prompt_templates_dict[prompt_template_id]["prefix"]
        if prompt_template is None:
            return ""
        else:
            prompt = prompt_template.format(
                bos_token=self.tokenizer.bos_token,
                eos_token=self.tokenizer.eos_token,
            )
            return prompt

    def generate_tokens(self, **kwargs):
        # Flamingo: Beam search with a beam size of 3
        # [FUNCTION DOC] Generate continuations for a prepared batch. kwargs: model, input_ids /
        # attention_mask (lists of 1-D tensors, stacked), pixel_values / pixel_attention_masks
        # (per-example tensors of shape (n_images, 3, H, W) and (n_images, H, W)), num_beams,
        # no_repeat_ngram_size, max_new_tokens. Images are zero-padded into a (batch,
        # max_images, 3, max_H, max_W) tensor plus a boolean mask, moved to the model device.
        # Bad words (newline, double newline, image token, token-around-image) are banned. For
        # DeepSpeed ZeRO-3 models it flags every submodule's _parameters._in_forward=True to
        # allow external-parameter discovery, and uses synced_gpus. Runs model.generate (on the
        # unwrapped model, no grad, use_cache, early_stopping). Returns only the newly
        # generated token ids (prompt length stripped).
        # KNOWN ISSUE: passes pixel_attention_mask, which VGPT2LMHeadModel.forward (the declared model_class)
        # does not accept; only works with models such as VLlama3 that take that argument.
        model = kwargs["model"]
        input_ids = torch.stack(kwargs["input_ids"]).to(model.device)
        attention_mask = torch.stack(kwargs["attention_mask"]).to(model.device)

        total_batch_size = len(kwargs["pixel_values"])
        max_num_images = max([i.size(0) for i in kwargs["pixel_values"]])
        max_height = max([i.size(2) for i in kwargs["pixel_values"]])
        max_width = max([i.size(3) for i in kwargs["pixel_values"]])
        pixel_values = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
        pixel_attention_mask = torch.zeros(total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool)
        for idx, (sample_images, sample_pixel_attention_mask) in enumerate(
            zip(kwargs["pixel_values"], kwargs["pixel_attention_masks"])
        ):
            im_batch_height, im_batch_width = sample_images.size()[2:]
            pixel_values[idx, : sample_images.shape[0], :, :im_batch_height, :im_batch_width] = sample_images
            pixel_attention_mask[idx, : sample_pixel_attention_mask.shape[0], :im_batch_height, :im_batch_width] = (
                sample_pixel_attention_mask
            )
        pixel_values = pixel_values.to(model.device)
        pixel_attention_mask = pixel_attention_mask.to(model.device)

        num_beams = kwargs["num_beams"]
        no_repeat_ngram_size = kwargs["no_repeat_ngram_size"]
        max_new_tokens = kwargs["max_new_tokens"]

        bad_words = ["\n", "\n\n", self.image_token, self.token_around_image]
        bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False)["input_ids"]

        unwrapped_model = extract_model_from_parallel(model)
        is_deepspeed_model = isinstance(model, DeepSpeedEngine)
        if is_deepspeed_model:
            if model.zero_optimization_partition_weights():
                # Enable automated discovery of external parameters by indicating that
                # we are in a forward pass.
                for module in model.module.modules():
                    module._parameters._in_forward = True
                    pass

        with torch.no_grad():
            generated_tokens = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                bad_words_ids=bad_words_ids,
                use_cache=True,
                early_stopping=True,
                synced_gpus=is_deepspeed_model,
            )

        generated_tokens = generated_tokens[:, input_ids.shape[1] :]
        return generated_tokens

    def format_tokens_to_texts(self, tokens) -> List[str]:
        # [FUNCTION DOC] Decode generated token ids with skip_special_tokens and truncate each text at the
        # first occurrence of any stop word (stop words joined as an unescaped regex
        # alternation). Returns a list of strings.
        texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        stop_words_pattern = r"|".join(self.stop_words)
        texts = [re.split(stop_words_pattern, text)[0] for text in texts]
        return texts

    def add_batch_metric(self, metric, **kwargs):
        # [FUNCTION DOC] Generate captions for the batch (generate_tokens + format_tokens_to_texts) and add
        # them to `metric` via metric.add_batch(generated_captions=..., plus the target_keys
        # values from kwargs). Returns metric.
        generated_tokens = self.generate_tokens(**kwargs)
        generated_captions = self.format_tokens_to_texts(generated_tokens)
        metric.add_batch(
            generated_captions=generated_captions,
            **{key: kwargs[key] for key in self.target_keys},
        )
        return metric


class TextCapsVgpt2ImageCaptioningInContextTextGenMetrics(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2ImageCaptioningInContext; declares only class attributes
    # (no new behavior). Dataset: 'HuggingFaceM4/TextCaps'. Metric
    # 'UnfoldedImageCaptioningMetrics' computing: BLEU_4, CIDER, METEOR, ROUGE_L, SPICE. Splits:
    # default_query_split_name='validation', default_support_split_name='train'. Columns:
    # image_column_name='image', reference_captions_column_name='reference_strs'.
    dataset_name: str = "HuggingFaceM4/TextCaps"
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
    reference_captions_column_name: str = "reference_strs"


class TextCapsVgpt2ImageCaptioningInContextBleuCiderMeteorRouge(TextCapsVgpt2ImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] Task configuration subclass of TextCapsVgpt2ImageCaptioningInContextTextGenMetrics; declares
    # only class attributes (no new behavior). Metric computing: BLEU_4, CIDER, METEOR, ROUGE_L.
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }


class TextCapsSampleVgpt2ImageCaptioningInContextTextGenMetrics(TextCapsVgpt2ImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] Same task configuration as TextCapsVgpt2ImageCaptioningInContextTextGenMetrics, but loads
    # dataset 'HuggingFaceM4/TextCaps-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class TextCapsSampleVgpt2ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVgpt2ImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] Same task configuration as TextCapsVgpt2ImageCaptioningInContextBleuCiderMeteorRouge, but
    # loads dataset 'HuggingFaceM4/TextCaps-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVgpt2ImageCaptioningInContextTextGenMetrics(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2ImageCaptioningInContext; declares only class attributes
    # (no new behavior). Dataset: 'HuggingFaceM4/common_gen'. Metric
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


class CommonGenVgpt2ImageCaptioningInContextBleuCiderMeteorRouge(CommonGenVgpt2ImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] Task configuration subclass of CommonGenVgpt2ImageCaptioningInContextTextGenMetrics;
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


class NoCapsVgpt2ImageCaptioningInContextTextGenMetrics(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2ImageCaptioningInContext; declares only class attributes
    # (no new behavior). Dataset: 'HuggingFaceM4/NoCaps'. Metric 'UnfoldedImageCaptioningMetrics'
    # computing: BLEU_4, CIDER, METEOR, ROUGE_L. Splits: default_query_split_name='validation',
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
            # ImageCaptioningMetrics.SPICE,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    # This does not exist yet... it would require adding a training split to the dataset (see `create_sample_evaluation_datasets_simplified.py`)
    image_column_name: str = "image"
    reference_captions_column_name: str = "annotations_captions"


class NoCapsSampleVgpt2ImageCaptioningInContextTextGenMetrics(NoCapsVgpt2ImageCaptioningInContextTextGenMetrics):
    # [CLASS DOC] Same task configuration as NoCapsVgpt2ImageCaptioningInContextTextGenMetrics, but loads
    # dataset 'HuggingFaceM4/NoCaps-Sample' - a small sample used for quick evaluation
    # runs/debugging.
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVgpt2ImageCaptioningInContextBleuCiderMeteorRouge(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2ImageCaptioningInContext; declares only class attributes
    # (no new behavior). Dataset: 'HuggingFaceM4/COCO' (config '2014_captions'). Metric
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


class CocoSampleVgpt2ImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVgpt2ImageCaptioningInContextBleuCiderMeteorRouge
):
    # [CLASS DOC] Same task configuration as CocoVgpt2ImageCaptioningInContextBleuCiderMeteorRouge, but loads
    # dataset 'HuggingFaceM4/COCO-2014_captions-Sample' (dataset_config=None) - a small sample
    # used for quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class IIIT5KVgpt2ImageCaptioningInContextExactMatch(Vgpt2ImageCaptioningInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2ImageCaptioningInContext; declares only class attributes
    # (no new behavior). Dataset: 'HuggingFaceM4/IIIT-5K'. Metric 'UnfoldedImageCaptioningMetrics'
    # computing: EXACT_MATCH. Splits: default_query_split_name='test',
    # default_support_split_name='train'. Columns: image_column_name='image',
    # reference_captions_column_name='label'. stop_words=['A photo']. prompt_templates_dict
    # overrides templates with ids [0] (prefix + per-example pattern).
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


class IIIT5KSampleVgpt2ImageCaptioningInContextExactMatch(IIIT5KVgpt2ImageCaptioningInContextExactMatch):
    # [CLASS DOC] Same task configuration as IIIT5KVgpt2ImageCaptioningInContextExactMatch, but loads dataset
    # 'HuggingFaceM4/IIIT-5K-Sample' - a small sample used for quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"
