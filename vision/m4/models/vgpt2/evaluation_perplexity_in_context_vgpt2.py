# [MODULE DOC] Zero-shot perplexity evaluation tasks for VGPT2 models on image-caption pair datasets or web
# documents (interleaved images and text), reporting per-example perplexity via
# MetricsPerplexity. Note: the base constructor always raises NotImplementedError, so none of
# these tasks can currently be built.
import os
from typing import Dict, List

import torch
from accelerate.utils import extract_model_from_parallel
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers import AutoTokenizer

from m4.evaluation.custom_metrics.perplexity_metrics import MetricsPerplexity
from m4.evaluation.tasks import BaseTask, Predictor
from m4.evaluation.utils import EvaluationVersion
from m4.training.types import DatasetTypes
from m4.training.utils import (
    FAKE_TOKEN_AROUND_IMAGE_V1,
    FAKE_TOKEN_AROUND_IMAGE_V2,
    IMAGE_TOKEN,
    build_image_transform,
)


class Vgpt2PerplexityInContext(BaseTask):
    # [CLASS DOC] Base perplexity task. Class attributes: image_column_name, text_column_name,
    # context_column_name, ds_type (DatasetTypes.IMAGE_CAPTION_PAIRS or WEB_DOCUMENTS),
    # add_end_of_doc_token / add_begin_of_doc_token, tokenizer_max_seq_len (1024), max_num_images
    # (70, padding size for web documents).
    model_class: str = "VGPT2LMHeadModel"
    predictor_class: Predictor = Predictor.in_contexter
    target_keys: List[str] = ["example_ids"]
    image_column_name: str = None
    text_column_name: str = None
    context_column_name: str = None

    ds_type: DatasetTypes = DatasetTypes.IMAGE_CAPTION_PAIRS

    add_end_of_doc_token: bool = True
    add_begin_of_doc_token: bool = False

    tokenizer_max_seq_len = 1024
    max_num_images = 70

    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] Parse kwargs (tokenizer_name, evaluation_version, optional tokenizer_use_fast,
        # vision_encoder_type, image_seq_len, image_size), build the image transform and
        # tokenizer (HF_TOKEN env), store image token id, eos token, pad id and
        # token_around_image (ValueError for unknown evaluation version).
        # KNOWN ISSUE: always raises NotImplementedError at the end (padding for variable image sizes not
        # implemented), so the class is unusable.
        super().__init__(**kwargs)
        self.tokenizer_name = kwargs.pop("tokenizer_name")
        evaluation_version = kwargs.pop("evaluation_version")
        tokenizer_use_fast = kwargs.pop("tokenizer_use_fast", False)
        vision_encoder_type = kwargs.pop("vision_encoder_type")
        self.image_seq_len = kwargs.pop("image_seq_len")
        image_size = kwargs.pop("image_size")
        self.image_transform = build_image_transform(
            max_image_size=image_size, image_size=None, eval=True, vision_encoder_type=vision_encoder_type
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, truncation_side="left", use_fast=tokenizer_use_fast, token=os.getenv("HF_TOKEN", True)
        )
        self.image_token = IMAGE_TOKEN
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.eos_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        if evaluation_version == EvaluationVersion.v1:
            self.token_around_image = FAKE_TOKEN_AROUND_IMAGE_V1
        elif evaluation_version == EvaluationVersion.v2:
            self.token_around_image = FAKE_TOKEN_AROUND_IMAGE_V2
        else:
            raise ValueError(f"Invalid evaluation version: {evaluation_version}")

        raise NotImplementedError(
            "Padding for various size images has not been implemented for that class yet. Ask Victor to do it. He's"
            " unsure the last time we used this and as such, won't be spending time on something we might not even"
            " touch in the future."
        )

    def get_info_from_dataset(self, dataset):
        # [FUNCTION DOC] No-op hook.
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

    def prepare_image_caption_pair_ds(self, exs: Dict) -> Dict:
        # [FUNCTION DOC] Build a batch for image-caption pairs: one prompt per example (image tokens + first
        # caption, plus eos/bos per flags), tokenized with padding/truncation; each image
        # transformed to (1, 3, H, W). Returns example_ids, input_ids, attention_mask,
        # pixel_values.
        # KNOWN ISSUE: context is read as exs[idx][column] although exs is a dict of columns (would fail), and
        # caption=text[idx][0] takes the first character when the text column holds plain strings.
        nb_exs = len(exs["id"])
        tot_texts = [
            self._create_image_caption_pair_prompt(
                caption=(exs[self.text_column_name][idx][0]),
                context=exs[idx][self.context_column_name] if self.context_column_name else None,
            )
            for idx in range(nb_exs)
        ]  # These are the tested example - size: batch_size
        tot_texts = [self._add_special_tokens_to_prompt(text) for text in tot_texts]

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

        # These are the tested images - size: batch_size
        pixel_values = [self.image_transform(img).unsqueeze(0) for img in exs[self.image_column_name]]

        example_ids: List[int] = exs["id"]
        return {
            "example_ids": example_ids,
            "input_ids": [tokens.input_ids[idx] for idx in range(len(tot_texts))],
            "attention_mask": [tokens.attention_mask[idx] for idx in range(len(tot_texts))],
            "pixel_values": pixel_values,
        }

    def _create_image_caption_pair_prompt(self, caption="", context=None):
        # [FUNCTION DOC] HELPER: Return token_around_image + image_token * image_seq_len + token_around_image
        # + caption; raises NotImplementedError when a context is given.
        if context is not None:
            raise NotImplementedError("Context not implemented for this task")
        prompt = f"{self.token_around_image}{self.image_token * self.image_seq_len}{self.token_around_image}{caption}"
        return prompt

    def _add_special_tokens_to_prompt(self, prompt):
        # [FUNCTION DOC] HELPER: Append the tokenizer eos token if add_end_of_doc_token and prepend bos if
        # add_begin_of_doc_token. Returns the prompt.
        if self.add_end_of_doc_token:
            prompt = f"{prompt}{self.tokenizer.eos_token}"
        if self.add_begin_of_doc_token:
            prompt = f"{self.tokenizer.bos_token}{prompt}"
        return prompt

    def prepare_webdoc_ds(self, exs: Dict) -> Dict:
        # [FUNCTION DOC] Build a batch from web documents (lists of images and texts with None gaps).
        # Documents are split on the END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED marker into sub-
        # documents (dropping those without images; ValueError if image/text counts diverge).
        # Each sub-document becomes text where each image is FAKE_TOKEN_AROUND_IMAGE_V2 +
        # IMAGE_TOKEN (closing fake token before the next text / at the end), images are
        # transformed and stacked. Token ids get eos/bos per flags, are right-padded with pad
        # id to tokenizer_max_seq_len (truncated) with an attention mask; images are zero-
        # padded to max_num_images. Returns example_ids, input_ids, attention_mask,
        # pixel_values (lists of tensors).
        # KNOWN ISSUE: uses FAKE_TOKEN_AROUND_IMAGE_V2 and one IMAGE_TOKEN per image regardless of
        # evaluation_version / image_seq_len; last_was_image is not reset between documents; example_ids count
        # can differ from the number of produced sub-documents.
        images_batch = exs[self.image_column_name]
        texts_batch = exs[self.text_column_name]

        tokenizer = self.tokenizer

        last_was_image = False
        all_images = []
        all_texts = []
        for raw_images, raw_texts in zip(images_batch, texts_batch):
            inds_of_texts_to_split = [
                i
                for i, text in enumerate(raw_texts)
                if text is not None and isinstance(text, str) and "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED" in text
            ]
            if inds_of_texts_to_split:
                splitted_raw_images, splitted_raw_texts = [], []
                previous_i = 0
                for i in inds_of_texts_to_split:
                    splitting = raw_texts[i].split("END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED")
                    part1, part2 = splitting[0], splitting[-1]

                    sub_doc_images = raw_images[previous_i:i] + [None]
                    sub_doc_texts = raw_texts[previous_i:i] + [part1.strip()]
                    if not any(sub_doc_images):  # This can happen if all images in raw_images[0:i] are all None
                        continue

                    splitted_raw_images.append(sub_doc_images)
                    splitted_raw_texts.append(sub_doc_texts)

                    if part2.strip() == "":
                        previous_i = i + 1
                    else:
                        raw_texts[i] = part2.strip()
                        previous_i = i

                if previous_i < len(raw_images) and any(raw_images[previous_i:]):
                    splitted_raw_images.append(raw_images[previous_i:])
                    splitted_raw_texts.append(raw_texts[previous_i:])

            else:
                splitted_raw_images, splitted_raw_texts = [raw_images], [raw_texts]

            # Sanity check
            if [len(ims) for ims in splitted_raw_images] != [len(txts) for txts in splitted_raw_texts]:
                raise ValueError(
                    "Number of images and texts don't match after splitting on `END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED`."
                    " Something core went wrong during the splitting and needs to be fixed."
                )

            for s_r_ims, s_r_txts in zip(splitted_raw_images, splitted_raw_texts):
                images, web_text = [], ""
                for image, text in zip(s_r_ims, s_r_txts):
                    if text is None and image is None:
                        continue

                    if image is not None:
                        web_text += f"{FAKE_TOKEN_AROUND_IMAGE_V2}{IMAGE_TOKEN}"
                        images.append(self.image_transform(image))
                        last_was_image = True
                    elif text is not None:
                        if last_was_image:
                            web_text += f"{FAKE_TOKEN_AROUND_IMAGE_V2}{text}"
                            last_was_image = False
                        else:
                            web_text += f" {text}" if web_text != "" else text

                if last_was_image:
                    web_text += f"{FAKE_TOKEN_AROUND_IMAGE_V2}"

                web_text = web_text.strip(" ")

                # This is mostly a sanity check. Cases like that should not happen at that point.
                if web_text == "" or len(images) == 0:
                    continue

                images = torch.stack(images)
                all_images.append(images)

                web_text_ids = tokenizer.encode(web_text, add_special_tokens=False)
                if self.add_end_of_doc_token:
                    web_text_ids += [tokenizer.eos_token_id]

                if self.add_begin_of_doc_token:
                    web_text_ids = [tokenizer.bos_token_id] + web_text_ids
                all_texts.append(web_text_ids)

        output_input_ids = []
        output_images = []
        output_attention_masks = []
        for images, text in zip(all_images, all_texts):
            padded_input_ids = [tokenizer.pad_token_id] * self.tokenizer_max_seq_len
            unpadded_seq_len = len(text)
            padded_input_ids[:unpadded_seq_len] = text[: self.tokenizer_max_seq_len]

            attention_mask = torch.zeros((self.tokenizer_max_seq_len,), dtype=torch.long)
            attention_mask[:unpadded_seq_len] = 1

            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, self.max_num_images)

            current_images = images[:local_max_num_images]

            padded_image_tensor = torch.zeros(self.max_num_images, *current_images.size()[1:])
            padded_image_tensor[: current_images.size(0)] = current_images

            output_images.append(padded_image_tensor)
            output_input_ids.append(torch.tensor(padded_input_ids))

            output_attention_masks.append(attention_mask)

        output_input_ids = torch.stack(output_input_ids)
        output_images = torch.stack(output_images)
        output_attention_masks = torch.stack(output_attention_masks)

        example_ids: List[int] = exs["id"]
        return {
            "example_ids": example_ids,
            "input_ids": [input_ids for input_ids in output_input_ids],
            "attention_mask": [attention_masks for attention_masks in output_attention_masks],
            "pixel_values": [pixels for pixels in output_images],
        }

    def prepare_dataset(self, exs: Dict, **kwargs) -> Dict:
        # [FUNCTION DOC] Validate kwargs["num_shots"] == 0 (ValueError otherwise) and dispatch on ds_type to
        # prepare_image_caption_pair_ds or prepare_webdoc_ds (ValueError for other types).
        """
        Prepare batch of examples.
        """
        num_shots: int = kwargs["num_shots"]
        if num_shots != 0:
            raise ValueError(
                f"Invalid num_shots selection: num_shots should equal 0 for perplexity but here num_shots={num_shots}"
            )

        if self.ds_type == DatasetTypes.IMAGE_CAPTION_PAIRS:
            # We have a image-caption pair dataset
            return self.prepare_image_caption_pair_ds(exs)
        elif self.ds_type == DatasetTypes.WEB_DOCUMENTS:
            # We have a webdoc dataset
            return self.prepare_webdoc_ds(exs)
        else:
            raise ValueError(f"Invalid dataset type: {self.ds_type}")

    def get_perplexities(self, **kwargs):
        # [FUNCTION DOC] Compute per-example perplexity: stack inputs on the model device (DeepSpeed ZeRO-3
        # forward flag set as in generation), switch the unwrapped model to eval and run it
        # under no_grad with labels=input_ids. Image tokens are removed from the attention mask
        # (in place) and image/pad tokens are excluded from the cross-entropy; loss is summed
        # per row, divided by the count of remaining tokens and exponentiated. Returns a 1-D
        # tensor.
        # KNOWN ISSUE: after boolean masking the flat loss is reshaped to (batch, -1), which fails or mixes
        # rows when examples have different numbers of kept tokens.
        model = kwargs["model"]
        input_ids = torch.stack(kwargs["input_ids"]).to(model.device)
        attention_mask = torch.stack(kwargs["attention_mask"]).to(model.device)
        pixel_values = torch.stack(kwargs["pixel_values"]).to(model.device)

        unwrapped_model = extract_model_from_parallel(model)
        is_deepspeed_model = isinstance(model, DeepSpeedEngine)
        if is_deepspeed_model:
            if model.zero_optimization_partition_weights():
                # Enable automated discovery of external parameters by indicating that
                # we are in a forward pass.
                for module in model.module.modules():
                    module._parameters._in_forward = True
                    pass
        unwrapped_model.eval()
        with torch.no_grad():
            logits = unwrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids,
            )["logits"]
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:]
        shift_attention_mask[shift_labels == self.image_token_id] = 0
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        non_ignored_token_count = torch.sum(shift_attention_mask == 1, dim=1, keepdim=True).flatten()
        mask = (shift_labels.view(-1) != self.image_token_id) & (shift_labels.view(-1) != self.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[mask], shift_labels.view(-1)[mask])
        loss = loss.reshape(logits.shape[0], -1).sum(dim=1)
        loss = loss / non_ignored_token_count

        perplexities = loss.exp()

        return perplexities

    def add_batch_metric(self, metric, **kwargs):
        # [FUNCTION DOC] Compute perplexities and call metric.add_batch(perplexities=..., example_ids).
        # Returns metric.
        perplexities = self.get_perplexities(**kwargs)
        metric.add_batch(
            perplexities=perplexities,
            **{key: kwargs[key] for key in self.target_keys},
        )
        return metric


class TextCapsVgpt2PerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2PerplexityInContext; declares only class attributes (no
    # new behavior). Dataset: 'HuggingFaceM4/TextCaps'. Metric 'PerplexityMetrics' computing:
    # PERPLEXITY. Splits: default_query_split_name='validation'. Columns:
    # image_column_name='image', text_column_name='reference_strs'.
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"


class TextCapsSampleVgpt2PerplexityInContext(TextCapsVgpt2PerplexityInContext):
    # [CLASS DOC] Same task configuration as TextCapsVgpt2PerplexityInContext, but loads dataset
    # 'HuggingFaceM4/TextCaps-Sample' - a small sample used for quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVgpt2PerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2PerplexityInContext; declares only class attributes (no
    # new behavior). Dataset: 'HuggingFaceM4/common_gen'. Metric 'PerplexityMetrics' computing:
    # PERPLEXITY. Splits: default_query_split_name='validation'. Columns:
    # image_column_name='image', context_column_name='concepts', text_column_name='target'.
    dataset_name: str = "HuggingFaceM4/common_gen"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    context_column_name: str = "concepts"
    text_column_name: str = "target"

    def _create_image_caption_pair_prompt(self, caption="", context=""):
        # [FUNCTION DOC] HELPER: Prompt "<fake><image><fake>Input: {context}. Output: {caption}" with a single
        # image token.
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}Input: {context}. Output: {caption}"
        )


class NoCapsVgpt2PerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2PerplexityInContext; declares only class attributes (no
    # new behavior). Dataset: 'HuggingFaceM4/NoCaps'. Metric 'PerplexityMetrics' computing:
    # PERPLEXITY. Splits: default_query_split_name='validation'. Columns:
    # image_column_name='image', text_column_name='annotations_captions'.
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    # This does not exist yet... it would require adding a training split to the dataset (see `create_sample_evaluation_datasets_simplified.py`)
    image_column_name: str = "image"
    text_column_name: str = "annotations_captions"


class NoCapsSampleVgpt2PerplexityInContext(NoCapsVgpt2PerplexityInContext):
    # [CLASS DOC] Same task configuration as NoCapsVgpt2PerplexityInContext, but loads dataset
    # 'HuggingFaceM4/NoCaps-Sample' - a small sample used for quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVgpt2PerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2PerplexityInContext; declares only class attributes (no
    # new behavior). Dataset: 'HuggingFaceM4/COCO' (config '2014_captions'). Metric
    # 'PerplexityMetrics' computing: PERPLEXITY. Splits: default_query_split_name='validation'.
    # Columns: image_column_name='image', text_column_name='sentences_raw'.
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "sentences_raw"


class CocoSampleVgpt2PerplexityInContext(CocoVgpt2PerplexityInContext):
    # [CLASS DOC] Same task configuration as CocoVgpt2PerplexityInContext, but loads dataset
    # 'HuggingFaceM4/COCO-2014_captions-Sample' (dataset_config=None) - a small sample used for
    # quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class IIIT5KVgpt2PerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2PerplexityInContext; declares only class attributes (no
    # new behavior). Dataset: 'HuggingFaceM4/IIIT-5K'. Metric 'PerplexityMetrics' computing:
    # PERPLEXITY. Splits: default_query_split_name='test'. Columns: image_column_name='image',
    # text_column_name='label'.
    dataset_name: str = "HuggingFaceM4/IIIT-5K"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "test"
    image_column_name: str = "image"
    text_column_name: str = "label"

    def _create_image_caption_pair_prompt(self, caption="", context=None):
        # [FUNCTION DOC] HELPER: Prompt "<fake><image><fake>A photo where it is written {caption}" with a
        # single image token; NotImplementedError if a context is given.
        if context is not None:
            raise NotImplementedError("Context not implemented for this task")
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}A photo where"
            f" it is written {caption}"
        )


class IIIT5KSampleVgpt2PerplexityInContext(IIIT5KVgpt2PerplexityInContext):
    # [CLASS DOC] Same task configuration as IIIT5KVgpt2PerplexityInContext, but loads dataset
    # 'HuggingFaceM4/IIIT-5K-Sample' - a small sample used for quick evaluation runs/debugging.
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"


class MiniGPTCaptionsVgpt2PerplexityInContext(Vgpt2PerplexityInContext):
    # [CLASS DOC] Task configuration subclass of Vgpt2PerplexityInContext; declares only class attributes (no
    # new behavior). Dataset: 'HuggingFaceM4/mini-GPT-captions'. Metric 'PerplexityMetrics'
    # computing: PERPLEXITY. Splits: default_query_split_name='train'. Columns:
    # image_column_name='image', text_column_name='reference_strs'.
    dataset_name: str = "HuggingFaceM4/mini-GPT-captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "train"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"
