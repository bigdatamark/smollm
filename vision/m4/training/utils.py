# [MODULE DOC] Grab-bag of m4 training utilities: special-token constants and OpenCLIP normalization stats,
# model
# freezing, Nougat-style document image augmentations (albumentations), image transforms and image splitting
# for vision encoders, tokenizer construction, NVML energy readings, analytic TFLOPs estimators, memory
# reporting, accelerate/DeepSpeed (ZeRO) helpers, state-dict loading, tensor statistics for activation
# tracking, a SIGTERM listener and LoRA unloading. Module-level `nougat_transform` is a composed random
# augmentation pipeline (bitmap, erosion/dilation, affine, shift-scale-rotate, grid/elastic distortion,
# brightness/contrast, JPEG compression, noise, blur) applied to PIL images, returning a numpy array.
import dataclasses
import gc
import json
import logging
import math
import signal
from contextlib import contextmanager
from enum import Enum
from functools import partial

import accelerate
import albumentations as alb
import cv2
import numpy as np
import psutil
import pynvml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from accelerate.state import AcceleratorState
from peft.utils import _get_submodules
from PIL import Image
from transformers import (  # AddedToken is needed for the eval of the tokenizer params # noqa: F401
    AddedToken,
    AutoTokenizer,
)

from m4.training.types import DatasetNames
from m4.utils.check_valid_tokenizer import check_valid_tokenizer


IMAGE_TOKEN = "<image>"
FAKE_TOKEN_AROUND_IMAGE_V2 = "<fake_token_around_image>"
FAKE_TOKEN_AROUND_IMAGE_V1 = "\n\n"
END_OF_UTTERANCE_TOKEN = "<end_of_utterance>"
# Originally taken from the values used in OpenCLIP
IMAGE_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
logger = logging.getLogger(__name__)


class LoggingTypes(Enum):
    # [CLASS DOC] Enum of where gradient/parameter statistics are logged: "jsonl", "wandb" or "print".
    """Types of logging to use for the gradient and parameter statistics"""

    JSONL = "jsonl"
    WANDB = "wandb"
    PRINT = "print"


class VisionEncoderTypes(Enum):
    # [CLASS DOC] Enum of supported vision encoder families: siglip, clip, vit.
    """Types of vision encoders"""

    siglip = "siglip"
    clip = "clip"
    vit = "vit"


class JSONEncoderForDataclasses(json.JSONEncoder):
    # [CLASS DOC] json.JSONEncoder subclass that serializes dataclass instances as dicts (via
    # dataclasses.asdict).
    """
    Use to serialize dataclass object, like so:
    json.dump(data, fp, indent=2, cls=JSONEncoderForDataclasses)
    """

    def default(self, obj):
        # [FUNCTION DOC] Returns asdict(obj) for dataclass instances, otherwise defers to the base encoder
        # (raises TypeError).
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def freeze_model(model, module_exceptions=[]):
    # [FUNCTION DOC] Sets requires_grad False on every module of `model`, except modules whose type is one of
    # the names in
    # module_exceptions ("LayerNorm", "Linear", "Embedding" mapped to nn classes), which get requires_grad
    # True.
    mapping = {
        "LayerNorm": nn.LayerNorm,
        "Linear": nn.Linear,
        "Embedding": nn.Embedding,
    }
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    for module in model.modules():
        if module_exceptions and any([isinstance(module, t) for t in module_exceptions_mapped]):
            module.requires_grad_(True)  # Explicitly setting it to true to avoid any mistakes
        else:
            module.requires_grad_(False)
    return model


# Code copied from Nougat repo: https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py
# Implements image augmentation


def alb_wrapper(transform):
    # [FUNCTION DOC] HELPER: wraps an albumentations transform into a callable taking a PIL image/array and
    # returning the
    # transformed numpy array.
    def f(im):
        # [FUNCTION DOC] Converts im to a numpy array, applies the wrapped transform and returns
        # result["image"].
        return transform(image=np.asarray(im))["image"]

    return f


class Erosion(alb.ImageOnlyTransform):
    # [CLASS DOC] Albumentations image-only transform performing morphological erosion with a random
    # elliptical kernel.
    """
    Apply erosion operation to an image.

    Erosion is a morphological operation that shrinks the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the erosion kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the erosion kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        # [FUNCTION DOC] scale: int (fixed size) or 2-element tuple/list (min, max) for kernel size;
        # always_apply and p are passed
        # to the base transform. Asserts a tuple/list has length 2.
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        # [FUNCTION DOC] Erodes img with an elliptical kernel whose width/height are drawn with numpy randint
        # in [scale0, scale1)
        # and returns the result.
        # KNOWN ISSUE: with an int scale (low == high) numpy randint raises ValueError; the upper bound is
        # exclusive
        # so the max kernel size is never used, contrary to the docstring.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(alb.ImageOnlyTransform):
    # [CLASS DOC] Albumentations image-only transform performing morphological dilation with a random
    # elliptical kernel.
    """
    Apply dilation operation to an image.

    Dilation is a morphological operation that expands the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the dilation kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the dilation kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        # [FUNCTION DOC] Same argument handling as Erosion.__init__ (int or 2-element tuple/list scale).
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        # [FUNCTION DOC] Dilates img with a random elliptical kernel sized in [scale0, scale1) and returns it.
        # KNOWN ISSUE: same randint problem as Erosion.apply (int scale raises ValueError; max is exclusive).
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(alb.ImageOnlyTransform):
    # [CLASS DOC] Albumentations image-only transform setting every pixel value below `lower` to `value`.
    """
    Apply a bitmap-style transformation to an image.

    This transformation replaces all pixel values below a certain threshold with a specified value.

    Args:
        value (int, optional): The value to replace pixels below the threshold with. Default is 0.
        lower (int, optional): The threshold value below which pixels will be replaced. Default is 200.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        # [FUNCTION DOC] Stores value (default 0) and lower threshold (default 200); always_apply/p go to the
        # base transform.
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        # [FUNCTION DOC] Returns a copy of img where elements < lower are replaced by value.
        img = img.copy()
        img[img < self.lower] = self.value
        return img


nougat_transform = alb_wrapper(
    alb.Compose(
        [
            Bitmap(p=0.05),
            alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.02),
            alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.03),
            alb.ShiftScaleRotate(
                shift_limit_x=(0, 0.04),
                shift_limit_y=(0, 0.03),
                scale_limit=(-0.15, 0.03),
                rotate_limit=2,
                border_mode=0,
                interpolation=2,
                value=(255, 255, 255),
                p=0.03,
            ),
            alb.GridDistortion(
                distort_limit=0.05,
                border_mode=0,
                interpolation=2,
                value=(255, 255, 255),
                p=0.04,
            ),
            alb.Compose(
                [
                    alb.Affine(translate_px=(0, 5), always_apply=True, cval=(255, 255, 255)),
                    alb.ElasticTransform(
                        p=1,
                        alpha=50,
                        sigma=120 * 0.1,
                        alpha_affine=120 * 0.01,
                        border_mode=0,
                        value=(255, 255, 255),
                    ),
                ],
                p=0.04,
            ),
            alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.03),
            alb.ImageCompression(95, p=0.07),
            alb.GaussNoise(20, p=0.08),
            alb.GaussianBlur((3, 3), p=0.03),
        ]
    )
)


def _convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates
    # a wrong background for transparent images. The call to `alpha_composite`
    # handles this case
    # [FUNCTION DOC] HELPER: converts a PIL image to RGB. RGB images are returned unchanged; others are
    # converted to RGBA and
    # alpha-composited over an opaque white background so transparent areas become white, then converted to
    # RGB.
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class ConditionalResize(object):
    # [CLASS DOC] Callable resize transform for PIL images used by the siglip pipeline, with optional random
    # scale-up during
    # training to oversample large images.
    def __init__(
        self,
        max_image_size,
        min_image_size=378,
        scale_up_frequency=0.7,
        scale_up_max=3.0,
        interpolation=transforms.InterpolationMode.BILINEAR,
        eval=False,
    ):
        # [FUNCTION DOC] Stores max_image_size, min_image_size (default 378, unused), scale_up_frequency
        # (default 0.7),
        # scale_up_max (default 3.0), interpolation (bilinear) and eval flag. Raises ValueError when training
        # (not eval) with a scale_up_frequency but scale_up_max None.
        self.scale_up_frequency = scale_up_frequency
        self.scale_up_max = scale_up_max
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.interpolation = interpolation
        self.eval = eval

        if not eval and scale_up_frequency is not None and scale_up_max is None:
            raise ValueError(
                f"`scale_up_max` ({scale_up_max}) cannot be None if images are going to be resizes on a"
                f" `scale_up_frequency` ({scale_up_frequency}) frequency basis"
            )

    def __call__(self, image, scale_up_factor=None):
        # [FUNCTION DOC] Computes a target size: applies scale_up_factor if given, else (training only) with
        # probability
        # scale_up_frequency multiplies by a triangular random factor in [1.5, scale_up_max]; then clamps the
        # longest side to max_image_size keeping aspect ratio and raises ValueError if still too large.
        # KNOWN ISSUE: the computed size is then discarded and the image is always resized to a square
        # max_image_size x max_image_size, so the aspect-ratio and scale-up logic has no effect
        # (min_image_size unused).
        new_width, new_height = image.size
        aspect_ratio = new_width / new_height
        if scale_up_factor is not None:
            new_width, new_height = int(scale_up_factor * new_width), int(scale_up_factor * new_height)
        elif not self.eval and self.scale_up_frequency is not None and np.random.random() < self.scale_up_frequency:
            # During training, we artificially scale up the images X% of the time to have more representation of large images
            scale_up_factor = np.random.triangular(left=1.5, mode=self.scale_up_max, right=self.scale_up_max)
            new_width, new_height = int(scale_up_factor * new_width), int(scale_up_factor * new_height)

        # Calculate the new size with respect to the aspect ratio
        if new_width >= new_height and new_width > self.max_image_size:
            new_width = self.max_image_size
            new_height = int(new_width / aspect_ratio)
        elif new_height > new_width and new_height > self.max_image_size:
            new_height = self.max_image_size
            new_width = int(new_height * aspect_ratio)

        if new_height > self.max_image_size or new_width > self.max_image_size:
            raise ValueError(
                f"Image size exceeds the maximum size of {self.max_image_size} with the new size of"
                f" ({new_width}, {new_height})"
            )

        # new_width = max(new_width, self.min_image_size)
        # new_height = max(new_height, self.min_image_size)
        new_width = self.max_image_size
        new_height = self.max_image_size

        resized_image = transforms.Resize(
            size=(new_height, new_width),
            interpolation=self.interpolation,
        )(image)
        return resized_image


# TODO(aps): Take parameters from config
def build_image_transform(
    image_size=384,
    min_image_size=378,
    max_image_size=None,
    eval=False,
    vision_encoder_type=VisionEncoderTypes.siglip,
    dataset_name=None,
    scale_up_frequency=None,
    scale_up_max=None,
):
    # [FUNCTION DOC] Builds the per-image transform for a vision encoder. Raises ValueError if both image_size
    # and
    # max_image_size are given, if scale_up_* is given with a fixed image_size, or for unknown encoder types.
    # CLIP/ViT: Compose of RGB conversion; eval -> bicubic resize to image_size square; train -> for
    # OCR/DocVQA
    # datasets bicubic resize + nougat_transform, otherwise RandomResizedCrop (scale 0.9-1.0); then ToTensor
    # and
    # OpenCLIP mean/std normalization. SigLIP: returns the nested `transform` function. Returns a callable.
    # Note: nougat_transform returns a numpy array which ToTensor accepts (HWC uint8).
    if image_size is not None and max_image_size is not None:
        raise ValueError(
            f"`image_size` ({image_size}) and `max_image_size` ({max_image_size}) can't be specified at the same."
        )
    if image_size is not None and (scale_up_frequency is not None or scale_up_max is not None):
        raise ValueError(
            "Both `scale_up_frequency` and `scale_up_max` should only be specified when images are systematically"
            " resized to squares."
        )

    if vision_encoder_type == VisionEncoderTypes.clip or vision_encoder_type == VisionEncoderTypes.vit:
        ops_to_compose = [_convert_to_rgb]
        if eval:
            ops_to_compose.append(
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
            )
        else:
            if dataset_name in [DatasetNames.OCR, DatasetNames.DOCVQA]:
                ops_to_compose.extend(
                    [
                        transforms.Resize(
                            (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        nougat_transform,
                    ]
                )
            else:
                ops_to_compose.append(
                    transforms.RandomResizedCrop(
                        (image_size, image_size),
                        scale=(0.9, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                )
        ops_to_compose.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_DATASET_MEAN, std=IMAGE_DATASET_STD),
            ]
        )
        transform = transforms.Compose(ops_to_compose)

    elif vision_encoder_type == VisionEncoderTypes.siglip:

        def transform(img, scale_up_factor=None):
            # [FUNCTION DOC] SigLIP transform for one image: builds a ConditionalResize (when max_image_size
            # is set, passing
            # scale_up_factor) or a bilinear square resize to image_size, composes RGB conversion, resize,
            # ToTensor and
            # normalization with mean/std 0.5, and returns the resulting tensor. A new ConditionalResize is
            # built per call.
            conditional_resize = ConditionalResize(
                max_image_size=max_image_size,
                min_image_size=min_image_size,
                scale_up_frequency=scale_up_frequency,
                scale_up_max=scale_up_max,
                interpolation=transforms.InterpolationMode.BILINEAR,
                eval=eval,
            )
            ops_to_compose = [
                _convert_to_rgb,
                (
                    partial(conditional_resize, scale_up_factor=scale_up_factor)
                    if max_image_size is not None
                    else transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR
                    )
                ),
            ]
            # if dataset_name in [DatasetNames.OCR, DatasetNames.DOCVQA] and not eval:
            #     ops_to_compose.append(nougat_transform)
            ops_to_compose.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            transformed_img = transforms.Compose(ops_to_compose)(img=img)
            return transformed_img

    else:
        raise ValueError("Wrong `vision_encoder_type`.")

    return transform


def image_splitting(
    image,
    vision_encoder_max_image_size,
    max_image_size,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    scale_up_factor=None,
):
    # [FUNCTION DOC] Splits a PIL image into crops for a vision encoder of side E =
    # vision_encoder_max_image_size.
    # Size selection: if scale_up_factor (not 1) is given, scales width/height by it; else with probability
    # pre_split_scale_up_frequency picks a random number of sub-images N along the longest side (1/2/3 with
    # probabilities .5/.3/.2 when max_image_size <= 1092, else 2/3/4/5 with .4/.3/.2/.1) and sets the longest
    # side to N*E with the other side rounded up to a multiple of E. Then the longest side is clamped to
    # max_image_size (or rounded up to a multiple of E), the other side derived from the aspect ratio and
    # rounded
    # up to a multiple of E; zero sizes fall back to E x E. Image is resized (Lanczos).
    # If either side exceeds E, the image is cut row-major into ceil(h/E) x ceil(w/E) near-equal crops and the
    # global image is resized to E x E; otherwise split counts are 0. Returns (list of crops followed by the
    # global image, num_splits_h, num_splits_w).
    # Note: when clamped to max_image_size that side is not rounded to a multiple of E, so crops may be
    # uneven.
    """
    Image splitting strategy.
    1) If one side of the original image is larger than `max_image_size`, resize it to `max_image_size` while preserving the aspect ratio.
    2) Divide the resulting image into `ceil(height / vision_encoder_max_image_size)` x `ceil(width / vision_encoder_max_image_size)`
    sub-images of approximately the same size each (up to the fact that `vision_encoder_max_image_size` does not divide `height` or
    `width`).
    3) [Optional] For all the crops and the original image, resize the largest side to `vision_encoder_max_image_size`, while preserving the aspect ratio.
    4) Returns the list of the crops and the original image, in addition to the number of splits for the height and the width.
    """

    width, height = image.size
    aspect_ratio = width / height

    if (scale_up_factor is not None) and (scale_up_factor != 1):
        width, height = int(scale_up_factor * width), int(scale_up_factor * height)

    elif (pre_split_scale_up_frequency is not None) and (np.random.random() < pre_split_scale_up_frequency):
        random_number = np.random.random()
        if max_image_size <= 1092:
            if random_number < 0.5:
                num_sub_images_longest_side = 1
            elif random_number < 0.8:
                num_sub_images_longest_side = 2
            else:
                num_sub_images_longest_side = 3
        else:
            if random_number < 0.4:
                num_sub_images_longest_side = 2
            elif random_number < 0.7:
                num_sub_images_longest_side = 3
            elif random_number < 0.9:
                num_sub_images_longest_side = 4
            else:
                num_sub_images_longest_side = 5
        if width >= height:
            width = num_sub_images_longest_side * vision_encoder_max_image_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / vision_encoder_max_image_size) * vision_encoder_max_image_size
        else:
            height = num_sub_images_longest_side * vision_encoder_max_image_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / vision_encoder_max_image_size) * vision_encoder_max_image_size


    if width >= height:
        if width > max_image_size:
            width = max_image_size
        else:
            width = math.ceil(width / vision_encoder_max_image_size) * vision_encoder_max_image_size
        height = int(width / aspect_ratio)
        height = math.ceil(height / vision_encoder_max_image_size) * vision_encoder_max_image_size
    elif height > width:
        if height > max_image_size:
            height = max_image_size
        else:
            height = math.ceil(height / vision_encoder_max_image_size) * vision_encoder_max_image_size
        width = int(height * aspect_ratio)
        width = math.ceil(width / vision_encoder_max_image_size) * vision_encoder_max_image_size

    if (width == 0) or (height == 0):
        # For some reasons it can happen (rarely) during a training. Don't know the cause.
        width, height = vision_encoder_max_image_size, vision_encoder_max_image_size

    image = image.resize((width, height), Image.LANCZOS)

    frames = []
    if height > vision_encoder_max_image_size or width > vision_encoder_max_image_size:
        # Calculate the number of splits
        num_splits_w = math.ceil(width / vision_encoder_max_image_size)
        num_splits_h = math.ceil(height / vision_encoder_max_image_size)
        # Calculate the optimal width and height for the sub-images
        optimal_width = math.ceil(width / num_splits_w)
        optimal_height = math.ceil(height / num_splits_h)

        # Iterate through each row and column
        for r in range(num_splits_h):
            for c in range(num_splits_w):
                # Calculate the starting point of the crop
                start_x = c * optimal_width
                start_y = r * optimal_height

                # Calculate the ending point of the crop
                end_x = min(start_x + optimal_width, width)
                end_y = min(start_y + optimal_height, height)

                # Crop the image
                crop = image.crop((start_x, start_y, end_x, end_y))
                frames.append(crop)

        # For the global image at the end, we resize it to match the vision_encoder_max_image_size, for cpu memory efficiency
        image = image.resize((vision_encoder_max_image_size, vision_encoder_max_image_size), Image.LANCZOS)

    else:
        num_splits_h, num_splits_w = 0, 0

    frames.append(image)

    return frames, num_splits_h, num_splits_w


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_add_tokens,
    tokenizer_add_special_tokens,
    tokenizer_params,
    additional_vocab_size,
    model_vocab_size=None,
    is_fine_tuning=False,
):
    # [FUNCTION DOC] Loads a fast AutoTokenizer (legacy=False) and adds tokens. tokenizer_params,
    # tokenizer_add_tokens and
    # tokenizer_add_special_tokens are Python-literal strings evaluated with eval (a dict, a list of
    # str/AddedToken, and an optional dict). If model_vocab_size exceeds the tokenizer length, pads the vocab
    # with "<fake_token_i>" tokens (raising ValueError if tokenizer_params contains
    # additional_special_tokens).
    # Adds existing additional special tokens plus the new tokens as additional_special_tokens and merges the
    # special-tokens dict. Then asserts layout: fine-tuning requires the last added token to be
    # "<end_of_utterance>" and present in the last additional_vocab_size ids; pretraining requires the last
    # two
    # to be "<fake_token_around_image>", "<image>" in that range. Finally runs check_valid_tokenizer.
    # Returns the tokenizer.
    # KNOWN ISSUE: eval of config strings executes arbitrary code; only safe with trusted configs.
    """
    We artificially separate `tokenizer_add_tokens` and `tokenizer_add_special_tokens` is a dictionary whose keys only takes into account special tokens (eos, pad, cls, etc.).
    On the contrary, `tokenizer_add_tokens` is a list of string of `AddedToken`.
    In practise, we use `tokenizer.add_special_tokens` to add all of these new special tokens or update the existing ones.

    NB: we constraint to tokenizer to be a fast tokenizer because with the slow tokenizer, we can't set the arguments of the added tokens (cf `.add_tokens`) and by default, the separators are stripped.
    """
    tokenizer_params = eval(tokenizer_params)
    assert isinstance(tokenizer_params, dict)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False, **tokenizer_params)

    if model_vocab_size is not None:
        if model_vocab_size > len(tokenizer):
            logger.warning(
                f"The model vocabulary size ({model_vocab_size}) is larger than the tokenizer vocabulary size "
                f"({len(tokenizer)}). Updating the tokenizer to match."
            )
            if "additional_special_tokens" in tokenizer_params:
                raise ValueError(
                    "You can't use `additional_special_tokens` in `tokenizer_params` with a model vocab "
                    "size > tokenizer vocab size. We need to adjust tokenizer before adding special "
                    "tokens. Please use `tokenizer_add_tokens` instead."
                )
            # We need to pad the tokenizer vocab with fake tokens
            tokenizer.add_tokens(["<fake_token_{}>".format(i) for i in range(model_vocab_size - len(tokenizer))])

    # This check ensures that the image token and the fake token around it will be in the `DecoupledEmbedding.additional_weight`.
    existing_special_tokens = (
        [*tokenizer.special_tokens_map_extended["additional_special_tokens"]]
        if "additional_special_tokens" in tokenizer.special_tokens_map_extended
        else []
    )
    add_special_tokens_dict = {"additional_special_tokens": existing_special_tokens + eval(tokenizer_add_tokens)}
    if tokenizer_add_special_tokens is not None:
        add_special_tokens_dict.update(eval(tokenizer_add_special_tokens))

    tokenizer.add_special_tokens(add_special_tokens_dict)
    if is_fine_tuning and len(eval(tokenizer_add_tokens)) > 0:
        assert str(eval(tokenizer_add_tokens)[-1]) == END_OF_UTTERANCE_TOKEN
        assert END_OF_UTTERANCE_TOKEN in tokenizer.convert_ids_to_tokens(
            [idx for idx in range(len(tokenizer) - additional_vocab_size, len(tokenizer))]
        )
    elif not is_fine_tuning and len(eval(tokenizer_add_tokens)) > 0:
        assert str(eval(tokenizer_add_tokens)[-1]) == IMAGE_TOKEN
        assert str(eval(tokenizer_add_tokens)[-2]) == FAKE_TOKEN_AROUND_IMAGE_V2
        assert IMAGE_TOKEN in tokenizer.convert_ids_to_tokens(
            [idx for idx in range(len(tokenizer) - additional_vocab_size, len(tokenizer))]
        )
        assert FAKE_TOKEN_AROUND_IMAGE_V2 in tokenizer.convert_ids_to_tokens(
            [idx for idx in range(len(tokenizer) - additional_vocab_size, len(tokenizer))]
        )
    # This verifies that `<image>` was correctly added to the tokenizer vocabulary
    # XXX: opt-1.3b fails here
    # assert tokenizer.is_fast == tokenizer_params.get("use_fast", True)

    check_valid_tokenizer(tokenizer)

    return tokenizer


def pynmvl_handle(accelerator):
    # [FUNCTION DOC] Returns an NVML device handle for the accelerator's local process index (initializing
    # NVML), or None
    # without CUDA.
    if not torch.cuda.is_available():
        return None

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(accelerator.local_process_index)


def pynvml_get_total_energy_in_joules(handle):
    # [FUNCTION DOC] Returns total GPU energy consumption in joules (NVML millijoules / 1000) for the handle,
    # or 0 without CUDA.
    if not torch.cuda.is_available():
        return 0
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000


def compute_tflops_per_batch_per_gpu(
    num_layers,
    batch_size,
    q_seq_len,
    k_seq_len,
    hidden_size,
    kv_in_dim,
    ff_exp_factor=None,
    grad_acc_size=1,
    swiglu=False,
    vocab_size=None,
    count_backward=False,
    use_grad_checkpointing=False,
):
    # [FUNCTION DOC] Analytic estimate of TFLOPs for a transformer stack on one GPU per batch. With
    # multiply-add factor 2, sums
    # per-layer: Q projection (b*q*h^2), K/V projections (b*k*2*h*kv_in_dim), QK^T and attention-times-V
    # (b*q*k*h each), softmax (q*k, no batch factor), output projection (b*q*h^2) and FFN (2 or, with swiglu,
    # 3
    # matrices of h x h*ff_exp_factor; 0 if no ff_exp_factor). Adds the LM head (b*q*h*vocab) if vocab_size.
    # Multiplies by (1 forward + 2 if count_backward + 1 if use_grad_checkpointing) and divides by 1e12.
    # Returns a 0-d torch tensor. grad_acc_size is accepted but unused.
    multiply_add_factor = torch.tensor(2)
    query_transformation = multiply_add_factor * batch_size * q_seq_len * hidden_size**2
    # k_seq_len == v_seq_len
    key_value_transformation = multiply_add_factor * batch_size * k_seq_len * (2 * hidden_size * kv_in_dim)
    attention_matrix_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * hidden_size
    attention_softmax = multiply_add_factor * q_seq_len * k_seq_len
    att_over_values_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * hidden_size
    post_attention_linear_proj = multiply_add_factor * batch_size * q_seq_len * hidden_size**2

    # There are usually 2 expansion_linear_layers because first one expands, and second one retracts back to hidden_size
    # When using a classic decoder, some blocks don't have those feed-forward layers
    # Swiglu duplicates the first linear layer, so we have to account for 3 of them instead of 2
    if ff_exp_factor and swiglu:
        expansion_linear_layers = 3 * (
            multiply_add_factor * batch_size * q_seq_len * (hidden_size * ff_exp_factor) * hidden_size
        )
    elif ff_exp_factor:
        expansion_linear_layers = 2 * (
            multiply_add_factor * batch_size * q_seq_len * (hidden_size * ff_exp_factor) * hidden_size
        )
    else:
        expansion_linear_layers = torch.tensor(0)

    transformer_block_flops = (
        query_transformation
        + key_value_transformation
        + attention_matrix_computation
        + attention_softmax
        + att_over_values_computation
        + post_attention_linear_proj
        + expansion_linear_layers
    )

    # This computation should only be added if the model has a language head
    if vocab_size:
        language_head_computation = multiply_add_factor * batch_size * q_seq_len * hidden_size * vocab_size
    else:
        language_head_computation = torch.tensor(0)

    forward_fact = 1
    backward_factor = 2 if count_backward else 0
    grad_checkpointing_factor = 1 if use_grad_checkpointing else 0
    model_flops = (forward_fact + backward_factor + grad_checkpointing_factor) * (
        num_layers * transformer_block_flops + language_head_computation
    )
    model_tflops = model_flops / (10**12)

    return model_tflops


def compute_linear_tflops_per_batch_per_gpu(
    batch_size,
    seq_len,
    in_features,
    out_features,
    count_backward=False,
    use_grad_checkpointing=False,
):
    # [FUNCTION DOC] TFLOPs of one linear layer: 2*batch*seq*in*out times (1 + 2 if backward + 1 if grad
    # checkpointing)
    # divided by 1e12; returns a 0-d tensor.
    forward_factor = 1
    backward_factor = 2 if count_backward else 0
    grad_checkpointing_factor = 1 if use_grad_checkpointing else 0
    multiply_add_factor = torch.tensor(2)
    linear_forward_flops = multiply_add_factor * batch_size * seq_len * in_features * out_features
    linear_flops = linear_forward_flops * (forward_factor + backward_factor + grad_checkpointing_factor)
    linear_tflops = linear_flops / (10**12)
    return linear_tflops


def compute_perceiver_tflops_per_batch_per_gpu(
    num_layers,
    batch_size,
    q_seq_len,
    vision_embed_seq_len,
    q_k_v_input_dim,
    attention_hidden_size,
    ff_exp_factor=None,
    count_backward=False,
    use_grad_checkpointing=False,
):
    # [FUNCTION DOC] TFLOPs estimate for a perceiver resampler: queries (q_seq_len latents) attend over vision
    # embeddings
    # concatenated with latents (k = vision_embed_seq_len + q_seq_len). Counts Q projection on latents, K/V on
    # vision embeddings only, attention matrix, softmax, attention-times-V, output projection and a 2-matrix
    # FFN
    # of width q_k_v_input_dim*ff_exp_factor; scaled per layer by num_layers and by
    # forward/backward/checkpoint
    # factors as above; divided by 1e12. Returns a 0-d tensor.
    multiply_add_factor = torch.tensor(2)
    query_transformation = multiply_add_factor * batch_size * q_seq_len * q_k_v_input_dim * attention_hidden_size
    # k_seq_len == v_seq_len
    key_value_transformation = (
        multiply_add_factor * batch_size * vision_embed_seq_len * (2 * attention_hidden_size * q_k_v_input_dim)
    )

    k_seq_len = vision_embed_seq_len + q_seq_len
    attention_matrix_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * attention_hidden_size
    attention_softmax = multiply_add_factor * q_seq_len * k_seq_len
    att_over_values_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * attention_hidden_size
    post_attention_linear_proj = multiply_add_factor * batch_size * q_seq_len * attention_hidden_size * q_k_v_input_dim

    # There are usually 2 expansion_linear_layers because first one expands, and second one retracts back to hidden_size
    # When using a classic decoder, some blocks don't have those feed-forward layers
    if ff_exp_factor:
        expansion_linear_layers = 2 * (
            multiply_add_factor * batch_size * q_seq_len * (q_k_v_input_dim * ff_exp_factor) * q_k_v_input_dim
        )
    else:
        expansion_linear_layers = torch.tensor(0)

    transformer_block_flops = (
        query_transformation
        + key_value_transformation
        + attention_matrix_computation
        + attention_softmax
        + att_over_values_computation
        + post_attention_linear_proj
        + expansion_linear_layers
    )

    forward_fact = 1
    backward_factor = 2 if count_backward else 0
    grad_checkpointing_factor = 1 if use_grad_checkpointing else 0
    model_flops = (forward_fact + backward_factor + grad_checkpointing_factor) * (num_layers * transformer_block_flops)
    model_tflops = model_flops / (10**12)

    return model_tflops


def mem_usage_formatted(logging_type=LoggingTypes.PRINT):
    # adapted from deepspeed's see_memory_usage

    # [FUNCTION DOC] Empties the CUDA cache, runs gc.collect, and reports CUDA allocated/max
    # allocated/reserved/max reserved (GB)
    # and used CPU virtual memory (GB and percent). Returns a dict of formatted strings, or with
    # LoggingTypes.PRINT a single "k: v | ..." string. Resets CUDA peak memory stats as a side effect.
    torch.cuda.empty_cache()

    # python doesn't do real-time garbage collection so do it explicitly to get the correct usage reports
    gc.collect()
    vm_stats = psutil.virtual_memory()

    mem = {
        "gpu mem alloc": f"{torch.cuda.memory_allocated()/2**30:0.2f}GB",
        "max alloc": f"{torch.cuda.max_memory_allocated()/2**30:0.2f}GB",
        "reserv": f"{torch.cuda.memory_reserved()/2**30:0.2f}GB",
        "max reserv": f"{torch.cuda.max_memory_reserved()/2**30:0.2f}GB",
        "cpu vm used": f"{(vm_stats.total-vm_stats.available)/2**30:0.2f}GB {vm_stats.percent}%",
    }

    if logging_type == LoggingTypes.PRINT:
        mem = " | ".join([f"{k}: {v}" for k, v in mem.items()]) + " | "

    # get the peak memory to report correct data, so reset the max_memory_allocated counter for the next call
    torch.cuda.reset_peak_memory_stats()

    return mem


def is_deepspeed_used():
    # [FUNCTION DOC] Returns True if accelerate is initialized with a DeepSpeed plugin.
    deepspeed_plugin = get_deepspeed_plugin()
    return deepspeed_plugin is not None


def get_deepspeed_stage():
    # [FUNCTION DOC] Returns the ZeRO stage from the DeepSpeed plugin config ("zero_optimization.stage"), or 0
    # if unused.
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is None:
        return 0
    ds_config = deepspeed_plugin.deepspeed_config
    stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    # from accelerate>=0.17.1 can do instead:
    # stage = deepspeed_plugin.zero_stage
    return stage


def is_deepspeed_zero3_used():
    # [FUNCTION DOC] Returns True if the ZeRO stage is 3.
    return get_deepspeed_stage() == 3


def accelerate_torch_dtype():
    # [FUNCTION DOC] Returns the torch dtype to use for from_pretrained: None if accelerate is not
    # initialized; from the
    # DeepSpeed config (fp16 -> float16, bf16 -> bfloat16, else None) when DeepSpeed is used, otherwise from
    # AcceleratorState.mixed_precision.
    """
    derive and return `torch_dtype` to be used in `from_pretrained` from either Deepspeed config or if
    Deepspeed isn't used than accelerator state
    """
    if not is_accelerate_initialized():
        return None

    accelerator_state = AcceleratorState()

    if is_deepspeed_used():
        deepspeed_plugin = accelerator_state.deepspeed_plugin
        ds_config = deepspeed_plugin.deepspeed_config
        if ds_config.get("fp16", {}).get("enabled", False):
            torch_dtype = torch.float16
        elif ds_config.get("bf16", {}).get("enabled", False):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None
    else:  # no Deepspeed
        if accelerator_state.mixed_precision == "fp16":
            torch_dtype = torch.float16
        elif accelerator_state.mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

    return torch_dtype


def is_accelerate_initialized():
    # [FUNCTION DOC] Returns whether the accelerate global state has been initialized.
    return accelerate.state.is_initialized()


def get_deepspeed_plugin():
    # [FUNCTION DOC] Returns AcceleratorState().deepspeed_plugin when accelerate is initialized, else None.
    if is_accelerate_initialized():
        return AcceleratorState().deepspeed_plugin
    else:
        return None


def get_deepspeed_engine(accelerator):
    # [FUNCTION DOC] Returns the DeepSpeed engine wrapped by an accelerate Accelerator.
    return accelerator.deepspeed_engine_wrapped.engine


def is_deepspeed_zero_init_enabled():
    # [FUNCTION DOC] Returns True if the DeepSpeed plugin has ZeRO-3 zero.Init enabled; False otherwise.
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is not None:
        return deepspeed_plugin.is_zero3_init_enabled()
    else:
        return False


@contextmanager
def hf_trainer_disable_zero3_init_context_manager():
    # monkey patch hack to emulate a context that has zero_init disabled as it's used in
    # modeling_utils.py in transformers for from_config and from_pretrained.
    # [FUNCTION DOC] Context manager that temporarily monkey-patches
    # transformers.modeling_utils.is_deepspeed_zero3_enabled to
    # return False (disabling zero.Init in from_pretrained/from_config), restoring it on exit.
    # KNOWN ISSUE: no try/finally, so an exception inside the block leaves the patch in place.
    import transformers.modeling_utils  # noqa

    orig = transformers.modeling_utils.is_deepspeed_zero3_enabled
    transformers.modeling_utils.is_deepspeed_zero3_enabled = lambda: False
    yield
    transformers.modeling_utils.is_deepspeed_zero3_enabled = orig


def deepspeed_zero_init_disabled_context_manager():
    # [FUNCTION DOC] Returns a list with one context manager disabling ZeRO-3 init: the DeepSpeed plugin's
    # zero3_init_context_manager(enable=False) if a plugin exists, else
    # hf_trainer_disable_zero3_init_context_manager().
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is not None:
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    else:
        return [hf_trainer_disable_zero3_init_context_manager()]


def deepspeed_gathered_parameters_context_manager(params, modify=True):
    # [FUNCTION DOC] Returns [deepspeed.zero.GatheredParameters(params, modifier_rank=0 if modify else None)]
    # when ZeRO-3
    # init is enabled (gathers sharded params; with modify, rank 0's changes are re-scattered on exit), else
    # [].
    # Meant to be used with transformers ContextManagers.
    """
    Under zero.Init returns a context manager that will gather the sharded param, otherwise returns an empty list

    If `modify` is `True`, gather the shards and once the context exits update the shards with the
    modified data - one wants that when modifying the gathered param. If one wants to just gather
    the shards in order to read the param and no modifications are done to it, use `modify=False` as
    it's more efficient.

    `params` - can be a single parameter, a list, or a tuple of parameters to collect.

    Example:

    from transformers.utils import ContextManagers
    from m4.training.utils import deepspeed_gathered_parameters_context_manager
    with ContextManagers(deepspeed_gathered_parameters_context_manager(module.weight, modify=True)):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


    """
    if is_deepspeed_zero_init_enabled():
        import deepspeed

        # 0 is for updating `params` shards after modifying it, `None` is for read-only (only gather)
        modifier_rank = 0 if modify else None
        return [deepspeed.zero.GatheredParameters(params, modifier_rank=modifier_rank)]
    else:
        return []


# adapted from https://github.com/huggingface/transformers/blob/a081f292ca8479eaf66d7396186021268f128829/src/transformers/modeling_utils.py#L438-L496
# as it appears to be a private function
def load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # Convert old format to new format if needed from a PyTorch state_dict
    # [FUNCTION DOC] Loads state_dict into model_to_load (adapted from transformers). Renames keys containing
    # "gamma"/"beta"
    # to "weight"/"bias" (mutating the caller's dict), copies it preserving _metadata, then recursively calls
    # each module's _load_from_state_dict with prefix start_prefix. Under ZeRO-3 init, gathers the module's
    # direct parameters present in the state dict and loads on rank 0 only. Returns the list of error
    # messages.
    # Note: the gamma/beta substring rename also affects unrelated keys containing those substrings.
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, state_dict, prefix=""):
        # [FUNCTION DOC] HELPER (recursive closure): loads entries with this module's prefix into `module` if
        # any key starts with
        # the prefix (using DeepSpeed gathering under ZeRO-3), then recurses into children with
        # "prefix<name>.".
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero_init_enabled():
                import deepspeed

                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def get_stats(var, ctx):
    # [FUNCTION DOC] Returns a dict of statistics for tensor var, keys prefixed with ctx: min/max/mean/std of
    # values and of
    # absolute values, L2 and L1 norms divided by numel, and nonzero count (all Python numbers). {} for None.
    if var is None:
        return {}
    var = var.float()
    abs_var = var.abs()
    return {
        f"{ctx}_var_min": var.min().item(),
        f"{ctx}_var_max": var.max().item(),
        f"{ctx}_var_mean": var.mean().item(),
        f"{ctx}_var_std": var.std().item(),
        f"{ctx}_abs_var_min": abs_var.min().item(),
        f"{ctx}_abs_var_max": abs_var.max().item(),
        f"{ctx}_abs_var_mean": abs_var.mean().item(),
        f"{ctx}_abs_var_std": abs_var.std().item(),
        f"{ctx}_var_norm_2": (var.norm(p=2) / var.numel()).item(),
        f"{ctx}_var_norm_1": (var.norm(p=1) / var.numel()).item(),
        f"{ctx}_nonzero": (var != 0).sum().item(),
    }


def get_stats_format(ctx):
    # [FUNCTION DOC] Returns the format spec for each get_stats key with prefix ctx: "e" (scientific) for all,
    # "" for nonzero.
    return {
        f"{ctx}_var_min": "e",
        f"{ctx}_var_max": "e",
        f"{ctx}_var_mean": "e",
        f"{ctx}_var_std": "e",
        f"{ctx}_abs_var_min": "e",
        f"{ctx}_abs_var_max": "e",
        f"{ctx}_abs_var_mean": "e",
        f"{ctx}_abs_var_std": "e",
        f"{ctx}_var_norm_2": "e",
        f"{ctx}_var_norm_1": "e",
        f"{ctx}_nonzero": "",
    }


# Handling SIGTERM signals (in case our job gets pre-empted)
class SigtermListener:
    # [CLASS DOC] Installs a SIGTERM handler so preempted jobs can stop gracefully; poll kill_now in the
    # training loop.
    def __init__(self):
        # [FUNCTION DOC] Sets kill_now False and registers exit_gracefully as the process-wide SIGTERM handler
        # (replacing any).
        self.kill_now = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        # [FUNCTION DOC] Signal handler: sets kill_now True (args ignored).
        self.kill_now = True


# Mostly copied from https://github.com/huggingface/peft/blob/v0.7.1/src/peft/tuners/mixed/model.py#L103
def lora_replace_module(parent, child_name, new_module, child) -> None:
    # [FUNCTION DOC] HELPER (from peft): replaces parent.child_name with new_module. Unwraps child (base_layer
    # or
    # quant_linear_module); if new_module has no base_layer, shares child's weight and bias with it; copies
    # bitsandbytes `state` if present (moving new_module to the child's device); moves submodules whose name
    # contains "lora" or "ranknum" to the child's weight device. Returns None.
    setattr(parent, child_name, new_module)
    # It's not necessary to set requires_grad here, as that is handled by
    # _mark_only_adapters_as_trainable

    # child layer wraps the original module, unpack it
    if hasattr(child, "base_layer"):
        child = child.get_base_layer()
    elif hasattr(child, "quant_linear_module"):
        # TODO maybe not necessary to have special treatment?
        child = child.quant_linear_module

    if not hasattr(new_module, "base_layer"):
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

    if getattr(child, "state", None) is not None:
        if hasattr(new_module, "base_layer"):
            new_module.base_layer.state = child.state
        else:
            new_module.state = child.state
        new_module.to(child.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora" in name:
            module.to(child.weight.device)
        if "ranknum" in name:
            module.to(child.weight.device)


# Inspired by _unload_and_optionally_merge https://github.com/huggingface/peft/blob/v0.7.1/src/peft/tuners/mixed/model.py#L256
def lora_unload(model):
    # [FUNCTION DOC] Removes LoRA wrappers without merging: for every non-lora module path, resolves (parent,
    # target, name)
    # with peft _get_submodules (skipping AttributeError) and, when target has base_layer, replaces it by its
    # base layer. LoRA weights are discarded, not merged. Mutates and returns the model.
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue

        if hasattr(target, "base_layer"):
            lora_replace_module(parent, target_name, target.get_base_layer(), target)
    return model
