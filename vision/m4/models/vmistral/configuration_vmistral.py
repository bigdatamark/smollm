# coding=utf-8
# [MODULE DOC] Configuration classes for VMistral: a vision sub-config (derived from a Hub vision model's config), a
# perceiver resampler sub-config, and the top-level composite VMistralConfig that combines Mistral
# text-decoder settings with the two sub-configs and training/freezing options. Also defines
# MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP (Hub URLs of the original Mistral configs) and a module logger (unused
# here).
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Mistral model configuration"""
import copy
import os
from typing import Union

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mistralai/Mistral-7B-v0.1": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json",
    "mistralai/Mistral-7B-Instruct-v0.1": (
        "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/config.json"
    ),
}


class VMistralVisionConfig(PretrainedConfig):
    # [CLASS DOC] Vision-encoder configuration (model_type "vmistral"; "hidden_size" is aliased to embed_dim). Stores the
    # vision model name and copies the key dimensions (embed_dim, image_size, patch_size, num_hidden_layers,
    # intermediate_size) from that model's Hub config.
    # KNOWN ISSUE: the docstring is copied from MistralConfig and describes a Mistral text model, and lists
    # embed_dim and image_size as constructor args, while the constructor actually only takes
    # vision_model_name.
    r"""
    This is the configuration class to store the configuration of a [`MistralModel`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        embed_dim (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer. (elsewhere referred to as `embed_dim`)
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
    """
    model_type = "vmistral"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        # Case for when vmistral is from the hub with no vision_model_name
        vision_model_name="HuggingFaceM4/siglip-so400m-14-384",
        **kwargs,
    ):
        # [FUNCTION DOC] Args: vision_model_name (str, default "HuggingFaceM4/siglip-so400m-14-384"), extra kwargs forwarded
        # to PretrainedConfig. Loads the named model's config with AutoConfig.from_pretrained
        # (trust_remote_code=True; network or cache access); if it has a nested vision_config (e.g. a
        # CLIP/SigLIP composite) that nested one is used. Sets embed_dim from embed_dim, else from
        # hidden_size; sets image_size, patch_size, num_hidden_layers and intermediate_size. Raises ValueError
        # for any of these attributes missing on the loaded config.
        self.vision_model_name = vision_model_name
        vision_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        # vmistral case (necessary for loading the vmistral model)
        if hasattr(vision_config, "embed_dim"):
            self.embed_dim = vision_config.embed_dim
        # clip case (necessary for initialization)
        elif hasattr(vision_config, "hidden_size"):
            self.embed_dim = vision_config.hidden_size
        else:
            raise ValueError("vision_config must have a hidden_size or embed_dim")

        if hasattr(vision_config, "image_size"):
            self.image_size = vision_config.image_size
        else:
            raise ValueError("vision_config must have an image_size")

        if hasattr(vision_config, "patch_size"):
            self.patch_size = vision_config.patch_size
        else:
            raise ValueError("vision_config must have a patch_size")

        if hasattr(vision_config, "num_hidden_layers"):
            self.num_hidden_layers = vision_config.num_hidden_layers
        else:
            raise ValueError("vision_config must have a num_hidden_layers")

        if hasattr(vision_config, "intermediate_size"):
            self.intermediate_size = vision_config.intermediate_size
        else:
            raise ValueError("vision_config must have an intermediate_size")

        super().__init__(**kwargs)


class VMistralPerceiverConfig(PretrainedConfig):
    # [CLASS DOC] Perceiver resampler configuration (model_type "vmistral") holding resampler hyper-parameters.
    # KNOWN ISSUE: docstring is copied from MistralConfig (starts with typo "TThis"), documents a
    # use_resampler arg that does not exist here and a wrong resampler_n_latents default (blank; actual 64).
    r"""
    TThis is the configuration class to store the configuration of a [`MistralModel`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_resampler (`bool`, *optional*, defaults to `False`):
            Whether or not to use the resampler
        resampler_n_latents (`int`, *optional*, defaults to ):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 6):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        qk_layer_norms_perceiver (`bool`, *optional*, defaults to `False`):
            Whether or not to use qk layer norms in perceiver
    """
    model_type = "vmistral"

    def __init__(
        self,
        hidden_act="silu",
        resampler_n_latents=64,
        resampler_depth=6,
        resampler_n_heads=16,
        num_key_value_heads=1,
        resampler_head_dim=96,
        qk_layer_norms_perceiver=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        # [FUNCTION DOC] Args (stored as attributes): hidden_act ("silu"), resampler_n_latents (64, number of output
        # latents), resampler_depth (6, number of resampler blocks), resampler_n_heads (16),
        # num_key_value_heads (1, for grouped KV attention in the resampler), resampler_head_dim (96),
        # qk_layer_norms_perceiver (False), attention_dropout (0.0); other kwargs go to PretrainedConfig.
        # Raises ValueError if num_key_value_heads > resampler_n_heads.
        self.hidden_act = hidden_act
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.num_key_value_heads = num_key_value_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver
        self.attention_dropout = attention_dropout
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" resampler_n_heads={self.resampler_n_heads}"
            )

        super().__init__(**kwargs)


class VMistralConfig(PretrainedConfig):
    # [CLASS DOC] Top-level VMistral configuration (model_type "vmistral", composite, ignores past_key_values at
    # inference). Holds Mistral decoder hyper-parameters plus image token id, additional (trainable) vocab
    # size, gating-alpha options, freezing options, NEFTune noise, flash-attention flag, resampler use, and
    # nested vision/perceiver configs.
    r"""
    This is the configuration class to store the configuration of a [`MistralModel`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        additional_vocab_size (`int`, *optional`, defaults to 0):
            Additional vocabulary size of the model, typically for the special "<img>" token. Additional vocab tokens
            are always trainable whereas regular vocab tokens can be frozen or not.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Mistral model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MistralModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Mistral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        alpha_initializer (`str`, *optional*, defaults to `"zeros"`):
            Initialization type for the alphas.
        alphas_initializer_range (`float`, *optional*, defaults to 0.0):
            The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross
            Attention.
        alpha_type (`str`, *optional*, defaults to `"float"`):
            Whether the gating alphas should be vectors or single floats.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size. If not specified, will default to `4096`.
        cross_layer_interval (`int`, *optional*, default to 1)
            Interval for cross attention (from text to image) layers.
        qk_layer_norms (`bool`, *optional*, defaults to `False`): Whether to add layer norm after q and k
        freeze_text_layers (`bool`, *optional*, defaults to `True`): Whether to freeze text layers
        freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing text layers when `freeze_text_layers` is `True`
        freeze_lm_head (`bool`, *optional*, defaults to `False`): Whether to freeze lm head
        freeze_vision_layers (`bool`, *optional*, defaults to `True`):  Whether to freeze vision layers
        freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
        use_resampler (`bool`, *optional*, defaults to `False`): Whether to use the Resampler
        vision_config (`IdeficsVisionConfig`,  *optional*): Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig`,  *optional*): Custom perceiver config or dict

    Example:
    ```python
    >>> from transformers import MistralModel, MistralConfig

    >>> # Initializing a Mistral 7B style configuration
    >>> configuration = MistralConfig()

    >>> # Initializing a model from the Mistral 7B style configuration
    >>> model = MistralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vmistral"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        additional_vocab_size=0,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        alpha_initializer="zeros",
        alphas_initializer_range=0.0,
        alpha_type="float",
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,  # None in the original configuration_mistral, we set it to the unk_token_id
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=32_001,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        cross_layer_interval=1,
        qk_layer_norms=False,
        freeze_text_layers=True,
        freeze_text_module_exceptions=[],
        freeze_lm_head=False,
        freeze_vision_layers=True,
        freeze_vision_module_exceptions=[],
        attention_dropout=0.0,
        neftune_noise_alpha=0.0,
        _flash_attn_2_enabled=True,
        use_resampler=True,
        vision_config=None,
        perceiver_config=None,
        **kwargs,
    ):
        # [FUNCTION DOC] Stores every argument as an attribute of the same name. Defaults: additional_vocab_size 0,
        # vocab_size 32000, hidden_size 4096, intermediate_size 14336, 32 layers, 32 heads,
        # num_key_value_heads 8 (None means equal to num_attention_heads), hidden_act "silu",
        # max_position_embeddings 131072, initializer_range 0.02, alpha_initializer "zeros",
        # alphas_initializer_range 0.0, alpha_type "float", rms_norm_eps 1e-6, use_cache True, pad/bos/eos ids
        # 0/1/2, image_token_id 32001, tie_word_embeddings False, rope_theta 10000, sliding_window 4096,
        # cross_layer_interval 1, qk_layer_norms False, freeze_text_layers True, freeze_text_module_exceptions
        # [], freeze_lm_head False, freeze_vision_layers True, freeze_vision_module_exceptions [],
        # attention_dropout 0.0, neftune_noise_alpha 0.0, _flash_attn_2_enabled True, use_resampler True.
        # perceiver_config / vision_config may be None (default sub-config built; the vision one loads from
        # the Hub), a dict (sub-config built from it) or a sub-config instance (used as is). pad/bos/eos ids,
        # tie_word_embeddings and remaining kwargs are passed to PretrainedConfig.
        # KNOWN ISSUE: a sub-config of any other type (e.g. a VLlama3 config object) is silently ignored,
        # leaving the attribute unset (AttributeError later).
        # KNOWN ISSUE: mutable list defaults ([]) for the freeze_*_module_exceptions args are shared across
        # instances.
        # KNOWN ISSUE: docstring says use_resampler defaults to False and pad_token_id has no default; code
        # uses True and 0.
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.image_token_id = image_token_id
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.alpha_initializer = alpha_initializer
        self.alphas_initializer_range = alphas_initializer_range
        self.alpha_type = alpha_type
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.cross_layer_interval = cross_layer_interval
        self.qk_layer_norms = qk_layer_norms
        self.freeze_vision_layers = freeze_vision_layers

        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        self.freeze_lm_head = freeze_lm_head

        self.use_resampler = use_resampler
        self._flash_attn_2_enabled = _flash_attn_2_enabled
        self.attention_dropout = attention_dropout
        self.neftune_noise_alpha = neftune_noise_alpha

        if perceiver_config is None:
            self.perceiver_config = VMistralPerceiverConfig()
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = VMistralPerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, VMistralPerceiverConfig):
            self.perceiver_config = perceiver_config

        if vision_config is None:
            self.vision_config = VMistralVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = VMistralVisionConfig(**vision_config)
        elif isinstance(vision_config, VMistralVisionConfig):
            self.vision_config = vision_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # IMPORTANT: Do not do any __init__ args-based checks in the constructor, since
        # PretrainedConfig.from_dict first instantiates the class with the config dict and only then
        # updates the config object with `kwargs` from from_pretrained, so during the instantiation
        # of this object many attributes have default values and haven't yet been overridden.
        # Do any required checks inside `from_pretrained` once the superclass' `from_pretrained` was run.

    def to_dict(self):
        # [FUNCTION DOC] Serializes the config: deep copy of the instance attribute mapping, with vision_config and
        # perceiver_config replaced by their own to_dict() mappings and "model_type" set to the class
        # model_type. Returns the dict.
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["vision_config"] = self.vision_config.to_dict()
        output["perceiver_config"] = self.perceiver_config.to_dict()
        output["model_type"] = self.__class__.model_type

        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # [FUNCTION DOC] Classmethod that simply delegates to PretrainedConfig.from_pretrained with the given name/path and
        # kwargs and returns its result (a VMistralConfig, or a (config, unused_kwargs) tuple if
        # return_unused_kwargs is set). Exists as a hook for post-load checks, none are done.
        outputs = super(VMistralConfig, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        return outputs
