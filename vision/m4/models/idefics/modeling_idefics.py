# coding=utf-8
# [MODULE DOC] PyTorch implementation of IDEFICS (Flamingo-style): a LLaMA decoder whose layers are
# interleaved with gated cross-attention layers attending to vision-encoder (optionally Perceiver-resampled)
# image features. Contents: mask helpers, LLaMA building blocks (RMSNorm, rotary embedding, MLP, attention
# using torch SDPA, decoder layer), IdeficsGatedCrossAttentionLayer, IdeficsPreTrainedModel (init, vision-
# model override hook), IdeficsModel (backbone), IdeficsForCausalLM (LM head with decoupled extra vocab, loss,
# generation hooks, TFLOPs estimate). Text layers / vision encoder are frozen per config at construction.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PretrainedConfig
from transformers.utils import (
    ContextManagers,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from m4.models import DecoupledEmbedding, DecoupledLinear
from m4.models.common import (
    expand_inputs_for_generation,
    prepare_inputs_for_generation,
    update_model_kwargs_for_generation,
)
from m4.models.custom_modules import VLOOMPreTrainedModelBase
from m4.models.idefics.configuration_idefics import IdeficsConfig
from m4.models.perceiver.perceiver import PerceiverResampler
from m4.training.utils import (
    compute_perceiver_tflops_per_batch_per_gpu,
    compute_tflops_per_batch_per_gpu,
    deepspeed_gathered_parameters_context_manager,
    freeze_model,
)
from m4.utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "IdeficsConfig"


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    # [FUNCTION DOC] HELPER: build an additive causal mask of shape (bsz, 1, tgt_len, tgt_len + past_len): 0
    # where a query may attend (key index <= query index, and all past positions), dtype-min elsewhere.
    # input_ids_shape is (bsz, tgt_len). Created on CPU; caller moves it to the device.
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    # [FUNCTION DOC] HELPER: convert a (bsz, src_len) 1/0 padding mask into an additive (bsz, 1, tgt_len,
    # src_len) mask in `dtype` (0 for keep, dtype-min for padded); tgt_len defaults to src_len.
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    # [CLASS DOC] LLaMA RMSNorm: x * rsqrt(mean(x^2) + eps) with a learned weight (init ones), no
    # bias/centering.
    def __init__(self, hidden_size, eps=1e-6):
        # [FUNCTION DOC] hidden_size: normalized dim; eps default 1e-6.
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # [FUNCTION DOC] Compute the variance in fp32 and normalize; cast to the weight dtype only when the
        # weight is fp16/bf16; return weight * normalized states.
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    # [CLASS DOC] Rotary position embedding tables: precomputes cos/sin caches of shape (1, 1, max_len, dim)
    # using inverse frequencies base^(-2i/dim); cache buffers are non-persistent.
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # [FUNCTION DOC] dim: head dim; max_position_embeddings default 2048; base default 10000; device for
        # inv_freq. Registers inv_freq (persistent buffer) and cos_cached / sin_cached.
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # [FUNCTION DOC] Return (cos, sin) truncated to seq_len and cast to x's dtype. If seq_len exceeds the
        # cache, rebuilds and re-registers larger caches on x's device (mutates module state).
        # KNOWN ISSUE: seq_len defaults to None but is compared with an int, so calling without seq_len raises
        # TypeError.
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    # [FUNCTION DOC] HELPER: split the last dim in two halves (x1, x2) and return concat(-x2, x1), used by
    # rotary embeddings.
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # [FUNCTION DOC] HELPER: gather cos/sin rows at position_ids (bsz, seq) and apply the rotation q*cos +
    # rotate_half(q)*sin (same for k). q/k are (bsz, heads, seq, head_dim). Returns (q_embed, k_embed).
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    # [CLASS DOC] SwiGLU feed-forward: down_proj(act(gate_proj(x)) * up_proj(x)), all bias-free linear layers.
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        # [FUNCTION DOC] Build gate/up (hidden_size -> intermediate_size) and down (intermediate_size ->
        # hidden_size) projections; hidden_act is resolved through ACT2FN.
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # [FUNCTION DOC] Apply the SwiGLU MLP to x (..., hidden_size).
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    # [CLASS DOC] Multi-head attention used both as causal self-attention (with rotary embeddings) and as
    # cross-attention from text to image features (no rotary). Computed with torch
    # scaled_dot_product_attention; attention weights are never returned.
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        config: PretrainedConfig = None,
        qk_layer_norms: bool = False,
    ):
        # [FUNCTION DOC] Inputs: hidden_size, num_heads (head_dim = hidden_size // num_heads; ValueError if
        # not divisible), dropout (default 0.0), is_cross_attention (default False), config (needed for cross-
        # attention to read vision_config.embed_dim as key/value input dim, else hidden_size, and for
        # rms_norm_eps when qk_layer_norms), qk_layer_norms (default False: adds per-head RMSNorm on q and k).
        # All projections are bias-free; also creates a LlamaRotaryEmbedding (used only for self-attention).
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            kv_input_dim = (
                self.hidden_size if not hasattr(config.vision_config, "embed_dim") else config.vision_config.embed_dim
            )
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            self.k_proj = nn.Linear(kv_input_dim, num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(
                kv_input_dim,
                num_heads * self.head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layer_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # [FUNCTION DOC] HELPER: reshape (bsz, seq, hidden) to contiguous (bsz, heads, seq, head_dim). Unused
        # in this file.
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # [FUNCTION DOC] Inputs: hidden_states (bsz, q_len, hidden); key_value_states (bsz, kv_len, kv_dim)
        # for cross-attention; attention_mask additive (bsz, 1, q_len, kv_seq_len) or None; position_ids (bsz,
        # q_len) for rotary; past_key_value (k, v) to prepend; output_attentions (ignored); use_cache. Self-
        # attention projects q/k/v from hidden_states and applies rotary embeddings; cross-attention projects
        # k/v from key_value_states. Concatenates the cache, optionally q/k RMS-normalizes, validates mask
        # shape (ValueError), runs SDPA (non-causal; causality comes from the mask), merges heads and applies
        # o_proj. Logs a one-time warning that attention weights are not available. Returns (output (bsz,
        # q_len, hidden), None, (k, v) if use_cache else None).
        # KNOWN ISSUE: dropout_p=self.dropout is passed to SDPA unconditionally, so attention dropout is also
        # applied in eval mode when config.dropout > 0.
        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = self.is_cross_attention or key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if not is_cross_attention:
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            _, kv_len, _ = key_value_states.size()  # Note that, in this case, `kv_len` == `kv_seq_len`
            key_states = self.k_proj(key_value_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = (
                self.v_proj(key_value_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if not is_cross_attention:
            cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        attn_weights = None
        logger.warning_once(
            "attn_weights are not extracted in scaled_dot_product_attention. The model returns None instead"
        )

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    # [CLASS DOC] Standard pre-norm LLaMA decoder block: x + dropout(self_attn(RMSNorm(x))), then x +
    # dropout(MLP(RMSNorm(x))).
    def __init__(self, config: IdeficsConfig):
        # [FUNCTION DOC] Build self-attention, SwiGLU MLP and the two RMSNorms from config (hidden_size,
        # num_attention_heads, dropout, intermediate_size, hidden_act, rms_norm_eps).
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            config=config,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # [FUNCTION DOC] Run the block (see class doc). attention_mask is the 4D additive causal+padding mask.
        # Returns a tuple (hidden_states,) + (attn weights,) if output_attentions + (present (k, v),) if
        # use_cache.
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class IdeficsGatedCrossAttentionLayer(nn.Module):
    # [CLASS DOC] Flamingo gated cross-attention block inserted before decoder layers: text attends to image
    # features, and both the attention output and the following MLP output are scaled by tanh(alpha) gates
    # before being added to the residual; text positions with no visible image get no cross-attention
    # contribution.
    def __init__(self, config: IdeficsConfig):
        # [FUNCTION DOC] Build cross-attention LlamaAttention (with optional qk norms), SwiGLU MLP, two
        # RMSNorms, tanh activations and the gate parameters alpha_cross_attn / alpha_dense. Gate init per
        # config.alpha_initializer: "zeros", "ones", or "normal"/"gaussian"/"random" (std
        # alphas_initializer_range); shape per config.alpha_type: "vector" (1, 1, hidden) or "float" (1,).
        # Raises ValueError for unknown alpha_type, NotImplementedError for unknown initializer. Stores the
        # dropout probability in self.config (misleading name).
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            is_cross_attention=True,
            dropout=config.dropout,
            config=config,
            qk_layer_norms=config.qk_layer_norms,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config.dropout

        self.act_cross_attn = nn.Tanh()
        self.act_dense = nn.Tanh()

        if config.alpha_initializer == "zeros":
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
                self.alpha_dense = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(torch.zeros(1))
                self.alpha_dense = nn.Parameter(torch.zeros(1))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        elif config.alpha_initializer == "ones":
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(torch.ones(1, 1, self.hidden_size))
                self.alpha_dense = nn.Parameter(torch.ones(1, 1, self.hidden_size))
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(torch.ones(1))
                self.alpha_dense = nn.Parameter(torch.ones(1))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        elif config.alpha_initializer in {"normal", "gaussian", "random"}:
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1, 1, self.hidden_size))
                )
                self.alpha_dense = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1, 1, self.hidden_size))
                )
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1))
                )
                self.alpha_dense = nn.Parameter(torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1)))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        else:
            raise NotImplementedError(f"Alpha initialization scheme {config.alpha_initializer} not yet implemented!")

        if not (hasattr(self, "alpha_cross_attn") and hasattr(self, "alpha_dense")):
            raise ValueError("Alpha parameters not initialized correctly!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # [FUNCTION DOC] Inputs: hidden_states (bsz, seq, hidden); attention_mask (unused);
        # image_hidden_states (bsz, n_img_tokens, vision_dim, required: ValueError if None);
        # image_attention_mask additive (bsz, 1, seq, n_img_tokens); output_attentions, use_cache;
        # past_key_value must be None (NotImplementedError otherwise). Computes residual + row_mask *
        # tanh(alpha_cross_attn) * dropout(cross_attn(norm(x))) where row_mask is 1 for text positions having
        # at least one attendable image token (mask value 0), then residual + tanh(alpha_dense) *
        # dropout(MLP(norm(x))). Returns (hidden_states,) + optional attn weights + optional present key/value
        # (always None, since use_cache is not passed to cross_attn).
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if image_hidden_states is None:
            raise ValueError(
                "`image_hidden_states` is required for Idefics cross attention module which are visual features to be"
                " conditioned on."
            )

        if past_key_value is not None:
            raise NotImplementedError("Past key value states are not implemented for Idefics cross attention module.")

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=image_hidden_states,
            attention_mask=image_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.config, training=self.training)
        # cross_attention_mask: zero-ing out any attention that is equal to 0.0 (image masks are composed of either very negative numbers or 0s).
        # If the batch contains no image, everything is zeroed out and hidden_states = residual
        cross_attention_mask = (
            ((image_attention_mask == 0.0).sum(dim=-1) > 0.0).to(dtype=hidden_states.dtype)
        ).permute(0, 2, 1)
        hidden_states = residual + cross_attention_mask * self.act_cross_attn(self.alpha_cross_attn) * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.config, training=self.training)
        hidden_states = residual + self.act_dense(self.alpha_dense) * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IdeficsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IdeficsPreTrainedModel(VLOOMPreTrainedModelBase):
    # [CLASS DOC] Common base for IDEFICS models: config_class IdeficsConfig, base_model_prefix "model",
    # gradient checkpointing support, no-split modules for device maps, custom weight init and the vision-
    # override hook used by VLOOMPreTrainedModelBase.from_pretrained.
    config_class = IdeficsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "IdeficsGatedCrossAttentionLayer"]

    def _init_weights(self, module):
        # [FUNCTION DOC] Initialize one module (called by post_init for every submodule), gathering ZeRO-3
        # shards when needed:
        # - gated cross-attention layers: every nn.Linear ~ N(0, 0.4 / (in_features * factor)), factor =
        #   2*num_hidden_layers for down_proj, else 1; biases zeroed.
        # - PerceiverResampler: latents ~ N(0, 1/vision embed_dim); linears ~ N(0, 0.4 / (vision embed_dim *
        #   factor)).
        # - nn.Embedding (incl. DecoupledEmbedding): N(0, 1/hidden_size), padding row zeroed.
        # - DecoupledLinear: only additional_fc ~ N(0, 1/in_features). Other modules (decoder linears, norms)
        #   are left as created / loaded.
        # KNOWN ISSUE: for the perceiver the factor checks for "c_proj" names, which do not exist in
        # PerceiverResampler, so the factor is always 1.
        def init_a_linear(module, mean=0.0, std=self.config.initializer_range):
            # [FUNCTION DOC] HELPER: under a DeepSpeed gathered-parameters context, fill module.weight with
            # N(mean, std) (std defaults to config.initializer_range) and zero module.bias if present.
            with ContextManagers(deepspeed_gathered_parameters_context_manager(module.weight, modify=True)):
                module.weight.data.normal_(mean=mean, std=std)
                if module.bias is not None:
                    with ContextManagers(deepspeed_gathered_parameters_context_manager(module.bias, modify=True)):
                        module.bias.data.zero_()

        if isinstance(module, IdeficsGatedCrossAttentionLayer):
            for sub_module_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    if "down_proj" in sub_module_name:
                        factor = 2 * self.config.num_hidden_layers
                    else:
                        factor = 1.0
                    init_a_linear(sub_module, std=(0.4 / (sub_module.in_features * factor)) ** 0.5)
        elif isinstance(module, PerceiverResampler):
            with ContextManagers(deepspeed_gathered_parameters_context_manager(module.latents, modify=True)):
                module.latents.data.normal_(mean=0.0, std=(1.0 / self.config.vision_config.embed_dim) ** 0.5)
            for sub_module_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    if "c_proj" in sub_module_name:
                        factor = 2 * self.config.num_hidden_layers
                    else:
                        factor = 1.0
                    init_a_linear(sub_module, std=(0.4 / (self.config.vision_config.embed_dim * factor)) ** 0.5)
        elif isinstance(module, nn.Embedding):
            with ContextManagers(deepspeed_gathered_parameters_context_manager(module.weight, modify=True)):
                module.weight.data.normal_(mean=0.0, std=(1.0 / self.config.hidden_size) ** 0.5)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, DecoupledLinear):
            if hasattr(module, "additional_fc"):
                init_a_linear(module.additional_fc, std=(1.0 / (module.additional_fc.in_features)) ** 0.5)

    def _set_gradient_checkpointing(self, module, value=False):
        # [FUNCTION DOC] Set gradient_checkpointing = value on IdeficsModel instances (legacy transformers
        # hook).
        if isinstance(module, IdeficsModel):
            module.gradient_checkpointing = value

    @classmethod
    def override_vision_model_wrapper(cls, model, config, vision_model_name, vision_model_params, torch_dtype):
        # [FUNCTION DOC] Hook called by VLOOMPreTrainedModelBase.from_pretrained for new models: take the
        # backbone (model.model if present), load pretrained vision weights via override_vision_model, then
        # re-apply freezing with freeze_relevant_params(config).
        # this can be called via from_pretrained from a class w/ head or w/o head so we extract the beheaded model version
        beheaded_model = model.model if hasattr(model, "model") else model
        cls.override_vision_model(beheaded_model, vision_model_name, vision_model_params, torch_dtype)
        beheaded_model.freeze_relevant_params(config)


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IdeficsModel(IdeficsPreTrainedModel):
    # [CLASS DOC] IDEFICS backbone: decoupled token embeddings, vision encoder (injected), optional Perceiver
    # resampler, num_hidden_layers LLaMA decoder layers with a gated cross-attention layer before every
    # cross_layer_interval-th layer, final RMSNorm.
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    def __init__(self, config: IdeficsConfig, vision_model=None):
        # [FUNCTION DOC] Build embed_tokens (DecoupledEmbedding with additional_vocab_size trainable rows,
        # base rows frozen if freeze_text_layers), store vision_model (default None; set later by
        # from_pretrained), optional perceiver, decoder layers, num_hidden_layers // cross_layer_interval
        # gated cross-attention layers and the final norm; run post_init (weight init) and
        # freeze_relevant_params.
        # KNOWN ISSUE: PerceiverResampler is called with 6 positional arguments but its constructor only
        # accepts config, so use_resampler=True raises TypeError.
        # KNOWN ISSUE: freeze_relevant_params runs freeze_model on self.vision_model, which is None by
        # default.
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = DecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            padding_idx=self.padding_idx,
        )

        # Load an uninitialized model and later in from_pretrained will load the pre-trained model -
        # this solves the losing of weights in `from_pretrained` on the main model
        self.vision_model = vision_model

        # Perceiver Resampler
        if config.use_resampler:
            self.perceiver_resampler = PerceiverResampler(
                self.config,
                self.config.vision_config.embed_dim,
                config.perceiver_config.resampler_depth,
                config.perceiver_config.resampler_n_heads,
                config.perceiver_config.resampler_head_dim,
                config.perceiver_config.resampler_n_latents,
            )

        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = nn.ModuleList(
            [IdeficsGatedCrossAttentionLayer(config) for _ in range(num_cross_layers)]
        )
        self.gradient_checkpointing = False

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_relevant_params(config)

    def freeze_relevant_params(self, config=None):
        # [FUNCTION DOC] Freeze text layers (decoder layers + final norm, except module_exceptions) if
        # config.freeze_text_layers, and the vision model (except freeze_vision_module_exceptions) if
        # config.freeze_vision_layers. config defaults to self.config.
        if config is None:
            config = self.config

        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)

        if config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    def freeze_text_layers(self, module_exceptions):
        # [FUNCTION DOC] Apply m4 freeze_model to self.layers and self.norm, keeping modules matching
        # module_exceptions trainable. Embeddings and cross-attention layers are not frozen here.
        for module in [self.layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    def get_input_embeddings(self):
        # [FUNCTION DOC] Return embed_tokens.
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # [FUNCTION DOC] Replace embed_tokens with value.
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # [FUNCTION DOC] HELPER: combine a causal mask (only when the current input has > 1 token) and the
        # expanded padding mask into one additive (bsz, 1, seq, seq + past) mask in inputs_embeds'
        # dtype/device; returns None if neither applies.
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # [FUNCTION DOC] Run the backbone. Inputs: input_ids (bsz, seq) or inputs_embeds (exactly one, else
        # ValueError); attention_mask (bsz, seq+past) 1/0; position_ids (derived from the mask cumsum, or a
        # range after the cache length); past_key_values per layer; pixel_values (bsz, n_images, 3, H, W) or
        # image_embeddings (bsz, n_images, img_seq, dim), not both (ValueError); image_attention_mask (bsz,
        # seq, n_images) 1/0 telling which images each text token may see; use_cache, output_attentions,
        # output_hidden_states, return_dict (defaults from config). Steps: if no images are given, builds one
        # dummy black image per sample with an all-zero image mask (text-only mode; keeps the DeepSpeed graph
        # static). Encodes images with vision_model (last_hidden_state), optionally resamples, reshapes to
        # (bsz, n_images*img_seq, dim), expands the image mask to every image token and inverts it to an
        # additive mask. Embeds tokens, builds the decoder mask, then for each layer runs (optionally under
        # gradient checkpointing, which disables the cache) the nested vblock: gated cross-attention when idx
        # % interval == 0, then the decoder layer. Applies the final norm. Returns
        # BaseModelOutputWithPast(last_hidden_state, past_key_values, hidden_states, attentions) or the tuple
        # of non-None values.
        # KNOWN ISSUE: text-only mode and the image path use input_ids for device/size, so inputs_embeds-only
        # calls fail; the image path also requires image_attention_mask (the "is None" fallback is unreachable
        # after .size() is called on it).
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        elif position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if pixel_values is None and image_embeddings is None:
            # This is the case where the model is used as a pure language model.
            # Mostly here to allow sft on text dialogues.
            # Create dummy tensors `pixel_values` and `image_attention_mask` to allow attending to something in
            # the cross-attention. the cross-attention to the image is then zeroed-out.
            # Something more efficient would have to completely bypass the call to the vision_model/perceicer resampler but it looks
            # like DS is complaining about conditional workflow (roughly this error: https://github.com/microsoft/DeepSpeed/issues/3156)
            # To make progress, i am commenting this solution for now (un-indent the whole block 861 -> 882 that should be in an else condition),
            # but TODO: come back to this later.
            # image_seq_len = (
            #     self.config.resampler_n_latents
            #     if self.config.use_resampler
            #     else 1 + int(self.vision_model.config.image_size / self.vision_model.config.patch_size) ** 2
            # )
            # image_hidden_size = self.config.vision_embed_dim
            # image_hidden_states = torch.zeros(
            #     input_ids.size(0), 1 * image_seq_len, image_hidden_size, device=input_ids.device, dtype=self.dtype
            # )
            # image_attention_mask = torch.zeros(
            #     input_ids.size(0), input_ids.size(1), 1 * image_seq_len, device=input_ids.device, dtype=torch.bool
            # )
            pixel_values = torch.zeros(
                input_ids.size(0),
                1,
                3,
                self.vision_model.config.image_size,
                self.vision_model.config.image_size,
                dtype=self.dtype,
                device=input_ids.device,
            )
            image_attention_mask = torch.zeros(
                input_ids.size(0), input_ids.size(1), 1, device=input_ids.device, dtype=torch.bool
            )

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeddings at the same time")
        elif pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype, device=input_ids.device)  # fp16 compatibility
            batch_size, num_images = pixel_values.size(0), pixel_values.size(1)
            pixel_values = pixel_values.contiguous().view(batch_size * num_images, *pixel_values.shape[2:])
            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(pixel_values=pixel_values).last_hidden_state
        elif image_embeddings is not None:
            batch_size, num_images, image_seq_len, image_hidden_size = image_embeddings.size()
            image_hidden_states = image_embeddings.to(dtype=self.dtype, device=input_ids.device)
            image_hidden_states = image_hidden_states.view(batch_size * num_images, image_seq_len, image_hidden_size)

        if self.config.use_resampler:
            image_hidden_states = self.perceiver_resampler(image_hidden_states)
        image_seq_len, image_hidden_size = image_hidden_states.size(1), image_hidden_states.size(2)
        image_hidden_states = image_hidden_states.view(batch_size, num_images * image_seq_len, image_hidden_size)
        # Make image_attention_mask compatible with hidden states
        text_seq_len = image_attention_mask.size(1)
        image_attention_mask = image_attention_mask.unsqueeze(-1)
        image_attention_mask = image_attention_mask.repeat(1, 1, 1, image_seq_len)
        image_attention_mask = image_attention_mask.view(batch_size, text_seq_len, num_images * image_seq_len)

        if image_hidden_states is not None:
            image_batch_size, image_sequence_length, _ = image_hidden_states.size()
            image_hidden_shape = (image_batch_size, image_sequence_length)
            if image_attention_mask is None:
                image_attention_mask = torch.ones(image_hidden_shape, device=device)
            image_attention_mask = self.invert_attention_mask(image_attention_mask)
        else:
            image_attention_mask = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            def vblock(
                main_block,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                image_hidden_states,
                image_attention_mask,
                output_attentions,
                use_cache,
                layer_idx,
                cross_layer_interval,
                gated_cross_attn_layers,
            ):
                # [FUNCTION DOC] HELPER: one decoder step (closure used for gradient checkpointing): if
                # layer_idx is a multiple of cross_layer_interval, apply gated_cross_attn_layers[layer_idx //
                # interval] to hidden_states with the image states and mask (no cache); then run main_block
                # (the decoder layer) and return its output tuple.
                # TODO(ls): Add cross attention values to respective lists
                if layer_idx % cross_layer_interval == 0:
                    xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
                    outputs = xblock(
                        hidden_states,
                        attention_mask=attention_mask,
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        past_key_value=None,  # not implemented
                    )
                    hidden_states = outputs[0]

                layer_outputs = main_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                return layer_outputs

            if self.gradient_checkpointing and self.training:
                past_key_value = None
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    vblock,
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    image_hidden_states,
                    image_attention_mask,
                    output_attentions,
                    use_cache,
                    idx,
                    self.cross_layer_interval,
                    self.gated_cross_attn_layers,
                )
            else:
                layer_outputs = vblock(
                    decoder_layer,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    image_hidden_states=image_hidden_states,
                    image_attention_mask=image_attention_mask,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    layer_idx=idx,
                    cross_layer_interval=self.cross_layer_interval,
                    gated_cross_attn_layers=self.gated_cross_attn_layers,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class IdeficsForCausalLM(IdeficsPreTrainedModel):
    # [CLASS DOC] IDEFICS with a causal LM head: lm_head is a bias-free DecoupledLinear (vocab_size outputs,
    # frozen if freeze_lm_head, plus additional_vocab_size trainable outputs). Provides loss, generation
    # hooks, weight tying and a TFLOPs estimate.
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config, vision_model=None):
        # [FUNCTION DOC] Build the IdeficsModel backbone (optional injected vision_model) and lm_head, then
        # post_init.
        super().__init__(config)
        self.model = IdeficsModel(config, vision_model=vision_model)

        self.lm_head = DecoupledLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            out_additional_features=config.additional_vocab_size,
            bias=False,
            partially_freeze=config.freeze_lm_head,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # [FUNCTION DOC] Return model.embed_tokens.
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # [FUNCTION DOC] Replace model.embed_tokens.
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        # [FUNCTION DOC] Return lm_head.
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # [FUNCTION DOC] Replace lm_head.
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # [FUNCTION DOC] Replace the backbone (self.model).
        self.model = decoder

    def get_decoder(self):
        # [FUNCTION DOC] Return the backbone (self.model).
        return self.model

    def tie_weights(self):
        # [FUNCTION DOC] Tie decoupled weights: if config.tie_word_embeddings (default True when missing),
        # share lm_head.weight with the embedding weight and, when extra vocab exists, additional_fc.weight
        # with additional_embedding.weight (AssertionError if their sizes differ). Then sync
        # lm_head.out_features / out_additional_features with the embedding sizes.
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of DecoupledLinear and DecoupledEmbedding.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()

        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings.weight = input_embeddings.weight
            if input_embeddings.num_additional_embeddings > 0:
                assert output_embeddings.out_additional_features == input_embeddings.num_additional_embeddings
                output_embeddings.additional_fc.weight = input_embeddings.additional_embedding.weight

        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
            if hasattr(output_embeddings, "out_additional_features") and hasattr(
                input_embeddings, "num_additional_embeddings"
            ):
                output_embeddings.out_additional_features = input_embeddings.num_additional_embeddings

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # [FUNCTION DOC] Run the backbone (same inputs as IdeficsModel.forward) and lm_head to get logits
        # (bsz, seq, vocab + additional_vocab). With labels, computes next-token cross-entropy: logits[:, :-1]
        # vs labels[:, 1:], keeping only positions where the shifted attention_mask is non-zero (when given);
        # -100 labels are ignored. Returns CausalLMOutputWithPast(loss, logits, past_key_values,
        # hidden_states, attentions) or a tuple (loss first when present).
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_embeddings=image_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # [FUNCTION DOC] Generation hook: delegate to m4.models.common.prepare_inputs_for_generation (requires
        # pixel_values) and drop token_type_ids. Note image_attention_mask is not forwarded by that helper.
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        unwanted_kwargs = ["token_type_ids"]
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        *args,
        **model_kwargs,
    ):
        # [FUNCTION DOC] Generation hook: delegate to m4.models.common.expand_inputs_for_generation.
        return expand_inputs_for_generation(*args, **model_kwargs)

    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # [FUNCTION DOC] Generation hook: delegate to m4.models.common.update_model_kwargs_for_generation.
        return update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        # [FUNCTION DOC] Reorder every cached tensor of every layer along the batch dim with beam_idx (beam
        # search); returns a new tuple of tuples.
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def get_model_tflops_per_batch_per_gpu(self, hparams, data_param, tokenizer, max_num_images):
        # [FUNCTION DOC] Estimate training TFLOPs per batch per GPU. Inputs: hparams (batch_size_per_gpu,
        # grad_acc_size, gradient_checkpointing, model_config["freeze_vision_layers"]), data_param
        # (max_seq_len), tokenizer (vocab_size), max_num_images. Sums compute_tflops_per_batch_per_gpu for:
        # language layers (SwiGLU, with vocab projection, backward counted), cross-attention layers (keys =
        # resampler latents or (image_size/patch)^2 + 1 tokens per image, times max_num_images), vision
        # encoder layers (backward only if not frozen) and, if use_resampler,
        # compute_perceiver_tflops_per_batch_per_gpu. Reads the vision model's (nested) config for
        # patch/hidden/intermediate sizes. Returns the total.
        config_vl_model = self.config

        language_embed_size = config_vl_model.hidden_size
        num_language_layers = config_vl_model.num_hidden_layers
        ffn_inner_size = config_vl_model.intermediate_size

        vision_config = self.model.vision_model.config
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        # Get vision model blocks infos
        vision_patch_size = vision_config.patch_size
        vision_hidden_size = vision_config.hidden_size
        num_vision_layers = vision_config.num_hidden_layers
        # The +1 is for the CLS token
        single_image_seq_len = (vision_config.image_size // vision_patch_size) ** 2 + 1
        vision_exp_factor = vision_config.intermediate_size // vision_hidden_size

        # Get language and cross-att blocks infos
        num_cross_attn_layers = num_language_layers // config_vl_model.cross_layer_interval
        language_seq_len = data_param.max_seq_len
        language_exp_factor = (ffn_inner_size // language_embed_size) if ffn_inner_size is not None else 4
        cross_att_exp_factor = (ffn_inner_size // language_embed_size) if ffn_inner_size is not None else 4
        k_v_cross_attn_seq_len = (
            (self.config.perceiver_config.resampler_n_latents * max_num_images)
            if self.config.use_resampler
            else (single_image_seq_len * max_num_images)
        )

        language_tflops_per_batch_per_gpu = compute_tflops_per_batch_per_gpu(
            num_layers=num_language_layers,
            batch_size=hparams.batch_size_per_gpu,
            q_seq_len=language_seq_len,
            k_seq_len=language_seq_len,
            hidden_size=language_embed_size,
            kv_in_dim=language_embed_size,
            ff_exp_factor=language_exp_factor,
            grad_acc_size=hparams.grad_acc_size,
            swiglu=True,
            vocab_size=tokenizer.vocab_size,
            count_backward=True,  # Always True regardless of freezing, because gradients are computed for cross-attentions
            use_grad_checkpointing=hparams.gradient_checkpointing,
        )
        cross_attention_tflops_per_batch_per_gpu = compute_tflops_per_batch_per_gpu(
            num_layers=num_cross_attn_layers,
            batch_size=hparams.batch_size_per_gpu,
            q_seq_len=language_seq_len,
            k_seq_len=k_v_cross_attn_seq_len,
            hidden_size=language_embed_size,
            kv_in_dim=vision_hidden_size,
            ff_exp_factor=cross_att_exp_factor,
            grad_acc_size=hparams.grad_acc_size,
            swiglu=True,
            vocab_size=None,
            count_backward=True,
            use_grad_checkpointing=hparams.gradient_checkpointing,
        )
        vision_tflops_per_batch_per_gpu = compute_tflops_per_batch_per_gpu(
            num_layers=num_vision_layers,
            batch_size=hparams.batch_size_per_gpu * max_num_images,
            q_seq_len=single_image_seq_len,
            k_seq_len=single_image_seq_len,
            hidden_size=vision_hidden_size,
            kv_in_dim=vision_hidden_size,
            ff_exp_factor=vision_exp_factor,
            grad_acc_size=hparams.grad_acc_size,
            swiglu=False,
            vocab_size=None,
            count_backward=not hparams.model_config["freeze_vision_layers"],
            use_grad_checkpointing=hparams.gradient_checkpointing,
        )
        if self.config.use_resampler:
            perceiver_tflops_per_batch_per_gpu = compute_perceiver_tflops_per_batch_per_gpu(
                num_layers=self.config.perceiver_config.resampler_depth,
                batch_size=hparams.batch_size_per_gpu * max_num_images,
                q_seq_len=self.config.perceiver_config.resampler_n_latents,
                vision_embed_seq_len=single_image_seq_len,
                q_k_v_input_dim=vision_hidden_size,
                attention_hidden_size=self.config.perceiver_config.resampler_n_heads
                * self.config.perceiver_config.resampler_head_dim,
                ff_exp_factor=cross_att_exp_factor,
                count_backward=True,
                use_grad_checkpointing=hparams.gradient_checkpointing,
            )
            flop_count = (
                language_tflops_per_batch_per_gpu
                + cross_attention_tflops_per_batch_per_gpu
                + vision_tflops_per_batch_per_gpu
                + perceiver_tflops_per_batch_per_gpu
            )
        else:
            flop_count = (
                language_tflops_per_batch_per_gpu
                + cross_attention_tflops_per_batch_per_gpu
                + vision_tflops_per_batch_per_gpu
            )
        return flop_count
