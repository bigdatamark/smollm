# [MODULE DOC] Perceiver Resampler (Flamingo-style) used by m4 models (e.g. vmistral / vllama3) to compress a
# variable-length sequence of vision embeddings into a fixed number of learned latents. Each layer lets the
# latents attend (multi-head, optional grouped-query KV heads, optional q/k RMSNorm) over the concatenation
# [context; latents], followed by a gated MLP, both with pre-RMSNorm and residuals. Supports eager attention
# and FlashAttention-2 (flash_attn imported only when available). Expects a config with hidden_size,
# rms_norm_eps, optional _flash_attn_2_enabled, and perceiver_config with resampler_n_heads,
# resampler_head_dim, num_key_value_heads, qk_layer_norms_perceiver, attention_dropout, hidden_act,
# resampler_n_latents, resampler_depth.
"""
perceiver.py
Generic interface to various configurations of the Perceiver Resampler, that simply takes in a series of (potentially
time-indexed) contextual embeddings, and "resamples" (compresses) them down to a pre-specified number of latents!
Note that the Perceiver in general resamples based solely off the *long-range* context; there's a nice opportunity here
to prime the Perceiver Resampler with say a single layer's worth of language embeddings (the target domain), and use
that to softly "retrieve & compress" what we need --> this would be a novel contribution we should explore.
References:
    - DeepMind's Flamingo: https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model
    - Code borrowed w/ love from: https://github.com/lucidrains/flamingo-pytorch
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.utils import is_flash_attn_2_available

from m4.models.common import MLP, RMSNorm
from m4.utils import logging


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    # [FUNCTION DOC] HELPER: from a (batch, seq) 0/1 attention mask return (flat indices of non-pad tokens,
    # int32 cumulative sequence lengths with a leading 0, max sequence length as python int), as required by
    # flash_attn varlen kernels.
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: expand key/value heads for grouped-query attention: (batch, kv_heads, seq,
    # head_dim) -> (batch, kv_heads * n_rep, seq, head_dim), each head repeated n_rep times consecutively;
    # returned unchanged if n_rep == 1.
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PerceiverAttention(nn.Module):
    # [CLASS DOC] Eager (matmul + softmax) non-causal cross-attention from latents (queries) to [context;
    # latents] (keys/values). Bias-free q/k/v/o projections; keys/values use num_key_value_heads heads,
    # repeated to num_heads.
    def __init__(self, config) -> None:
        # [FUNCTION DOC] Read sizes from config (see module doc), build q_proj (hidden -> heads*head_dim),
        # k_proj/v_proj (hidden -> kv_heads*head_dim), o_proj (heads*head_dim -> hidden) and, if
        # qk_layer_norms_perceiver, per-head RMSNorms q_layer_norm/k_layer_norm. Sets is_causal False.
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.perceiver_config.resampler_n_heads
        self.head_dim = config.perceiver_config.resampler_head_dim
        self.num_key_value_heads = config.perceiver_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver
        self.attention_dropout = config.perceiver_config.attention_dropout

        if self.qk_layer_norms:
            self.q_layer_norm = RMSNorm(self.head_dim)
            self.k_layer_norm = RMSNorm(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # [FUNCTION DOC] Inputs: latents (bsz, n_latents, hidden), context (bsz, seq, hidden), attention_mask
        # optional additive float mask of shape (bsz, 1, n_latents, seq + n_latents), position_ids (unused; no
        # rotary embedding), past_key_value optional (k, v) tuple concatenated before the new keys/values,
        # output_attentions, use_cache. Projects queries from latents and keys/values from [context; latents],
        # optionally RMS-normalizes q and k (after caching), repeats kv heads, computes softmax(q k^T /
        # sqrt(head_dim) + mask) in fp32 then casts back, multiplies by v, merges heads and applies o_proj.
        # Raises ValueError on unexpected attention weight / mask / output shapes. Returns (output (bsz,
        # n_latents, hidden), attention weights or None, (k, v) cache or None). Attention dropout is not
        # applied in this eager path.
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!
        :param context: Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
        :param latents: Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
        :return: Tensor of shape [bsz, n_latents, embed_dim] representing attention over latents w/ cross from context.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        query_states = self.q_proj(latents).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(torch.cat([context, latents], dim=-2))
            .view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(torch.cat([context, latents], dim=-2))
            .view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PerceiverFlashAttention2(PerceiverAttention):
    # [CLASS DOC] FlashAttention-2 variant of PerceiverAttention with identical weights; handles padding via
    # unpadding and varlen kernels. Attention mask here is a 2D (bsz, seq + n_latents) 0/1 padding mask.
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Delegates to PerceiverAttention.__init__.
        super().__init__(*args, **kwargs)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # [FUNCTION DOC] Same inputs/outputs as PerceiverAttention.forward but attention_mask is a 2D padding
        # mask. Projects q from latents and k/v from [context; latents]; with a cache, optionally trims the
        # cache to config.sliding_window (raising ValueError if the trimmed cache length is not sliding_window
        # - 1) and adjusts the mask, then concatenates. Repeats kv heads, uses attention_dropout only in
        # training mode, casts fp32 inputs back to the pre-quantization or q_proj weight dtype (logging a one-
        # time warning), transposes to (bsz, seq, heads, dim) and calls _flash_attention_forward (non-causal,
        # no sliding window), then o_proj.
        # KNOWN ISSUE: q/k layer norms (qk_layer_norms_perceiver) are never applied in this path, so outputs
        # differ from the eager PerceiverAttention when that option is on.
        # KNOWN ISSUE: attn_weights is never assigned, so output_attentions=True raises UnboundLocalError.
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        # Query, Key, Value Projections --> Note that in Flamingo, latents are *concatenated* with context prior to attn!
        #   Note: This results in queries w/ `seq = n_latents`, and keys, values with `seq = len(context) + n_latents`
        query_states = self.q_proj(latents)
        key_states = self.k_proj(torch.cat([context, latents], dim=-2))
        value_states = self.v_proj(torch.cat([context, latents], dim=-2))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            if hasattr(self.config, "sliding_window") and kv_seq_len > self.config.sliding_window:
                slicing_tokens = kv_seq_len - self.config.sliding_window

                past_key = past_key_value[0]
                past_value = past_key_value[1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        "past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1,"
                        f" head_dim`), got {past_key.shape}"
                    )

                past_key_value = (past_key, past_value)

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to the fact"
                " you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=False,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        # [FUNCTION DOC] HELPER: run flash attention on (bsz, seq, heads, dim) tensors. If attention_mask is
        # given, unpad q/k/v via _upad_input, call flash_attn_varlen_func with cumulative lengths and re-pad
        # the output to (bsz, query_length, ...); otherwise call flash_attn_func directly. dropout is the
        # dropout probability, softmax_scale defaults to 1/sqrt(head_dim) inside flash_attn,
        # use_sliding_windows adds window_size=(sliding_window, sliding_window). Causality follows
        # self.is_causal (False). Returns the attention output (bsz, query_length, heads, dim).
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # [FUNCTION DOC] HELPER: unpad q/k/v for varlen flash attention. Keys/values are gathered with the
        # mask indices; queries are gathered the same way when query_length equals the kv length, treated as
        # one token per batch when query_length == 1, else unpadded using the last query_length mask columns
        # (assumes left padding). Returns (q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q,
        # max_seqlen_k)). For the resampler, query_length (n_latents) is normally smaller than the kv length,
        # so the last branch is used.
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class PerceiverLayer(nn.Module):
    # [CLASS DOC] One resampler block: latents = latents + attn(RMSNorm(latents), RMSNorm(context)); latents =
    # latents + MLP(RMSNorm(latents)) where MLP is the gated MLP with intermediate size 4 * hidden_size.
    # Chooses PerceiverFlashAttention2 when config._flash_attn_2_enabled is true, else PerceiverAttention.
    def __init__(self, config):
        # [FUNCTION DOC] Read hidden size, activation, n_latents, depth and rms eps from config; build
        # input_latents_norm, input_context_norm, self_attn, post_attention_layernorm and mlp.
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.rms_norm_eps
        self.intermediate_size = self.hidden_size * 4

        self.input_latents_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = (
            PerceiverAttention(config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else PerceiverFlashAttention2(config)
        )
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = MLP(
            activation=self.hidden_act,
            input_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            output_size=self.hidden_size,
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # [FUNCTION DOC] Inputs: latents (bsz, n_latents, hidden), context (bsz, seq, hidden), attention_mask
        # optional 2D (bsz, seq) padding mask over the context (defaults to all ones), plus position_ids /
        # past_key_value / output_attentions / use_cache. Extends the mask with ones for the latents, converts
        # it to a 4D additive mask (eager) or passes it as-is (flash), runs attention and MLP with residuals.
        # Returns a tuple (latents,) plus attention weights if output_attentions and the cache if use_cache.
        # KNOWN ISSUE: position_ids, past_key_value, output_attentions and use_cache are not forwarded to
        # self_attn, so the returned attention weights and cache are always None.
        # KNOWN ISSUE: reads config._flash_attn_2_enabled directly (AttributeError if the attribute is absent)
        # while __init__ uses getattr with a False default.
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        if attention_mask is None:
            attention_mask = torch.ones(
                size=(context.size(0), context.size(1)),
                dtype=torch.bool,
                device=context.device,
            )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    size=(attention_mask.size(0), latents.size(1)),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=(
                _prepare_4d_attention_mask(attention_mask, latents.dtype, tgt_len=self.n_latents)
                if not self.config._flash_attn_2_enabled
                else attention_mask
            ),
        )
        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PerceiverResampler(nn.Module):
    # [CLASS DOC] Full resampler: a learned (n_latents, hidden) latent array (initialized to ones), broadcast
    # over the batch, refined by `resampler_depth` PerceiverLayers over the context, then a final RMSNorm.
    def __init__(
        self,
        config,
    ) -> None:
        # [FUNCTION DOC] Read sizes from config and create latents parameter, the ModuleList of PerceiverLayer
        # and final norm.
        # KNOWN ISSUE: the docstring documents embed_dim/depth/n_heads/head_dim/n_latents parameters, but the
        # constructor only takes a config.
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim].
        :param embed_dim: Dimensionality of embeddings being fed to the Perceiver Resampler (also dimensionality of
                          latent embeddings *returned* by the Perceiver Resampler. Could be e.g., VIT embed_dim, ResNet
                          pool dim, and so on.
        :param depth: Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
        :param n_heads: Number of heads in each Transformer block (for multi-headed self-attention).
        :param head_dim: Dimensionality of each head projection in the Transformer block.
        :param n_latents: Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.rms_norm_eps

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.ones(self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        self.layers = nn.ModuleList([PerceiverLayer(config) for _ in range(self.depth)])
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Inputs: context (bsz, seq, hidden) embeddings and optional (bsz, seq) padding mask.
        # Repeats latents to the batch size, runs each layer (no cache, no attention outputs) and returns
        # normalized latents (bsz, n_latents, hidden).
        latents = repeat(self.latents, "seq embed -> bsz seq embed", bsz=context.shape[0])

        for perceiver_layer in self.layers:
            layer_outputs = perceiver_layer(
                latents,
                context,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            latents = layer_outputs[0]

        latents = self.norm(latents)

        return latents
