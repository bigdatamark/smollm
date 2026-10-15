# [MODULE DOC] Shared building blocks for m4 vision-language models: generation helpers (input expansion for
# beam search / multiple return sequences, per-step kwarg update, input preparation that requires
# pixel_values) and small torch modules (gated MLP, single linear projection MLP, RMSNorm). Pure torch; no
# I/O.
import torch
from torch import nn
from transformers.activations import ACT2FN


def expand_inputs_for_generation(
    input_ids,
    expand_size=1,
    is_encoder_decoder=False,
    attention_mask=None,
    encoder_outputs=None,
    **model_kwargs,
):
    # [FUNCTION DOC] HELPER: repeat every batch row `expand_size` times (row-interleaved, e.g. [a,a,b,b]) for
    # beam search / num_return_sequences generation. Inputs: input_ids (batch x seq LongTensor); expand_size
    # int (default 1); is_encoder_decoder bool (default False); attention_mask optional tensor;
    # encoder_outputs optional model output; **model_kwargs extra generation kwargs. Builds an index vector
    # [0..0,1..1,...] on input_ids' device and index_selects along dim 0 for input_ids and, when present,
    # model_kwargs["token_type_ids"]. When attention_mask is given it is expanded into
    # model_kwargs["attention_mask"] AND model_kwargs["pixel_values"] is expanded too (KeyError if
    # pixel_values is absent). For encoder-decoder, expands encoder_outputs.last_hidden_state in place and
    # stores it in model_kwargs["encoder_outputs"]; raises ValueError if encoder_outputs is None. Returns
    # (expanded input_ids, mutated model_kwargs).
    # KNOWN ISSUE: pixel_values is only expanded when attention_mask is not None; without a mask images are
    # not repeated and batch sizes mismatch.
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        model_kwargs["pixel_values"] = model_kwargs["pixel_values"].index_select(0, expanded_return_idx)

    if is_encoder_decoder:
        if encoder_outputs is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
    return input_ids, model_kwargs


def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
    # [FUNCTION DOC] HELPER: update generation kwargs after one decoding step. Inputs: outputs (model output
    # mapping), model_kwargs (dict, mutated in place), is_encoder_decoder (default False). Ensures key
    # "past_key_values" exists (default None); sets model_kwargs["past"] to the first available of
    # outputs.past_key_values / mems / past_buckets_states, else None. Appends a copy of the last
    # token_type_id column to token_type_ids if present, and for decoder-only models appends a column of ones
    # to attention_mask. Returns model_kwargs.
    # KNOWN ISSUE: the cache is stored under "past" while "past_key_values" is left at its previous value, so
    # callers reading past_key_values never see the updated cache (legacy transformers API).
    # must have this key set to at least None
    model_kwargs["past_key_values"] = model_kwargs.get("past_key_values", None)

    # update past
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs.past_key_values
    elif "mems" in outputs:
        model_kwargs["past"] = outputs.mems
    elif "past_buckets_states" in outputs:
        model_kwargs["past"] = outputs.past_buckets_states
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention masks
    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return model_kwargs


def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    # [FUNCTION DOC] HELPER: build the model forward kwargs for one generation step. Inputs: input_ids (batch
    # x seq), past (cached key/values or None), **kwargs (token_type_ids, attention_mask, position_ids,
    # use_cache, pixel_values). If past is truthy, keeps only the last token of input_ids and token_type_ids.
    # If attention_mask is given without position_ids, derives position_ids as cumulative sum of the mask
    # minus 1 with padded positions set to 1 (last column only when past). Raises ValueError if pixel_values
    # is missing. Returns a dict with input_ids, past_key_values, use_cache, position_ids, attention_mask,
    # token_type_ids, pixel_values.
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    pixel_values = kwargs.get("pixel_values", None)
    if pixel_values is None:
        raise ValueError("pixel values cannot be None")

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
    }


class MLP(nn.Module):
    # [CLASS DOC] Gated (SwiGLU-style) feed-forward block: output = down_proj(act(gate_proj(x)) * up_proj(x));
    # all three linear layers have no bias.
    def __init__(self, activation, input_size, intermediate_size, output_size):
        # [FUNCTION DOC] Store sizes and build gate_proj and up_proj (input_size -> intermediate_size) and
        # down_proj (intermediate_size -> output_size), all bias-free; activation (str) is looked up in
        # transformers ACT2FN (KeyError for unknown names).
        super().__init__()
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size

        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[activation]

    def forward(self, x):
        # [FUNCTION DOC] Apply the gated MLP to x (..., input_size) and return (..., output_size).
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SimpleMLP(nn.Module):
    # [CLASS DOC] Single bias-free linear projection input_size -> output_size, used where an MLP slot needs
    # no hidden layer.
    def __init__(self, input_size, output_size):
        # [FUNCTION DOC] Store sizes and create the bias-free nn.Linear `proj`.
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        # [FUNCTION DOC] Return proj(x).
        return self.proj(x)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm
class RMSNorm(nn.Module):
    # [CLASS DOC] Root-mean-square layer norm (copy of LlamaRMSNorm): scales by 1/sqrt(mean(x^2)+eps) over the
    # last dim and multiplies by a learned per-feature weight; no mean centering, no bias.
    def __init__(self, hidden_size, eps=1e-6):
        # [FUNCTION DOC] hidden_size: size of last dim; eps (default 1e-6) stored as variance_epsilon. Weight
        # initialized to ones.
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # [FUNCTION DOC] Compute in float32: x * rsqrt(mean(x^2, last dim) + eps), cast back to the input
        # dtype, then multiply by weight (result dtype follows weight/input promotion).
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
