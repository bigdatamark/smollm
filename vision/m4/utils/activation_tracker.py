# [MODULE DOC] Activation debugging for m4 training (adapted from transformers' debug_utils): forward hooks
# record
# per-module statistics of inputs/outputs (via m4.training.utils.get_stats) into JSONL-able dicts and abort
# training when nan/inf activations appear.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

"""
Adapted from https://github.com/huggingface/transformers/blob/f93c90d21749b61bd89152a7fe99a839df29ed94/src/transformers/debug_utils.py
"""

import json

from transformers.utils import ExplicitEnum, is_torch_available, logging

from m4.training.utils import get_stats


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class ActivationTracker:
    # [CLASS DOC] Attaches a forward hook to every submodule of a model. While tracing is on and not in
    # validation mode,
    # each hook call appends one dict {"name", "type", <stats per input/output tensor>} to self.jsonl_stats.
    # Tracks batch count (incremented when the top-level model's hook fires) and can abort after N batches.
    # Callers toggle tracing with activate_hooks/deactivate_hooks, tag records with fill_in_batch_idx, write
    # them with dump_stats and clear them with reset_jsonl_stats.
    """
    This debug class helps detect and understand where the model starts getting very large or very small, and more
    importantly `nan` or `inf` activation elements.

    This class will plug hooks into the model and record the activation values of the model into a list of dictionaries: `jsonl_stats`.

    Recording is only active during training, not during validation, and when `trace_activation` is set to True.
    In practise, since this tracking requires additional computation, we only track activations every X steps.

    In the case of gradient accumulation, all the batches being accumulated are being recorded and identified by the `batch_idx` key.

    Args:
        model (`nn.Module`):
            The model to debug.
        abort_after_batch_num  (`int``, *optional*):
            Whether to abort after a certain batch number has finished
    """

    def __init__(
        self,
        model,
        abort_after_batch_num=None,
    ):
        # [FUNCTION DOC] Args: model (nn.Module to track), abort_after_batch_num (int or None; raise once
        # batch_number exceeds it).
        # Initializes empty stats, batch_number 0, detected_overflow False, builds the module-name map and
        # registers
        # forward hooks on all modules (side effect: mutates the model's hook registry).
        # KNOWN ISSUE: self.trace_activation is not initialized here; a forward pass before activate_hooks()
        # or
        # deactivate_hooks() is called raises AttributeError in forward_hook.
        self.model = model
        self.is_validation = False
        self.abort_after_batch_num = abort_after_batch_num

        self.jsonl_stats = []
        self.batch_number = 0
        self.detected_overflow = False
        self.analyse_model()

        self.register_forward_hook()

    def analyse_model(self):
        # extract the fully qualified module names, to be able to report at run time. e.g.:
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # for shared weights only the first shared module name will be registered
        # [FUNCTION DOC] HELPER: builds self.module_names, a mapping from module object to its fully qualified
        # name from
        # named_modules().
        # KNOWN ISSUE: the comment says only the first name of shared modules is registered, but the dict
        # comprehension keeps the last name encountered.
        self.module_names = {m: name for name, m in self.model.named_modules()}

    def analyse_variable(self, var, ctx, current_module_stats):
        # [FUNCTION DOC] HELPER: if var is a tensor, merges get_stats(var, ctx) into current_module_stats
        # (mutated in place and
        # returned) and sets self.detected_overflow when detect_overflow finds nan/inf. Non-tensors are
        # ignored.
        if torch.is_tensor(var):
            dict_stats = get_stats(var, ctx)
            current_module_stats.update(dict_stats)
            # self.expand_frame(text_stats)
            if detect_overflow(var, ctx):
                self.detected_overflow = True
        return current_module_stats

    def create_frame(self, module, input, output):
        # [FUNCTION DOC] HELPER: collects stats for a module call. Inputs: each element of a tuple input
        # labelled "input[i]" (or
        # "input"); outputs labelled "output[i]", "output[i][j]" for nested tuples, or "output". If any stats
        # were
        # gathered and no record with the same (name, type) already exists in jsonl_stats (dedup for gradient
        # checkpointing re-running forwards), appends {"name", "type", **stats}.
        # Note: the dedup also drops repeated calls of the same module within the window (e.g. shared modules
        # or
        # multiple accumulated batches before reset), which may lose records.
        module_name = f"{self.module_names[module]}"
        module_type = f"{module.__class__.__name__}"
        current_module_stats = {}

        # inputs
        if isinstance(input, tuple):
            for i, x in enumerate(input):
                current_module_stats = self.analyse_variable(x, f"input[{i}]", current_module_stats)
        else:
            current_module_stats = self.analyse_variable(input, "input", current_module_stats)

        # outputs
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # possibly a tuple of tuples
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        current_module_stats = self.analyse_variable(y, f"output[{i}][{j}]", current_module_stats)
                else:
                    current_module_stats = self.analyse_variable(x, f"output[{i}]", current_module_stats)
        else:
            current_module_stats = self.analyse_variable(output, "output", current_module_stats)
        if current_module_stats:
            # When we activate gradient checkpointing, the forward hook will be called twice for some (not all) modules.
            # That will lead to double (repeated) entries in the list.
            # This is a hack to avoid these double entries.
            if (module_name, module_type) not in [(x["name"], x["type"]) for x in self.jsonl_stats]:
                self.jsonl_stats.append(
                    {
                        "name": module_name,
                        "type": module_type,
                        **current_module_stats,
                    }
                )

    def register_forward_hook(self):
        # [FUNCTION DOC] Applies _register_forward_hook to the model and all submodules recursively.
        self.model.apply(self._register_forward_hook)

    def _register_forward_hook(self, module):
        # [FUNCTION DOC] HELPER: registers self.forward_hook as a forward hook on a single module.
        module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        # - input is a tuple of packed inputs (could be non-Tensors)
        # - output could be a Tensor or a tuple of Tensors and non-Tensors

        # [FUNCTION DOC] Forward hook (module, input, output). Increments batch_number when module is the root
        # model (the root's
        # hook fires last in a forward). Records a frame when tracing and not validating. Raises ValueError if
        # an
        # overflow was detected, or if abort_after_batch_num is set and batch_number exceeds it.
        trace_activation = self.trace_activation

        # count batch numbers - the very first forward hook of the batch will be called when the
        # batch completes - i.e. it gets called very last - we know this batch has finished
        if module == self.model:
            self.batch_number += 1

        if trace_activation and not self.is_validation:
            self.create_frame(module, input, output)

        if self.detected_overflow:
            # now we can abort, as it's pointless to continue running
            raise ValueError(
                "DebugUnderflowOverflow: inf/nan detected, aborting as there is no point running further. "
                "Please scroll up above this traceback to see the activation values prior to this event."
            )

        # abort after certain batch if requested to do so
        if self.abort_after_batch_num is not None and self.batch_number > self.abort_after_batch_num:
            raise ValueError(
                f"DebugUnderflowOverflow: aborting after {self.batch_number} batches due to"
                f" `abort_after_batch_num={self.abort_after_batch_num}` arg"
            )

    def fill_in_batch_idx(self, batch_idx):
        # [FUNCTION DOC] Assigns batch_idx to every record lacking "batch_idx"; raises ValueError if an
        # existing record has a
        # larger batch_idx (indices must be non-decreasing). No-op on empty stats.
        if not self.jsonl_stats:
            return
        for r in self.jsonl_stats:
            if "batch_idx" not in r:
                r["batch_idx"] = batch_idx
            else:
                if not (r["batch_idx"] <= batch_idx):
                    raise ValueError("`batch_idx` should be increasing")

    def dump_stats(self, log_activations_filename, curr_opt_step):
        # [FUNCTION DOC] Appends every record, after setting its "step" key to curr_opt_step, as one JSON line
        # to the file
        # log_activations_filename (opened in append mode).
        with open(log_activations_filename, "a") as file:
            # append stats to file
            for r in self.jsonl_stats:
                r["step"] = curr_opt_step
                file.write(json.dumps(r) + "\n")

    def reset_jsonl_stats(self):
        # [FUNCTION DOC] Clears the recorded stats list.
        self.jsonl_stats = []

    def activate_hooks(self):
        # [FUNCTION DOC] Turns tracing on (trace_activation = True).
        self.trace_activation = True

    def deactivate_hooks(self):
        # [FUNCTION DOC] Turns tracing off (trace_activation = False).
        self.trace_activation = False

    def is_eval(self):
        # [FUNCTION DOC] Marks validation mode; frames are not recorded while set.
        self.is_validation = True

    def is_train(self):
        # [FUNCTION DOC] Marks training mode; frames are recorded when tracing is active.
        self.is_validation = False


def detect_overflow(var, ctx):
    # [FUNCTION DOC] Returns True if tensor var contains any nan or inf, printing "<ctx> has nans"/"<ctx> has
    # infs" for each.
    # Contains disabled (if 0) blocks for printing large-magnitude counts and min/max/var/mean.
    """
    Report whether the tensor contains any `nan` or `inf` entries.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the tensor in question.

    This function contains a few other helper features that you can enable and tweak directly if you want to track
    various other things.

    Args:
        var: the tensor variable to check
        ctx: the message to print as a context

    Return:
        `True` if `inf` or `nan` was detected, `False` otherwise
    """
    detected = False
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    if torch.isinf(var).any().item():
        detected = True
        print(f"{ctx} has infs")

    # if needed to monitor large elements can enable the following
    if 0:  # and detected:
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f"{ctx}:  n100={n100.numel()}")
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f"{ctx}: n1000={n1000.numel()}")
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f"{ctx}: n10000={n10000.numel()}")

    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e}")

    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})")

    return detected


class DebugOption(ExplicitEnum):
    # [CLASS DOC] String enum of debug options: "underflow_overflow" and "tpu_metrics_debug".
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
