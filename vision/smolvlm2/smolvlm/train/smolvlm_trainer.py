# [MODULE DOC] Custom Hugging Face Trainer for SmolVLM2 with per-component learning rates and simplified
# saving.
import os
import torch
import logging

import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import Trainer, PreTrainedModel
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

class SmolVLMTrainer(Trainer):
    # [CLASS DOC] transformers.Trainer subclass overriding create_optimizer (separate parameter groups for the
    # vision
    # encoder, the connector and the language model) and save_model.
    """
    A specialized Trainer that supports:
      - Distinct LR for vision tower vs. connector vs. LLM.
      - Save model logic that can handle large models or PEFT.
    """

    def create_optimizer(self):
        # [FUNCTION DOC] Builds (once) and returns self.optimizer. Trainable parameters are split by name:
        # containing
        # "vision_model" -> vision group (args.vision_tower_lr), "connector" -> connector group
        # (args.connector_lr), else language model (args.language_model_lr). Each component with lr > 0 yields
        # a
        # weight-decay group (non-bias, non-LayerNorm params, args.weight_decay) and a no-decay group. If no
        # groups result, falls back to all trainable params with args.learning_rate. Logs the groups and
        # instantiates
        # the optimizer class from Trainer.get_optimizer_cls_and_kwargs(args).
        # Note: a component with lr <= 0 is excluded from the optimizer but its params keep requires_grad;
        # tune_vision_tower is not consulted here.
        if self.optimizer is not None:
            return self.optimizer  # Already created

        # Deepspeed or SageMaker MP users can rely on parent's create_optimizer
        # (which then calls this if needed)

        model = self.model
        args = self.args

        # Collect param names that should receive weight decay
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        # Prepare param groups
        vision_params = []
        connector_params = []
        llm_params = []
        
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Decide group
            if "vision_model" in n:
                vision_params.append(n)
            elif "connector" in n:
                connector_params.append(n)
            else:
                llm_params.append(n)

        # We'll build up param groups based on user-defined LR
        # If e.g. vision_tower_lr=0 => we do not train the vision tower
        #   or you can skip the param group if LR=0
        def make_group(param_names, lr_value):
            # returns two subgroups: {decay: True}, {decay: False}
            # so that weight decay is only applied for non-bias,non-LN
            # [FUNCTION DOC] HELPER: returns [] if lr_value <= 0, else two param-group dicts (decay / no
            # decay) with that lr for the
            # named parameters. Membership tests on lists make this quadratic in parameter count.
            if lr_value <= 0:
                return []
            decay = {
                "params": [p for n, p in model.named_parameters() 
                           if n in param_names and n in decay_parameters],
                "weight_decay": args.weight_decay,
                "lr": lr_value,
            }
            no_decay = {
                "params": [p for n, p in model.named_parameters()
                           if n in param_names and n not in decay_parameters],
                "weight_decay": 0.0,
                "lr": lr_value,
            }
            return [decay, no_decay]

        groups = []
        groups += make_group(vision_params, args.vision_tower_lr)
        groups += make_group(connector_params, args.connector_lr)
        groups += make_group(llm_params, args.language_model_lr)

        # Fallback if no param groups are created (e.g. all lrs=0).
        if not groups:
            logger.warning("No param groups found. Possibly all LRs=0 or no requires_grad. "
                           "Falling back to default group.")
            groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "weight_decay": args.weight_decay,
                       "lr": args.learning_rate}]
            
        # Function to log details of each parameter group
        def log_param_groups(groups: List[Dict[str, Any]]):
            # [FUNCTION DOC] HELPER: logs number of params, weight decay and lr of each group ("unnamed_group"
            # since no names are set).
            logger.info("Parameter Groups Configuration:")
            for group in groups:
                group_name = group.get("name", "unnamed_group")
                num_params = len(group["params"])
                weight_decay = group.get("weight_decay", 0.0)
                lr = group.get("lr", 0.0)
                logger.info(
                    f"  - Group '{group_name}': "
                    f"Number of Params = {num_params}, "
                    f"Weight Decay = {weight_decay}, "
                    f"Learning Rate = {lr}"
                )
    
        # Log the parameter groups
        log_param_groups(groups)

        # Let HF parse the correct optimizer class
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False, state_dict=None):
        # [FUNCTION DOC] Saves to output_dir (default args.output_dir). With DeepSpeed delegates to
        # Trainer.save_model. Otherwise
        # uses the given state_dict or the model's state_dict and, if args.should_save, calls
        # model.save_pretrained(output_dir, state_dict=...). Tokenizer/processor and training args are not
        # saved.
        # KNOWN ISSUE: the fallback branch calls PreTrainedModel.unwrap_model, which does not exist (it is
        # only
        # reached for models without state_dict, so effectively dead code that would raise AttributeError).
        """
        Saves the model. Supports big models or PEFT seamlessly if not using DeepSpeed.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # If the user is using DeepSpeed, super().save_model handles the Zero partitions
        if self.is_deepspeed_enabled:
            super().save_model(output_dir, _internal_call=_internal_call)
            return

        # If we have state_dict, use it; else gather from self.model
        if state_dict is None:
            if hasattr(self.model, "state_dict"):
                state_dict = self.model.state_dict()
            else:
                # PEFT adapter has `get_base_model`, or it's a normal model
                state_dict = PreTrainedModel.unwrap_model(self.model).state_dict()

        # Let model handle the actual serialization
        if self.args.should_save:
            # typical structure: yourmodel.save_pretrained(output_dir, state_dict=state_dict)
            self.model.save_pretrained(output_dir, state_dict=state_dict)
