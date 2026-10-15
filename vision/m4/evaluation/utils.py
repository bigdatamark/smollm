# [MODULE DOC] Evaluation helpers: model loading (by task model_class or from a checkpoint's config), prompt-template id
# selection, batch splitting, and the EvaluationVersion enum.
import json
import logging
import os
from enum import Enum
from pathlib import Path

import transformers
from deepspeed.utils.z3_leaf_module import set_z3_leaf_modules

from m4.models.__init__ import model_type_to_modeling_class
from m4.models.idefics import modeling_idefics
from m4.models.vgpt2 import modeling_vgpt2
from m4.models.vllama3 import modeling_vllama3
from m4.models.vmistral import modeling_vmistral
from m4.training.setup_language_model import model_name_to_classes


logger = logging.getLogger(__name__)


def get_prompt_template_id(args, task):
    # [FUNCTION DOC] Return task.selected_prompt_template_id if args.hparams.use_selected_prompt_template_ids, else
    # args.tasks.prompt_template_id.
    if args.hparams.use_selected_prompt_template_ids:
        prompt_template_id = task.selected_prompt_template_id
    else:
        prompt_template_id = args.tasks.prompt_template_id
    return prompt_template_id


def get_model(args, task):
    # [FUNCTION DOC] Load the model for a task: picks the module among m4's vgpt2/idefics/vmistral/vllama3 modeling modules whose
    # key is a substring of task.model_class (lower-cased), falling back to transformers; gets the class named
    # task.model_class from it and calls from_pretrained(args.tasks.model_name) with dtype
    # args.tasks.model_precision.value and token from env HF_TOKEN (or the cached login). Logs, sets eval mode and
    # returns the model.
    model_name = args.tasks.model_name

    supported_custom_modules = {
        "vgpt2": modeling_vgpt2,
        "idefics": modeling_idefics,
        "vmistral": modeling_vmistral,
        "vllama3": modeling_vllama3,
    }
    parent_class = (
        [v for k, v in supported_custom_modules.items() if k in task.model_class.lower()] + [transformers]
    )[0]
    model_class = getattr(parent_class, task.model_class)
    model = model_class.from_pretrained(
        model_name, torch_dtype=args.tasks.model_precision.value, token=os.getenv("HF_TOKEN", True)
    )
    logger.info(f"Loaded model {model.__class__.__name__} from checkpoint {model_name}")
    model.eval()
    return model


def get_model_from_config_file(args, is_deepspeed=False):
    # [FUNCTION DOC] Load the model for evaluation from args.tasks.model_name. If <model_name>/config.json exists, the class is
    # looked up from its model_type in m4.models.model_type_to_modeling_class (None if unknown -> AttributeError
    # later); otherwise via model_name_to_classes(model_name). Loads with the requested dtype and HF_TOKEN. If a
    # sibling "<path before /unwrapped_model>/unwrapped_adapter" folder exists, loads and enables that (PEFT)
    # adapter. Puts the model in eval mode; if is_deepspeed, marks the model class as a ZeRO-3 leaf module.
    # Returns the model.
    model_name = args.tasks.model_name
    if os.path.exists(f"{model_name}/config.json"):
        with open(f"{model_name}/config.json", "r") as f:
            model_config = json.loads(f.read())
            model_class = model_type_to_modeling_class.get(model_config["model_type"], None)
    else:
        _, model_class = model_name_to_classes(model_name)
    model = model_class.from_pretrained(
        model_name, torch_dtype=args.tasks.model_precision.value, token=os.getenv("HF_TOKEN", True)
    )
    adapter_folder = Path(str(model_name).split("/unwrapped_model")[0] + "/unwrapped_adapter")
    if adapter_folder.exists():
        model.load_adapter(adapter_folder)
        model.enable_adapters()
        logger.info("Loaded adapter for model")
    logger.info(f"Loaded model {model.__class__.__name__} from checkpoint {model_name}")
    model.eval()
    if is_deepspeed:
        set_z3_leaf_modules(model, [model_class])

    return model


def split_batch(batch, chunk_size):
    # [FUNCTION DOC] HELPER: generator splitting a column-dict batch (mapping key -> list/sequence of equal length) into
    # consecutive sub-batches of at most chunk_size rows, each a dict with the same keys.
    keys = list(batch.keys())
    for i in range(0, len(batch[keys[0]]), chunk_size):
        yield {key: batch[key][i : i + chunk_size] for key in keys}


class EvaluationVersion(Enum):
    # [CLASS DOC] Enum of evaluation protocol versions (v1, v2), selected via TaskParams.evaluation_version.
    v1 = "v1"
    v2 = "v2"
