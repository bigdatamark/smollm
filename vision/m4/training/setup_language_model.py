# [MODULE DOC] Maps a language-model backbone name to the m4 vision-language config/model classes.
# model_name2classes is
# an ordered mapping from regex (gpt2, idefics, mistral, llama, smollm) to [config class, model class]; the
# first regex found in the lower-cased name wins.
import re

from m4.models.idefics.configuration_idefics import IdeficsConfig
from m4.models.idefics.modeling_idefics import IdeficsForCausalLM
from m4.models.vgpt2.configuration_vgpt2 import VGPT2Config
from m4.models.vgpt2.modeling_vgpt2 import VGPT2LMHeadModel
from m4.models.vllama3.configuration_vllama3 import VLlama3Config
from m4.models.vllama3.modeling_vllama3 import VLlama3ForCausalLM
from m4.models.vmistral.configuration_vmistral import VMistralConfig
from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM


model_name2classes = {
    r"gpt2": [VGPT2Config, VGPT2LMHeadModel],
    r"idefics": [IdeficsConfig, IdeficsForCausalLM],
    r"mistral": [VMistralConfig, VMistralForCausalLM],
    r"llama": [VLlama3Config, VLlama3ForCausalLM],
    r"smollm": [VLlama3Config, VLlama3ForCausalLM],
}


def model_name_to_classes(model_name_or_path):
    # [FUNCTION DOC] Return the [config_class, model_class] pair for the first regex in model_name2classes
    # that matches
    # model_name_or_path (case-insensitive search). Raises ValueError listing supported regexes if none match.
    """returns config_class, model_class for a given model name or path"""

    model_name_lowcase = model_name_or_path.lower()
    for rx, classes in model_name2classes.items():
        if re.search(rx, model_name_lowcase):
            return classes
    else:
        raise ValueError(
            f"Unknown type of backbone LM. Got {model_name_or_path}, supported regexes:"
            f" {list(model_name2classes.keys())}."
        )
