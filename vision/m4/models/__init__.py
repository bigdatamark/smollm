# [PACKAGE DOC] m4.models package: registry of the vision-language model families implemented in m4. Importing
# it imports every model family (vgpt2, idefics, vmistral, vllama3) and the decoupled embedding/linear
# modules, re-exporting DecoupledEmbedding, DecoupledLinear, IdeficsConfig, IdeficsForCausalLM, VGPT2Config,
# VGPT2LMHeadModel, VLlama3Config, VLlama3ForCausalLM, VMistralConfig, VMistralForCausalLM. Module-level
# mappings:
# - _SUPPORTED_MODELS: model_type string ("vgpt2", "idefics", "vmistral", "vllama3") -> config class.
# - model_type_to_modeling_class: same keys -> causal-LM modeling class. Used by training/eval code to pick
#   the classes from a config's model_type. The legacy "vllama" alias is commented out in both.
from m4.models.custom_modules import DecoupledEmbedding, DecoupledLinear
from m4.models.idefics.configuration_idefics import IdeficsConfig
from m4.models.idefics.modeling_idefics import IdeficsForCausalLM
from m4.models.vgpt2.configuration_vgpt2 import VGPT2Config
from m4.models.vgpt2.modeling_vgpt2 import VGPT2LMHeadModel
from m4.models.vllama3.configuration_vllama3 import VLlama3Config
from m4.models.vllama3.modeling_vllama3 import VLlama3ForCausalLM
from m4.models.vmistral.configuration_vmistral import VMistralConfig
from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM


_SUPPORTED_MODELS = {
    "vgpt2": VGPT2Config,
    # "vllama": IdeficsConfig,
    "idefics": IdeficsConfig,
    "vmistral": VMistralConfig,
    "vllama3": VLlama3Config,
}

model_type_to_modeling_class = {
    "vgpt2": VGPT2LMHeadModel,
    # "vllama": IdeficsForCausalLM,
    "idefics": IdeficsForCausalLM,
    "vmistral": VMistralForCausalLM,
    "vllama3": VLlama3ForCausalLM,
}
