# [MODULE DOC] Helpers to load a pretrained vision backbone via transformers AutoModel and extract the vision
# tower.
# vision_model_name2model maps a regex (clip, siglip, vit) to a function that extracts the vision module from
# the loaded model (model.vision_model for clip/siglip, the model itself for vit).
import re

from transformers import AutoModel


# map to check the supported cv archs and also how to extract the model - in some arch, we want to
# go through a specific prefix to get to the model as in `model.vision_model` for clip
vision_model_name2model = {
    r"clip": lambda model: model.vision_model,
    r"siglip": lambda model: model.vision_model,
    r"vit": lambda model: model,
}


def vision_model_name_to_model(model_name_or_path, model):
    # [FUNCTION DOC] Return the vision sub-module of model selected by the first regex matching the
    # lower-cased
    # model_name_or_path. Raises ValueError (not an assertion as the docstring says) if no regex matches.
    """returns the model if supported, asserts otherwise"""

    model_name_lowcase = model_name_or_path.lower()
    for rx, lookup in vision_model_name2model.items():
        if re.search(rx, model_name_lowcase):
            return lookup(model)
    else:
        raise ValueError(
            f"Unknown type of backbone vision model. Got {model_name_or_path}, supported regexes:"
            f" {list(vision_model_name2model.keys())}."
        )


def get_vision_model(config):
    # [FUNCTION DOC] Load config.vision_model_name with AutoModel.from_pretrained (trust_remote_code=True),
    # passing the kwargs
    # obtained by evaluating the string config.vision_model_params with eval, and return its vision tower.
    # Downloads/loads weights.
    # KNOWN ISSUE: uses eval on a config string (arbitrary code execution risk).
    vision_model_name = config.vision_model_name
    vision_model_params = eval(config.vision_model_params)

    model = AutoModel.from_pretrained(vision_model_name, **vision_model_params, trust_remote_code=True)
    return vision_model_name_to_model(vision_model_name, model)
