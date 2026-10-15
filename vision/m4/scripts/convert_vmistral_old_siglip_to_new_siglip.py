# [MODULE DOC] CLI script migrating a VMistral checkpoint built with the old nested SigLIP wrapper to the new
# SigLIP layout: loads <checkpoint_path>/unwrapped_model, builds a fresh model from its config with
# vision_config.vision_model_name set to --new_siglip_model_path, copies weights after renaming
# "vision_model.vision_model" key fragments to "vision_model", optionally saves to --new_checkpoint_path and
# reloads the result to print it.
# KNOWN ISSUE: --save is declared type=str with default True, so any value passed (even "False") is truthy;
# help strings of the arguments are copy-pasted and wrong.
import argparse

from m4.models.vmistral.configuration_vmistral import VMistralConfig
from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM


KEYS_TO_MODIFY_MAPPING = {
    "vision_model.vision_model": "vision_model",
}


def rename_state_dict(state_dict):
    # [FUNCTION DOC] HELPER: returns a new dict with the same values where every key has each
    # KEYS_TO_MODIFY_MAPPING substring replaced by its target.
    model_state_dict = {}

    for key, value in state_dict.items():
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        model_state_dict[key] = value

    return model_state_dict


def save_model_and_config(new_checkpoint_path, new_model):
    # [FUNCTION DOC] HELPER: save_pretrained new_model (weights and config) into new_checkpoint_path.
    new_model.save_pretrained(new_checkpoint_path)


def check_loaded_model(new_checkpoint_path):
    # [FUNCTION DOC] HELPER: reloads VMistralForCausalLM from new_checkpoint_path (is_resume=False,
    # new_model=False) and prints it; raises if the directory was not saved.
    new_model = VMistralForCausalLM.from_pretrained(
        new_checkpoint_path, is_resume=False, new_model=False, trust_remote_code=True
    )
    print(f"model: {new_model}")


def convert_checkpoint(checkpoint_path, new_checkpoint_path, new_siglip_model_path, save=False):
    # [FUNCTION DOC] Performs the migration described in the module doc. checkpoint_path is a string opt_step
    # directory, save (default False) controls writing to new_checkpoint_path. load_state_dict is strict, so
    # any unmatched key raises. Always calls check_loaded_model, which fails when save is false and nothing
    # exists at the target.
    model_name = checkpoint_path + "/unwrapped_model"
    model = VMistralForCausalLM.from_pretrained(model_name, is_resume=False, new_model=False, trust_remote_code=True)

    config_path = checkpoint_path + "/unwrapped_model/config.json"
    new_config = VMistralConfig.from_pretrained(
        config_path,
        new_model=False,
    )
    new_config.vision_config.vision_model_name = new_siglip_model_path
    new_model = VMistralForCausalLM.from_config(new_config)

    state_dict = rename_state_dict(model.state_dict())
    new_model.load_state_dict(state_dict)

    if save:
        save_model_and_config(new_checkpoint_path, new_model)

    check_loaded_model(new_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/temp_266_dir/opt_step-38000"
        ),
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument("--new_checkpoint_path", default="new_model", type=str, help="Path to fairseq checkpoint")
    parser.add_argument(
        "--new_siglip_model_path",
        default="/fsx/leo/repos/siglip-so400m-14-384-flash-attn2",
        type=str,
        help="Path to new siglip model",
    )
    parser.add_argument("--save", default=True, type=str, help="Path to image")

    args = parser.parse_args()

    convert_checkpoint(args.checkpoint_path, args.new_checkpoint_path, args.new_siglip_model_path, args.save)
