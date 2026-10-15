# [MODULE DOC] CLI script merging a LoRA adapter into its base model. Reads
# opt_step_dir/unwrapped_model/config.json to pick the m4 modeling class (ValueError if unsupported), loads
# the base model in bf16 and the PEFT adapter from unwrapped_adapter, merges (safe_merge, progress bar) and
# saves the merged model to output_dir/unwrapped_model; optionally copies opt_step_dir/tokenizer to
# output_dir/tokenizer. Adds the repo root to sys.path.
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel


repodir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, repodir)

import m4.models
from m4.testing_utils import read_json_file


def get_args():
    # [FUNCTION DOC] HELPER: parses positional opt_step_dir, output_dir and save_tokenizer; returns the
    # namespace.
    # KNOWN ISSUE: save_tokenizer is a positional argument with action store_true, so it cannot be passed as a
    # flag and is effectively always set; copytree then fails if the output tokenizer dir already exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("opt_step_dir", type=str, help="opt_step_dir with unwrapped_model and unwrapped_adapter")
    parser.add_argument(
        "output_dir", type=str, help="output_dir where the opt_step_dir of the merged lora model will be saved "
    )
    parser.add_argument("save_tokenizer", action="store_true", help="whether to copy the tokenizer too")
    return parser.parse_args()


def main():
    # [FUNCTION DOC] Runs the merge described in the module doc. Creates the output directory; raises on load
    # or copy failures.
    args = get_args()

    source_dir = Path(args.opt_step_dir)
    output_dir = Path(args.output_dir)

    checkpoint_dir = source_dir / "unwrapped_model"
    adapter_dir = source_dir / "unwrapped_adapter"
    config_file_path = source_dir / "unwrapped_model" / "config.json"

    config = read_json_file(config_file_path)
    config_class = m4.models._SUPPORTED_MODELS.get(config["model_type"], None)
    if config_class is None:
        raise ValueError(f"{config['model_type']=} isn't supported by m4")
    modeling_class = m4.models.model_type_to_modeling_class.get(config["model_type"], None)
    model = modeling_class.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)

    merged_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    checkpoint_out_dir = f"{output_dir}/unwrapped_model"
    os.makedirs(checkpoint_out_dir, exist_ok=True)
    merged_model.save_pretrained(checkpoint_out_dir)

    if args.save_tokenizer:
        tokenizer_input_dir = source_dir / "tokenizer"
        tokenizer_out_dir = f"{output_dir}/tokenizer"
        shutil.copytree(tokenizer_input_dir, tokenizer_out_dir)


if __name__ == "__main__":
    main()
