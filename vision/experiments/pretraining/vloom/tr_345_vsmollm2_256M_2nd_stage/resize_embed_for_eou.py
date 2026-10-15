# [MODULE DOC] Checkpoint surgery script used before resuming a SmolVLM training stage: adds one new token row (e.g. for an
# end-of-utterance token) to the *additional* embedding matrix and to the additional output projection
# (additional_fc) of a decoupled-embedding model, initialised as the mean of all existing rows. Edits
# unwrapped_model/model.safetensors in place, so running it twice adds two rows.
# Usage: python resize_embed_for_eou.py <opt_step_dir>. The json import and commented index code are unused.
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_args():
    # [FUNCTION DOC] HELPER: parse the single positional CLI argument opt_step_dir (str): a training checkpoint directory that
    # contains an unwrapped_model/ subfolder with model.safetensors. Returns the argparse Namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument("opt_step_dir", type=str, help="opt_step_dir with unwrapped_model")
    return parser.parse_args()


def main():
    # [FUNCTION DOC] Entry point: parse args; append init_new(embed_tokens.weight, additional_embedding.weight) as one extra row
    # to model.embed_tokens.additional_embedding.weight and save; likewise append init_new(lm_head.weight,
    # additional_fc.weight) to additional_fc.weight and save; prints a message after each. Returns None.
    args = get_args()
    source_dir = Path(args.opt_step_dir) / "unwrapped_model"

    # with open(source_dir / "model.safetensors.index.json", "r") as f:
    #     index = json.load(f)

    # _ = index["metadata"].pop("total_size")
    # with open(source_dir / "model.safetensors.index.json", "w") as json_file:
    #     json.dump(index, json_file, indent=4)

    tensor_files_mapping = {}
    for k in [
        "additional_fc.weight",
        "lm_head.weight",
        "model.embed_tokens.additional_embedding.weight",
        "model.embed_tokens.weight",
    ]:
        tensor_files_mapping[k] = "model.safetensors"

    def load(weight_name):
        # [FUNCTION DOC] HELPER: read and return one tensor (by weight name) from the safetensors file mapped to it in source_dir.
        tensor_file = tensor_files_mapping[weight_name]
        with safe_open(source_dir / tensor_file, framework="pt") as f:
            return f.get_tensor(weight_name)

    def init_new(matrix, matrix_additional):
        # [FUNCTION DOC] HELPER: return a 1 x hidden row equal to the mean of all rows of matrix and matrix_additional stacked
        # vertically (used as the initialisation of new token rows).
        return torch.vstack([matrix, matrix_additional]).mean(dim=0, keepdim=True)

    def save_modified(weight_name, new_weight):
        # [FUNCTION DOC] HELPER: rewrite the whole safetensors file holding weight_name in place: load every tensor into memory,
        # replace weight_name with new_weight and save with metadata {"format": "pt"}. Overwrites the checkpoint.
        tensor_file = tensor_files_mapping[weight_name]
        all_tensors = {}
        with safe_open(source_dir / tensor_file, framework="pt") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
        all_tensors[weight_name] = new_weight
        save_file(all_tensors, source_dir / tensor_file, metadata={"format": "pt"})

    embed = load("model.embed_tokens.weight")
    embed_additional = load("model.embed_tokens.additional_embedding.weight")
    embed_additional = torch.vstack([embed_additional, init_new(embed, embed_additional)])
    save_modified("model.embed_tokens.additional_embedding.weight", embed_additional)
    print("Resized embedding matrix")

    lm_head = load("lm_head.weight")
    lm_head_additional = load("additional_fc.weight")
    lm_head_additional = torch.vstack([lm_head_additional, init_new(lm_head, lm_head_additional)])
    save_modified("additional_fc.weight", lm_head_additional)
    print("Resized lm head matrix")


if __name__ == "__main__":
    main()
