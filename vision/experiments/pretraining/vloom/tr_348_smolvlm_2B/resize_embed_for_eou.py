# [MODULE DOC] Checkpoint surgery script used before resuming a SmolVLM training stage: pads the *additional* (newly added
# tokens) embedding matrix and the matching additional output projection (additional_fc) of a decoupled-embedding
# model so their row counts become a multiple of 128. All four tensors live in unwrapped_model/model.safetensors,
# which is modified in place. Usage: python resize_embed_for_eou.py <opt_step_dir>.
# KNOWN ISSUE: despite the name ("for EOU" token), no new row is added when the additional matrices are already a
# multiple of 128; the script only pads. The json import and commented index-editing code are unused.
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
    # [FUNCTION DOC] Entry point: parse args, then for (model.embed_tokens.weight, model.embed_tokens.additional_embedding.weight)
    # and (lm_head.weight, additional_fc.weight) pad the additional matrix with pad_to_multiple_of_128 (using the
    # main matrix for the mean init), save it back in place and print the resulting row count. Returns None.
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

    def pad_to_multiple_of_128(tensor, embedding_matrix):
        # [FUNCTION DOC] HELPER: append rows to tensor (n x hidden) until n is the next multiple of 128 (unchanged if already a
        # multiple). Every appended row is the same vector: the mean over all rows of embedding_matrix and tensor.
        current_size = tensor.size(0)
        target_size = ((current_size + 127) // 128) * 128  # Round up to next multiple of 128
        padding_needed = target_size - current_size
        
        if padding_needed == 0:
            return tensor
            
        # Create padding rows by averaging existing embeddings
        padding = init_new(embedding_matrix, tensor).repeat(padding_needed, 1)
        return torch.vstack([tensor, padding])

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
    embed_additional = pad_to_multiple_of_128(embed_additional, embed)
    save_modified("model.embed_tokens.additional_embedding.weight", embed_additional)
    print(f"Resized embedding matrix to {embed_additional.size(0)} rows")

    lm_head = load("lm_head.weight")
    lm_head_additional = load("additional_fc.weight")
    lm_head_additional = pad_to_multiple_of_128(lm_head_additional, lm_head)
    save_modified("additional_fc.weight", lm_head_additional)
    print(f"Resized lm head matrix to {lm_head_additional.size(0)} rows")



if __name__ == "__main__":
    main()
