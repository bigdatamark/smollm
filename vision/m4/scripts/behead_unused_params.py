# [MODULE DOC] Script that edits a safetensors checkpoint in place (--model_dir, default
# /fsx/m4/victor/idefics2) to remove unused weights: with --behead_siglip_pooling drops every tensor whose key
# contains "vision_model.head"; with --behead_perceiver_rmsnorm drops keys containing "q_layer_norm" or
# "k_layer_norm". Each model*.safetensors shard is rewritten (pt metadata) only when tensors were skipped,
# then model.safetensors.index.json is rewritten without the dropped keys and without metadata.total_size.
# Prints every skipped/popped key.
# KNOWN ISSUE: shards are only rewritten when skipped > 1, so a shard with exactly one removed tensor keeps it
# while the index no longer references it.
"""
There is always the multi attention pooling head that we never use/train. remove that from the ultimate checkpoint. remove the unused qk layer norms in the perceiver too.
"""
import argparse
import glob
import json

from safetensors import safe_open
from safetensors.torch import save_file


def get_args():
    # [FUNCTION DOC] HELPER: parses --model_dir, --behead_siglip_pooling and --behead_perceiver_rmsnorm;
    # returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/fsx/m4/victor/idefics2",
        help="model_dir where weights and index.json are saved. operations are done in place.",
    )
    parser.add_argument("--behead_siglip_pooling", action="store_true", help="Behead the siglip pooling head")
    parser.add_argument(
        "--behead_perceiver_rmsnorm", action="store_true", help="Behead the q/k RMSNorms in the perceiver"
    )
    return parser.parse_args()


args = get_args()
model_path = args.model_dir

safetensor_files = glob.glob(f"{model_path}/model*.safetensors")
for file in safetensor_files:
    tensors = {}
    skipped = 0
    with safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if args.behead_siglip_pooling:
                if "vision_model.head" in key:
                    print("Skipping", key)
                    skipped += 1
                    continue
            if args.behead_perceiver_rmsnorm:
                if "q_layer_norm" in key or "k_layer_norm" in key:
                    print("Skipping", key)
                    skipped += 1
                    continue
            tensors[key] = f.get_tensor(key)
    if skipped > 1:
        save_file(tensors, file, metadata={"format": "pt"})
print("Finished saving the weights")

with open(f"{model_path}/model.safetensors.index.json", "r") as f:
    data = json.load(f)
    data["metadata"].pop("total_size", None)  # Doesn't seem to be really used so ditching it, perhaps i am wrong here
    keys_to_iterate = list(data["weight_map"].keys())
    for k in keys_to_iterate:
        if args.behead_siglip_pooling:
            if "vision_model.head" in k:
                data["weight_map"].pop(k)
                print("Popping", k)
        if args.behead_perceiver_rmsnorm:
            if "q_layer_norm" in k or "k_layer_norm" in k:
                data["weight_map"].pop(k)
                print("Popping", k)

with open(f"{model_path}/model.safetensors.index.json", "w") as f:
    json_object = json.dumps(data, indent=4)
    f.write(json_object)
print("Finished saving the weights mapping")
