# [MODULE DOC] Script that annotates a Hugging Face dataset shard with educational-quality scores from a
# sequence-classification (regression) model such as the FineWeb-Edu classifier, then pushes the scored
# shard to the Hub. CLI args: --model_name, --dataset_name, --dataset_config, --output_dataset_name,
# --output_dataset_config, --text_column (default "text"), --shard and --num_shards (both required).
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def main(args):
    # [FUNCTION DOC] Entry point. Loads tokenizer and model (bfloat16) onto CUDA if available else CPU;
    # loads the "train"
    # split of the dataset (cache dir hard-coded to /scratch/cosmo/cache/, 12 processes); keeps only rows
    # whose
    # index modulo num_shards equals shard; maps compute_scores over batches of 512; then pushes to the Hub as
    # private dataset `output_dataset_name` with config "<output_dataset_config>_<shard>" (max shard 4096MB).
    # Side effects: network/GPU use; the push is retried forever on any exception (error printed each time).
    # KNOWN ISSUE: the push retry loop has no limit or backoff, so a persistent error loops infinitely.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        cache_dir="/scratch/cosmo/cache/",
        num_proc=12,
    )
    dataset = dataset.filter(
        lambda x, i: i % args.num_shards == args.shard, with_indices=True, num_proc=12
    )

    def compute_scores(batch):
        # [FUNCTION DOC] Batched map function: tokenizes batch[text_column] (longest padding, truncation)
        # onto the device, runs
        # the model without gradients and takes the squeezed float logits. Adds "score" (raw float list) and
        # "int_score" (each score clamped to [0, 5] and rounded to int) to the batch and returns it.
        inputs = tokenizer(
            batch[args.text_column],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()

        batch["score"] = logits.tolist()
        batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
        return batch

    dataset = dataset.map(compute_scores, batched=True, batch_size=512)

    while True:
        try:
            config_name = f"{args.output_dataset_config}_{args.shard}"
            dataset.push_to_hub(
                args.output_dataset_name,
                config_name=config_name,
                private=True,
                max_shard_size="4096MB",
            )
            break
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="HuggingFaceFW/fineweb-edu-classifier"
    )
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument(
        "--output_dataset_name", type=str, default="HuggingFaceFW/fineweb-edu"
    )
    parser.add_argument("--output_dataset_config", type=str, default="default")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)

    args = parser.parse_args()
    main(args)
