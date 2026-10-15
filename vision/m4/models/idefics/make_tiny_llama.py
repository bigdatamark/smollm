#!/usr/bin/env python
# [MODULE DOC] Script (run directly; all work happens at import time) that builds a tiny random-weight
# LlamaForCausalLM (vocab 32000, hidden 16, intermediate 64, 2 layers, 4 heads) for fast tests. Side effects:
# creates directory ./tiny-random-LlamaForCausalLM in the CWD, loads the tokenizer from the placeholder local
# path "path_to_llama_7b" (must be edited to a real LLaMA tokenizer path or it fails), runs one generate()
# smoke test on "This is a test", converts the model to fp16, saves model + tokenizer, reloads the model to
# verify, and prints an upload reminder.

# This script creates a super tiny model that is useful inside tests, when we just want to test that
# the machinery works, without needing to check the quality of the outcomes.
#
# usage: adjust the configs if wanted, but otherwise just run the script

from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


mname_tiny = "tiny-random-LlamaForCausalLM"

path = Path(mname_tiny)
path.mkdir(parents=True, exist_ok=True)

config = LlamaConfig()
config.update(
    dict(
        vocab_size=32000,
        hidden_size=16,
        intermediate_size=16 * 4,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
)
model = LlamaForCausalLM(config)
tokenizer = LlamaTokenizer.from_pretrained("path_to_llama_7b")

# Test w/ one text
query = "This is a test"
query_tokens = tokenizer(query, return_tensors="pt")

input = {
    "input_ids": query_tokens["input_ids"],
    "attention_mask": query_tokens["attention_mask"],
}

out_gen = model.generate(**input)
text = tokenizer.batch_decode(out_gen)

# Save model + config + tokenizer
model.half()  # makes it smaller
model.save_pretrained(path)
tokenizer.save_pretrained(path)

# test we can load it back
model = LlamaForCausalLM.from_pretrained(path)

print(f"Generated {mname_tiny} - Upload the generated folder to the hub")
