# [MODULE DOC] One-off script that cleans an evaluation results JSONL file in place (hardcoded local
# PATH_JSONL). For every record with a "score" field it parses a string score (single quotes replaced by
# double quotes, then JSON) into a mapping and removes the keys listed in BANNED_KEYS (distributions,
# BLEU/METEOR/ROUGE variants, per-bucket accuracies, etc.), then rewrites the whole file one JSON object per
# line.
import json


PATH_JSONL = "/Users/hugolaurencon/Desktop/tr_209_ift_mixture_test_final_evaluations.jsonl"


BANNED_KEYS = [
    "kl_distribution",
    "entropy_distribution",
    "kl_mean",
    "Bleu_1",
    "Bleu_1_all",
    "Bleu_2",
    "Bleu_2_all",
    "Bleu_3",
    "Bleu_3_all",
    "Bleu_4",
    "Bleu_4_all",
    "METEOR",
    "METEOR_all",
    "CIDEr_all",
    "ROUGE_L",
    "ROUGE_L_all",
    "per_bucket_accuracy",
    "std_per_bucket_accuracy",
    "entropy_mean",
]


jsonl_data = []
with open(PATH_JSONL, "r") as file:
    for line in file:
        json_data = json.loads(line)
        jsonl_data.append(json_data)


for idx, data in enumerate(jsonl_data):
    if "score" in data:
        if type(data["score"]) == str:
            data["score"] = json.loads(data["score"].replace("'", '"'))
        for banned_key in BANNED_KEYS:
            if banned_key in data["score"]:
                data["score"].pop(banned_key)
    jsonl_data[idx] = data


with open(PATH_JSONL, "w") as file:
    for item in jsonl_data:
        item_json = json.dumps(item)
        file.write(item_json + "\n")
