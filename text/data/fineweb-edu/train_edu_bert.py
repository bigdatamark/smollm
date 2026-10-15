# [MODULE DOC] Script that trains the FineWeb-Edu quality classifier: a BERT-like encoder with a
# single-output regression
# head on LLM-annotated educational scores (0-5), with frozen embeddings/encoder (only the head trains).
# CLI args: --base_model_name, --dataset_name, --target_column (default "score"), --checkpoint_dir,
# --output_model_name (Hub repo to push to).
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(eval_pred):
    # [FUNCTION DOC] Trainer metric callback. Input: eval_pred tuple (logits array, labels array). Rounds
    # logits and clips to
    # [0, 5] as int predictions, rounds labels to int, and computes macro precision/recall/F1 and accuracy via
    # the `evaluate` library (loaded on every call, may hit network). Prints an sklearn classification report
    # and confusion matrix. Returns a dict with keys precision, recall, f1_macro, accuracy.
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    # [FUNCTION DOC] Entry point. Loads the "train" split (cache /scratch/cosmo/cache/), clips the target
    # column to int in
    # [0, 5], casts it to a 6-class ClassLabel and does a stratified 90/10 train/test split (seed 42). Loads
    # the
    # model with num_labels=1 and no dropout, a tokenizer with max length min(model max positions, 512), using
    # EOS as pad token if none. Tokenizes with preprocess, freezes model.bert embeddings and encoder
    # parameters,
    # and trains with Trainer (lr 3e-4, 20 epochs, batch 256/128, eval/save every 1000 steps, bf16, best
    # model by
    # f1_macro, push_to_hub to output_model_name). Saves the final model to <checkpoint_dir>/final.
    # KNOWN ISSUE: assumes the model exposes a `.bert` attribute; non-BERT base models will raise
    # AttributeError.
    dataset = load_dataset(
        args.dataset_name, split="train", cache_dir="/scratch/cosmo/cache/", num_proc=8
    )
    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)},
        num_proc=8,
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42, stratify_by_column=args.target_column
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=1,
        classifier_dropout=0.0,
        hidden_dropout_prob=0.0,
        output_hidden_states=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        model_max_length=min(model.config.max_position_embeddings, 512),
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        # [FUNCTION DOC] Batched map function: tokenizes examples["text"] with truncation and adds "labels"
        # as float32 values of the
        # target column (regression targets). Returns the tokenized batch.
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        learning_rate=3e-4,
        num_train_epochs=20,
        seed=0,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-m"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW/fineweb-edu-llama3-annotations",
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/fsx/anton/cosmopedia/edu_score/bert_snowflake_regression",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="HuggingFaceTB/fineweb-edu-scorer"
    )
    args = parser.parse_args()

    main(args)
