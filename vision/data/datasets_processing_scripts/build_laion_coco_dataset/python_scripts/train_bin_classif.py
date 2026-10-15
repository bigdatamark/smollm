# [MODULE DOC] Experiment script training the SigLIP-based binary "bad pair" classifier used by
# 05_binary_classification.py.
# Loads the SigLIP processor at import. Labels: 1 for indices in the hand-annotated NEGATIVE_LABELS list, 0
# otherwise, over a 10,000-row LAION-COCO sample loaded from disk; train = rows 0-899, test = rows 900-999.
# For each learning rate in lr_values ([1e-4]) and 5 independent trials ("folds", no actual k-fold split; the
# StratifiedKFold object is unused): builds a frozen-SigLIP classifier on cuda:0, trains 15 epochs with
# BCELoss
# and AdamW (weight decay 0.1, batch 8, printing loss every 20 steps), and after each epoch evaluates on the
# test
# set (threshold 0.5), printing precision/recall/F1 and confusion matrix, then saves the state dict and the
# confusion matrix (.npy) under /fsx/m4/victor/bin_classif_models/ with metrics in the filename.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor


DEVICE = torch.device("cuda:0")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
# processor = AutoProcessor.from_pretrained("/fsx/m4/victor/siglip-so400m-patch14-384-ocred/")


class MyCustomBinaryClassification(nn.Module):
    # [CLASS DOC] Binary classifier on top of SigLIP (google/siglip-so400m-patch14-384): concatenation of
    # image and text
    # embeddings fed to an MLP (Linear 2h->h, ReLU, Dropout 0.1, Linear h->1, Sigmoid).
    def __init__(self, freeze_siglip=True):
        # [FUNCTION DOC] Loads SigLIP; if freeze_siglip (default True) disables its gradients. Input size = 2
        # x text hidden size.
        super().__init__()
        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        # self.siglip = AutoModel.from_pretrained("/fsx/m4/victor/siglip-so400m-patch14-384-ocred/")
        if freeze_siglip:
            self.siglip.requires_grad_(False)
        self.freeze_siglip = freeze_siglip
        input_size = self.siglip.config.text_config.hidden_size * 2
        self.fc1 = nn.Linear(input_size, int(input_size / 2))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(int(input_size / 2), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values):
        # [FUNCTION DOC] Runs SigLIP on input_ids/attention_mask/pixel_values (no_grad + eval if frozen),
        # concatenates image_embeds and
        # text_embeds and returns probabilities of shape (batch,).
        if self.freeze_siglip:
            with torch.no_grad():
                self.siglip.eval()
                outputs = self.siglip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        else:
            outputs = self.siglip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        feature = torch.cat([image_features, text_features], dim=-1)
        return self.sigmoid(self.fc2(self.dropout(self.activation(self.fc1(feature)))).squeeze(-1))


NEGATIVE_LABELS = [
    14,
    18,
    20,
    21,
    23,
    24,
    25,
    27,
    28,
    29,
    30,
    37,
    41,
    42,
    52,
    55,
    56,
    57,
    59,
    60,
    64,
    67,
    70,
    75,
    82,
    87,
    88,
    89,
    92,
    100,
    104,
    107,
    112,
    113,
    114,
    115,
    116,
    122,
    125,
    133,
    137,
    138,
    140,
    141,
    145,
    147,
    150,
    152,
    156,
    158,
    159,
    163,
    167,
    168,
    170,
    172,
    173,
    181,
    189,
    191,
    192,
    195,
    196,
    197,
    201,
    203,
    208,
    218,
    223,
    225,
    234,
    243,
    247,
    251,
    259,
    264,
    268,
    271,
    272,
    273,
    274,
    276,
    281,
    287,
    293,
    294,
    296,
    303,
    304,
    305,
    312,
    313,
    317,
    325,
    328,
    330,
    331,
    338,
    340,
    350,
    352,
    366,
    367,
    376,
    377,
    378,
    379,
    385,
    386,
    387,
    393,
    398,
    400,
    405,
    411,
    412,
    416,
    417,
    419,
    428,
    431,
    440,
    442,
    443,
    445,
    448,
    459,
    463,
    466,
    468,
    473,
    479,
    483,
    484,
    486,
    497,
    501,
    502,
    505,
    512,
    514,
    515,
    520,
    527,
    531,
    535,
    538,
    542,
    543,
    547,
    548,
    555,
    566,
    567,
    568,
    575,
    576,
    577,
    584,
    590,
    592,
    594,
    595,
    596,
    610,
    618,
    620,
    621,
    622,
    627,
    635,
    641,
    643,
    648,
    650,
    652,
    653,
    659,
    666,
    668,
    676,
    680,
    684,
    687,
    703,
    706,
    709,
    712,
    715,
    716,
    721,
    723,
    724,
    725,
    726,
    728,
    729,
    731,
    733,
    739,
    743,
    750,
    753,
    759,
    772,
    774,
    780,
    784,
    786,
    787,
    793,
    794,
    799,
    804,
    806,
    807,
    808,
    812,
    813,
    816,
    817,
    818,
    822,
    823,
    826,
    827,
    828,
    836,
    837,
    838,
    839,
    840,
    841,
    844,
    846,
    850,
    852,
    853,
    856,
    859,
    860,
    865,
    866,
    868,
    871,
    872,
    874,
    879,
    880,
    882,
    884,
    885,
    893,
    895,
    896,
    898,
    899,
    901,
    904,
    905,
    908,
    909,
    911,
    919,
    921,
    928,
    930,
    937,
    944,
    945,
    949,
    951,
    952,
    955,
    956,
    960,
    962,
    965,
    966,
    971,
    975,
    977,
    980,
    981,
    984,
    988,
    989,
    994,
    998,
]
LABELS = [1 if idx in NEGATIVE_LABELS else 0 for idx in range(10_000)]
# Load dataset
dataset = load_from_disk("/fsx/m4/victor/laion_coco_visualization/ds_laion_coco_10000")
dataset = dataset.add_column("labels", LABELS)

# dataset = dataset.select(list(range(0, 1000)))
train_data = dataset.select(list(range(0, 900)))
test_data = dataset.select(list(range(900, 1000)))

# positive_samples = [idx for idx, item in enumerate(train_data) if item["labels"] == 1]
# negative_samples = [idx for idx, item in enumerate(train_data) if item["labels"] == 0]
# subset_size = min(len(positive_samples), len(negative_samples))
# balanced_subset = random.sample(positive_samples, subset_size) + random.sample(negative_samples, subset_size)
# train_data = train_data.select(balanced_subset)

print("Dataset len", len(train_data))


def _convert_to_rgb(image):
    # [FUNCTION DOC] HELPER: returns the image unchanged if already RGB; otherwise alpha-composites it (as
    # RGBA) over a white
    # background and returns the RGB result.
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


# Function to tokenize and prepare data
def tokenize_batch(batch):
    # [FUNCTION DOC] HELPER: DataLoader collate function. Input: list of rows with "text", "image", "labels".
    # Tokenizes texts with
    # padding and attention mask, adds pixel_values for RGB-converted images, moves everything to DEVICE, and
    # returns
    # (inputs mapping, float label tensor on DEVICE).
    inputs = processor.tokenizer(
        [b["text"] for b in batch], return_tensors="pt", return_attention_mask=True, padding=True
    )
    inputs.update(
        processor(
            images=[_convert_to_rgb(b["image"]) for b in batch],
        )
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float, device=DEVICE)
    return inputs, labels


# Set up k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_values = [1e-4]

for lr in lr_values:
    print(f"\nGrid Search: Learning Rate = {lr}\n")

    # for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset, dataset["labels"])):
    for fold in range(5):  # fold is more like a trial
        model = MyCustomBinaryClassification(True).to(DEVICE)

        def lorify(model):
            # [FUNCTION DOC] HELPER (unused; call is commented out): wraps the model with a PEFT LoRA adapter
            # (r=16, alpha=16, dropout 0.1)
            # on SigLIP attention projections and MLP layers, re-enables gradients on layer norms and the
            # classifier head
            # fc1/fc2, and returns the wrapped model.
            lora_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=16,
                bias="none",
                target_modules=[
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.q_proj",
                    "self_attn.out_proj",
                    "mlp.fc1",
                    "mlp.fc2",
                ],
            )
            model = get_peft_model(model, lora_config)

            for n, p in model.named_modules():
                if "layer_norm" in n or "layernorm" in n:
                    p.requires_grad_(True)
            model.base_model.fc1.requires_grad_(True)
            model.base_model.fc2.requires_grad_(True)
            return model

        # model = lorify(model)

        # train_data = dataset.select(train_idx)
        # test_data = dataset.select(test_idx)

        train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=tokenize_batch)
        test_dataloader = DataLoader(
            test_data, batch_size=16, shuffle=False, collate_fn=tokenize_batch, drop_last=False
        )

        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

        num_epochs = 15
        eval_every = 20

        for epoch in range(num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                inputs, labels = batch
                optimizer.zero_grad()
                logits = model(**inputs)
                loss = criterion(logits, labels)
                if step % eval_every == 0:
                    print(round(loss.item(), 3))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                all_logits = []

                for eval_batch in test_dataloader:
                    eval_inputs, eval_labels = eval_batch
                    logits = model(**eval_inputs)
                    # eval_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    eval_preds = (logits >= 0.5).float().cpu().numpy()

                    all_logits.extend(logits.cpu().numpy())
                    all_preds.extend(eval_preds)
                    all_labels.extend(eval_labels.cpu().numpy())

                precision = precision_score(y_true=all_labels, y_pred=all_preds, zero_division=0.0)
                recall = recall_score(y_true=all_labels, y_pred=all_preds)
                f1 = f1_score(y_true=all_labels, y_pred=all_preds)
                confusion_mat = confusion_matrix(all_labels, all_preds)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}, Step {step}:")
                print(f"Current loss: {round(loss.item(), 2)}")
                print(f"Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}")
                print("Confusion Matrix:")
                print(confusion_mat)
                # print("precision recall curve")
                # precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true=all_labels, probas_pred=all_logits)
                # gathered = [(x, y, z) for x, y, z in zip(precision_curve, recall_curve, thresholds_curve)]
                # print(gathered)
            saving_path = (
                f"/fsx/m4/victor/bin_classif_models/fold-{fold+1}_epoch-{epoch + 1}_precision-{round(precision, 2)}_recall-{round(recall, 2)}_f1-{round(f1, 2)}.model.pt"
            )
            torch.save(model.state_dict(), saving_path)
            saving_path = (
                f"/fsx/m4/victor/bin_classif_models/fold-{fold+1}_epoch-{epoch + 1}_precision-{round(precision, 2)}_recall-{round(recall, 2)}_f1-{round(f1, 2)}.confusion_matrix"
            )
            np.save(saving_path, confusion_mat)
            model.train()
