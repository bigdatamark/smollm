# [MODULE DOC] CLIP image-text similarity. On import, loads `openai/clip-vit-base-patch32` model and
# processor (network /
# HF cache) and moves the model to `cuda:0` if available (module globals `model`, `processor`, `device`).
# Running as a script scores 3 captions against a COCO image downloaded from the web and prints the result.
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL = "openai/clip-vit-base-patch32"
NUM_MAX_WORDS = 50

model = CLIPModel.from_pretrained(CLIP_MODEL)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model = model.to(device)
else:
    device = "cpu"


def compute_clip_score(texts, image, num_max_words=NUM_MAX_WORDS):
    # [FUNCTION DOC] Compute CLIP similarity between each text and the image(s). texts: list of str, each
    # truncated to its first
    # `num_max_words` space-separated words (default 50; None disables). image: PIL/array/tensor or a list of
    # them.
    # Tokenizes with padding, runs the model on `device`, returns a 1-D CPU tensor of length len(texts):
    # dot products of the (normalized) text and image embeddings for the FIRST image only (proportional to
    # logits).
    # Gradients are not disabled. KNOWN ISSUE: docstring names the parameter `images`; extra images are
    # ignored.
    """
    Args
        texts: List[str]
        images: (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`):
                The image to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
    Output is of size nb_of_text. Element j-th correponds to the cosine similarity between the the image and text j.
    """
    if num_max_words is not None:
        texts = [" ".join(t.split(" ")[:num_max_words]) for t in texts]
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    outputs = model(**inputs)
    outputs = torch.matmul(
        outputs.text_embeds, outputs.image_embeds.t()
    ).T  # We couls also return the `logits_per_image`, but these two are exactly proportional
    if device != "cpu":
        return outputs.cpu()[0]
    else:
        return outputs[0]


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "this is a book",
    ]
    images = [image]
    out = compute_clip_score(texts, images)
    print(out)
