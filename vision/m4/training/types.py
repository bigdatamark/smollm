# [MODULE DOC] Enumerations naming the dataset sources and dataset formats understood by the m4 training data
# pipeline.
# Pure constant definitions; no side effects on import.
from enum import Enum


class DatasetNames(Enum):
    # [CLASS DOC] Enum of dataset source names (pmd, laion, laion_coco, tikz, cm4, wiki, image_website_code,
    # vqav2_task_finetuning, ocr, docvqa, sft); each member's value is the lowercase string used in configs.
    PMD = "pmd"
    LAION = "laion"
    LAION_COCO = "laion_coco"
    TIKZ = "tikz"
    CM4 = "cm4"
    WIKI = "wiki"
    IMAGE_WEBSITE_CODE = "image_website_code"
    VQAV2_TASK_FINETUNING = "vqav2_task_finetuning"
    OCR = "ocr"
    DOCVQA = "docvqa"
    SFT = "sft"


class DatasetTypes(Enum):
    # [CLASS DOC] Enum of dataset layout types: web documents ("wd"), image-caption pairs ("icp"), VQAv2 task
    # finetuning,
    # OCR, DocVQA and SFT. Values are the short strings used in configs and by the data-loading code.
    WEB_DOCUMENTS = "wd"
    IMAGE_CAPTION_PAIRS = "icp"
    VQAV2_TASK_FINETUNING = "vqav2_task_finetuning"
    OCR = "ocr"
    DOCVQA = "docvqa"
    SFT = "sft"
