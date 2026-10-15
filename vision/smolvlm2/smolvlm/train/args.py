# [MODULE DOC] Dataclasses defining the command-line arguments for SmolVLM2 training, parsed with
# HfArgumentParser in
# smolvlm.train.train: model options, data/mixture options, and TrainingArguments extending transformers'
# TrainingArguments with quantization, LoRA, per-component learning rates and sequence-parallel fields.
# arguments.py

from dataclasses import dataclass, field
from typing import Optional, List

import transformers


@dataclass
class ModelArguments:
    # [CLASS DOC] Model options: model_name_or_path (default "HuggingFaceTB/SmolVLM_converted_4"),
    # trust_remote_code
    # (False), padding_side ("right"), apply_diagonal_block_attention (False; block-diagonal attention for
    # packing), frames_per_clip (1; frames averaged together after the vision encoder), fps (1.0, video
    # sampling).
    """
    Arguments related to the model/config/trust_remote_code.
    """
    model_name_or_path: str = field(
        default="HuggingFaceTB/SmolVLM_converted_4", 
        metadata={"help": "Path to pretrained model or model identifier."}
    )
    trust_remote_code: bool = field(
        default=False, 
        metadata={"help": "Allow custom code from the model repo."}
    )
    padding_side: str = field(
        default="right", 
        metadata={"help": "Tokenizer padding side. Usually 'right' for LLMs."}
    )
    apply_diagonal_block_attention: bool = field(
        default=False, 
        metadata={"help": "Apply diagonal cross attention. Important when doing sequence pack."}
    )
    frames_per_clip: int = field(
        default=1,
        metadata={"help": "Number of frames to group together and average after vision encoder."}
    )
    fps: float = field(
        default=1.0,
        metadata={"help": "FPS for video sampling if needed."}
    ) 

@dataclass
class DataArguments:
    # [CLASS DOC] Data options: data_folder (media root, "./data"), data_mixture (YAML of sub-datasets),
    # add_media_intro_outro,
    # mask_user_tokens (False), mask_system_tokens (True), video_target_size (384) and image_target_size
    # (1536)
    # in pixels, max_frames (25), packed (sequence packing), loss_reduction ("token", "sample", "square" or
    # "none" weighting of packed sub-samples) and loss_reduction_all_gather.
    # KNOWN ISSUE: help texts of video_target_size, image_target_size and max_frames are copy-pasted
    # "FPS for video sampling if needed." and do not describe these fields.
    """
    Arguments related to the data for training (e.g., paths, dataset config).
    """
    data_folder: str = field(
        default="./data",
        metadata={"help": "Root folder for images or videos if needed."}
    )
    data_mixture: str = field(
        default="./data_mixture.yaml",
        metadata={"help": "YAML that describes multiple sub-datasets."}
    )
    add_media_intro_outro: bool = field(default=False, metadata={"help": "Add extra user messages around media files"})
    mask_user_tokens: bool = field(default=False, metadata={"help": "Mask user prompts."})
    mask_system_tokens: bool = field(default=True, metadata={"help": "Mask system prompts."})
    video_target_size: int = field(
        default=384,
        metadata={"help": "FPS for video sampling if needed."}
    )
    image_target_size: int = field(
        default=1536,
        metadata={"help": "FPS for video sampling if needed."}
    )
    max_frames: int = field(
        default=25,
        metadata={"help": "FPS for video sampling if needed."}
    )
    packed: bool = field(default=False, metadata={"help": "Use sequnce packing."})
    loss_reduction: str = field(
        default="token",
        metadata={
            "help": "How to weight sub-samples in a packed sequence. "
                    "Options: 'token', 'sample', 'square', or 'none'."
        }
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={
            "help": "Whether we all-gather partial sums if in distributed mode."
        }
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # [CLASS DOC] transformers.TrainingArguments plus: cache_dir, optim ("adamw_torch"), remove_unused_columns
    # (False),
    # disable_flash_attn2, model_dtype (string such as "torch.bfloat16"), model_max_length (16384),
    # quantization
    # (bits 16/8/4, double_quant, quant_type "nf4"), LoRA (peft_enable, lora_rank 64, lora_alpha 32,
    # lora_dropout 0.05, target_modules, lora_bias, use_dora), learning rates for vision tower (2e-6),
    # connector
    # (1e-5) and language model (1e-5), tune_vision_tower, sequence-parallel settings (-1 = disabled) and
    # debug_e2e.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache dir for huggingface transformers."}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use with Trainer."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove dataset columns not used by the model input."}
    )
    disable_flash_attn2: bool = field(
        default=False,
        metadata={"help": "Disable flash-attention v2 if True, fallback to SDPA or older attention."}
    )
    model_dtype: str = field(
        default="torch.bfloat16",
        metadata={"help": "Data type for the model. E.g. torch.bfloat16, torch.float16, etc."}
    )
    model_max_length: int = field(
        default=16384,
        metadata={"help": "Max sequence length (token positions). For rope scaling, etc."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Use double quantization (applicable to 4-bit or 8-bit modes)."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type: 'nf4', 'fp4', etc."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "Precision: 16 (fp16/bf16), 8, or 4 bits. 0 or 16 means standard float."}
    )
    peft_enable: bool = field(
        default=False,
        metadata={"help": "Enable LoRA/PEFT-based training adapters."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "Rank dimension for LoRA."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha scaling factor."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability within LoRA adapters."}
    )
    target_modules: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Module names to apply LoRA to, e.g. ['q_proj','k_proj']."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Type of LoRA bias to use: none, all, or lora_only."}
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Use DoRA if needed (experimental)."})
    vision_tower_lr: float = field(
        default=2e-6,
        metadata={"help": "Learning rate for vision tower submodule if tune_vision_tower=True."}
    )
    tune_vision_tower: bool = field(
        default=False,
        metadata={"help" : "Enable / Disable vision tower tunning"}
    )
    connector_lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for the multi-modal connector/merger if tune_mm_connector=True."}
    )
    language_model_lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for the core LLM if tune_language_model=True."}
    )
    seq_parallel_size: int = field(
        default=-1,
        metadata={"help": "Sequence parallel size if used, otherwise -1 means disabled."}
    )
    seq_parallel_ring_size: int = field(
        default=-1,
        metadata={"help": "Ring size for ring-based sequence parallel attention."}
    )
    seq_parallel_ring_type: str = field(
        default="ring_varlen",
        metadata={"help": "Type of ring attention if using ring-based SP."}
    )
    debug_e2e: bool = field(
        default=False,
        metadata={"help": "Enable end-to-end debug prints or additional checks."}
    )



