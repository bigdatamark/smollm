# [MODULE DOC] Command-line / config-file parameters of the evaluation launcher, built with simple_parsing dataclasses.
# Parameters (root) = hparams (Hparams) + tasks (TaskParams). get_config supports --load_config/--save_config
# (YAML/JSON via simple_parsing Serializable). Run directly, it parses and prints the config.
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable, list_field

from m4.evaluation.utils import EvaluationVersion
from m4.training.utils import VisionEncoderTypes


logger = logging.getLogger(__name__)


@dataclass
class CfgFileConfig:
    # [CLASS DOC] Two optional Paths: load_config (file whose values become parser defaults) and save_config (where the final
    # parsed config is written).
    """Config file args"""

    load_config: Optional[Path] = None  # load config file
    save_config: Optional[Path] = None  # save config to specified file


@dataclass
class Hparams:
    # [CLASS DOC] General run hyper-parameters: commit_hash; batch_size (deprecated alias) / batch_size_per_gpu (default 1000)
    # / batch_size_per_gpu_dl (dataloader batch size, defaults to batch_size_per_gpu); mini_batch_size (deprecated,
    # unused here); select_n_examples (evaluate only the first N); device; only_load_datasets (prepare datasets and
    # stop); save_to_jsonl (results file); dir_path_load_from_disk (local datasets root); timeout (seconds, default
    # 21600 = 6h, used for the process-group timeout); seed (42); show_gpu_mem_util; is_test;
    # use_selected_prompt_template_ids (use each task's selected_prompt_template_id instead of
    # tasks.prompt_template_id).
    # KNOWN ISSUE: mini_batch_size is marked deprecated in favor of batch_size_per_gpu but, unlike batch_size,
    # it is never migrated or checked in Parameters.__post_init__ (silently ignored).
    """General Hyperparameters"""

    commit_hash: Optional[str] = None
    # deprecated in favor of batch_size_per_gpu
    batch_size: Optional[int] = None
    batch_size_per_gpu: int = 1000
    batch_size_per_gpu_dl: Optional[int] = None
    # deprecated in favor of batch_size_per_gpu
    mini_batch_size: Optional[int] = None
    select_n_examples: Optional[int] = None
    device: Optional[torch.device] = None
    only_load_datasets: bool = False
    save_to_jsonl: Optional[Path] = None
    dir_path_load_from_disk: Optional[Path] = None
    timeout: int = 1800 * 12  # 6h
    seed: int = 42
    show_gpu_mem_util: bool = False
    is_test: bool = False
    use_selected_prompt_template_ids: bool = False


class ShotSelectionMode(Enum):
    # [CLASS DOC] How in-context support examples are chosen: random, rices (retrieval by image-embedding similarity using
    # the vision encoder), or first_without_image.
    random = "random"
    rices = "rices"
    first_without_image = "first_without_image"


@dataclass
class InContextParams:
    # [CLASS DOC] In-context learning settings: num_shots (default 0), shot_selection_mode (default rices) and
    # vision_encoder_name (model used for RICES embeddings, default openai/clip-vit-base-patch32).
    """In context learning parameters"""

    num_shots: int = 0
    shot_selection_mode: ShotSelectionMode = ShotSelectionMode.rices
    vision_encoder_name: str = "openai/clip-vit-base-patch32"


@dataclass
class TextGenerationParams:
    # [CLASS DOC] Generation settings for open-ended tasks: num_beams (3), no_repeat_ngram_size (0 = disabled),
    # max_new_tokens (15).
    """Text generation parameters"""

    num_beams: int = 3
    no_repeat_ngram_size: int = 0
    max_new_tokens: int = 15


class ModelPrecision(Enum):
    # [CLASS DOC] Enum mapping a CLI name (fp32/bf16/fp16) to the torch dtype used to load the model (value).
    fp32 = torch.float32
    bf16 = torch.bfloat16
    fp16 = torch.float16


class DatasetSplit(Enum):
    # [CLASS DOC] Which split to evaluate: default (task's default query split), validation, test, or server_check (small split
    # used to check submission-server formats).
    default = "default"
    validation = "validation"
    test = "test"
    server_check = "server_check"


@dataclass
class TaskParams:
    # [CLASS DOC] Task-level parameters: evaluation_version (EvaluationVersion, default v2), model_name (path or Hub id of the
    # checkpoint), model_precision, tokenizer_name, vision_encoder_max_image_size (384), vision_encoder_type
    # (default siglip), do_tasks (list of task class names to run), in_context_params, tokenizer_use_fast,
    # text_generation_params, dataset_split, prompt_template_id, save_generations (store generated texts),
    # scale_up_images and image_size_after_scaling (4000) for upscaling small images.
    """Task parameters"""

    evaluation_version: EvaluationVersion = EvaluationVersion.v2
    model_name: str = "openai/clip-vit-base-patch32"
    model_precision: ModelPrecision = ModelPrecision.fp32
    tokenizer_name: Optional[str] = None
    vision_encoder_max_image_size: int = 384
    vision_encoder_type: VisionEncoderTypes = VisionEncoderTypes.siglip
    do_tasks: List[str] = list_field("Food101ClipLinearProberAcc")
    in_context_params: InContextParams = InContextParams()
    tokenizer_use_fast: bool = True
    text_generation_params: TextGenerationParams = TextGenerationParams()
    dataset_split: DatasetSplit = DatasetSplit.default
    prompt_template_id: int = 0
    save_generations: bool = False
    scale_up_images: bool = False
    image_size_after_scaling: int = 4000


@dataclass
class Parameters(Serializable):
    # [CLASS DOC] Root config (simple_parsing Serializable, so it supports load/save): hparams and tasks.
    """base options."""

    hparams: Hparams = Hparams()
    tasks: TaskParams = TaskParams()

    def __post_init__(self, should_verify: bool = True):
        # [FUNCTION DOC] Normalise deprecated batch-size options: if hparams.batch_size is set, raise ValueError when
        # batch_size_per_gpu > 1 (ambiguous), otherwise warn and copy it into batch_size_per_gpu. Always resets
        # hparams.batch_size to None and defaults batch_size_per_gpu_dl to batch_size_per_gpu. should_verify is unused.
        # KNOWN ISSUE: since batch_size_per_gpu defaults to 1000, setting the deprecated batch_size without also
        # lowering batch_size_per_gpu to <= 1 always raises.
        """Post-initialization code"""
        # # Used to set different hyperparameters depending on the model
        # if self.hparams.arch == "FlamingoBase":
        #     self.model_param: FlamingoBase = FlamingoBase()

        # deprecation
        if self.hparams.batch_size is not None:
            if self.hparams.batch_size_per_gpu > 1:
                raise ValueError(
                    "as hparams.batch_size is deprecated - don't know how to proceed with both hparams.batch_size>1"
                    " and hparams.batch_size_per_gpu > 1"
                )
            else:
                logger.warn(
                    "will use the deprecated hparams.batch_size, but transition to hparams.batch_size_per_gpu instead"
                )
                self.hparams.batch_size_per_gpu = self.hparams.batch_size
        self.hparams.batch_size = None

        if self.hparams.batch_size_per_gpu_dl is None:
            self.hparams.batch_size_per_gpu_dl = self.hparams.batch_size_per_gpu

    @classmethod
    def parse(cls):
        # [FUNCTION DOC] Classmethod: parse sys.argv into a Parameters instance with a simple_parsing ArgumentParser and return it
        # (no config-file support; see get_config).
        parser = ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance


def get_config(args=None):
    # [FUNCTION DOC] Two-pass parsing: first parse only --load_config/--save_config (ignoring other args); if load_config is set,
    # load Parameters from that file and use it as defaults; then parse all args into Parameters, print it, save it
    # to save_config if requested, and return it. The args parameter is unused (sys.argv is always read).
    # taken from https://github.com/lebrice/SimpleParsing/issues/45#issuecomment-1035224416
    cfgfile_parser = ArgumentParser(add_help=False)
    cfgfile_parser.add_arguments(CfgFileConfig, dest="cfgfile")
    cfgfile_args, _ = cfgfile_parser.parse_known_args()

    cfgfile: CfgFileConfig = cfgfile_args.cfgfile

    file_config: Optional[Parameters] = None
    if cfgfile.load_config is not None:
        file_config = Parameters.load(cfgfile.load_config)

    parser = ArgumentParser()

    # add cfgfile args so they appear in the help message
    parser.add_arguments(CfgFileConfig, dest="cfgfile")
    parser.add_arguments(Parameters, dest="my_preferences", default=file_config)

    args = parser.parse_args()

    prefs: Parameters = args.my_preferences
    print(prefs)

    if cfgfile.save_config is not None:
        prefs.save(cfgfile.save_config, indent=4)
    return prefs


if __name__ == "__main__":
    config = get_config()
    print(config)
