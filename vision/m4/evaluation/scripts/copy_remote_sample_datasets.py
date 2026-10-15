# [MODULE DOC] Script that downloads the test and train splits of every VGPT2 "Sample" task dataset from the Hub and saves
# them with save_to_disk under /gpfsscratch/rech/cnw/commun/local_datasets/<dataset_name>/<split> (for offline
# evaluation via hparams.dir_path_load_from_disk). Tasks are instantiated with dummy model/tokenizer settings.
# MIN_DATASET_SIZE, DEFAULT_NUM_EX_PER_CLASS and ALREADY_COPIED_DATASETS are unused.
# KNOWN ISSUE: tasks no longer define test_split_name/train_split_name (they use
# default_query_split_name/default_support_split_name etc.), so this raises AttributeError with current tasks.
from pathlib import Path

from datasets import load_dataset

from m4.evaluation.tasks import VGPT2_SAMPLE_TASKS, Predictor
from m4.evaluation.utils import EvaluationVersion


MIN_DATASET_SIZE = 100
DEFAULT_NUM_EX_PER_CLASS = 3

ALREADY_COPIED_DATASETS = set()

model_name = "gpt2"  # Not used but necessary to load the task
tokenizer_name = "t5-base"  # Not used but necessary to load the task
image_size = 224  # Not used but necessary to load the task
evaluation_version = EvaluationVersion.v2  # Not used but necessary to load the task

save_dir = Path("/gpfsscratch/rech/cnw/commun/local_datasets")


def load_and_save_dataset(task, split, save_dir):
    # [FUNCTION DOC] Load task.dataset_name (config task.dataset_config) split split from the Hub with auth, print task/size info,
    # and save it to save_dir/<dataset_name>/<split>.
    dataset_split = load_dataset(task.dataset_name, name=task.dataset_config, split=split, use_auth_token=True)
    print("********************************************************")
    print(task.__class__.__name__)
    print(len(dataset_split))
    print(f"Dataset name is {task.dataset_name} and split is {split} and config is {task.dataset_config}")
    print("********************************************************")

    dataset_split.save_to_disk(save_dir / task.dataset_name / split)


if __name__ == "__main__":
    for task_objet in VGPT2_SAMPLE_TASKS[Predictor.in_contexter]:
        task = task_objet(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            image_size=image_size,
            evaluation_version=evaluation_version,
        )

        load_and_save_dataset(task, task.test_split_name, save_dir)
        load_and_save_dataset(task, task.train_split_name, save_dir)
