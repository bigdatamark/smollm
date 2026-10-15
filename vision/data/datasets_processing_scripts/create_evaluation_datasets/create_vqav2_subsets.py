# [MODULE DOC] Script (top-level code): splits the test_query_set of
# HuggingFaceM4/VQAv2_modif_support_query_sets into
# NUM_SUBSETS (6) contiguous near-equal parts (numpy array_split) to parallelize evaluation; each part, with
# the
# other splits unchanged, is pushed privately to <NAME_DS>_part_<i>.
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


NUM_SUBSETS = 6

NAME_DS = "HuggingFaceM4/VQAv2_modif_support_query_sets"


ds_subsets = [load_dataset(NAME_DS, use_auth_token=True) for _ in range(NUM_SUBSETS)]

num_test_examples = ds_subsets[0]["test_query_set"].num_rows

selected_indices = np.array_split(range(num_test_examples), NUM_SUBSETS)

for idx_ds in range(NUM_SUBSETS):
    ds_subsets[idx_ds]["test_query_set"] = ds_subsets[idx_ds]["test_query_set"].select(selected_indices[idx_ds])

for idx_ds in tqdm(range(NUM_SUBSETS)):
    ds_subsets[idx_ds].push_to_hub(repo_id=NAME_DS + f"_part_{idx_ds}", private=True)
