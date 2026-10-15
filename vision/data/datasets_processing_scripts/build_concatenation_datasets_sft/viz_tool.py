# [MODULE DOC] Streamlit app for browsing SFT datasets saved on disk. Deploy with "streamlit run viz_tool.py"
# (port forwarding
# may be needed on the cluster). Sidebar lets the user choose between the original folder
# (/fsx/hugo/fine_tuning_datasets) and the multi-turn merged folder, pick a dataset sub-folder and a number of
# examples to render.
"""
To deploy: streamlit run viz_tool.py
If on HFC, probably will need to do port forwarding to get it in your local browser.
"""

import glob
import os

import streamlit as st
from datasets import load_from_disk


def display_examples(dataset, num_examples=300):
    # [FUNCTION DOC] Renders, for i in 0..num_examples-1 (default 300), a header, every non-None image of
    # example i (width 250 with
    # its size as caption) and each turn's user and assistant texts as HTML <pre> blocks (newlines turned into
    # <br>),
    # followed by a divider. Input dataset: indexable HF dataset with "images" and "texts" columns. Returns
    # None.
    # KNOWN ISSUE: no bound check; raises IndexError if the dataset has fewer than num_examples rows (slider
    # min is 100).
    st.header(f"Sample Examples - First {num_examples}")

    for i in range(num_examples):
        st.subheader(f"Example {i}")

        example = dataset[i]

        images = [im for im in example["images"] if im is not None]
        for im in images:
            st.image(im, width=250, caption=f"Image dimension: {im.size}")

        texts = example["texts"]
        for turn in texts:
            display_text = f"<strong>User:</strong> {turn['user']}\n".replace("\n", "<br>")
            st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)
            display_text = f"<strong>Assistant:</strong> {turn['assistant']}\n".replace("\n", "<br>")
            st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)

        st.divider()


def main():
    # [FUNCTION DOC] Streamlit entry point: sets page config, reads the sidebar radio to pick DATASET_FOLDER,
    # lists its
    # sub-folders as dataset choices (with an empty first choice), loads the selected one and, if loaded,
    # shows a
    # slider (100-2000, default 200), the dataset length and calls display_examples.
    st.set_page_config(page_title="Prompted set viewer", layout="wide")

    merged_or_not = st.sidebar.radio(
        label="Multi-turn merged or not?",
        options=["Original", "Multi-turn merged"],
        index=1,
    )
    if merged_or_not == "Original":
        DATASET_FOLDER = "/fsx/hugo/fine_tuning_datasets"
    elif merged_or_not == "Multi-turn merged":
        DATASET_FOLDER = "/fsx/hugo/fine_tuning_datasets_merge_image_individual"
    else:
        DATASET_FOLDER = None

    def load_hf_dataset(dataset_path):
        # [FUNCTION DOC] HELPER: returns load_from_disk(DATASET_FOLDER/dataset_path), or None if dataset_path
        # is None/empty or loading
        # raises any exception.
        if dataset_path is None or dataset_path == "":
            return None

        try:
            dataset = load_from_disk(f"{DATASET_FOLDER}/{dataset_path}")
            return dataset
        except Exception:
            return None

    all_datasets = [os.path.basename(p) for p in glob.glob(DATASET_FOLDER + "/*")]
    datasets_list = [""] + all_datasets

    st.sidebar.header("Prompted set viewer")
    selected_dataset = st.sidebar.selectbox(
        label="Select a dataset",
        options=datasets_list,
        index=0,
    )

    dataset = load_hf_dataset(selected_dataset)

    if dataset is not None:
        num_examples = st.sidebar.slider(
            label="Number of examples to display", value=200, min_value=100, max_value=2000
        )
        st.sidebar.write(f"Dataset length: `{len(dataset)}`")
        display_examples(dataset, num_examples=num_examples)


if __name__ == "__main__":
    main()
