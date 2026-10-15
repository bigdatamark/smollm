# [MODULE DOC] Streamlit app to browse an interleaved web-document dataset saved on disk (columns texts,
# images, metadata
# JSON). As a script: wide layout, dataset at ./large_files/web_docs_final.
import json
import random

import streamlit as st
from datasets import load_from_disk


class Visualization:
    # [CLASS DOC] Stateful page: holds the dataset path, the (subsampled) dataset and the current document.
    def __init__(self, path_web_documents_dataset):
        # [FUNCTION DOC] Store the dataset path. No I/O.
        self.path_web_documents_dataset = path_web_documents_dataset

    def visualization(self):
        # [FUNCTION DOC] Render the page: title, dataset load + size choice, document choice, document
        # display.
        self.set_title()
        self.load_dataset()
        self.choose_document()
        self.display_document()

    def set_title(self):
        # [FUNCTION DOC] Show the page title.
        st.title("Visualization of web documents")

    def load_dataset(self):
        # [FUNCTION DOC] Load the dataset from disk and keep only its first 100/300/1000/3000 rows (user
        # choice).
        # KNOWN ISSUE: selecting more rows than the dataset holds raises an IndexError.
        st.header("Select the size of the dataset")

        self.dataset = load_from_disk(self.path_web_documents_dataset)

        opt_sizes = ["100", "300", "1000", "3000"]
        size_dataset = st.selectbox(
            "Select the size of the dataset",
            options=opt_sizes,
        )

        self.dataset = self.dataset.select(range(int(size_dataset)))

    def choose_document(self):
        # [FUNCTION DOC] Random-document button plus index input; stores the selected row in self.current_doc.
        st.header("Choose a document")
        if st.button("Select a random document"):
            dct_idx = random.randint(a=0, b=self.dataset.num_rows - 1)
        else:
            dct_idx = 0
        idx = st.number_input(
            f"Select a document among the first {self.dataset.num_rows} ones",
            min_value=0,
            max_value=self.dataset.num_rows - 1,
            value=dct_idx,
            step=1,
            help=f"Index between 0 and {self.dataset.num_rows-1}",
        )
        self.current_doc = self.dataset[idx]

    def display_document(self):
        # [FUNCTION DOC] Display the current document node by node: text nodes as <pre> blocks with newlines
        # turned into <br>, image
        # nodes as an <img> tag pointing at the metadata `src` url (max width 1000px).
        st.header("Document")
        texts = self.current_doc["texts"]
        images = self.current_doc["images"]
        metadata = json.loads(self.current_doc["metadata"])
        for text, image, meta in zip(texts, images, metadata):
            if text:
                display_text = f"{text}\n".replace(
                    "\n", "<br>"
                )  # .replace(" ", "&nbsp;") Preserve white spaces, but creates text outside the width of the window
                st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)
            elif image:
                st.markdown(
                    f'<img src="{meta["src"]}" style="max-width: 1000px; height: auto;" />', unsafe_allow_html=True
                )
                st.text("\n")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    path_web_documents_dataset = "./large_files/web_docs_final"  # Find at s3://m4-datasets/trash/web_docs_final/
    visualization = Visualization(path_web_documents_dataset=path_web_documents_dataset)
    visualization.visualization()
