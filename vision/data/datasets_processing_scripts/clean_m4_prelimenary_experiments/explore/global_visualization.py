# [MODULE DOC] Streamlit app comparing two DOM-tree simplification configurations of the m4 web-document
# extraction
# pipeline (m4.sourcing.data_collection). Loads a hard-coded local shard filtered on a user CSS selector,
# lets the user pick a document, and in "Extraction" mode shows a text diff (diff_viewer) and the images of
# the documents extracted with simplificator v1 (default rules) vs v2 (extra footer/site-info removal and
# "more-link" replaced by an END_OF_DOCUMENT token). When run as main: wide layout, num_docs=1000, template
# assets/DOM_tree_viz.html next to this file.
import os
import pathlib
import random
from functools import partial

import diff_viewer

# import pandas as pd
import streamlit as st
from datasets import load_from_disk
from jinja2 import Template
from selectolax.parser import HTMLParser

from m4.sourcing.data_collection.processors import (  # TextMediaPairsExtractor,
    DOMTreeSimplificator,
    PreExtractionSimplificator,
)
from m4.sourcing.data_collection.utils import InterestingAttributesSetCategory, make_selectolax_tree


class Visualization:
    # [CLASS DOC] Holds the loaded examples, Jinja DOM-visualization template and the simplificator/extractor
    # objects, and
    # renders the Streamlit page sections.
    def __init__(self, num_docs, dom_viz_template_path):
        # [FUNCTION DOC] Inputs: num_docs (int, upper bound for the document index selector) and
        # dom_viz_template_path (path of a
        # Jinja HTML template). Shows a "CSS rule" text input (default "[class~='more-link']"), loads and
        # filters
        # the dataset (self.examples), loads the template (self.dom_viz_template), and builds two
        # DOMTreeSimplificator configs (v1 / v2) and two PreExtractionSimplificator configs (merge / not merge
        # consecutive text nodes, Wikipedia interesting-attributes set).
        self.num_docs = num_docs

        css_rule = st.text_input("CSS rule", value="[class~='more-link']")

        def filter_doc_with_matching_class(example, css_rule):
            # [FUNCTION DOC] HELPER: filter predicate; True if example["html"] has a node matching css_rule
            # (selectolax css_first).
            current_html = example["html"]
            tree = HTMLParser(current_html)
            matche = tree.css_first(css_rule)
            if matche:
                return True
            return False

        @st.cache_resource  # it is caching but is incredibly slow when N is big.
        def load_examples(num_docs, css_rule):
            # [FUNCTION DOC] HELPER: loads the hard-coded shard_215 from disk, prints its size, filters it to
            # documents matching
            # css_rule (32 processes), prints the new size and returns it. Cached with st.cache_resource;
            # num_docs unused.
            dataset = load_from_disk(
                "/home/lucile/data/web_document_dataset_45M_sharded_ftered_2_line_deduplicated_with_html/train/shard_215"
            )
            # dataset = dataset.select(range(num_docs))
            print(f"Loaded {len(dataset)} examples.")
            dataset = dataset.filter(partial(filter_doc_with_matching_class, css_rule=css_rule), num_proc=32)
            # dataset = dataset.filter(lambda x: x["document_url"] == "https://acrusteaten.com/tag/meatless-monday/", num_proc=32)

            print(f"Loaded {len(dataset)} examples.")
            return dataset

        self.examples = load_examples(num_docs, css_rule)

        def load_dom_viz_template(dom_viz_template_path):
            # [FUNCTION DOC] HELPER: reads the file at dom_viz_template_path and returns it as a jinja2
            # Template.
            with open(dom_viz_template_path, "r") as file:
                template_string = file.read()
            return Template(template_string)

        self.dom_viz_template = load_dom_viz_template(dom_viz_template_path)

        self.dom_tree_simplificator_v1 = DOMTreeSimplificator(
            strip_multiple_linebreaks=True,
            strip_multiple_spaces=True,
            remove_html_comments=True,
            replace_line_break_tags=True,
            unwrap_tags=True,
            strip_tags=True,
            strip_special_divs=True,
            remove_dates=True,
            remove_empty_leaves=True,
            unnest_nodes=True,
            remake_tree=True,
        )

        self.dom_tree_simplificator_v2 = DOMTreeSimplificator(
            strip_multiple_linebreaks=True,
            strip_multiple_spaces=True,
            remove_html_comments=True,
            replace_line_break_tags=True,
            unwrap_tags=True,
            strip_tags=True,
            strip_special_divs=True,
            remove_dates=True,
            remove_empty_leaves=True,
            unnest_nodes=True,
            remake_tree=True,
            css_rules=[
                "[class~='footer']",
                "[class~='site-info']",
            ],
            css_rules_replace_with_text={"[class~='more-link']": "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED"},
        )
        self.pre_extraction_simplificator_not_merge_texts = PreExtractionSimplificator(
            only_text_image_nodes=True,
            format_texts=True,
            merge_consecutive_text_nodes=False,
            interesting_attributes_set_cat=InterestingAttributesSetCategory.WIKIPEDIA,
        )
        self.pre_extraction_simplificator_merge_texts = PreExtractionSimplificator(
            only_text_image_nodes=True,
            format_texts=True,
            merge_consecutive_text_nodes=True,
            interesting_attributes_set_cat=InterestingAttributesSetCategory.WIKIPEDIA,
        )

        # self.extractor = TextMediaPairsExtractor(
        #     dom_tree_simplificator=self.dom_tree_simplificator,
        #     pre_extraction_simplificator=self.pre_extraction_simplificator_merge_texts,
        #     also_extract_images_not_in_simplified_dom_tree=True,
        #     extract_clip_scores=True,
        # )

    def visualization(self):
        # [FUNCTION DOC] Entry point: renders the title, then the mode selector, the document selector, and
        # the
        # simplification and extraction sections (each only renders when its mode is selected).
        st.title(
            "Visualization of DOM tree simplification strategies, "
            "web document rendering, and text-image pair extractions"
        )
        self.choose_mode()
        self.choose_example()
        self.simplification_mode()
        self.extraction_mode()

    def choose_mode(self):
        # [FUNCTION DOC] Renders a selectbox ("Simplification" / "Extraction", default Extraction) and stores
        # the choice in
        # self.mode.
        st.header("Mode")
        self.mode = st.selectbox(
            label="Choose a mode",
            options=["Simplification", "Extraction"],
            index=1,
        )

    def choose_example(self):
        # [FUNCTION DOC] Renders a "random document" button and a number input for the document index
        # (0..num_docs-1; the
        # random button preselects a random index) and stores self.current_example = self.examples[idx].
        # KNOWN ISSUE: the bound is self.num_docs (1000), not the size of the filtered dataset, so indices
        # beyond
        # len(self.examples) raise IndexError.
        st.header("Document")
        if st.button("Select a random document"):
            dct_idx = random.randint(a=0, b=self.num_docs - 1)
        else:
            dct_idx = 0
        idx = st.number_input(
            f"Select a document among the first {self.num_docs} ones",
            min_value=0,
            max_value=self.num_docs - 1,
            value=dct_idx,
            step=1,
            help=f"Index between 0 and {self.num_docs-1}",
        )
        self.current_example = self.examples[idx]

    def get_dom_viz_html(self, html):
        # [FUNCTION DOC] Returns the DOM-visualization page for an HTML string: the template rendered with
        # body_html = the <body>
        # HTML of the page after removing <script> tags.
        def get_body_html_string(html):
            # [FUNCTION DOC] HELPER: parses html with make_selectolax_tree, strips script tags and returns
            # the body's HTML string.
            tree = make_selectolax_tree(html)
            tree.strip_tags(["script"])
            return tree.body.html

        body_html = get_body_html_string(html)
        rendered_dom = self.dom_viz_template.render(body_html=body_html)
        return rendered_dom

    def simplification_mode(self):
        # [FUNCTION DOC] In "Simplification" mode: simplifies the current document's HTML and renders side by
        # side (raw vs
        # simplified) the rendered pages, the DOM-tree visualizations and the HTML source.
        # KNOWN ISSUE: calls self.dom_tree_simplificator, which is never defined (only _v1/_v2 exist), so
        # this mode
        # raises AttributeError.
        if self.mode == "Simplification":
            current_html = self.current_example["html"]
            current_url = self.current_example["document_url"]

            simplified_current_html = self.dom_tree_simplificator(current_html, type_return="str")

            def display_rendered_webpages():
                # [FUNCTION DOC] HELPER: shows the URL and two columns with the raw and simplified HTML
                # rendered in iframes.
                st.header("Rendered webpage")
                st.markdown(f"Webpage url: {current_url}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Raw html rendering")
                    st.components.v1.html(current_html, height=450, scrolling=True)
                with col2:
                    st.subheader("Simplified html rendering")
                    st.components.v1.html(simplified_current_html, height=450, scrolling=True)

            def display_dom_trees():
                # [FUNCTION DOC] HELPER: shows two columns with the DOM-tree visualizations of the raw and
                # simplified HTML.
                st.header("DOM trees")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Raw DOM tree")
                    rendered_dom = self.get_dom_viz_html(current_html)
                    st.components.v1.html(rendered_dom, height=600, scrolling=True)
                with col2:
                    st.subheader("Simplified DOM tree")
                    simplified_rendered_dom = self.get_dom_viz_html(simplified_current_html)
                    st.components.v1.html(simplified_rendered_dom, height=600, scrolling=True)

            def display_html_codes():
                # [FUNCTION DOC] HELPER: shows two columns with the raw and simplified HTML source (wrapped
                # in <xmp> tags).
                st.header("HTML codes")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Raw HTML code")
                    st.components.v1.html("<xmp>" + current_html + "</xmp>", height=450, scrolling=True)
                with col2:
                    st.subheader("Simplified HTML code")
                    st.components.v1.html("<xmp>" + simplified_current_html + "</xmp>", height=450, scrolling=True)

            display_rendered_webpages()
            display_dom_trees()
            display_html_codes()

    def extraction_mode(self):
        # [FUNCTION DOC] In "Extraction" mode: simplifies the current document with v1 and v2 simplificators
        # (selectolax trees),
        # extracts text and image markdown from each, shows a diff of the texts (v1 old, v2 new) and the
        # images of
        # each version in two columns.
        if self.mode == "Extraction":

            def get_text_and_images(simplified_current_html_tree):
                # [FUNCTION DOC] HELPER: runs the merging PreExtractionSimplificator on a simplified tree
                # (page_url = current_url from the
                # enclosing scope) and returns (text, images): text is the concatenation of every "-text"
                # node followed by
                # a blank line; images is a list of markdown image strings for every "img" node's
                # media_info["src"].
                current_list_nodes_merge_texts = self.pre_extraction_simplificator_merge_texts(
                    simplified_current_html_tree, page_url=current_url
                )
                list_nodes = current_list_nodes_merge_texts
                texts = []
                images = []
                for node in list_nodes:
                    if node.tag == "-text":
                        # print(node.text)
                        texts.append(f"{node.text}\n\n")
                    elif node.tag == "img":
                        images.append(f"![img]({node.media_info['src']})\n\n")
                text = "".join(texts)
                return text, images

            current_html = self.current_example["html"]
            current_url = self.current_example["document_url"]
            st.write(f"Current url: {current_url}")

            simplified_current_html_tree_v1 = self.dom_tree_simplificator_v1(
                current_html, type_return="selectolax_tree"
            )
            text_v1, images_v1 = get_text_and_images(simplified_current_html_tree_v1)

            simplified_current_html_tree_v2 = self.dom_tree_simplificator_v2(
                current_html, type_return="selectolax_tree"
            )
            text_v2, images_v2 = get_text_and_images(simplified_current_html_tree_v2)

            diff_viewer.diff_viewer(old_text=text_v1, new_text=text_v2, lang="none")

            img_col_1, img_col_2 = st.columns(2)
            with img_col_1:
                st.subheader("V1")
                st.markdown("".join(images_v1))
            with img_col_2:
                st.subheader("V2")
                st.markdown("".join(images_v2))
            # current_list_nodes_not_merge_texts = self.pre_extraction_simplificator_not_merge_texts(
            #     simplified_current_html_tree, page_url=current_url
            # )
            # current_list_nodes_merge_texts = self.pre_extraction_simplificator_merge_texts(
            #     simplified_current_html_tree, page_url=current_url
            # )

            # extracted_images = self.extractor(html_str=current_html, page_url=current_url)

            # # For simplicity, only doing this replacement on the extracted images.
            # # Doing that before the extraction (i.e. in the DOM simplification) would be possible be would require
            # # more significant changes
            # replacement_dict = {
            #     elem["unformatted_src"]: elem["src"]
            #     for elem in extracted_images
            #     if elem["src"] != elem["unformatted_src"]
            # }

            # def replace_relative_paths(html_string):
            #     if replacement_dict:
            #         for k, v in replacement_dict.items():
            #             html_string = html_string.replace(k, v)
            #     return html_string

            # def display_rendered_webpages():
            #     st.header("Rendered webpage")
            #     st.markdown(f"Webpage url: {current_url}")

            #     display_raw_html_rendering = st.checkbox("Raw html rendering", value=True)
            #     display_simplified_html_rendering = st.checkbox("Simplified html rendering", value=True)
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         display_pre_extraction_visualization = st.checkbox(
            #             "Web document rendering (pre-extraction visualization)", value=True
            #         )
            #     with col2:
            #         if display_pre_extraction_visualization:
            #             merge_text_nodes = st.checkbox("Merge text nodes", value=True)

            #     list_display_pages = [
            #         [display_raw_html_rendering, "raw_html_rendering"],
            #         [display_simplified_html_rendering, "simplified_html_rendering"],
            #         [display_pre_extraction_visualization, "pre_extraction_visualization"],
            #     ]
            #     list_display_pages = [
            #         page_to_display
            #         for should_display_page, page_to_display in list_display_pages
            #         if should_display_page
            #     ]

            #     def display_specific_rendered_webpage(page_to_display, col):
            #         with col:
            #             if page_to_display == "raw_html_rendering":
            #                 st.subheader("Raw html rendering")
            #                 st.components.v1.html(replace_relative_paths(current_html), height=800, scrolling=True)
            #             elif page_to_display == "simplified_html_rendering":
            #                 st.subheader("Simplified html rendering")
            #                 st.components.v1.html(
            #                     replace_relative_paths(simplified_current_html), height=800, scrolling=True
            #                 )
            #             elif page_to_display == "pre_extraction_visualization":
            #                 st.subheader("Web document rendering (pre-extraction visualization)")

            #                 def list_nodes_to_visu():
            #                     if not merge_text_nodes:
            #                         list_nodes = current_list_nodes_not_merge_texts
            #                         reduce_levels = {
            #                             v: i + 1
            #                             for i, v in enumerate(sorted(list(set([node.level for node in list_nodes]))))
            #                         }
            #                         last_level = None
            #                         markdown = ""
            #                         for node in list_nodes:
            #                             if node.tag in ["-text", "img"]:
            #                                 current_level = reduce_levels[node.level]
            #                                 if last_level != current_level:
            #                                     markdown += (
            #                                         "#" * min(current_level, 6) + f" Level: {current_level}\n\n"
            #                                     )
            #                                     last_level = current_level
            #                                     path_in_tree_str = [tag for tag, _ in node.path_in_tree]
            #                                     markdown += f"**{'/'.join(path_in_tree_str)}**\n\n"
            #                                 if node.tag == "-text":
            #                                     markdown += f"{node.text}\n\n"
            #                                 elif node.tag == "img":
            #                                     markdown += f"![img]({node.media_info['src']})\n\n"
            #                         st.markdown(markdown)

            #                     else:
            #                         list_nodes = current_list_nodes_merge_texts
            #                         for node in list_nodes:
            #                             if node.tag == "-text":
            #                                 print(node.text)
            #                                 st.text(f"{node.text}\n\n")
            #                             elif node.tag == "img":
            #                                 st.markdown(f"![img]({node.media_info['src']})\n\n")

            #                 list_nodes_to_visu()

            #     if list_display_pages:
            #         columns = st.columns(len(list_display_pages))
            #         for page_to_display, col in zip(list_display_pages, columns):
            #             display_specific_rendered_webpage(page_to_display, col)

            # def display_extraction():
            #     st.header("Extracted content")
            #     if not extracted_images:
            #         st.write("No extracted content")
            #     else:
            #         df = pd.DataFrame(
            #             extracted_images,
            #             columns=[
            #                 "src",
            #                 "unformatted_src",
            #                 "format",
            #                 "rendered_width",
            #                 "rendered_height",
            #                 "original_width",
            #                 "original_height",
            #                 "formatted_filename",
            #                 "alt_text",
            #                 "extracted_text",
            #                 "clip_score_image_formatted_filename",
            #                 "clip_score_image_alt_text",
            #                 "clip_score_image_extracted_text",
            #                 "image_in_simplified_dom_tree",
            #             ],
            #         )

            #         for i, link in enumerate(df["src"]):
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 st.image(link, width=500, use_column_width=True)
            #             with col2:
            #                 src = f'<a target="_blank" href="{link}">{link.split("/")[-1]}</a>'
            #                 unformatted_src = df["unformatted_src"][i]
            #                 format = df["format"][i]
            #                 rendered_width = df["rendered_width"][i]
            #                 rendered_height = df["rendered_height"][i]
            #                 original_width = df["original_width"][i]
            #                 original_height = df["original_height"][i]
            #                 formatted_filename = df["formatted_filename"][i]
            #                 alt_text = df["alt_text"][i]
            #                 extracted_text = df["extracted_text"][i]
            #                 clip_score_image_formatted_filename = df["clip_score_image_formatted_filename"][i]
            #                 clip_score_image_alt_text = df["clip_score_image_alt_text"][i]
            #                 clip_score_image_extracted_text = df["clip_score_image_extracted_text"][i]
            #                 image_in_simplified_dom_tree = df["image_in_simplified_dom_tree"][i]

            #                 st.components.v1.html(
            #                     (
            #                         f"<p><strong>Source</strong>: {src}</p><p><strong>Unformated source</strong>:"
            #                         f" {unformatted_src}</p><p><strong>Format</strong>:"
            #                         f" {format}</p><p><strong>Rendered width</strong>:"
            #                         f" {rendered_width}</p><p><strong>Rendered height</strong>:"
            #                         f" {rendered_height}</p><p><strong>Original width</strong>:"
            #                         f" {original_width}</p><p><strong>Original height</strong>:"
            #                         f" {original_height}</p><p><strong>Formatted filename</strong>:"
            #                         f" {formatted_filename}</p><p><strong>Alt-text</strong>:"
            #                         f" {alt_text}</p><p><strong>Extracted text</strong>:"
            #                         f" {extracted_text}</p><p><strong>Clip score image/formatted filename</strong>:"
            #                         f" {clip_score_image_formatted_filename:.4f}</p><p><strong>Clip score"
            #                         f" image/alt-text</strong>: {clip_score_image_alt_text:.4f}</p><p><strong>Clip"
            #                         " score image/extracted text</strong>:"
            #                         f" {clip_score_image_extracted_text:.4f}</p><p><strong>Image in simplified DOM"
            #                         f" tree</strong>: {image_in_simplified_dom_tree}</p>"
            #                     ),
            #                     height=500,
            #                     scrolling=True,
            #                 )
            #             st.write("-----")

            # def display_dom_tree():
            #     st.header("Simplified DOM tree")
            #     simplified_rendered_dom = self.get_dom_viz_html(simplified_current_html)
            #     st.components.v1.html(simplified_rendered_dom, height=600, scrolling=True)

            # def display_html_code():
            #     st.header("Simplified HTML code")
            #     st.components.v1.html("<xmp>" + simplified_current_html + "</xmp>", height=450, scrolling=True)

            # display_rendered_webpages()
            # display_extraction()
            # display_dom_tree()
            # display_html_code()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    num_docs = 1_000
    dom_viz_template_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets/DOM_tree_viz.html")
    visualization = Visualization(num_docs=num_docs, dom_viz_template_path=dom_viz_template_path)
    visualization.visualization()
