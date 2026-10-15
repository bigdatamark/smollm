# [MODULE DOC] Converts a (simplified) selectolax DOM tree into a flat, ordered list of lightweight nodes
# (text, img, figure) ready for pair/document extraction: optionally keeps only text/image nodes, normalizes
# texts (restoring #BR_TAG# markers as newlines) and merges consecutive text nodes using separators inferred
# from the tags separating them (TAG_TO_SEP).
import re

from m4.sourcing.data_collection.utils import (
    MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET,
    TAG_TO_SEP,
    InterestingAttributesSetCategory,
    simplify_media_node,
)


class Node:
    # [CLASS DOC] Lightweight tree node: path_in_tree (list of [tag, node index] pairs from the root to this
    # node), media_info (mapping from simplify_media_node for media tags, else None), text (own text for
    # "-text" nodes, else ""), children (list of Node).
    def __init__(self, path_in_tree, media_info, text, children):
        # [FUNCTION DOC] Stores the four attributes as given.
        self.path_in_tree = path_in_tree
        self.media_info = media_info
        self.text = text
        self.children = children

    @property
    def tag(self):
        # [FUNCTION DOC] Tag of this node: the tag of the last path_in_tree entry.
        return self.path_in_tree[-1][0]

    @property
    def level(self):
        # [FUNCTION DOC] Depth of this node: length of path_in_tree.
        return len(self.path_in_tree)


class Tree:
    # [CLASS DOC] Recursive conversion of a selectolax node into Node objects, numbering nodes in pre-order.
    def __init__(
        self,
        selectolax_root_node,
        page_url,
        interesting_attributes_set_cat=InterestingAttributesSetCategory.COMMONCRAWL,
    ):
        # [FUNCTION DOC] Builds the Node tree from selectolax_root_node (page_url used to resolve media URLs;
        # interesting_attributes_set_cat selects which tags are media). Sets num_nodes and tree.
        self.num_nodes = 0
        self.interesting_attributes_set_cat = interesting_attributes_set_cat
        self.tree = self.make_tree(selectolax_root_node, page_url)

    def make_tree(self, selectolax_node, page_url, path_in_tree=[]):
        # [FUNCTION DOC] HELPER: recursively converts selectolax_node: media tags become leaf Nodes with
        # media_info, "-text" nodes become leaf Nodes with their raw own text, other tags become Nodes whose
        # children are the converted direct children (including text nodes). path_in_tree (default empty list,
        # not mutated) is the parent path. Increments num_nodes.
        tag = selectolax_node.tag
        path_in_tree = path_in_tree + [[tag, self.num_nodes]]
        self.num_nodes += 1

        if tag in MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET[self.interesting_attributes_set_cat]:
            return Node(
                path_in_tree=path_in_tree,
                media_info=simplify_media_node(selectolax_node, page_url=page_url),
                text="",
                children=[],
            )

        elif tag == "-text":
            return Node(
                path_in_tree=path_in_tree,
                media_info=None,
                text=selectolax_node.text(deep=False, separator="", strip=False),
                children=[],
            )

        return Node(
            path_in_tree=path_in_tree,
            media_info=None,
            text="",
            children=[
                self.make_tree(child, page_url=page_url, path_in_tree=path_in_tree)
                for child in selectolax_node.iter(include_text=True)
            ],
        )

    def traverse(self):
        # [FUNCTION DOC] Returns all Nodes in pre-order (depth-first) as a list.
        def traverse_recursive(node, list_nodes):
            # [FUNCTION DOC] HELPER: appends node then recurses into its children, accumulating into
            # list_nodes.
            list_nodes.append(node)
            for child_node in node.children:
                traverse_recursive(child_node, list_nodes=list_nodes)

        list_nodes = []
        traverse_recursive(self.tree, list_nodes=list_nodes)
        return list_nodes


class PreExtractionSimplificator:
    # [CLASS DOC] Callable configured with only_text_image_nodes, format_texts, merge_consecutive_text_nodes
    # (all default True) and interesting_attributes_set_cat.
    def __init__(
        self,
        only_text_image_nodes=True,
        format_texts=True,
        merge_consecutive_text_nodes=True,
        interesting_attributes_set_cat=InterestingAttributesSetCategory.COMMONCRAWL,
    ):
        # [FUNCTION DOC] Stores the options.
        self.only_text_image_nodes = only_text_image_nodes
        self.format_texts = format_texts
        self.merge_consecutive_text_nodes = merge_consecutive_text_nodes
        self.interesting_attributes_set_cat = interesting_attributes_set_cat

    def __call__(self, selectolax_tree, page_url):
        # [FUNCTION DOC] Builds a Tree from selectolax_tree.root, flattens it in pre-order and applies the
        # enabled steps in order (filter, format, merge). Returns the list of Node.
        tree = Tree(
            selectolax_tree.root, page_url=page_url, interesting_attributes_set_cat=self.interesting_attributes_set_cat
        )
        list_nodes = tree.traverse()

        if self.only_text_image_nodes:
            list_nodes = self._only_text_image_nodes(list_nodes)
        if self.format_texts:
            list_nodes = self._format_texts(list_nodes)
        if self.merge_consecutive_text_nodes:
            list_nodes = self._merge_consecutive_text_nodes(list_nodes)

        return list_nodes

    def _only_text_image_nodes(self, list_nodes):
        # [FUNCTION DOC] HELPER: keeps "-text" nodes, "figure" nodes and "img" nodes with non-empty
        # media_info.
        list_nodes = [
            node
            for node in list_nodes
            if (node.tag == "-text") or (node.tag == "figure") or ((node.tag == "img") and (node.media_info))
        ]
        return list_nodes

    def _format_texts(self, list_nodes):
        # [FUNCTION DOC] HELPER: formats every node's text with format_one_text (mutating nodes) and drops
        # text nodes left empty.
        def format_one_text(text):
            # [FUNCTION DOC] HELPER: replaces newlines/tabs by spaces, collapses repeated spaces, then splits
            # on "#BR_TAG#", strips each piece and joins with newlines, preserving one leading and one
            # trailing space if the text had them. Empty stays empty.
            if text == "":
                return text
            text = text.replace("\n", " ")
            text = text.replace("\t", " ")
            text = re.sub(r"[ ]{2,}", " ", text)
            beg_sep = " " == text[0]
            end_sep = (" " == text[-1]) and (len(text) > 1)
            text = "\n".join([el.strip() for el in text.split("#BR_TAG#")])
            text = beg_sep * " " + text + end_sep * " "
            return text

        for idx, node in enumerate(list_nodes):
            list_nodes[idx].text = format_one_text(node.text)
        list_nodes = [node for node in list_nodes if (node.tag != "-text") or ((node.tag == "-text") and (node.text))]
        return list_nodes

    def _merge_consecutive_text_nodes(self, list_nodes):
        # [FUNCTION DOC] HELPER: walks the list merging each run of adjacent text nodes pairwise. For a pair,
        # trailing/leading "\n\n", "\n" or " " are trimmed and recorded as separators, plus TAG_TO_SEP
        # separators of the tags on both paths after their first differing position; the strongest separator
        # ("\n\n" > "\n" > " " > "") joins the texts and the merged node takes the second node's path. A text
        # node with no following text node gets path [["-text", 0]] and stripped text. Finally drops
        # whitespace-only text nodes. Mutates nodes and returns the new list.
        # KNOWN ISSUE: if one path is a prefix of the other, start_diff_path stays 0, so separators of all
        # ancestor tags (e.g. html/body) are added and the merge may use a stronger separator than intended.
        current_idx = 0
        while current_idx <= len(list_nodes) - 1:
            if list_nodes[current_idx].tag != "-text":
                current_idx += 1
            else:
                if (current_idx == len(list_nodes) - 1) or (
                    (current_idx + 1 <= len(list_nodes) - 1) and (list_nodes[current_idx + 1].tag != "-text")
                ):
                    list_nodes[current_idx].path_in_tree = [["-text", 0]]
                    list_nodes[current_idx].text = list_nodes[current_idx].text.strip()
                    current_idx += 1
                else:
                    seps = set()

                    text_1 = list_nodes[current_idx].text
                    text_2 = list_nodes[current_idx + 1].text

                    for char in ["\n\n", "\n", " "]:
                        if text_1.endswith(char):
                            seps.add(char)
                            text_1 = text_1[: -len(char)]
                        if text_2.startswith(char):
                            seps.add(char)
                            text_2 = text_2[len(char) :]

                    path_1 = list_nodes[current_idx].path_in_tree
                    path_2 = list_nodes[current_idx + 1].path_in_tree

                    start_diff_path = 0
                    for i in range(min(len(path_1), len(path_2))):
                        if path_1[i] != path_2[i]:
                            start_diff_path = i
                            break
                    for tag, _ in path_1[start_diff_path:] + path_2[start_diff_path:]:
                        if tag in TAG_TO_SEP:
                            seps.add(TAG_TO_SEP[tag])

                    if "\n\n" in seps:
                        sep = "\n\n"
                    elif "\n" in seps:
                        sep = "\n"
                    elif " " in seps:
                        sep = " "
                    else:
                        sep = ""

                    list_nodes[current_idx].path_in_tree = path_2
                    list_nodes[current_idx].text = text_1 + sep + text_2
                    del list_nodes[current_idx + 1]

        list_nodes = [
            node for node in list_nodes if (node.tag != "-text") or ((node.tag == "-text") and (node.text.strip()))
        ]
        return list_nodes
