# [MODULE DOC] DOM tree simplification for raw HTML pages, built on selectolax (via make_selectolax_tree).
# Applies a configurable sequence of string-level cleanups and tree-level transformations to reduce a page to
# the nodes that matter for text/media extraction (tags in INTERESTING_TAGS_SET and media tags).
import re

from m4.sourcing.data_collection.utils import (
    INTERESTING_TAGS_SET,
    MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET,
    UNWRAP_TAGS,
    InterestingAttributesSetCategory,
    get_media_src,
    make_selectolax_tree,
)


class DOMTreeSimplificator:
    # [CLASS DOC] Configurable, callable HTML simplifier. Each boolean option enables one step (see __call__
    # for order).
    def __init__(
        self,
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
        preserve_img_children=False,
        remove_everything_after_node_id=None,
        css_rules=None,
        css_rules_replace_with_text=None,
        interesting_attributes_set_cat=InterestingAttributesSetCategory.COMMONCRAWL,
    ):
        # [FUNCTION DOC] Stores options: strip_multiple_linebreaks, strip_multiple_spaces,
        # remove_html_comments, replace_line_break_tags, unwrap_tags, strip_tags, strip_special_divs,
        # remove_dates, remove_empty_leaves, unnest_nodes, remake_tree (all default True);
        # preserve_img_children (False); remove_everything_after_node_id (list of element ids or None);
        # css_rules (list of CSS selectors whose nodes are deleted, or None); css_rules_replace_with_text
        # (mapping selector -> replacement text, or None); interesting_attributes_set_cat (which media-tag set
        # to use, default COMMONCRAWL). Raises ValueError if preserve_img_children and strip_tags are both
        # set.
        self.strip_multiple_linebreaks = strip_multiple_linebreaks
        self.strip_multiple_spaces = strip_multiple_spaces
        self.remove_html_comments = remove_html_comments
        self.replace_line_break_tags = replace_line_break_tags
        self.unwrap_tags = unwrap_tags
        self.strip_tags = strip_tags
        self.strip_special_divs = strip_special_divs
        self.remove_dates = remove_dates
        self.remove_empty_leaves = remove_empty_leaves
        self.unnest_nodes = unnest_nodes
        self.remake_tree = remake_tree
        self.preserve_img_children = preserve_img_children
        self.remove_everything_after_node_id = remove_everything_after_node_id
        self.css_rules = css_rules
        self.css_rules_replace_with_text = css_rules_replace_with_text
        self.interesting_attributes_set_cat = interesting_attributes_set_cat
        if self.preserve_img_children and self.strip_tags:
            raise ValueError("`preserve_img_children` and `strip_tags` are incompatible.")

    def __call__(
        self,
        html_str,
        type_return,
    ):
        # [FUNCTION DOC] Simplifies html_str. type_return must be "str" (returns HTML string) or
        # "selectolax_tree" (returns the tree), else ValueError. Order: string steps (collapse repeated
        # newlines, collapse repeated spaces, remove comments, replace <br> variants with the #BR_TAG#
        # marker), parse, then css removals, css text replacements, unwrap, strip non-interesting tags, cut
        # after node ids, preserve-img stripping, special divs, dates, empty leaves, unnesting, reparse.
        if type_return not in ["str", "selectolax_tree"]:
            raise ValueError("`type_return` must be `str` or `selectolax_tree`")

        if self.strip_multiple_linebreaks:
            html_str = self._strip_multiple_linebreaks(html_str)
        if self.strip_multiple_spaces:
            html_str = self._strip_multiple_spaces(html_str)
        if self.remove_html_comments:
            html_str = self._remove_html_comments(html_str)
        if self.replace_line_break_tags:
            html_str = self._replace_line_break_tags(html_str)

        selectolax_tree = make_selectolax_tree(html_str)

        if self.css_rules:
            selectolax_tree = self._remove_nodes_matching_css_rules(selectolax_tree)
        if self.css_rules_replace_with_text:
            selectolax_tree = self._replace_nodes_matching_css_rules_with_text(selectolax_tree)
        if self.unwrap_tags:
            selectolax_tree = self._unwrap_html_tree(selectolax_tree)
        if self.strip_tags:
            selectolax_tree = self._strip_html_tree(selectolax_tree)
        if self.remove_everything_after_node_id:
            selectolax_tree = self._remove_everything_after_node_id(selectolax_tree)
        if self.preserve_img_children:
            selectolax_tree = self._remove_non_interesting_nodes_but_preserve_very_interesting_children(
                selectolax_tree
            )

        if self.strip_special_divs:
            selectolax_tree = self._strip_special_divs(selectolax_tree)
        if self.remove_dates:
            selectolax_tree = self._remove_dates(selectolax_tree)
        if self.remove_empty_leaves:
            selectolax_tree = self._remove_empty_leaves(selectolax_tree)
        if self.unnest_nodes:
            selectolax_tree = self._unnest_nodes(selectolax_tree)
        if self.remake_tree:
            selectolax_tree = self._remake_tree(selectolax_tree)

        if type_return == "str":
            return selectolax_tree.html
        elif type_return == "selectolax_tree":
            return selectolax_tree

    def _strip_multiple_linebreaks(self, html_str):
        # [FUNCTION DOC] HELPER: replaces runs of 2+ newlines with one newline.
        html_str = re.sub(r"[\n]{2,}", "\n", html_str)
        return html_str

    def _strip_multiple_spaces(self, html_str):
        # [FUNCTION DOC] HELPER: replaces runs of 2+ spaces with one space.
        html_str = re.sub(r"[ ]{2,}", " ", html_str)
        return html_str

    def _remove_html_comments(self, html_str):
        # [FUNCTION DOC] HELPER: removes all <!-- ... --> comments (non-greedy, across lines). Unclosed
        # comments are kept.
        # KNOWN ISSUE: the pattern places the (?s) global flag mid-expression, which is an error on Python
        # 3.11+ (re.error).
        html_str = re.sub(r"<!--(?s).*?-->", "", html_str)
        return html_str

    def _replace_line_break_tags(self, html_str):
        # [FUNCTION DOC] HELPER: replaces "<br>", "<br/>", "<br />" and "</br>" with the literal marker
        # "#BR_TAG#" (restored later as newlines by the pre-extraction step).
        html_str = re.sub("<br>|<br/>|<br />|</br>", "#BR_TAG#", html_str)
        return html_str

    def _unwrap_html_tree(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: unwraps tags in UNWRAP_TAGS (keeping their content), then strips any
        # remaining empty ones; mutates and returns the tree.
        selectolax_tree.unwrap_tags(UNWRAP_TAGS)

        # `.unwrap_tags` won't unwrap+remove tags which are empty leaves (i.e. either with no children tags or no text).
        # For instance, `<a href="https://twitter.com/share"><img src="blo.png"></a>` will be unwrapped and mentions
        # of `a` will be removed, but `<a href="https://twitter.com/share"></a>` will stay as is.
        # As a consequence, we call `strip_tags` to remove these unwrap empty tags for good.
        selectolax_tree.strip_tags(UNWRAP_TAGS)

        return selectolax_tree

    def _remove_digits_string(self, string):
        # [FUNCTION DOC] HELPER: returns string with all digit runs removed (so "h1" -> "h" for tag set
        # lookups).
        string = re.sub(r"\d+", "", string)
        return string

    def _remove_nodes_matching_css_rules(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: repeatedly finds the first non-<html> node matching any selector in css_rules
        # and decomposes it (with its subtree), restarting after each deletion until none match. Returns the
        # tree.
        modification = True
        while modification:
            found_a_node = False
            for css_rule in self.css_rules:
                for node in selectolax_tree.css(css_rule):
                    if node.tag != "html":
                        node.decompose()
                        found_a_node = True
                        break
            if not found_a_node:
                modification = False
        return selectolax_tree

    def _replace_nodes_matching_css_rules_with_text(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: replaces every node matching each selector in css_rules_replace_with_text
        # with the mapped text. Returns the tree.
        for css_rule, text in self.css_rules_replace_with_text.items():
            for node in selectolax_tree.css(css_rule):
                node.replace_with(text)
        return selectolax_tree

    def _remove_non_interesting_nodes_but_preserve_very_interesting_children(
        self, selectolax_tree, very_interesting_children_tags=["img"]
    ):
        # [FUNCTION DOC] HELPER: alternative to _strip_html_tree that keeps images: walks the tree; for each
        # non-interesting node (tag, digits removed, not in INTERESTING_TAGS_SET) moves any descendant whose
        # tag is in very_interesting_children_tags (default ["img"]) to just before the node and marks the
        # node for removal; recurses into interesting nodes. Marked nodes are then decomposed non-recursively
        # (children kept). Returns the tree.
        def recursive_strip_and_preserve_interesting_children(current_node, nodes_to_remove):
            # [FUNCTION DOC] HELPER: recursive walk over current_node's direct children implementing the step
            # above; appends to and returns nodes_to_remove.
            for node in current_node.iter():
                if self._remove_digits_string(node.tag) not in INTERESTING_TAGS_SET:
                    nodes_to_move = []
                    if node.child is not None:
                        for children_node in node.child.traverse():
                            if children_node.tag in very_interesting_children_tags:
                                nodes_to_move.append(children_node)
                    if len(nodes_to_move) > 0:
                        for children_node in nodes_to_move:
                            node.insert_before(children_node)
                    nodes_to_remove.append(node)
                else:
                    nodes_to_remove = recursive_strip_and_preserve_interesting_children(node, nodes_to_remove)
            return nodes_to_remove

        nodes_to_remove = []

        recursive_strip_and_preserve_interesting_children(selectolax_tree.root, nodes_to_remove)
        for node in nodes_to_remove:
            node.decompose(recursive=False)
        return selectolax_tree

    def _strip_html_tree(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: collects the distinct tags of all nodes whose digit-stripped tag is not in
        # INTERESTING_TAGS_SET and strips them (content kept, tag removed). Returns the tree.
        """
        Strips all nodes with tags NOT in INTERESTING_TAGS_SET and has
        counterintuitively nothing to do with the STRIP_TAGS list
        """
        strip_tags_l = [
            node.tag
            for node in selectolax_tree.root.traverse()
            if self._remove_digits_string(node.tag) not in INTERESTING_TAGS_SET
        ]
        strip_tags_l = list(set(strip_tags_l))
        selectolax_tree.strip_tags(strip_tags_l)
        return selectolax_tree

    def _remove_everything_after_node_id(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: for every node whose id is in remove_everything_after_node_id, decomposes all
        # its following siblings and the node itself (e.g. cut reference/notes sections). Returns the tree.
        # Find all nodes with id "ref" or "note"
        nodes_to_remove = selectolax_tree.select(
            ", ".join([f"#{node_id}" for node_id in self.remove_everything_after_node_id])
        ).matches

        # Remove each node and its siblings
        for node in nodes_to_remove:
            current_node = node.next
            while current_node:
                next_node = current_node.next
                current_node.decompose()
                current_node = next_node
            node.decompose()
        return selectolax_tree

    def _strip_special_divs(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: repeatedly removes (with subtree) the first div whose id, class or title
        # equals one of footer, header, navigation, nav, navbar, menu, until none remain. Returns the tree.
        special_div_ids = ["footer", "header", "navigation", "nav", "navbar", "menu"]
        modification = True
        while modification:
            # Traverse the tree to find one node to remove, and remove it right then
            # to avoid the recursivity problem with `decompose`
            found_a_node = False
            for node in selectolax_tree.root.traverse():
                if node.tag == "div":
                    attributes = node.attributes
                    if (
                        ("id" in attributes and attributes["id"] in special_div_ids)
                        or ("class" in attributes and attributes["class"] in special_div_ids)
                        or ("title" in attributes and attributes["title"] in special_div_ids)
                    ):
                        node.decompose()
                        found_a_node = True
                        break
            if not found_a_node:
                modification = False
        return selectolax_tree

    def _remove_dates(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: for divs whose class attribute contains "date", removes their direct text
        # children. Returns the tree.
        nodes_to_remove = []
        for node in selectolax_tree.root.traverse():
            if node.tag == "div":
                if node.attributes:
                    if "class" in node.attributes:
                        if node.attributes["class"]:
                            if "date" in node.attributes["class"]:
                                nodes_to_remove += [
                                    child for child in node.iter(include_text=True) if child.tag == "-text"
                                ]
        for node in nodes_to_remove:
            node.decompose()
        return selectolax_tree

    def _remove_empty_leaves(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: iteratively removes (non-recursively) nodes that are non-media leaves without
        # children and without text (except <html>), and media nodes without a source (get_media_src), until
        # nothing changes, so emptiness propagates upward. Returns the tree.
        """
        Function used to remove empty leaves iteratively, so it also ends up also removing nodes
        that are higher up in the tree.
        """
        modification = True
        while modification:
            nodes_to_remove = [
                node
                for node in selectolax_tree.root.traverse()
                if (
                    (node.tag not in MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET[self.interesting_attributes_set_cat])
                    and (not [child for child in node.iter()])
                    and (not node.text().strip())
                    and (node.tag != "html")
                )
                or (
                    (node.tag in MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET[self.interesting_attributes_set_cat])
                    and not get_media_src(node)
                )
            ]
            if nodes_to_remove:
                for node in nodes_to_remove:
                    node.decompose(recursive=False)
            else:
                modification = False
        return selectolax_tree

    def _unnest_nodes(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: repeatedly replaces a node by its single child when both share the same tag
        # and the node has no own text (e.g. div>div), until no such pair exists. Returns the tree.
        modification = True
        while modification:
            modification = False
            for node in selectolax_tree.root.traverse():
                children = [child for child in node.iter()]
                if len(children) == 1:
                    child = children[0]
                    if node.tag == child.tag:
                        text = node.text(deep=False).strip()
                        if not text:
                            node.replace_with(child)
                            modification = True
                            break
        return selectolax_tree

    def _remake_tree(self, selectolax_tree):
        # [FUNCTION DOC] HELPER: reparses the tree's HTML into a fresh selectolax tree (merges adjacent text
        # nodes) and returns it.
        """It could be interesting to remake a tree after the
        simplifications since it can now merge some text nodes
        that couldn't be merged before"""
        return make_selectolax_tree(selectolax_tree.html)
