# [MODULE DOC] Script (WebSight v0.2 step 3): filters generated (idea, html_code) pairs. A page is rejected
# if its
# lowercased HTML contains any banned substring (LIST_BAN_SUBSTRINGS: lorem ipsum, the example image
# keywords from the prompt, "description", raster image extensions) or if its visible text has too few
# words. When run as main: loads PATH_HTML_CODES (JSON list of pairs), keeps pairs passing HTMLFiltering
# (tqdm progress) and dumps them as JSON to PATH_SAVE_HTML_CODES_FILTERED.
import json
import re
from string import punctuation, whitespace
from typing import List, Union

from selectolax.parser import HTMLParser
from tqdm import tqdm


LIST_BAN_SUBSTRINGS = ["lorem", "?gym", "?cake", "description", ".png", ".jpg", ".jpeg"]
THRESHOLD_MIN_NUM_WORDS = 10

PATH_HTML_CODES = "/fsx/hugo/websight_v02_generated_html_codes/all_generated_html_code.json"
PATH_SAVE_HTML_CODES_FILTERED = "/fsx/hugo/websight_v02_generated_html_codes/all_generated_html_code_filtered.json"


class HTMLWordCounter:
    # [CLASS DOC] Callable that counts the words of the visible text of an HTML string, using selectolax.
    def __call__(self, html_str: str) -> int:
        # [FUNCTION DOC] Removes HTML comments, parses with selectolax, extracts root text (deep, space
        # separator, unstripped),
        # turns newlines into spaces, splits on single spaces, strips punctuation/whitespace from each token
        # and
        # drops empty tokens. Returns the number of remaining tokens (int).
        html_str = self._remove_html_comments(html_str=html_str)

        selectolax_tree = self._make_selectolax_tree(html_str=html_str)
        text = selectolax_tree.root.text(deep=True, separator=" ", strip=False)
        text = self._replace_breaking_lines_by_whitespaces(text=text)

        words = self._get_words_from_text(text=text)
        words = self._remove_punctuations_and_whitespaces(words)
        words = self._remove_empty_elements_from_list(words)
        num_words = len(words)
        return num_words

    def _remove_html_comments(self, html_str: str) -> str:
        # [FUNCTION DOC] HELPER: removes "<!-- ... -->" comments (non-greedy, across lines) from html_str and
        # returns it.
        # KNOWN ISSUE: the pattern puts the inline flag "(?s)" mid-pattern, which is deprecated and raises
        # re.error
        # ("global flags not at the start of the expression") on Python 3.11+.
        html_str = re.sub(r"<!--(?s).*?-->", "", html_str)
        return html_str

    def _make_selectolax_tree(self, html_str: str) -> HTMLParser:
        # [FUNCTION DOC] HELPER: returns a selectolax HTMLParser tree for html_str.
        selectolax_tree = HTMLParser(html_str)
        return selectolax_tree

    def _replace_breaking_lines_by_whitespaces(self, text: str) -> str:
        # [FUNCTION DOC] HELPER: returns text with every newline replaced by a space.
        text = text.replace("\n", " ")
        return text

    def _get_words_from_text(self, text: str) -> List[str]:
        # [FUNCTION DOC] HELPER: splits text on single spaces (empty strings kept) and returns the list.
        words = text.split(" ")
        return words

    def _remove_punctuations_and_whitespaces(self, list_: List[str]) -> List[str]:
        # [FUNCTION DOC] HELPER: returns a new list where each string is stripped of leading/trailing
        # punctuation and whitespace.
        list_ = [el.strip(punctuation + whitespace) for el in list_]
        return list_

    def _remove_empty_elements_from_list(self, list_: Union[List[str], None]) -> List[str]:
        # [FUNCTION DOC] HELPER: returns the list without empty (falsy) strings. Annotation allows None but
        # None would raise.
        list_ = [el for el in list_ if el]
        return list_


class HTMLFiltering:
    # [CLASS DOC] Callable predicate deciding whether a generated HTML page is kept.
    def __init__(self, list_ban_substrings: List[str], threshold_min_num_words: int) -> None:
        # [FUNCTION DOC] Stores list_ban_substrings (list of lowercase str), threshold_min_num_words (int)
        # and creates an
        # HTMLWordCounter.
        self.list_ban_substrings = list_ban_substrings
        self.threshold_min_num_words = threshold_min_num_words
        self.html_word_counter = HTMLWordCounter()

    def __call__(self, html_str: str) -> bool:
        # [FUNCTION DOC] Lowercases html_str; returns False if any banned substring occurs, False if the word
        # count is below 10,
        # True otherwise.
        # KNOWN ISSUE: compares against the hard-coded 10 instead of self.threshold_min_num_words (same value
        # today,
        # but the parameter is ignored).
        html_str = html_str.lower()
        if any([(ban_substring in html_str) for ban_substring in self.list_ban_substrings]):
            return False
        if self.html_word_counter(html_str=html_str) < 10:
            return False
        return True


if __name__ == "__main__":
    with open(PATH_HTML_CODES) as f:
        ideas_and_html_codes = json.load(f)

    html_filtering = HTMLFiltering(
        list_ban_substrings=LIST_BAN_SUBSTRINGS, threshold_min_num_words=THRESHOLD_MIN_NUM_WORDS
    )
    ideas_and_html_codes_filtered = [
        (idea, html_code) for idea, html_code in tqdm(ideas_and_html_codes) if html_filtering(html_str=html_code)
    ]

    with open(PATH_SAVE_HTML_CODES_FILTERED, "w") as f:
        json.dump(ideas_and_html_codes_filtered, f)
