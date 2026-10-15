# [MODULE DOC] Quality filtering of web documents (and reusable text/image checks). FilteringFunctions holds
# stateless checks (each check_* returns True to keep); WebDocumentFilteringNodeLevel removes individual
# images and text paragraphs from a document (used with Dataset.map); WebDocumentFilteringDocLevel keeps or
# drops whole documents (used with Dataset.filter). Text statistics follow the BigScience/ROOTS filtering
# recipes; lang-id uses FastText, perplexity uses SentencePiece + KenLM models loaded from disk.
import json
import re
from collections import Counter

import fasttext
import kenlm
import numpy as np
import sentencepiece

from m4.sourcing.data_collection.utils import SPECIAL_CHARACTERS


class FilteringFunctions:
    # [CLASS DOC] Namespace of static helper and check methods; check_* return True when the input passes
    # (should be kept).
    @staticmethod
    def check_format(image_metadata, valid_formats):
        # [FUNCTION DOC] False only if image_metadata has a "format" not in valid_formats.
        if "format" in image_metadata:
            if image_metadata["format"] not in valid_formats:
                return False
        return True

    @staticmethod
    def check_size_image(
        image_metadata,
        original_width_min_cutoff,
        original_width_max_cutoff,
        original_height_min_cutoff,
        original_height_max_cutoff,
        rendered_width_min_cutoff,
        rendered_width_max_cutoff,
        rendered_height_min_cutoff,
        rendered_height_max_cutoff,
        aspect_ratio_max_cutoff,
    ):
        # [FUNCTION DOC] True iff original width and height lie in their [min, max] cutoffs, rendered
        # width/height (if present in image_metadata) in theirs, and width/height ratio in
        # [1/aspect_ratio_max_cutoff, aspect_ratio_max_cutoff]. Rendered cutoffs may be None only when
        # rendered sizes are absent (else TypeError).
        if not (original_width_min_cutoff <= image_metadata["original_width"] <= original_width_max_cutoff):
            return False
        if not (original_height_min_cutoff <= image_metadata["original_height"] <= original_height_max_cutoff):
            return False
        if "rendered_width" in image_metadata:
            if not (rendered_width_min_cutoff <= image_metadata["rendered_width"] <= rendered_width_max_cutoff):
                return False
        if "rendered_height" in image_metadata:
            if not (rendered_height_min_cutoff <= image_metadata["rendered_height"] <= rendered_height_max_cutoff):
                return False
        if not (
            1 / aspect_ratio_max_cutoff
            <= image_metadata["original_width"] / image_metadata["original_height"]
            <= aspect_ratio_max_cutoff
        ):
            return False
        return True

    @staticmethod
    def check_number_images(number_images, number_images_min_cutoff, number_images_max_cutoff):
        # [FUNCTION DOC] True iff number_images is within [number_images_min_cutoff,
        # number_images_max_cutoff].
        if (number_images < number_images_min_cutoff) or (number_images > number_images_max_cutoff):
            return False
        return True

    @staticmethod
    def remove_empty_el_from_list(list_):
        # [FUNCTION DOC] HELPER: returns the list without falsy elements.
        return [el for el in list_ if el]

    @staticmethod
    def remove_non_printing_characters(text, non_printing_characters_re):
        # [FUNCTION DOC] HELPER: deletes all matches of the compiled non_printing_characters_re from text.
        return non_printing_characters_re.sub("", text)

    @staticmethod
    def standardize_whitespace(
        text,
        whitespace=[
            " ",
            " ",
            " ",
            " ",
            " ",
            "　",
            " ",
            " ",
            " ",
            " ",
            "￼",
            "",
            " ",
        ],
    ):
        # [FUNCTION DOC] HELPER: maps each character found in the whitespace list (Unicode space variants and
        # a few special characters) to a plain space.
        """There are different whitespace characters."""
        whitespace = set(whitespace)
        text = "".join([char if char not in whitespace else " " for char in text])
        return text

    @staticmethod
    def split_on_whitespace(
        text,
        new_line=False,
        tab=False,
    ):
        # [FUNCTION DOC] HELPER: splits text on spaces (plus newlines if new_line, tabs if tab; both default
        # False) and drops empty parts.
        """This method also removes concatenated spaces."""
        sep = [" "] + new_line * ["\n"] + tab * ["\t"]
        sep = "|".join(sep)
        split_text = re.split(sep, text)
        split_text = FilteringFunctions.remove_empty_el_from_list(split_text)
        return split_text

    @staticmethod
    def strip(text, strip_characters):
        # [FUNCTION DOC] HELPER: strips leading/trailing characters contained in the strip_characters set;
        # empty text returned as is.
        """Way faster than text.strip(strip_characters)
        since strip_characters is a set instead of a str,
        and it contains a lot of elements (all the emojis)."""
        if not text:
            return text
        beg_ind = 0
        end_ind = len(text)
        for i in range(len(text)):
            if text[i] in strip_characters:
                beg_ind += 1
            else:
                break
        for i in range(1, len(text) + 1):
            if text[-i] in strip_characters:
                end_ind -= 1
            else:
                break
        text_stripped = text[beg_ind:end_ind]
        return text_stripped

    @staticmethod
    def get_words_from_text(text, lower_case=True, strip_words=True, strip_characters=SPECIAL_CHARACTERS):
        # [FUNCTION DOC] HELPER: splits on spaces/newlines/tabs, lowercases if lower_case (default True), and
        # if strip_words (default True) strips strip_characters (default SPECIAL_CHARACTERS) from each word
        # and drops empty ones. ValueError if strip_words with strip_characters None. Returns the word list.
        """Get words from a text. Non reversible since the text
        is split on multiple characters, words are stripped of
        special characters and characters are converted to lower case.
        Useful to compute ratios, like the stopword ratio."""
        if strip_words and strip_characters is None:
            raise ValueError("strip_characters must be provided if strip_words is True.")
        words = FilteringFunctions.split_on_whitespace(text=text, new_line=True, tab=True)
        if lower_case:
            words = [word.lower() for word in words]
        if strip_words:
            words = [FilteringFunctions.strip(word, strip_characters) for word in words]
            words = FilteringFunctions.remove_empty_el_from_list(words)
        return words

    @staticmethod
    def check_number_words(text, strip_characters, number_words_min_cutoff, number_words_max_cutoff):
        # [FUNCTION DOC] True iff the normalized word count of text is within [number_words_min_cutoff,
        # number_words_max_cutoff].
        words = FilteringFunctions.get_words_from_text(
            text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
        )
        number_words = len(words)
        if (number_words < number_words_min_cutoff) or (number_words > number_words_max_cutoff):
            return False
        return True

    @staticmethod
    def compute_character_repetition_ratio(text, character_repetition_length):
        # [FUNCTION DOC] Counts character n-grams of length character_repetition_length; takes the k most
        # frequent counts where k = min(floor(sqrt(number of distinct n-grams)), number of n-grams occurring
        # more than once) and returns their sum over the total count (0 if text shorter than n).
        def get_freq_character_ngrams(text, n):
            # [FUNCTION DOC] HELPER: Counter of all overlapping length-n substrings of text.
            character_ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
            freq_character_ngrams = Counter(character_ngrams)
            return freq_character_ngrams

        freq_character_ngrams = get_freq_character_ngrams(text=text, n=character_repetition_length)
        if len(freq_character_ngrams) == 0:
            return 0
        freq_character_ngrams = list(freq_character_ngrams.values())
        freq_character_ngrams = sorted(freq_character_ngrams, reverse=True)
        val_one = len([el for el in freq_character_ngrams if el == 1])
        num_rep_character_ngrams = min(
            int(np.sqrt(len(freq_character_ngrams))),
            len(freq_character_ngrams) - val_one,
        )
        character_repetition_ratio = sum(freq_character_ngrams[:num_rep_character_ngrams]) / sum(freq_character_ngrams)
        return character_repetition_ratio

    @staticmethod
    def check_character_repetition_ratio(
        text,
        character_repetition_length,
        character_repetition_max_cutoff,
    ):
        # [FUNCTION DOC] True iff the character repetition ratio is <= character_repetition_max_cutoff.
        character_repetition_ratio = FilteringFunctions.compute_character_repetition_ratio(
            text=text, character_repetition_length=character_repetition_length
        )
        if character_repetition_ratio > character_repetition_max_cutoff:
            return False
        return True

    @staticmethod
    def compute_word_repetition_ratio(text, strip_characters, word_repetition_length):
        # [FUNCTION DOC] Counts word n-grams (n = word_repetition_length, normalized words) and returns the
        # share of n-gram occurrences belonging to n-grams seen more than once (0 if no n-gram).
        def get_freq_word_ngrams(text, strip_characters, n):
            # [FUNCTION DOC] HELPER: Counter of space-joined overlapping word n-grams of text.
            words = FilteringFunctions.get_words_from_text(
                text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
            )
            word_ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            freq_word_ngrams = Counter(word_ngrams)
            return freq_word_ngrams

        freq_word_ngrams = get_freq_word_ngrams(text=text, strip_characters=strip_characters, n=word_repetition_length)
        if len(freq_word_ngrams) == 0:
            return 0
        freq_word_ngrams = list(freq_word_ngrams.values())
        word_repetition_ratio = sum(freq for freq in freq_word_ngrams if freq > 1) / sum(freq_word_ngrams)
        return word_repetition_ratio

    @staticmethod
    def check_word_repetition_ratio(
        text,
        strip_characters,
        word_repetition_length,
        word_repetition_max_cutoff,
    ):
        # [FUNCTION DOC] True iff the word repetition ratio is <= word_repetition_max_cutoff.
        word_repetition_ratio = FilteringFunctions.compute_word_repetition_ratio(
            text=text, strip_characters=strip_characters, word_repetition_length=word_repetition_length
        )
        cond = word_repetition_ratio <= word_repetition_max_cutoff
        return cond

    @staticmethod
    def compute_special_character_ratio(text, special_characters):
        # [FUNCTION DOC] Fraction of characters of text in special_characters (0 for empty text).
        if len(text) == 0:
            return 0
        special_character_ratio = len([char for char in text if char in special_characters]) / len(text)
        return special_character_ratio

    @staticmethod
    def check_special_character_ratio(text, special_characters, special_character_ratio_max_cutoff):
        # [FUNCTION DOC] True iff the special character ratio is <= special_character_ratio_max_cutoff.
        special_character_ratio = FilteringFunctions.compute_special_character_ratio(
            text=text, special_characters=special_characters
        )
        if special_character_ratio > special_character_ratio_max_cutoff:
            return False
        return True

    @staticmethod
    def compute_stopword_ratio(text, strip_characters, stopwords):
        # [FUNCTION DOC] Fraction of normalized words that are in stopwords (0 if no words).
        words = FilteringFunctions.get_words_from_text(
            text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
        )
        if not words:
            return 0
        stopword_ratio = len([word for word in words if word in stopwords]) / len(words)
        return stopword_ratio

    @staticmethod
    def check_stopword_ratio(text, strip_characters, stopwords, stopword_ratio_min_cutoff):
        # [FUNCTION DOC] True iff the stopword ratio is >= stopword_ratio_min_cutoff.
        stopword_ratio = FilteringFunctions.compute_stopword_ratio(
            text=text, strip_characters=strip_characters, stopwords=stopwords
        )
        if stopword_ratio < stopword_ratio_min_cutoff:
            return False
        return True

    @staticmethod
    def compute_flagged_word_ratio(text, strip_characters, flagged_words):
        # [FUNCTION DOC] Fraction of normalized words that are in flagged_words (e.g. adult content words); 0
        # if no words.
        words = FilteringFunctions.get_words_from_text(
            text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
        )
        if not words:
            return 0
        flagged_word_ratio = len([word for word in words if word in flagged_words]) / len(words)
        return flagged_word_ratio

    @staticmethod
    def check_flagged_word_ratio(
        text,
        strip_characters,
        flagged_words,
        flagged_word_ratio_max_cutoff,
    ):
        # [FUNCTION DOC] True iff the flagged word ratio is <= flagged_word_ratio_max_cutoff.
        flagged_word_ratio = FilteringFunctions.compute_flagged_word_ratio(
            text=text,
            strip_characters=strip_characters,
            flagged_words=flagged_words,
        )
        if flagged_word_ratio > flagged_word_ratio_max_cutoff:
            return False
        return True

    @staticmethod
    def compute_punctuation_ratio(text, punctuation, min_nb_words=-1):
        # [FUNCTION DOC] Tokenizes text into runs of word characters/apostrophes and single punctuation marks
        # (from punctuation) and returns the fraction of tokens that are punctuation. Returns 0 if no tokens,
        # and 1.0 if min_nb_words > 0 and there are fewer tokens than that (short texts are not judged).
        # KNOWN ISSUE: the pattern uses an invalid "\w" escape in a non-raw f-string (SyntaxWarning) and
        # inserts the punctuation characters unescaped into a character class, so characters like "]", "^" or
        # "\" would break it.
        # The regepx is used to extract all words and punctuation marks from the text string.
        # `[\w']+`: This matches any sequence of one or more word characters (letters, digits, or underscores) or apostrophes.
        # The square brackets indicate a character class, which means "match any of these characters".
        # The backslash before the w means to escape it as it has a special meaning in regex,
        # and the plus sign means "one or more occurrences". We choose to ignore other special characters because in practise,
        # it's a good enough approximation to filter web documents that don't contain long sequences of (English) text.
        # `|`: This is the "or" operator in regex. It separates the two alternative patterns that the regex can match.
        # `[punctuation]`: This matches any one of the punctuation marks specified inside the square brackets.
        # This is inspired from https://stackoverflow.com/a/367292/6226208
        punc_splitters = "".join(punctuation)
        pattern = r"{}".format(f"[\w']+|[{punc_splitters}]")
        punctuation_splitted_words = re.findall(pattern, text)
        if not punctuation_splitted_words:
            return 0
        if min_nb_words > 0:
            if len(punctuation_splitted_words) < min_nb_words:
                # If the sequence is too short, we don't want to rely on punctuation to filter it out.
                return 1.0
        punctuation_ratio = len([word for word in punctuation_splitted_words if word in punctuation]) / len(
            punctuation_splitted_words
        )
        return punctuation_ratio

    @staticmethod
    def check_punctuation_ratio(
        text,
        punctuation,
        punctuation_ratio_min_cutoff,
        min_nb_words=-1,
    ):
        # [FUNCTION DOC] True iff the punctuation ratio (min_nb_words default -1, i.e. disabled) is >=
        # punctuation_ratio_min_cutoff.
        punctuation_ratio = FilteringFunctions.compute_punctuation_ratio(
            text=text,
            punctuation=punctuation,
            min_nb_words=min_nb_words,
        )
        if punctuation_ratio < punctuation_ratio_min_cutoff:
            return False
        return True

    @staticmethod
    def compute_common_word_ratio(text, strip_characters, common_words):
        # [FUNCTION DOC] Fraction of normalized words present in common_words (0 if no words).
        words = FilteringFunctions.get_words_from_text(
            text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
        )
        if not words:
            return 0
        common_word_ratio = len([word for word in words if word in common_words]) / len(words)
        return common_word_ratio

    @staticmethod
    def check_common_word_ratio(
        text,
        strip_characters,
        common_words,
        common_word_ratio_min_cutoff,
    ):
        # [FUNCTION DOC] True iff the common word ratio is >= common_word_ratio_min_cutoff.
        common_word_ratio = FilteringFunctions.compute_common_word_ratio(
            text=text,
            strip_characters=strip_characters,
            common_words=common_words,
        )
        if common_word_ratio < common_word_ratio_min_cutoff:
            return False
        return True

    @staticmethod
    def compute_lang_id_pred_score(text, lang_id_model):
        # [FUNCTION DOC] Lowercases text, replaces newlines by spaces and runs the FastText model; returns
        # (top language code without the "__label__" prefix, its probability).
        text = text.lower().replace("\n", " ")
        tot_pred_lang_id = lang_id_model.predict(text)
        pred_lang_id = tot_pred_lang_id[0][0].replace("__label__", "")
        score_pred_lang_id = tot_pred_lang_id[1][0]
        return pred_lang_id, score_pred_lang_id

    @staticmethod
    def check_lang_id(text, lang_id_model, target_lang_id, lang_id_min_cutoff):
        # [FUNCTION DOC] True iff the predicted language equals target_lang_id and its score is >=
        # lang_id_min_cutoff.
        pred_lang_id, score_pred_lang_id = FilteringFunctions.compute_lang_id_pred_score(
            text=text, lang_id_model=lang_id_model
        )
        if (pred_lang_id != target_lang_id) or (score_pred_lang_id < lang_id_min_cutoff):
            return False
        return True

    @staticmethod
    def replace_digits_with_zeros(text, digits_re):
        # [FUNCTION DOC] HELPER: replaces every match of digits_re in text with "0".
        return digits_re.sub("0", text)

    @staticmethod
    def replace_unicode_punctuation(text, unicode_punctuation):
        # [FUNCTION DOC] HELPER: maps each character through the unicode_punctuation mapping (unmapped
        # characters kept).
        return "".join(unicode_punctuation.get(c, c) for c in text)

    @staticmethod
    def normalization(
        text,
        remove_non_printing_characters,
        strip,
        lower_case,
        standardize_whitespace,
        replace_digits_with_zeros,
        replace_unicode_punctuation,
        non_printing_characters_re,
        digits_re,
        unicode_punctuation,
    ):
        # [FUNCTION DOC] HELPER: optionally removes non-printing characters, strips, (returns early if empty)
        # lowercases, standardizes whitespace, replaces digits with zeros and normalizes Unicode punctuation,
        # according to the boolean flags.
        if remove_non_printing_characters:
            text = FilteringFunctions.remove_non_printing_characters(
                text=text, non_printing_characters_re=non_printing_characters_re
            )
        if strip:
            text = text.strip()
        if not text:
            return text
        if lower_case:
            text = text.lower()
        if standardize_whitespace:
            text = FilteringFunctions.standardize_whitespace(text=text)
        if replace_digits_with_zeros:
            text = FilteringFunctions.replace_digits_with_zeros(text=text, digits_re=digits_re)
        if replace_unicode_punctuation:
            text = FilteringFunctions.replace_unicode_punctuation(text=text, unicode_punctuation=unicode_punctuation)
        return text

    @staticmethod
    def tokenization(text, sentencepiece_model, join_on_whitespace):
        # [FUNCTION DOC] HELPER: SentencePiece-encodes text into pieces; returns them joined by spaces if
        # join_on_whitespace, else the list.
        text_tokenized = sentencepiece_model.encode_as_pieces(text)
        if join_on_whitespace:
            text_tokenized = " ".join(text_tokenized)
        return text_tokenized

    @staticmethod
    def compute_perplexity_score(
        text, non_printing_characters_re, digits_re, unicode_punctuation, sentencepiece_model, kenlm_model
    ):
        # [FUNCTION DOC] Normalizes text (CCNet-style: non-printing removed, stripped, whitespace
        # standardized, digits zeroed, punctuation normalized, case kept), SentencePiece-tokenizes it, then
        # sums KenLM log10 scores and token counts (+1 per line for end of sentence) over lines; returns 10 **
        # (-total score / total length) rounded to 1 decimal.
        text = FilteringFunctions.normalization(
            text=text,
            remove_non_printing_characters=True,
            strip=True,
            lower_case=False,
            standardize_whitespace=True,
            replace_digits_with_zeros=True,
            replace_unicode_punctuation=True,
            non_printing_characters_re=non_printing_characters_re,
            digits_re=digits_re,
            unicode_punctuation=unicode_punctuation,
        )
        text = FilteringFunctions.tokenization(
            text=text, sentencepiece_model=sentencepiece_model, join_on_whitespace=True
        )
        doc_log_score, doc_length = 0, 0
        for line in text.split("\n"):
            log_score = kenlm_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        pp_score = 10.0 ** (-doc_log_score / doc_length)
        pp_score = round(pp_score, 1)
        return pp_score

    @staticmethod
    def check_perplexity_score(
        text,
        non_printing_characters_re,
        digits_re,
        unicode_punctuation,
        sentencepiece_model,
        kenlm_model,
        perplexity_score_max_cutoff,
    ):
        # [FUNCTION DOC] True iff the perplexity score is <= perplexity_score_max_cutoff.
        perplexity_score = FilteringFunctions.compute_perplexity_score(
            text=text,
            non_printing_characters_re=non_printing_characters_re,
            digits_re=digits_re,
            unicode_punctuation=unicode_punctuation,
            sentencepiece_model=sentencepiece_model,
            kenlm_model=kenlm_model,
        )
        if perplexity_score > perplexity_score_max_cutoff:
            return False
        return True


class WebDocumentFilteringNodeLevel:
    # [CLASS DOC] Node-level filter: callable on a web document (texts, images, metadata JSON) returning the
    # cleaned document. Uses __slots__ and __reduce__ so it can be pickled for multiprocessing.
    # Needed to make multiprocessing work
    __slots__ = (
        "cond_check_format",
        "valid_formats",
        "cond_check_size_image",
        "original_width_min_cutoff",
        "original_width_max_cutoff",
        "original_height_min_cutoff",
        "original_height_max_cutoff",
        "rendered_width_min_cutoff",
        "rendered_width_max_cutoff",
        "rendered_height_min_cutoff",
        "rendered_height_max_cutoff",
        "aspect_ratio_max_cutoff",
        "cond_remove_non_printing_characters",
        "non_printing_characters_re",
        "cond_standardize_whitespace",
        "cond_check_number_words_node_level",
        "strip_characters",
        "number_words_node_level_min_cutoff",
        "number_words_node_level_max_cutoff",
        "cond_check_character_repetition_ratio_node_level",
        "character_repetition_length_node_level",
        "character_repetition_node_level_max_cutoff",
        "cond_check_word_repetition_ratio_node_level",
        "word_repetition_length_node_level",
        "word_repetition_node_level_max_cutoff",
        "cond_check_special_character_ratio_node_level",
        "special_character_ratio_node_level_max_cutoff",
        "cond_check_stopword_ratio_node_level",
        "stopwords",
        "stopword_ratio_node_level_min_cutoff",
        "cond_check_flagged_word_ratio_node_level",
        "flagged_words",
        "flagged_word_ratio_node_level_max_cutoff",
        "cond_check_punctuation_ratio_node_level",
        "min_number_words_to_check_punctuation_ratio_node_level",
        "punctuation",
        "punctuation_ratio_node_level_min_cutoff",
        "cond_check_common_word_ratio_node_level",
        "path_common_words",
        "common_words",
        "common_word_ratio_node_level_min_cutoff",
        "cond_check_lang_id_node_level",
        "path_lang_id_model",
        "lang_id_model",
        "lang_id_node_level_min_cutoff",
        "cond_check_perplexity_score_node_level",
        "digits_re",
        "unicode_punctuation",
        "path_sentencepiece_model",
        "sentencepiece_model",
        "path_kenlm_model",
        "kenlm_model",
        "perplexity_score_node_level_max_cutoff",
    )

    def __init__(
        self,
        cond_check_format,
        valid_formats,
        cond_check_size_image,
        original_width_min_cutoff,
        original_width_max_cutoff,
        original_height_min_cutoff,
        original_height_max_cutoff,
        rendered_width_min_cutoff,
        rendered_width_max_cutoff,
        rendered_height_min_cutoff,
        rendered_height_max_cutoff,
        aspect_ratio_max_cutoff,
        cond_remove_non_printing_characters,
        non_printing_characters_re,
        cond_standardize_whitespace,
        cond_check_number_words_node_level,
        strip_characters,
        number_words_node_level_min_cutoff,
        number_words_node_level_max_cutoff,
        cond_check_character_repetition_ratio_node_level,
        character_repetition_length_node_level,
        character_repetition_node_level_max_cutoff,
        cond_check_word_repetition_ratio_node_level,
        word_repetition_length_node_level,
        word_repetition_node_level_max_cutoff,
        cond_check_special_character_ratio_node_level,
        special_character_ratio_node_level_max_cutoff,
        cond_check_stopword_ratio_node_level,
        stopwords,
        stopword_ratio_node_level_min_cutoff,
        cond_check_flagged_word_ratio_node_level,
        flagged_words,
        flagged_word_ratio_node_level_max_cutoff,
        cond_check_punctuation_ratio_node_level,
        min_number_words_to_check_punctuation_ratio_node_level,
        punctuation,
        punctuation_ratio_node_level_min_cutoff,
        cond_check_common_word_ratio_node_level,
        path_common_words,
        common_word_ratio_node_level_min_cutoff,
        cond_check_lang_id_node_level,
        path_lang_id_model,
        lang_id_node_level_min_cutoff,
        cond_check_perplexity_score_node_level,
        digits_re,
        unicode_punctuation,
        path_sentencepiece_model,
        path_kenlm_model,
        perplexity_score_node_level_max_cutoff,
    ):
        # [FUNCTION DOC] Stores every switch/threshold (format, original/rendered size and aspect ratio for
        # images; non-printing removal, whitespace standardization and the node-level text checks: number of
        # words, character/word repetition, special characters, stopwords, flagged words, punctuation with a
        # minimum word count, common words, lang id, perplexity) plus the resources (regexes, character/word
        # sets, punctuation list, Unicode punctuation map). Always loads common words JSON from
        # path_common_words; loads the FastText model only if lang-id check is enabled and the
        # SentencePiece/KenLM models only if the perplexity check is enabled.
        self.cond_check_format = cond_check_format
        self.valid_formats = valid_formats

        self.cond_check_size_image = cond_check_size_image
        self.original_width_min_cutoff = original_width_min_cutoff
        self.original_width_max_cutoff = original_width_max_cutoff
        self.original_height_min_cutoff = original_height_min_cutoff
        self.original_height_max_cutoff = original_height_max_cutoff
        self.rendered_width_min_cutoff = rendered_width_min_cutoff
        self.rendered_width_max_cutoff = rendered_width_max_cutoff
        self.rendered_height_min_cutoff = rendered_height_min_cutoff
        self.rendered_height_max_cutoff = rendered_height_max_cutoff
        self.aspect_ratio_max_cutoff = aspect_ratio_max_cutoff

        self.cond_remove_non_printing_characters = cond_remove_non_printing_characters
        self.non_printing_characters_re = non_printing_characters_re

        self.cond_standardize_whitespace = cond_standardize_whitespace

        self.cond_check_number_words_node_level = cond_check_number_words_node_level
        self.strip_characters = strip_characters
        self.number_words_node_level_min_cutoff = number_words_node_level_min_cutoff
        self.number_words_node_level_max_cutoff = number_words_node_level_max_cutoff

        self.cond_check_character_repetition_ratio_node_level = cond_check_character_repetition_ratio_node_level
        self.character_repetition_length_node_level = character_repetition_length_node_level
        self.character_repetition_node_level_max_cutoff = character_repetition_node_level_max_cutoff

        self.cond_check_word_repetition_ratio_node_level = cond_check_word_repetition_ratio_node_level
        self.word_repetition_length_node_level = word_repetition_length_node_level
        self.word_repetition_node_level_max_cutoff = word_repetition_node_level_max_cutoff

        self.cond_check_special_character_ratio_node_level = cond_check_special_character_ratio_node_level
        self.special_character_ratio_node_level_max_cutoff = special_character_ratio_node_level_max_cutoff

        self.cond_check_stopword_ratio_node_level = cond_check_stopword_ratio_node_level
        self.stopwords = stopwords
        self.stopword_ratio_node_level_min_cutoff = stopword_ratio_node_level_min_cutoff

        self.cond_check_flagged_word_ratio_node_level = cond_check_flagged_word_ratio_node_level
        self.flagged_words = flagged_words
        self.flagged_word_ratio_node_level_max_cutoff = flagged_word_ratio_node_level_max_cutoff

        self.cond_check_punctuation_ratio_node_level = cond_check_punctuation_ratio_node_level
        self.min_number_words_to_check_punctuation_ratio_node_level = (
            min_number_words_to_check_punctuation_ratio_node_level
        )
        self.punctuation = punctuation
        self.punctuation_ratio_node_level_min_cutoff = punctuation_ratio_node_level_min_cutoff

        self.cond_check_common_word_ratio_node_level = cond_check_common_word_ratio_node_level
        self.path_common_words = path_common_words
        with open(path_common_words) as f:
            self.common_words = json.load(f)
        self.common_word_ratio_node_level_min_cutoff = common_word_ratio_node_level_min_cutoff

        self.cond_check_lang_id_node_level = cond_check_lang_id_node_level
        self.path_lang_id_model = path_lang_id_model
        if cond_check_lang_id_node_level:
            self.lang_id_model = fasttext.load_model(path_lang_id_model)
        self.lang_id_node_level_min_cutoff = lang_id_node_level_min_cutoff

        self.cond_check_perplexity_score_node_level = cond_check_perplexity_score_node_level
        self.digits_re = digits_re
        self.unicode_punctuation = unicode_punctuation
        self.path_sentencepiece_model = path_sentencepiece_model
        if cond_check_perplexity_score_node_level:
            self.sentencepiece_model = sentencepiece.SentencePieceProcessor()
            self.sentencepiece_model.load(path_sentencepiece_model)
        self.path_kenlm_model = path_kenlm_model
        if cond_check_perplexity_score_node_level:
            self.kenlm_model = kenlm.Model(path_kenlm_model)
        self.perplexity_score_node_level_max_cutoff = perplexity_score_node_level_max_cutoff

    def __call__(self, web_document):
        # [FUNCTION DOC] For each position of the document: an image gets original size from the PIL image,
        # percentage rendered sizes converted to pixels (dropped if unparsable) and its lowercase format, then
        # is removed if it fails the format or size checks; a text is optionally cleaned (non-printing,
        # whitespace), split into paragraphs on blank lines, and each paragraph failing any enabled check is
        # removed (the END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED paragraph is always kept); a text with no
        # paragraph left is removed, otherwise replaced by the cleaned remaining paragraphs; positions with
        # neither text nor image are removed. Rewrites texts, images and metadata (JSON) of web_document and
        # returns it.
        texts = web_document["texts"]
        images = web_document["images"]
        metadata = json.loads(web_document["metadata"])

        indices_to_remove = set()

        for ind, (text, image, meta) in enumerate(zip(texts, images, metadata)):
            if image is not None:
                meta["original_width"], meta["original_height"] = image.size
                for side in ["width", "height"]:
                    if (
                        (f"rendered_{side}" in meta)
                        and isinstance(meta[f"rendered_{side}"], str)
                        and ("%" in meta[f"rendered_{side}"])
                    ):
                        try:
                            meta[f"rendered_{side}"] = round(
                                meta[f"original_{side}"] * int(meta[f"rendered_{side}"].replace("%", "")) / 100
                            )
                        except Exception:
                            del meta[f"rendered_{side}"]
                if image.format:
                    meta["format"] = image.format.lower()

                if self.cond_check_format:
                    if not FilteringFunctions.check_format(image_metadata=meta, valid_formats=self.valid_formats):
                        indices_to_remove.add(ind)
                        continue

                if self.cond_check_size_image:
                    if not FilteringFunctions.check_size_image(
                        image_metadata=meta,
                        original_width_min_cutoff=self.original_width_min_cutoff,
                        original_width_max_cutoff=self.original_width_max_cutoff,
                        original_height_min_cutoff=self.original_height_min_cutoff,
                        original_height_max_cutoff=self.original_height_max_cutoff,
                        rendered_width_min_cutoff=self.rendered_width_min_cutoff,
                        rendered_width_max_cutoff=self.rendered_width_max_cutoff,
                        rendered_height_min_cutoff=self.rendered_height_min_cutoff,
                        rendered_height_max_cutoff=self.rendered_height_max_cutoff,
                        aspect_ratio_max_cutoff=self.aspect_ratio_max_cutoff,
                    ):
                        indices_to_remove.add(ind)
                        continue

                metadata[ind] = meta

            elif text is not None:
                if self.cond_remove_non_printing_characters:
                    text = FilteringFunctions.remove_non_printing_characters(
                        text=text, non_printing_characters_re=self.non_printing_characters_re
                    )

                if self.cond_standardize_whitespace:
                    text = FilteringFunctions.standardize_whitespace(text=text)

                paragraphs = text.split("\n\n")
                paragraphs_indices_to_remove = set()

                for ind_par, paragraph in enumerate(paragraphs):
                    if paragraph == "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED":
                        continue

                    if self.cond_check_number_words_node_level:
                        if not FilteringFunctions.check_number_words(
                            text=paragraph,
                            strip_characters=self.strip_characters,
                            number_words_min_cutoff=self.number_words_node_level_min_cutoff,
                            number_words_max_cutoff=self.number_words_node_level_max_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_character_repetition_ratio_node_level:
                        if not FilteringFunctions.check_character_repetition_ratio(
                            text=paragraph,
                            character_repetition_length=self.character_repetition_length_node_level,
                            character_repetition_max_cutoff=self.character_repetition_node_level_max_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_word_repetition_ratio_node_level:
                        if not FilteringFunctions.check_word_repetition_ratio(
                            text=paragraph,
                            strip_characters=self.strip_characters,
                            word_repetition_length=self.word_repetition_length_node_level,
                            word_repetition_max_cutoff=self.word_repetition_node_level_max_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_special_character_ratio_node_level:
                        if not FilteringFunctions.check_special_character_ratio(
                            text=paragraph,
                            special_characters=self.strip_characters,
                            special_character_ratio_max_cutoff=self.special_character_ratio_node_level_max_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_stopword_ratio_node_level:
                        if not FilteringFunctions.check_stopword_ratio(
                            text=paragraph,
                            strip_characters=self.strip_characters,
                            stopwords=self.stopwords,
                            stopword_ratio_min_cutoff=self.stopword_ratio_node_level_min_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_flagged_word_ratio_node_level:
                        if not FilteringFunctions.check_flagged_word_ratio(
                            text=paragraph,
                            strip_characters=self.strip_characters,
                            flagged_words=self.flagged_words,
                            flagged_word_ratio_max_cutoff=self.flagged_word_ratio_node_level_max_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_punctuation_ratio_node_level:
                        if not FilteringFunctions.check_punctuation_ratio(
                            text=paragraph,
                            punctuation=self.punctuation,
                            punctuation_ratio_min_cutoff=self.punctuation_ratio_node_level_min_cutoff,
                            min_nb_words=self.min_number_words_to_check_punctuation_ratio_node_level,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_common_word_ratio_node_level:
                        if not FilteringFunctions.check_common_word_ratio(
                            text=paragraph,
                            strip_characters=self.strip_characters,
                            common_words=self.common_words,
                            common_word_ratio_min_cutoff=self.common_word_ratio_node_level_min_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_lang_id_node_level:
                        if not FilteringFunctions.check_lang_id(
                            text=paragraph,
                            lang_id_model=self.lang_id_model,
                            target_lang_id="en",
                            lang_id_min_cutoff=self.lang_id_node_level_min_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                    if self.cond_check_perplexity_score_node_level:
                        if not FilteringFunctions.check_perplexity_score(
                            text=paragraph,
                            non_printing_characters_re=self.non_printing_characters_re,
                            digits_re=self.digits_re,
                            unicode_punctuation=self.unicode_punctuation,
                            sentencepiece_model=self.sentencepiece_model,
                            kenlm_model=self.kenlm_model,
                            perplexity_score_max_cutoff=self.perplexity_score_node_level_max_cutoff,
                        ):
                            paragraphs_indices_to_remove.add(ind_par)
                            continue

                paragraphs = [
                    el for ind_par, el in enumerate(paragraphs) if ind_par not in paragraphs_indices_to_remove
                ]
                if not paragraphs:
                    indices_to_remove.add(ind)
                else:
                    texts[ind] = "\n\n".join(paragraphs)

            else:
                indices_to_remove.add(ind)

        web_document["texts"] = [el for ind, el in enumerate(texts) if ind not in indices_to_remove]
        web_document["images"] = [el for ind, el in enumerate(images) if ind not in indices_to_remove]
        web_document["metadata"] = json.dumps([el for ind, el in enumerate(metadata) if ind not in indices_to_remove])

        return web_document

    # Needed to make multiprocessing work
    def __reduce__(self):
        # [FUNCTION DOC] HELPER: pickling support: recreates the object from its constructor arguments (models
        # reloaded from their paths).
        return (
            self.__class__,
            (
                self.cond_check_format,
                self.valid_formats,
                self.cond_check_size_image,
                self.original_width_min_cutoff,
                self.original_width_max_cutoff,
                self.original_height_min_cutoff,
                self.original_height_max_cutoff,
                self.rendered_width_min_cutoff,
                self.rendered_width_max_cutoff,
                self.rendered_height_min_cutoff,
                self.rendered_height_max_cutoff,
                self.aspect_ratio_max_cutoff,
                self.cond_remove_non_printing_characters,
                self.non_printing_characters_re,
                self.cond_standardize_whitespace,
                self.cond_check_number_words_node_level,
                self.strip_characters,
                self.number_words_node_level_min_cutoff,
                self.number_words_node_level_max_cutoff,
                self.cond_check_character_repetition_ratio_node_level,
                self.character_repetition_length_node_level,
                self.character_repetition_node_level_max_cutoff,
                self.cond_check_word_repetition_ratio_node_level,
                self.word_repetition_length_node_level,
                self.word_repetition_node_level_max_cutoff,
                self.cond_check_special_character_ratio_node_level,
                self.special_character_ratio_node_level_max_cutoff,
                self.cond_check_stopword_ratio_node_level,
                self.stopwords,
                self.stopword_ratio_node_level_min_cutoff,
                self.cond_check_flagged_word_ratio_node_level,
                self.flagged_words,
                self.flagged_word_ratio_node_level_max_cutoff,
                self.cond_check_punctuation_ratio_node_level,
                self.min_number_words_to_check_punctuation_ratio_node_level,
                self.punctuation,
                self.punctuation_ratio_node_level_min_cutoff,
                self.cond_check_common_word_ratio_node_level,
                self.path_common_words,
                self.common_word_ratio_node_level_min_cutoff,
                self.cond_check_lang_id_node_level,
                self.path_lang_id_model,
                self.lang_id_node_level_min_cutoff,
                self.cond_check_perplexity_score_node_level,
                self.digits_re,
                self.unicode_punctuation,
                self.path_sentencepiece_model,
                self.path_kenlm_model,
                self.perplexity_score_node_level_max_cutoff,
            ),
        )


class WebDocumentFilteringDocLevel:
    # [CLASS DOC] Document-level predicate for Dataset.filter, configured like the node-level filter but with
    # doc-level thresholds.
    def __init__(
        self,
        cond_check_number_images,
        number_images_min_cutoff,
        number_images_max_cutoff,
        cond_check_number_words_doc_level,
        strip_characters,
        number_words_doc_level_min_cutoff,
        number_words_doc_level_max_cutoff,
        cond_check_character_repetition_ratio_doc_level,
        character_repetition_length_doc_level,
        character_repetition_doc_level_max_cutoff,
        cond_check_word_repetition_ratio_doc_level,
        word_repetition_length_doc_level,
        word_repetition_doc_level_max_cutoff,
        cond_check_special_character_ratio_doc_level,
        special_character_ratio_doc_level_max_cutoff,
        cond_check_stopword_ratio_doc_level,
        stopwords,
        stopword_ratio_doc_level_min_cutoff,
        cond_check_flagged_word_ratio_doc_level,
        flagged_words,
        flagged_word_ratio_doc_level_max_cutoff,
        cond_check_punctuation_ratio_doc_level,
        punctuation,
        punctuation_ratio_doc_level_min_cutoff,
        cond_check_common_word_ratio_doc_level,
        path_common_words,
        common_word_ratio_doc_level_min_cutoff,
        cond_check_lang_id_doc_level,
        path_lang_id_model,
        lang_id_doc_level_min_cutoff,
        cond_check_perplexity_score_doc_level,
        non_printing_characters_re,
        digits_re,
        unicode_punctuation,
        path_sentencepiece_model,
        path_kenlm_model,
        perplexity_score_doc_level_max_cutoff,
    ):
        # [FUNCTION DOC] Stores switches/thresholds for number of images, number of words, character/word
        # repetition, special characters, stopwords, flagged words, punctuation, common words, lang id and
        # perplexity, plus the resources. Loads common words always and the FastText / SentencePiece+KenLM
        # models only when their checks are enabled.
        self.cond_check_number_images = cond_check_number_images
        self.number_images_min_cutoff = number_images_min_cutoff
        self.number_images_max_cutoff = number_images_max_cutoff

        self.cond_check_number_words_doc_level = cond_check_number_words_doc_level
        self.strip_characters = strip_characters
        self.number_words_doc_level_min_cutoff = number_words_doc_level_min_cutoff
        self.number_words_doc_level_max_cutoff = number_words_doc_level_max_cutoff

        self.cond_check_character_repetition_ratio_doc_level = cond_check_character_repetition_ratio_doc_level
        self.character_repetition_length_doc_level = character_repetition_length_doc_level
        self.character_repetition_doc_level_max_cutoff = character_repetition_doc_level_max_cutoff

        self.cond_check_word_repetition_ratio_doc_level = cond_check_word_repetition_ratio_doc_level
        self.word_repetition_length_doc_level = word_repetition_length_doc_level
        self.word_repetition_doc_level_max_cutoff = word_repetition_doc_level_max_cutoff

        self.cond_check_special_character_ratio_doc_level = cond_check_special_character_ratio_doc_level
        self.special_character_ratio_doc_level_max_cutoff = special_character_ratio_doc_level_max_cutoff

        self.cond_check_stopword_ratio_doc_level = cond_check_stopword_ratio_doc_level
        self.stopwords = stopwords
        self.stopword_ratio_doc_level_min_cutoff = stopword_ratio_doc_level_min_cutoff

        self.cond_check_flagged_word_ratio_doc_level = cond_check_flagged_word_ratio_doc_level
        self.flagged_words = flagged_words
        self.flagged_word_ratio_doc_level_max_cutoff = flagged_word_ratio_doc_level_max_cutoff

        self.cond_check_punctuation_ratio_doc_level = cond_check_punctuation_ratio_doc_level
        self.punctuation = punctuation
        self.punctuation_ratio_doc_level_min_cutoff = punctuation_ratio_doc_level_min_cutoff

        self.cond_check_common_word_ratio_doc_level = cond_check_common_word_ratio_doc_level
        self.path_common_words = path_common_words
        with open(path_common_words) as f:
            self.common_words = json.load(f)
        self.common_word_ratio_doc_level_min_cutoff = common_word_ratio_doc_level_min_cutoff

        self.cond_check_lang_id_doc_level = cond_check_lang_id_doc_level
        self.path_lang_id_model = path_lang_id_model
        if cond_check_lang_id_doc_level:
            self.lang_id_model = fasttext.load_model(path_lang_id_model)
        self.lang_id_doc_level_min_cutoff = lang_id_doc_level_min_cutoff

        self.cond_check_perplexity_score_doc_level = cond_check_perplexity_score_doc_level
        self.non_printing_characters_re = non_printing_characters_re
        self.digits_re = digits_re
        self.unicode_punctuation = unicode_punctuation
        self.path_sentencepiece_model = path_sentencepiece_model
        if cond_check_perplexity_score_doc_level:
            self.sentencepiece_model = sentencepiece.SentencePieceProcessor()
            self.sentencepiece_model.load(path_sentencepiece_model)
        self.path_kenlm_model = path_kenlm_model
        if cond_check_perplexity_score_doc_level:
            self.kenlm_model = kenlm.Model(path_kenlm_model)
        self.perplexity_score_doc_level_max_cutoff = perplexity_score_doc_level_max_cutoff

    def __call__(self, web_document):
        # [FUNCTION DOC] Joins all non-empty texts with blank lines and counts non-empty images, then applies
        # the enabled checks in order (image count, word count, char repetition, word repetition, special
        # characters, stopwords, flagged words, punctuation without minimum word count, common words, English
        # lang id, perplexity). Returns False on the first failure, else True.
        texts = web_document["texts"]
        images = web_document["images"]

        full_text = "\n\n".join([text for text in texts if text])
        all_images = [image for image in images if image]

        if self.cond_check_number_images:
            if not FilteringFunctions.check_number_images(
                number_images=len(all_images),
                number_images_min_cutoff=self.number_images_min_cutoff,
                number_images_max_cutoff=self.number_images_max_cutoff,
            ):
                return False

        if self.cond_check_number_words_doc_level:
            if not FilteringFunctions.check_number_words(
                text=full_text,
                strip_characters=self.strip_characters,
                number_words_min_cutoff=self.number_words_doc_level_min_cutoff,
                number_words_max_cutoff=self.number_words_doc_level_max_cutoff,
            ):
                return False

        if self.cond_check_character_repetition_ratio_doc_level:
            if not FilteringFunctions.check_character_repetition_ratio(
                text=full_text,
                character_repetition_length=self.character_repetition_length_doc_level,
                character_repetition_max_cutoff=self.character_repetition_doc_level_max_cutoff,
            ):
                return False

        if self.cond_check_word_repetition_ratio_doc_level:
            if not FilteringFunctions.check_word_repetition_ratio(
                text=full_text,
                strip_characters=self.strip_characters,
                word_repetition_length=self.word_repetition_length_doc_level,
                word_repetition_max_cutoff=self.word_repetition_doc_level_max_cutoff,
            ):
                return False

        if self.cond_check_special_character_ratio_doc_level:
            if not FilteringFunctions.check_special_character_ratio(
                text=full_text,
                special_characters=self.strip_characters,
                special_character_ratio_max_cutoff=self.special_character_ratio_doc_level_max_cutoff,
            ):
                return False

        if self.cond_check_stopword_ratio_doc_level:
            if not FilteringFunctions.check_stopword_ratio(
                text=full_text,
                strip_characters=self.strip_characters,
                stopwords=self.stopwords,
                stopword_ratio_min_cutoff=self.stopword_ratio_doc_level_min_cutoff,
            ):
                return False

        if self.cond_check_flagged_word_ratio_doc_level:
            if not FilteringFunctions.check_flagged_word_ratio(
                text=full_text,
                strip_characters=self.strip_characters,
                flagged_words=self.flagged_words,
                flagged_word_ratio_max_cutoff=self.flagged_word_ratio_doc_level_max_cutoff,
            ):
                return False

        if self.cond_check_punctuation_ratio_doc_level:
            if not FilteringFunctions.check_punctuation_ratio(
                text=full_text,
                punctuation=self.punctuation,
                punctuation_ratio_min_cutoff=self.punctuation_ratio_doc_level_min_cutoff,
            ):
                return False

        if self.cond_check_common_word_ratio_doc_level:
            if not FilteringFunctions.check_common_word_ratio(
                text=full_text,
                strip_characters=self.strip_characters,
                common_words=self.common_words,
                common_word_ratio_min_cutoff=self.common_word_ratio_doc_level_min_cutoff,
            ):
                return False

        if self.cond_check_lang_id_doc_level:
            if not FilteringFunctions.check_lang_id(
                text=full_text,
                lang_id_model=self.lang_id_model,
                target_lang_id="en",
                lang_id_min_cutoff=self.lang_id_doc_level_min_cutoff,
            ):
                return False

        if self.cond_check_perplexity_score_doc_level:
            if not FilteringFunctions.check_perplexity_score(
                text=full_text,
                non_printing_characters_re=self.non_printing_characters_re,
                digits_re=self.digits_re,
                unicode_punctuation=self.unicode_punctuation,
                sentencepiece_model=self.sentencepiece_model,
                kenlm_model=self.kenlm_model,
                perplexity_score_max_cutoff=self.perplexity_score_doc_level_max_cutoff,
            ):
                return False

        return True

    # Needed to make multiprocessing work
    def __reduce__(self):
        # [FUNCTION DOC] HELPER: pickling support: recreates the object from its constructor arguments.
        return (
            self.__class__,
            (
                self.cond_check_number_images,
                self.number_images_min_cutoff,
                self.number_images_max_cutoff,
                self.cond_check_number_words_doc_level,
                self.strip_characters,
                self.number_words_doc_level_min_cutoff,
                self.number_words_doc_level_max_cutoff,
                self.cond_check_character_repetition_ratio_doc_level,
                self.character_repetition_length_doc_level,
                self.character_repetition_doc_level_max_cutoff,
                self.cond_check_word_repetition_ratio_doc_level,
                self.word_repetition_length_doc_level,
                self.word_repetition_doc_level_max_cutoff,
                self.cond_check_special_character_ratio_doc_level,
                self.special_character_ratio_doc_level_max_cutoff,
                self.cond_check_stopword_ratio_doc_level,
                self.stopwords,
                self.stopword_ratio_doc_level_min_cutoff,
                self.cond_check_flagged_word_ratio_doc_level,
                self.flagged_words,
                self.flagged_word_ratio_doc_level_max_cutoff,
                self.cond_check_punctuation_ratio_doc_level,
                self.punctuation,
                self.punctuation_ratio_doc_level_min_cutoff,
                self.cond_check_common_word_ratio_doc_level,
                self.path_common_words,
                self.common_word_ratio_doc_level_min_cutoff,
                self.cond_check_lang_id_doc_level,
                self.path_lang_id_model,
                self.lang_id_doc_level_min_cutoff,
                self.cond_check_perplexity_score_doc_level,
                self.non_printing_characters_re,
                self.digits_re,
                self.unicode_punctuation,
                self.path_sentencepiece_model,
                self.path_kenlm_model,
                self.perplexity_score_doc_level_max_cutoff,
            ),
        )
