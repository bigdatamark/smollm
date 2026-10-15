# [MODULE DOC] Script that cleans the "argilla/ifeval-like-data" dataset into a verified IFEval-style
# dataset: normalizes
# per-instruction kwargs, drops rows with invalid kwargs or mutually conflicting instructions, scores each
# response with lm-eval-harness's IFEval checker and keeps only rows passing prompt-level strict accuracy.
# Module constants: INSTRUCTION_ARGS (instruction id -> list of kwarg names it takes), ALL_ARGUMENTS (set of
# all kwarg names), IFEVAL_INSTRUCTION_CONFLICTS (instruction id -> set of ids it conflicts with, including
# itself), LANGUAGE_TO_CODE (language name -> ISO code). Running as main pushes the result to the Hub as
# config "filtered" of the same repo.
import json
from typing import TYPE_CHECKING

from datasets import load_dataset
from lm_eval.tasks.ifeval.utils import process_results

if TYPE_CHECKING:
    from datasets import Dataset


INSTRUCTION_ARGS = {
    "keywords:existence": ["keywords"],
    "keywords:frequency": ["keyword", "frequency", "relation"],
    "keywords:forbidden_words": ["forbidden_words"],
    "keywords:letter_frequency": ["letter", "let_frequency", "let_relation"],
    "language:response_language": ["language"],
    "length_constraints:number_sentences": ["num_sentences", "relation"],
    "length_constraints:number_paragraphs": ["num_paragraphs"],
    "length_constraints:number_words": ["num_words", "relation"],
    "length_constraints:nth_paragraph_first_word": [
        "num_paragraphs",
        "nth_paragraph",
        "first_word",
    ],
    "detectable_content:number_placeholders": ["num_placeholders"],
    "detectable_content:postscript": ["postscript_marker"],
    "detectable_format:number_bullet_lists": ["num_bullets"],
    "detectable_format:constrained_response": [],
    "detectable_format:number_highlighted_sections": ["num_highlights"],
    "detectable_format:multiple_sections": ["section_spliter", "num_sections"],
    "detectable_format:json_format": [],
    "detectable_format:title": [],
    "combination:two_responses": [],
    "combination:repeat_prompt": ["prompt_to_repeat"],
    "startend:end_checker": ["end_phrase"],
    "change_case:capital_word_frequency": ["capital_frequency", "capital_relation"],
    "change_case:english_capital": [],
    "change_case:english_lowercase": [],
    "punctuation:no_comma": [],
    "startend:quotation": [],
}

ALL_ARGUMENTS = {
    "keywords",
    "keyword",
    "frequency",
    "relation",
    "forbidden_words",
    "letter",
    "let_frequency",
    "let_relation",
    "language",
    "num_sentences",
    "num_paragraphs",
    "num_words",
    "nth_paragraph",
    "first_word",
    "num_placeholders",
    "postscript_marker",
    "num_bullets",
    "num_highlights",
    "section_spliter",
    "num_sections",
    "prompt_to_repeat",
    "end_phrase",
    "capital_frequency",
    "capital_relation",
}

IFEVAL_INSTRUCTION_CONFLICTS = {
    "keywords:existence": {"keywords:existence"},
    "keywords:frequency": {"keywords:frequency"},
    "keywords:forbidden_words": {"keywords:forbidden_words"},
    "keywords:letter_frequency": {"keywords:letter_frequency"},
    "language:response_language": {
        "change_case:english_capital",
        "change_case:english_lowercase",
        "startend:end_checker",
        "keywords:frequency",
        "keywords:forbidden_words",
        "detectable_format:multiple_sections",
        "keywords:existence",
        "language:response_language",
    },
    "length_constraints:number_sentences": {"length_constraints:number_sentences"},
    "length_constraints:number_paragraphs": {
        "length_constraints:number_sentences",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
    },
    "length_constraints:number_words": {"length_constraints:number_words"},
    "length_constraints:nth_paragraph_first_word": {
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
    },
    "detectable_content:number_placeholders": {
        "detectable_content:number_placeholders"
    },
    "detectable_content:postscript": {"detectable_content:postscript"},
    "detectable_format:number_bullet_lists": {"detectable_format:number_bullet_lists"},
    "detectable_format:constrained_response": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "startend:end_checker",
        "keywords:forbidden_words",
        "length_constraints:number_sentences",
        "combination:repeat_prompt",
        "combination:two_responses",
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "change_case:english_lowercase",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "keywords:letter_frequency",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "keywords:existence",
        "length_constraints:nth_paragraph_first_word",
        "detectable_format:title",
        "change_case:capital_word_frequency",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "language:response_language",
    },
    "detectable_format:number_highlighted_sections": {
        "detectable_format:number_highlighted_sections"
    },
    "detectable_format:multiple_sections": {
        "detectable_format:multiple_sections",
        "detectable_format:number_highlighted_sections",
        "language:response_language",
    },
    "detectable_format:json_format": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "startend:end_checker",
        "keywords:letter_frequency",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_sentences",
        "language:response_language",
        "combination:repeat_prompt",
        "detectable_format:title",
        "change_case:capital_word_frequency",
        "combination:two_responses",
        "detectable_format:json_format",
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "change_case:english_lowercase",
    },
    "detectable_format:title": {"detectable_format:title"},
    "combination:two_responses": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "startend:end_checker",
        "keywords:letter_frequency",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_sentences",
        "combination:repeat_prompt",
        "change_case:capital_word_frequency",
        "combination:two_responses",
        "detectable_format:json_format",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "change_case:english_lowercase",
    },
    "combination:repeat_prompt": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "startend:end_checker",
        "keywords:letter_frequency",
        "keywords:forbidden_words",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_sentences",
        "language:response_language",
        "combination:repeat_prompt",
        "change_case:capital_word_frequency",
        "combination:two_responses",
        "detectable_format:json_format",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "change_case:english_lowercase",
    },
    "startend:end_checker": {"startend:end_checker"},
    "change_case:capital_word_frequency": {
        "change_case:english_capital",
        "change_case:capital_word_frequency",
        "change_case:english_lowercase",
    },
    "change_case:english_capital": {"change_case:english_capital"},
    "change_case:english_lowercase": {
        "change_case:english_capital",
        "change_case:english_lowercase",
    },
    "punctuation:no_comma": {"punctuation:no_comma"},
    "startend:quotation": {"startend:quotation", "detectable_format:title"},
}

LANGUAGE_TO_CODE = {
    "English": "en",
    "Spanish": "es",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Hindi": "hi",
    "French": "fr",
    "Russian": "ru",
    "German": "de",
    "Japanese": "ja",
    "Italian": "it",
    "Bengali": "bn",
    "Ukrainian": "uk",
    "Thai": "th",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Bulgarian": "bg",
    "Korean": "ko",
    "Polish": "pl",
    "Hebrew": "he",
    "Persian": "fa",
    "Vietnamese": "vi",
    "Nepali": "ne",
    "Swahili": "sw",
    "Kannada": "kn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Malayalam": "ml",
    "Finnish": "fi",
}


def build_instruction_kwargs(row: dict) -> dict:
    # [FUNCTION DOC] Map function. Input row with "kwargs" (JSON string of one flat mapping of all args) and
    # "instruction_id_list". Returns {"valid_kwargs_json": False} if kwargs is None, not valid JSON, or a
    # "language" value is not a key of LANGUAGE_TO_CODE. Otherwise builds, for each instruction id in order, a
    # mapping of just that instruction's args (per INSTRUCTION_ARGS; language names converted to ISO codes)
    # and
    # returns {"kwargs": JSON string of that list, "valid_kwargs_json": True}.
    # KNOWN ISSUE: a kwarg name missing from the JSON raises KeyError (uncaught) instead of marking the row
    # invalid; an unknown instruction id also raises KeyError; a language already given as a code ("en") is
    # rejected as invalid.
    """Builds the list of `kwargs` for each instruction in `instruction_id_list`."""
    kwargs = row["kwargs"]

    if kwargs is None:
        return {"valid_kwargs_json": False}

    try:
        kwargs = json.loads(row["kwargs"])
    except json.JSONDecodeError:
        return {"valid_kwargs_json": False}

    instruction_id_list = row["instruction_id_list"]
    kwargs_list = []
    for instruction_id in instruction_id_list:
        args = INSTRUCTION_ARGS[instruction_id]
        instruction_kwargs = {}
        for arg in args:
            value = kwargs[arg]
            # Fix "English" instead of "en"
            if arg == "language":
                if value in LANGUAGE_TO_CODE:
                    value = LANGUAGE_TO_CODE[value]
                else:
                    return {"valid_kwargs_json": False}
            instruction_kwargs[arg] = value
        kwargs_list.append(instruction_kwargs)

    return {"kwargs": json.dumps(kwargs_list), "valid_kwargs_json": True}


def filter_not_valid_rows(row: dict) -> bool:
    # [FUNCTION DOC] Filter predicate. Returns False if row["valid_kwargs_json"] is falsy, or if any
    # instruction in
    # instruction_id_list conflicts (per IFEVAL_INSTRUCTION_CONFLICTS, excluding itself) with another
    # instruction present in the list; True otherwise.
    """Filters out rows which their JSON kwargs are not valid or that the instructions
    in their `instruction_id_list` conflict each other."""
    valid_kwargs_json = row["valid_kwargs_json"]
    if not valid_kwargs_json:
        return False

    instruction_id_list = row["instruction_id_list"]
    for instruction_id in instruction_id_list:
        conflicts = IFEVAL_INSTRUCTION_CONFLICTS[instruction_id]
        if any(
            conflict in instruction_id_list
            for conflict in conflicts
            if conflict != instruction_id
        ):
            return False

    return True


def get_ifeval_results(row: dict) -> dict:
    # [FUNCTION DOC] Map function. Copies the row, parses its "kwargs" JSON string into a list and calls
    # lm-eval's IFEval
    # process_results with [row["response"]]. Returns its result dict (prompt/inst level strict/loose acc);
    # on any exception returns a dict with both prompt-level accs False and both inst-level lists empty.
    """Checks if the `response` correct is OK using the IFEval benchmark code from `lm-evaluation-harness`."""
    results = [row["response"]]
    doc = row.copy()
    doc["kwargs"] = json.loads(doc["kwargs"])
    try:
        return process_results(doc, results)
    except Exception:
        return {
            "prompt_level_strict_acc": False,
            "inst_level_strict_acc": [],
            "prompt_level_loose_acc": False,
            "inst_level_loose_acc": [],
        }


def get_dataset() -> "Dataset":
    # [FUNCTION DOC] Loads the "train" split of argilla/ifeval-like-data (network), applies
    # build_instruction_kwargs and
    # filter_not_valid_rows, adds an integer "key" column (0..n-1), renames "instruction" to "prompt", keeps
    # columns key/prompt/response/instruction_id_list/kwargs, adds IFEval results and keeps rows whose
    # prompt_level_strict_acc is True. Returns the resulting Dataset.
    dataset = load_dataset("argilla/ifeval-like-data", split="train")
    dataset = dataset.map(build_instruction_kwargs)
    dataset = dataset.filter(filter_not_valid_rows)
    dataset = dataset.add_column("key", list(range(len(dataset))))
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.select_columns(
        ["key", "prompt", "response", "instruction_id_list", "kwargs"]
    )
    dataset = dataset.map(get_ifeval_results)
    dataset = dataset.filter(lambda x: x["prompt_level_strict_acc"])
    return dataset


if __name__ == "__main__":
    dataset = get_dataset()
    dataset.push_to_hub("argilla/ifeval-like-data", config_name="filtered")
