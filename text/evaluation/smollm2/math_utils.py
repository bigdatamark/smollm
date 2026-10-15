# [MODULE DOC] Math answer normalization/extraction utilities (adapted from the Qwen2.5-Math eval parser)
# used by the
# SmolLM2 lighteval math tasks: pull the final answer out of a model completion or a ground-truth solution
# and canonicalize LaTeX/plain-text answers so they can be compared as strings. Module data: unit_texts, a
# list of MathQA unit words (plus their plural "s" forms) that strip_string removes.
# Disclaimer: this code is adapted from the Qwen2.5-Math eval suite.
# Source: https://github.com/QwenLM/Qwen2.5-Math/blob/cb098e13/evaluation/parser.py

import re
from word2number import w2n


def _fix_fracs(string):
    # [FUNCTION DOC] HELPER: rewrites shorthand LaTeX fractions into braced form, e.g. "\frac12" ->
    # "\frac{1}{2}" and
    # "\frac1{2}" -> "\frac{1}{2}"; parts already starting with "{" are kept. Returns the original string
    # unchanged if a non-braced \frac is followed by fewer than 2 characters.
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    # [FUNCTION DOC] HELPER: converts a plain "a/b" (exactly one slash, both sides integers or containing
    # "sqrt", and the
    # string round-trips exactly) into "\frac{a}{b}"; otherwise returns the input unchanged.
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    # [FUNCTION DOC] HELPER: adds braces to unbraced square roots, e.g. "\sqrt3" -> "\sqrt{3}" (applies to
    # word characters
    # directly following \sqrt).
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    # [FUNCTION DOC] HELPER: converts an English number phrase ("twenty one") to its digit string via
    # word2number; returns the
    # text unchanged if conversion fails (any exception swallowed).
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):
    # [FUNCTION DOC] Canonicalizes an answer string. Input: any value (converted to str), skip_unit (default
    # False) to keep unit
    # words. Steps in order: strip, drop newlines and trailing dots, remove "\!", normalize array/bmatrix to
    # pmatrix, tfrac/dfrac -> frac, \neq/\leq/\geq -> \ne/\le/\ge, drop \left/\right and escaped braces,
    # drop a
    # trailing \text{...} unit, (unless skip_unit) remove unit_texts words bounded by non-word chars (2
    # passes),
    # drop degree symbols, dollar signs and \( \), convert word numbers, unwrap \text{...}, remove "x=",
    # "y\in",
    # "z\to"-style prefixes, \emptyset -> "{}", (-\infty,\infty) -> \mathbb{R}, drop percent signs, add
    # leading
    # zeros to ".5"-style decimals, unwrap alnum-only brackets, map "infinity"/"inf" to \infty, delete "and"
    # and
    # \mathbf and \mbox{...}, map j to i when no i is present, drop trailing zero decimals ("2.000" -> "2"),
    # drop short "k=" prefixes, fix sqrt, remove all spaces, fix fracs and a/b. Returns the normalized string.
    # KNOWN ISSUE: the quote-removal replace calls discard their results, so quotes are never removed.
    # KNOWN ISSUE: the "+\inity" replacement is a typo that never matches; the bracket-unwrap condition can
    # never
    # be true (a string with brackets is not alphanumeric); deleting every "and"/"inf" substring can corrupt
    # words or LaTeX commands (e.g. "\land", "\inf").
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
            string.startswith("{")
            and string.endswith("}")
            and string.isalnum()
            or string.startswith("(")
            and string.endswith(")")
            and string.isalnum()
            or string.startswith("[")
            and string.endswith("]")
            and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer(pred_str, data_name, use_last_number=True):
    # [FUNCTION DOC] Extracts the predicted final answer from a completion. Inputs: pred_str (model text),
    # data_name (dataset
    # name: drives multiple-choice handling and unit skipping), use_last_number (default True). Removes the
    # "ки" marker, then takes, by priority: text between "final answer is $" and "$. I hope" (Minerva style);
    # the content of the last \boxed{...} with balanced braces (or text up to the next "$" if unbraced;
    # returns
    # "" if nothing follows "boxed"); the text after the last "he answer is" / "final answer is"; the text
    # after
    # the first Chinese "答案是" up to a blank line; else the last number in the comma-stripped text (or "" ).
    # For sat_math/aqua/mmlu* the last standalone A-E letter is used if present. Then removes newlines and a
    # leading ":" / trailing "." or "/", and normalizes with strip_string (skip_unit for
    # carp_en/minerva_math).
    # Returns the answer string.
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # choice answer
    if (
            data_name in ["sat_math", "aqua"]
            or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def parse_math_answer(text: str, data_name):
    # [FUNCTION DOC] Extracts the ground-truth answer from a reference solution. data_name
    # "math"/"minerva_math": uses
    # extract_answer on the text; "gsm8k": takes the part after "####"; any other name raises
    # NotImplementedError. Then maps \neq/\leq/\geq to \ne/\le/\ge and returns the answer string.
    # KNOWN ISSUE: for gsm8k the answer is not stripped (keeps leading space) and text with more than one
    # "####"
    # raises ValueError on unpacking.
    # parse ground truth
    if data_name in ["math", "minerva_math"]:
        gt_cot = text
        gt_ans = extract_answer(gt_cot, data_name)
    elif data_name == "gsm8k":
        gt_cot, gt_ans = text.split("####")
    else:
        raise NotImplementedError(f"`{data_name}`")
    # post process
    gt_ans = (
        gt_ans.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    return gt_ans