# [MODULE DOC] Step 3 of the image-to-website-code pipeline: extracts a single self-contained HTML page from
# each raw model
# generation. argv[1] = NUM_FILE; reads a JSON list of generations from
# /fsx/hugo/deepseek_html_css_scale/gen/NUM_FILE.json, applies process_html_css, drops failures (None) and
# writes
# the JSON list of pages to /fsx/hugo/process_html_css/gen/NUM_FILE.json, then prints a message.
# KNOWN ISSUE: input folder (deepseek_html_css_scale) differs from the folder step 2 writes
# (deepseek_html_css).
import json
import sys


NUM_FILE = sys.argv[1]
PATH_HTML_CSS_FILE = f"/fsx/hugo/deepseek_html_css_scale/gen/{NUM_FILE}.json"
PATH_SAVE_PROCESSED_FILE = f"/fsx/hugo/process_html_css/gen/{NUM_FILE}.json"


def process_html_css(html_css_file):
    # [FUNCTION DOC] Input: one generation string expected to contain exactly two fenced code blocks (4
    # occurrences of three
    # backticks): HTML first, CSS second; otherwise returns None. The CSS block is trimmed of leading lines
    # before the
    # first line containing "{" (None if no "{"); the HTML is cut to the <body>...</body> region inclusive
    # (None if
    # either tag is missing). Returns "<html>\n<style>\n" + css + "</style>\n" + body + "\n</html>".
    # Note the leading text after each opening fence (e.g. the language tag "html") is kept in the HTML slice
    # but
    # discarded by the body cut.
    substring_start_code = "```"
    positions_code = []
    start_pos = html_css_file.find(substring_start_code)
    while start_pos != -1:
        positions_code.append(start_pos)
        start_pos = html_css_file.find(substring_start_code, start_pos + 1)

    if len(positions_code) != 4:
        return None

    html = html_css_file[positions_code[0] + len(substring_start_code) : positions_code[1]]
    css = html_css_file[positions_code[2] + len(substring_start_code) : positions_code[3]]

    if css.find("{") == -1:
        return None
    css = css.split("\n")
    start_line = 0
    for line in css:
        if "{" not in line:
            start_line += 1
        else:
            break
    css = "\n".join(css[start_line:])

    beg_body_substring = "<body>"
    end_body_substring = "</body>"
    beg_body_pos_html = html.find(beg_body_substring)
    end_body_pos_html = html.find(end_body_substring)
    if beg_body_pos_html == -1 or end_body_pos_html == -1:
        return None

    html = html[beg_body_pos_html : end_body_pos_html + len(end_body_substring)].strip("\n")

    final_file = f"<html>\n<style>\n{css}</style>\n{html}\n</html>"
    return final_file


with open(PATH_HTML_CSS_FILE, "r") as f:
    html_css_files = json.load(f)

processed_files = [process_html_css(html_css_file) for html_css_file in html_css_files]
processed_files = [el for el in processed_files if el is not None]

with open(PATH_SAVE_PROCESSED_FILE, "w") as f:
    json.dump(processed_files, f)

print("Successfully done")
