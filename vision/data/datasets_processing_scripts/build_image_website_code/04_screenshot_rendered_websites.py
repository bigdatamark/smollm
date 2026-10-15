# [MODULE DOC] Step 4 of the image-to-website-code pipeline: renders the HTML pages with headless Chromium
# (playwright) and
# builds HF datasets of (screenshot, code) pairs. Input JSON files COMMON_PATH_JSON_HTML_CODES with indices
# 0..NUM_JSON_FILES-1, each a list of HTML strings; screenshots are written temporarily to
# COMMON_PATH_SCREENSHOT;
# one dataset per JSON file is saved to COMMON_PATH_SAVE_DATASETS_IMAGE_TO_WEBSITE_CODE. As __main__,
# processes
# indices in chunks of 250 with a pool of NUM_PROCESSES workers (sleeping 10s before each chunk), printing
# each
# finished index; any exception in a chunk silently abandons the rest of that chunk.
# KNOWN ISSUE: the last chunk includes indices >= NUM_JSON_FILES (10057 is not a multiple of 250), whose
# missing
# files raise and abort the remainder of that chunk; failures are swallowed, so unprocessed files go
# unnoticed.
import json
import multiprocessing
import os
import time

from datasets import Dataset
from PIL import Image
from playwright.sync_api import sync_playwright
from tqdm import tqdm


NUM_PROCESSES = 10
NUM_JSON_FILES = 10057
COMMON_PATH_JSON_HTML_CODES = "/fsx/hugo/image_website_code/new_gen/{NUM}.json"
COMMON_PATH_SCREENSHOT = "/fsx/hugo/image_website_code/output_screenshots/{NAME}.png"
COMMON_PATH_SAVE_DATASETS_IMAGE_TO_WEBSITE_CODE = "/fsx/hugo/image_website_code/datasets_image_to_website_code/{NUM}"


def process_one_json_html_code(idx_json):
    # [FUNCTION DOC] Worker for one JSON index idx_json: loads the list of HTML codes, opens one browser
    # context (fixed desktop
    # Chrome user agent), and for each code opens a page, sets the content, waits for network idle, takes a
    # full-page
    # PNG screenshot to disk and opens it with PIL. Saves Dataset.from_dict({"image", "code"}) to disk,
    # deletes the
    # screenshots with a shell rm on the "idx_json_*" glob, and returns idx_json.
    # KNOWN ISSUE: pages are never closed (only the context at the end), so memory grows with the list length.
    path_one_json_html_codes = COMMON_PATH_JSON_HTML_CODES.format(NUM=str(idx_json))
    with open(path_one_json_html_codes, "r") as f:
        list_html_codes = json.load(f)

    list_screenshot_websites = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0"
                " Safari/537.36"
            )
        )
        for idx_html_code, html_code in enumerate(list_html_codes):
            page = context.new_page()
            page.set_content(html_code)
            page.wait_for_load_state("networkidle")
            output_path_screenshot = COMMON_PATH_SCREENSHOT.format(NAME=f"{idx_json}_{idx_html_code}")
            page.screenshot(path=output_path_screenshot, full_page=True)
            list_screenshot_websites.append(Image.open(output_path_screenshot))
        context.close()
        browser.close()

    ds = Dataset.from_dict({"image": list_screenshot_websites, "code": list_html_codes})
    path_save_datasets_image_to_website_code = COMMON_PATH_SAVE_DATASETS_IMAGE_TO_WEBSITE_CODE.format(
        NUM=str(idx_json)
    )
    ds.save_to_disk(path_save_datasets_image_to_website_code)
    path_screenshots_to_remove = COMMON_PATH_SCREENSHOT.format(NAME=f"{idx_json}_*")
    os.system(f"rm -r {path_screenshots_to_remove}")
    return idx_json


if __name__ == "__main__":
    size_chunk = 250
    for idx_chunk in tqdm(range(0, NUM_JSON_FILES, size_chunk)):
        time.sleep(10)
        try:
            with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                with tqdm(total=size_chunk) as pbar:
                    for idx_json in pool.imap_unordered(
                        process_one_json_html_code, range(idx_chunk, idx_chunk + size_chunk)
                    ):
                        print(f"Index {idx_json} done")
                        pbar.update()
        except Exception:
            pass
    print("All processes completed.")
