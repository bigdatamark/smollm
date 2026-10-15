# [MODULE DOC] Script (step 15_02): queries the Spawning opt-out API for every URL found in step 15_01 and
# keeps only the
# opted-out ones. Requires env var SPAWNING_TOKEN (read at import time, KeyError if absent), used in the
# global
# `headers` mapping as "Authorization: API <token>".
# When run as main: syncs the url dataset from S3 (MAX_NUM_RETRIES_SYNC times), filters it in batches of
# BATCH_SIZE with opt_in_out (column "url"), saves the remaining (opt-out) rows to /scratch and syncs to S3.
# Requests are sent in chunks of API_BATCH_SIZE urls, max 10 concurrent and 20 per second, with retry delays
# taken from RETRY_TIMES (5x1s, 5x10s, 10x10min, 10x1h) and a TIMEOUT of 60s per request.
import asyncio
import os
import time
from typing import List

import aiohttp
from aiolimiter import AsyncLimiter
from datasets import load_from_disk


PATH_S3_DS_DOC_AND_IMAGE_URLS_OBELICS = "s3://m4-datasets-us-east-1/webdocs/ds_doc_and_image_urls_obelics"
PATH_SAVE_DISK_DS_DOC_AND_IMAGE_URLS_OBELICS = "/scratch/ds_doc_and_image_urls_obelics"

PATH_SAVE_DISK_DS_OPT_OUT_DOC_AND_IMAGE_URLS_OBELICS = "/scratch/ds_opt_out_doc_and_image_urls_obelics"
PATH_SAVE_S3_DS_OPT_OUT_DOC_AND_IMAGE_URLS_OBELICS = (
    "s3://m4-datasets-us-east-1/webdocs/ds_opt_out_doc_and_image_urls_obelics"
)

MAX_NUM_RETRIES_SYNC = 3

TOKEN = os.environ["SPAWNING_TOKEN"]

BATCH_SIZE = 100_000

headers = {"Authorization": f"API {TOKEN}"}
API_BATCH_SIZE = 1000
RETRY_TIMES = ([1] * 5) + ([10] * 5) + ([10 * 60] * 10) + ([60 * 60] * 10)
TIMEOUT = 60


async def check_spawning(image_urls: List[str], session: aiohttp.ClientSession, semaphore, limiter) -> dict:
    # [FUNCTION DOC] Async. POSTs a newline-joined list of urls to the Spawning v2 "query/urls" endpoint and
    # returns the parsed
    # JSON response (a mapping containing "urls": one entry per input url, each with an "optOut" bool).
    # Inputs: image_urls (list of str), an aiohttp session (already carrying auth headers), an asyncio
    # semaphore and an aiolimiter.AsyncLimiter, both acquired around the request.
    # Empty input -> returns {"urls": []} without a request; a single url is padded with "" since the API
    # needs at least 2. Any exception (network, non-JSON, missing "urls") triggers a retry after sleeping
    # the next RETRY_TIMES delay. When all retries fail, writes the urls to ./erroring_urls.txt (overwritten)
    # and raises RuntimeError with the last response body.
    # KNOWN ISSUE: uses blocking time.sleep inside a coroutine, which freezes the whole event loop (all other
    # in-flight requests) during each retry delay; asyncio.sleep was likely intended.
    url = "https://opts-api.spawningaiapi.com/api/v2/query/urls"
    if not image_urls:
        return {"urls": []}
    elif len(image_urls) == 1:
        image_urls = image_urls + [""]  # the API requires > 1 urls
    async with semaphore:
        async with limiter:
            resp_body = None
            for retry_time in RETRY_TIMES:
                try:
                    async with session.post(
                        url=url,
                        data="\n".join(image_urls),
                        timeout=TIMEOUT,
                    ) as resp:
                        resp_body = await resp.text()
                        spawning_response = await resp.json()
                        assert "urls" in spawning_response, str(resp_body)
                        return spawning_response
                except Exception:
                    pass
                time.sleep(retry_time)
            with open("erroring_urls.txt", "w") as f:
                f.write("\n".join(image_urls))
            raise RuntimeError(str(resp_body))


async def opt_in_out_task(image_urls: List[str], session, semaphore, limiter) -> tuple:
    # [FUNCTION DOC] Async. Calls check_spawning on one chunk of urls and returns the list of "optOut"
    # booleans, in
    # response order.
    # KNOWN ISSUE: annotated as returning tuple but returns a list. For a 1-url chunk the padding "" url makes
    # the API answer 2 entries, so 2 booleans are returned for 1 url, misaligning the filter mask downstream.
    spawning_response = await check_spawning(image_urls, session, semaphore, limiter)
    urls_responses = spawning_response["urls"]
    urls_opt_out = [urls_response["optOut"] for urls_response in urls_responses]
    return urls_opt_out


async def parallel_opt_in_out_task(urls: list) -> tuple:
    # [FUNCTION DOC] Async. Splits urls (list of str) into chunks of API_BATCH_SIZE, runs opt_in_out_task on
    # each as
    # concurrent tasks sharing one aiohttp session (with global auth headers), a Semaphore(10) and an
    # AsyncLimiter(20 per 1s), waits for all, and returns the flattened list of opt-out booleans in input
    # order. Exceptions from a task are re-raised by task.result().
    # KNOWN ISSUE: annotated as returning tuple but returns a list.
    tasks = []
    semaphore = asyncio.Semaphore(value=10)
    limiter = AsyncLimiter(20, time_period=1)
    async with aiohttp.ClientSession(headers=headers) as session:
        for offset in range(0, len(urls), API_BATCH_SIZE):
            tasks.append(
                asyncio.create_task(
                    opt_in_out_task(urls[offset : offset + API_BATCH_SIZE], session, semaphore, limiter)
                )
            )
        await asyncio.wait(tasks)
    tasks_results = [task.result() for task in tasks]
    tasks_results = [sub_el for el in tasks_results for sub_el in el]
    return tasks_results


def opt_in_out(batch: dict) -> None:
    # [FUNCTION DOC] Batched datasets.filter function: receives the list of urls of a batch
    # (input_columns=["url"]) and
    # returns a list of booleans (True = opted out, row kept) by running parallel_opt_in_out_task with
    # asyncio.run (a fresh event loop per batch).
    # KNOWN ISSUE: annotated `-> None` and `batch: dict`, but it actually takes a list and returns a list.
    opt_out_bool = asyncio.run(parallel_opt_in_out_task(batch))
    return opt_out_bool


if __name__ == "__main__":
    command_sync_s3 = (
        f"aws s3 sync {PATH_S3_DS_DOC_AND_IMAGE_URLS_OBELICS} {PATH_SAVE_DISK_DS_DOC_AND_IMAGE_URLS_OBELICS}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds = load_from_disk(PATH_SAVE_DISK_DS_DOC_AND_IMAGE_URLS_OBELICS)

    ds = ds.filter(opt_in_out, input_columns=["url"], batched=True, batch_size=BATCH_SIZE)

    ds.save_to_disk(PATH_SAVE_DISK_DS_OPT_OUT_DOC_AND_IMAGE_URLS_OBELICS)

    command_sync_s3 = (
        "aws s3 sync"
        f" {PATH_SAVE_DISK_DS_OPT_OUT_DOC_AND_IMAGE_URLS_OBELICS} {PATH_SAVE_S3_DS_OPT_OUT_DOC_AND_IMAGE_URLS_OBELICS}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
