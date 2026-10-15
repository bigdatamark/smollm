# [MODULE DOC] LAION step 4.3: finds opted-out image URLs via the Spawning API. Same logic as LAION-COCO
# 02_01_find_opt_out.py
# but on the dataset of all LAION URLs (ds_urls_laion_dataset_filtered_dedup, from 04_02); requires env var
# SPAWNING_TOKEN. As __main__: syncs input from S3, keeps rows whose URL is opted out (batched filter of
# 100k),
# saves to /scratch/ds_opt_out_laion_urls and syncs to s3://m4-datasets/LAION_data/ds_opt_out_laion_urls/.
import asyncio
import os
import time
from typing import List

import aiohttp
from aiolimiter import AsyncLimiter
from datasets import load_from_disk


PATH_DS_IMG_URLS_S3 = "s3://m4-datasets/LAION_data/ds_urls_laion_dataset_filtered_dedup/"
PATH_DS_IMG_URLS_LOCAL = "/scratch/ds_laion_urls"

PATH_SAVE_DISK_OPT_OUT = "/scratch/ds_opt_out_laion_urls"
PATH_SAVE_S3_OPT_OUT = "s3://m4-datasets/LAION_data/ds_opt_out_laion_urls/"

MAX_NUM_RETRIES_SYNC = 3

TOKEN = os.environ["SPAWNING_TOKEN"]

BATCH_SIZE = 100_000

headers = {"Authorization": f"API {TOKEN}"}
API_BATCH_SIZE = 1000
RETRY_TIMES = ([1] * 5) + ([10] * 5) + ([10 * 60] * 10) + ([60 * 60] * 10)
TIMEOUT = 60


async def check_spawning(image_urls: List[str], session: aiohttp.ClientSession, semaphore, limiter) -> dict:
    # [FUNCTION DOC] HELPER: async POST of newline-joined image_urls to the Spawning v2 query endpoint inside
    # the semaphore and rate
    # limiter. Empty input -> {"urls": []}; a single URL is padded with "" (API needs >1). Retries on any
    # exception
    # with the RETRY_TIMES schedule (1s x5, 10s x5, 10min x10, 1h x10), TIMEOUT 60s, requiring a "urls" key;
    # after
    # exhausting retries writes the URLs to erroring_urls.txt (cwd) and raises RuntimeError(last body).
    # KNOWN ISSUE: blocking time.sleep inside the coroutine stalls the whole event loop during backoff.
    # KNOWN ISSUE: padding a 1-URL batch yields 2 results, breaking the filter's result length.
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
    # [FUNCTION DOC] HELPER: awaits check_spawning for one chunk and returns the list of "optOut" booleans
    # (annotated tuple).
    spawning_response = await check_spawning(image_urls, session, semaphore, limiter)
    urls_responses = spawning_response["urls"]
    urls_opt_out = [urls_response["optOut"] for urls_response in urls_responses]
    return urls_opt_out


async def parallel_opt_in_out_task(urls: list) -> tuple:
    # [FUNCTION DOC] HELPER: splits urls into API_BATCH_SIZE (1000) chunks, runs them concurrently on a shared
    # authenticated aiohttp
    # session (semaphore 10, 20 req/s limiter), and returns the flattened opt-out booleans in input order; re-
    # raises
    # any task exception.
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
    # [FUNCTION DOC] Batched filter predicate over the "url" column: runs parallel_opt_in_out_task with
    # asyncio.run and returns the
    # opt-out booleans (True keeps the row). Annotated -> None, wrongly.
    opt_out_bool = asyncio.run(parallel_opt_in_out_task(batch))
    return opt_out_bool


if __name__ == "__main__":
    command_sync_s3 = f"aws s3 sync {PATH_DS_IMG_URLS_S3} {PATH_DS_IMG_URLS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds = load_from_disk(PATH_DS_IMG_URLS_LOCAL)

    ds = ds.filter(opt_in_out, input_columns=["url"], batched=True, batch_size=BATCH_SIZE)

    ds.save_to_disk(PATH_SAVE_DISK_OPT_OUT)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_OPT_OUT} {PATH_SAVE_S3_OPT_OUT}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
