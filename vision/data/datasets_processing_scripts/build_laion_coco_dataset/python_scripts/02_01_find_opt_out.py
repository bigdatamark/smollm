# [MODULE DOC] LAION-COCO step 2.1: finds image URLs that opted out of AI training via the Spawning API.
# Requires env var SPAWNING_TOKEN (KeyError otherwise). As __main__: syncs the URL dataset (column "url") from
# S3 to /scratch, filters it in batches of BATCH_SIZE (100k) keeping only rows whose URL is opted OUT, saves
# the
# resulting dataset locally and syncs it to PATH_SAVE_S3_DS_OPT_OUT_IMAGE_URLS_LAION_COCO. Network-heavy.
import asyncio
import os
import time
from typing import List

import aiohttp
from aiolimiter import AsyncLimiter
from datasets import load_from_disk


PATH_S3_DS_IMAGE_URLS_LAION_COCO = "s3://m4-datasets-us-east-1/LAION_data/ds_laion_coco_urls/"
PATH_SAVE_DISK_DS_IMAGE_URLS_LAION_COCO = "/scratch/ds_image_urls_laion_coco"

PATH_SAVE_DISK_DS_OPT_OUT_IMAGE_URLS_LAION_COCO = "/scratch/ds_opt_out_image_urls_laion_coco"
PATH_SAVE_S3_DS_OPT_OUT_IMAGE_URLS_LAION_COCO = (
    "s3://m4-datasets-us-east-1/LAION_data/ds_opt_out_image_urls_laion_coco"
)

MAX_NUM_RETRIES_SYNC = 3

TOKEN = os.environ["SPAWNING_TOKEN"]

BATCH_SIZE = 100_000

headers = {"Authorization": f"API {TOKEN}"}
API_BATCH_SIZE = 1000
RETRY_TIMES = ([1] * 5) + ([10] * 5) + ([10 * 60] * 10) + ([60 * 60] * 10)
TIMEOUT = 60


async def check_spawning(image_urls: List[str], session: aiohttp.ClientSession, semaphore, limiter) -> dict:
    # [FUNCTION DOC] HELPER: async POST of newline-joined image_urls to the Spawning v2 query endpoint,
    # guarded by an asyncio
    # semaphore and an AsyncLimiter. Returns {"urls": []} for an empty list; pads a single URL with "" (API
    # needs >1).
    # Retries on any exception following RETRY_TIMES (sleep 1s x5, 10s x5, 10min x10, 1h x10), each request
    # with
    # TIMEOUT=60s, and requires a "urls" key in the JSON reply. After all retries, writes the URLs to
    # erroring_urls.txt (cwd) and raises RuntimeError with the last response body. Returns the parsed JSON
    # mapping.
    # KNOWN ISSUE: uses blocking time.sleep inside a coroutine, stalling all other concurrent requests during
    # backoff.
    # KNOWN ISSUE: the "" padding yields 2 results for a 1-URL batch, making the filter's boolean list length
    # wrong.
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
    # [FUNCTION DOC] HELPER: awaits check_spawning for one batch and returns the list of "optOut" booleans,
    # one per queried URL
    # (annotated as tuple but returns a list).
    spawning_response = await check_spawning(image_urls, session, semaphore, limiter)
    urls_responses = spawning_response["urls"]
    urls_opt_out = [urls_response["optOut"] for urls_response in urls_responses]
    return urls_opt_out


async def parallel_opt_in_out_task(urls: list) -> tuple:
    # [FUNCTION DOC] HELPER: splits urls into chunks of API_BATCH_SIZE (1000), runs opt_in_out_task
    # concurrently on one shared
    # aiohttp session (auth headers), semaphore of 10 and 20 requests/second limiter, waits for all and
    # returns the
    # flattened list of opt-out booleans in input order. Re-raises a task's exception when collecting results.
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
    # [FUNCTION DOC] Batched Dataset.filter predicate: given the list of URLs of a batch, runs
    # parallel_opt_in_out_task in a fresh
    # event loop and returns the list of opt-out booleans (True = keep row = opted out). Annotated -> None,
    # wrongly.
    opt_out_bool = asyncio.run(parallel_opt_in_out_task(batch))
    return opt_out_bool


if __name__ == "__main__":
    command_sync_s3 = f"aws s3 sync {PATH_S3_DS_IMAGE_URLS_LAION_COCO} {PATH_SAVE_DISK_DS_IMAGE_URLS_LAION_COCO}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds = load_from_disk(PATH_SAVE_DISK_DS_IMAGE_URLS_LAION_COCO)

    ds = ds.filter(opt_in_out, input_columns=["url"], batched=True, batch_size=BATCH_SIZE)

    ds.save_to_disk(PATH_SAVE_DISK_DS_OPT_OUT_IMAGE_URLS_LAION_COCO)

    command_sync_s3 = (
        "aws s3 sync"
        f" {PATH_SAVE_DISK_DS_OPT_OUT_IMAGE_URLS_LAION_COCO} {PATH_SAVE_S3_DS_OPT_OUT_IMAGE_URLS_LAION_COCO}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
