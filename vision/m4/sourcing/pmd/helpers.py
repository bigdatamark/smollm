# [MODULE DOC] Shared helpers for PMD loaders: collapsing extra columns into a JSON `meta` column, a cached
# media downloader
# (M4HTTPClient) that respects robots.txt and stores files via cache_path.cached_path, image validation, and a
# picklable threaded download iterator usable inside `datasets` map. SIDE EFFECT on import: disables HTTPS
# certificate verification globally (ssl default context replaced by an unverified one).
import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import fields
from datetime import datetime
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
from datasets import Dataset
from datasets.utils.file_utils import DownloadConfig

from m4.sourcing.pmd.cache_path import cached_path


def json_serializer(o):
    # [FUNCTION DOC] HELPER: `default` hook for json.dumps: exceptions -> repr string, datetimes -> str;
    # otherwise TypeError.
    if isinstance(o, BaseException):
        return repr(o)

    if isinstance(o, datetime):
        return str(o)

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def collapse_columns_to_meta(
    batch: Dict,
    columns_to_collapse: List[str],
    meta_column_name: str,
) -> Dict:
    # [FUNCTION DOC] Batched map helper: for each row, JSON-encode (indent 2, exceptions/datetimes
    # serialized) a mapping of the
    # `columns_to_collapse` (list, order kept) to their values, delete those columns from the batch, and
    # store the
    # JSON strings under `meta_column_name`. Asserts the meta column does not already exist unless it is
    # collapsed
    # too. Mutates and returns the batch.
    # Order matters
    assert isinstance(columns_to_collapse, list)

    # {meta_column_name} needs to :be either inserted in
    #   - not be a column_name of the dataset in question
    #   - part of the columns we're going to collapse.
    assert meta_column_name not in batch or meta_column_name in columns_to_collapse

    # Aggregate all values into a single dict
    metas = [
        json.dumps(
            {column_name: value for column_name, value in zip(columns_to_collapse, values)},
            default=json_serializer,
            indent=2,
        )
        for values in zip(*[batch[column_name] for column_name in columns_to_collapse])
    ]

    # Remove columns from batch
    for column_name in columns_to_collapse:
        del batch[column_name]

    batch[meta_column_name] = metas

    return batch


# ---- Download medias helper ----


class RobotsDisallow(BaseException):
    # [CLASS DOC] Exception (BaseException subclass) signaling that robots.txt forbids downloading a url;
    # returned, not raised.
    """Exception class when robots.txt prevents us from downloading the urls"""

    pass


def compute_cache_path(cache_dir, hash):
    # [FUNCTION DOC] HELPER: return the absolute path cache_dir/<hash[:3]>/<hash[3:6]>/<hash> as str,
    # creating parent dirs.
    cached_path = Path(cache_dir) / hash[:3] / hash[3:6] / hash
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    return str(cached_path.absolute())


# LRU cache that caches exception as well ... let's see if this works.
def lru_cache_with_exception(maxsize=128, typed=False):
    # [FUNCTION DOC] HELPER: decorator factory like functools.lru_cache(maxsize, typed) but also caches
    # raised exceptions: the
    # call result or exception is memoized and the exception is re-raised on each cached call.
    _lru_cache = lru_cache(maxsize, typed)

    def wrap_exception(unwrapped_function):
        # [FUNCTION DOC] HELPER: wrap a function so it returns (result, None) or (None, exception) instead of
        # raising.
        def func(*args, **kwargs):
            # [FUNCTION DOC] HELPER: call the wrapped function, capturing any BaseException into the returned
            # pair.
            try:
                return unwrapped_function(*args, **kwargs), None
            except BaseException as err:
                return None, err

        return func

    def unwrap_exception(wrapped_function):
        # [FUNCTION DOC] HELPER: wrap a pair-returning function back so it returns the result or raises the
        # stored exception.
        def func(*args, **kwargs):
            # [FUNCTION DOC] HELPER: unpack the pair; raise the exception if present else return the result.
            result, exception = wrapped_function(*args, **kwargs)
            if exception is not None:
                raise exception
            else:
                return result

        return func

    def decorating_function(user_function):
        # [FUNCTION DOC] HELPER: apply wrap -> lru_cache -> unwrap to `user_function`.
        return unwrap_exception(_lru_cache(wrap_exception(user_function)))

    return decorating_function


# https://stackoverflow.com/a/28052583
ssl._create_default_https_context = ssl._create_unverified_context


class M4HTTPClient:
    # [CLASS DOC] HTTP client downloading media into a datasets-style cache (sharded by hash prefix, no ETag
    # versioning), with
    # robots.txt checks and an offline mode that only reads the cache.
    def __init__(self, cache_dir: Path, retries: int, offline_mode: bool, user_agent: Optional[str] = None):
        # [FUNCTION DOC] cache_dir: cache root; retries: max retries for downloads; offline_mode: never
        # download; user_agent: one of
        # "Googlebot-Image/1.0", "Googlebot-Video/1.0" or None (asserted). Builds the DownloadConfig used for
        # downloads.
        super(M4HTTPClient, self).__init__()
        # Hack found: https://github.com/igorbrigadir/DownloadConceptualCaptions/blob/efb16f028936e6c628b6ee435765d6e1771b0f2d/download_data.py#L13
        assert user_agent in ["Googlebot-Image/1.0", "Googlebot-Video/1.0", None]
        self.user_agent = user_agent
        self.datasets_download_config = DownloadConfig(
            cache_dir=cache_dir,
            user_agent=self.user_agent,
            num_proc=1,  # We handle this via `.map`
            max_retries=retries,
            # TODO @thomasw21 Not really sure we care about versioning ...
            use_etag=False,
        )
        self.offline_mode = offline_mode

    def check_robots_txt(self, url):
        # [FUNCTION DOC] Return True if the site's robots.txt (scheme://netloc/robots.txt) allows
        # `self.user_agent` to fetch the url.
        # Network access (cached per robots url); may raise from robots fetching errors.
        parsed_url = urllib.parse.urlparse(url)
        robots_txt_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        robots_parser = self.__get_robots_txt__(robots_txt_url)
        return robots_parser.can_fetch(self.user_agent, url)

    # TODO @thomasw21: Maybe lru if we're scared of it being too big at some point.
    @lru_cache_with_exception(maxsize=None)
    def __get_robots_txt__(self, robots_txt_url):
        # [FUNCTION DOC] HELPER: memoized (unbounded, exceptions cached; key includes self). Fetch and parse
        # robots.txt with a 10s
        # timeout; 4xx -> allow all; timeout -> allow all; other URL errors are raised. Returns the
        # RobotFileParser.
        # KNOWN ISSUE: an HTTP 5xx leaves the parser unread and not allow_all, so can_fetch then returns False
        # (urls are treated as disallowed).
        robots_parser = urllib.robotparser.RobotFileParser(robots_txt_url)

        # equivalent to `robots_parser.read()` but with a timeout
        try:
            f = urllib.request.urlopen(robots_parser.url, timeout=10)
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                # robots.txt could not be queried to check
                robots_parser.allow_all = True
            elif err.code >= 400 and err.code < 500:
                robots_parser.allow_all = True
        except urllib.error.URLError as err:
            if isinstance(err.reason, socket.timeout):
                # We couldn't find robots.txt, we assume we can query the media.
                robots_parser.allow_all = True
            else:
                # unknown exception
                # print(robots_txt_url, err)
                raise err
        else:
            raw = f.read()
            robots_parser.parse(raw.decode("utf-8").splitlines())

        return robots_parser

    def cache_path(self, url: str) -> Union[str, BaseException]:
        # [FUNCTION DOC] Return the local file path (str) for `url` or an exception object (never raises
        # except in offline mode
        # for unexpected errors). First tries the cache only; in offline mode returns FileNotFoundError when
        # missing.
        # Online: returns RobotsDisallow if robots.txt forbids the url, else downloads via cached_path.
        # KNOWN ISSUE: the cache-only lookup passes `local_files_only` both explicitly and in the
        # DownloadConfig fields
        # expansion, raising TypeError (duplicate keyword) which is swallowed online (raised in offline
        # mode), so the
        # local-first lookup never succeeds and robots.txt is checked even for cached files.
        try:
            # Try querying the file locally first
            try:
                return cached_path(
                    url,
                    compute_cache_path=compute_cache_path,
                    **{
                        field.name: getattr(self.datasets_download_config, field.name)
                        for field in fields(self.datasets_download_config)
                    },
                    local_files_only=True,
                )
            except FileNotFoundError as e:
                # We ignore the exception when the file could not be found in the cache. In offline mode, we return the exception
                if self.offline_mode:
                    return e
            except BaseException as e:
                # We ignore this exception as this will be caught down the line. However this should never happen ...
                if self.offline_mode:
                    # In offline mode, we raise exception as it's something that shouldn't happen
                    raise e

            # check if robots.txt allows us to download the the url
            if not self.check_robots_txt(url):
                return RobotsDisallow("Unable to query the url due to `robots.txt` restrictions")

            # Return file path or exception
            return cached_path(
                url, compute_cache_path=compute_cache_path, download_config=self.datasets_download_config
            )
        except BaseException as e:
            return e


def fetch_single_image(
    image_url: str, http_client: M4HTTPClient
) -> Union[Tuple[str, None], Tuple[None, BaseException]]:
    # [FUNCTION DOC] Download (or read from cache) `image_url` with `http_client` and verify it is a valid
    # image with PIL.
    # Returns (path, None) on success or (None, exception) on any failure; never raises.
    path_or_exception = http_client.cache_path(image_url)
    if isinstance(path_or_exception, str):
        path = path_or_exception
        try:
            # Check that it's an image
            with PIL.Image.open(path) as image:
                image.verify()
            return path, None
        except BaseException as exception:
            return None, exception
    else:
        exception = path_or_exception
        return None, exception


def batch_iter(dset: Dataset, transform: Callable[[Dict], List[Any]], batch_size: int = 1000):
    # [FUNCTION DOC] HELPER: generator over a Dataset in slices of `batch_size` rows, yielding each element
    # of transform(batch).
    num_rows = len(dset)
    for start in range(0, num_rows, batch_size):
        batch = dset[start : start + batch_size]
        for elt in transform(batch):
            yield elt


class PickableMediaDownloadGenerator:
    # [CLASS DOC] Iterator of download results (path/exception pairs) for all media urls of a dataset,
    # computed lazily by a
    # thread pool (`imap`, order preserved). The pool is only created on first `next` so the object stays
    # picklable
    # for `datasets` fingerprinting. Usable as a context manager that terminates the pool.
    def __init__(
        self,
        download_media_url: Callable[[str], Union[Tuple[str, None], Tuple[None, BaseException]]],
        get_media_urls: Callable[[Dict], List[str]],
        dset: Dataset,
        batch_size: int,
        num_threads_per_proc: int,
    ):
        # [FUNCTION DOC] Store the download function, the batch -> list of urls function, the dataset, batch
        # size and number of
        # threads; the pool and iterator are created lazily.
        self.download_media_url = download_media_url
        self.get_media_urls = get_media_urls
        self.dset = dset
        self.batch_size = batch_size
        self.num_threads_per_proc = num_threads_per_proc

        # This is used to trick the pickle algorithm as we load thread_pool and _media_iterator AFTER fingerprinting
        self._has_started = False
        self._thread_pool = None
        self._media_iterator = None

    def load_media_iterator(self):
        # [FUNCTION DOC] HELPER: create the ThreadPool and the ordered imap iterator applying
        # download_media_url to every url.
        self._thread_pool = ThreadPool(self.num_threads_per_proc)
        self._media_iterator = self._thread_pool.imap(
            self.download_media_url,
            iterable=batch_iter(dset=self.dset, transform=self.get_media_urls, batch_size=self.batch_size),
        )

    def __iter__(self):
        # [FUNCTION DOC] Return self (the object is its own iterator).
        return self

    def __next__(self):
        # [FUNCTION DOC] Return the next download result, starting the pool on first call. Raises
        # StopIteration when exhausted.
        if self._has_started is False:
            self.load_media_iterator()
            self._has_started = True
        return next(self._media_iterator)

    def __enter__(self):
        # [FUNCTION DOC] Return self.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] Terminate and join the thread pool if started, reset the started flag and delete the
        # pool attribute.
        # KNOWN ISSUE: the pool attribute is deleted rather than set to None, so a second exit or a restart
        # path that
        # reads `_thread_pool` raises AttributeError.
        if self._thread_pool is not None:
            self._has_started = False
            self._thread_pool.terminate()
            self._thread_pool.join()
            del self._thread_pool
