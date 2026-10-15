# [MODULE DOC] Extraction of the HTML document from a raw WARC record (bytes) with warcio and BeautifulSoup.
# Raises the recursion limit to 10000 at import time because decode_contents can recurse deeply.
import io
import sys

from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
from warcio.archiveiterator import WARCIterator


# For `soup.decode_content` that can hit the limit
sys.setrecursionlimit(10000)


class HtmlExtractor:
    # [CLASS DOC] Callable used with Dataset.map over rows with warc/warc_error/html/html_error fields.
    def __call__(self, example):
        # [FUNCTION DOC] Processes one example mapping: returns it unchanged if it already has html and no
        # html_error (resume); if warc_error is set, sets html "" and html_error "no WARC"; otherwise fills
        # html/html_error from get_html_from_warc. Mutates and returns the example.
        if example["html"] and not example["html_error"]:
            return example

        warc, warc_error = example["warc"], example["warc_error"]
        if warc_error:
            example["html"] = ""
            example["html_error"] = "no WARC"
            return example

        html, html_error = self.get_html_from_warc(warc=warc)
        example["html"] = html
        example["html_error"] = html_error
        return example

    def get_html_from_warc(self, warc):
        # [FUNCTION DOC] Parses warc bytes and returns a (html_str, error_str) tuple. Takes the payload of the
        # first "response" record and its WARC-Identified-Content-Charset; if no charset, uses the first
        # encoding guessed by bs4 EncodingDetector. Then parses with html.parser and serializes with
        # decode_contents, checking the result is encodable. Any failure returns ("", message); a missing page
        # or encoding returns ("", "Not page or encoding"); success returns (html, "").
        page, encoding = None, None

        with io.BytesIO(warc) as stream:
            try:
                for record in WARCIterator(stream):
                    if record.rec_type == "response":
                        page = record.content_stream().read()
                        encoding = record.rec_headers["WARC-Identified-Content-Charset"]
                        break
            except Exception as e:
                return "", str(e)

        if not encoding:
            try:
                for enc in EncodingDetector(page, is_html=True).encodings:
                    # take the first detected encoding
                    encoding = enc
                    break
            except Exception as e:
                return "", str(e)

        if (not page) or (not encoding):
            return "", "Not page or encoding"

        try:
            soup = BeautifulSoup(page, "html.parser", from_encoding=encoding)
        except Exception as e:
            return "", str(e)

        try:
            html_str = soup.decode_contents(formatter="html")
        except Exception as e:
            return "", str(e)

        try:
            html_str.encode()
        except Exception as e:
            return "", str(e)

        return html_str, ""
