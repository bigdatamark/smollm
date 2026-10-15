# [MODULE DOC] Downloads individual Common Crawl WARC records from the public "commoncrawl" S3 bucket with
# boto3 byte-range requests (requires AWS credentials/network).
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ProxyConnectionError


class WarcDownloader:
    # [CLASS DOC] Picklable callable used with Dataset.map over metadata rows (warc_filename,
    # warc_record_offset, warc_record_length, warc, warc_error).
    def __init__(self):
        # [FUNCTION DOC] Creates an S3 client with standard retry mode and up to 10 attempts.
        config_boto = Config(
            # region_name="us-east-1", # Location of the CC data, commenting this line since it doesn't help
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.client = boto3.client("s3", config=config_boto)

    def __call__(self, example):
        # [FUNCTION DOC] Returns example unchanged if warc is already non-empty and warc_error is empty
        # (resume). Otherwise downloads the record and sets warc (bytes) and warc_error (string). Mutates and
        # returns the example.
        if example["warc"] and not example["warc_error"]:
            return example

        warc_filename = example["warc_filename"]
        warc_record_offset = example["warc_record_offset"]
        warc_record_length = example["warc_record_length"]

        warc, warc_error = self.get_warc_from_metadata(
            client=self.client,
            warc_filename=warc_filename,
            warc_record_offset=warc_record_offset,
            warc_record_length=warc_record_length,
        )
        example["warc"] = warc
        example["warc_error"] = warc_error
        return example

    def get_warc_from_metadata(self, client, warc_filename, warc_record_offset, warc_record_length):
        # [FUNCTION DOC] Fetches bytes [offset, offset + length - 1] of key warc_filename in bucket
        # commoncrawl with client. Returns (bytes, "") on success or (b"", repr(error)) on
        # ClientError/ProxyConnectionError; other errors propagate.
        try:
            response = client.get_object(
                Bucket="commoncrawl",
                Key=warc_filename,
                Range=f"bytes={warc_record_offset}-{warc_record_offset + warc_record_length - 1}",
            )
        except (ClientError, ProxyConnectionError) as e:
            return b"", repr(e)
        return response["Body"].read(), ""

    # Needed to make multiprocessing work
    def __reduce__(self):
        # [FUNCTION DOC] HELPER: pickling support for multiprocessing: recreate via the no-argument
        # constructor (new S3 client).
        return (self.__class__, ())
