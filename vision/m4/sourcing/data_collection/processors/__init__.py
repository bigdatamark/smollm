# [PACKAGE DOC] Package of the data collection processors. Re-exports the main processor classes so callers
# can import them from the package: DOMTreeSimplificator, HtmlExtractor, ImageDeduplicator,
# TextMediaPairsExtractor, PairFiltering, PreExtractionSimplificator, WarcDownloader,
# CommonCrawlWebDocumentExtractor, FilteringFunctions, WebDocumentFilteringDocLevel/NodeLevel,
# WebDocumentImageDeduplication and WebDocumentLineDeduplication.
from m4.sourcing.data_collection.processors.dom_tree_simplificator import DOMTreeSimplificator
from m4.sourcing.data_collection.processors.html_extractor import HtmlExtractor
from m4.sourcing.data_collection.processors.image_deduplicator import ImageDeduplicator
from m4.sourcing.data_collection.processors.pair_extractor import TextMediaPairsExtractor
from m4.sourcing.data_collection.processors.pair_filtering import PairFiltering
from m4.sourcing.data_collection.processors.pre_extraction_simplificator import PreExtractionSimplificator
from m4.sourcing.data_collection.processors.warc_downloader import WarcDownloader
from m4.sourcing.data_collection.processors.web_document_extractor import CommonCrawlWebDocumentExtractor
from m4.sourcing.data_collection.processors.web_document_filtering import (
    FilteringFunctions,
    WebDocumentFilteringDocLevel,
    WebDocumentFilteringNodeLevel,
)
from m4.sourcing.data_collection.processors.web_document_image_deduplication import WebDocumentImageDeduplication
from m4.sourcing.data_collection.processors.web_document_line_deduplication import WebDocumentLineDeduplication
