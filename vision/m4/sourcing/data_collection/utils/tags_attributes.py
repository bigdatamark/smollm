# [MODULE DOC] Tag and attribute vocabularies used to simplify HTML DOM trees: UNWRAP_TAGS (inline style tags
# whose
# content is kept but tag removed), structure / media / media_wiki (media without "figure") tag lists,
# INTERESTING_TAGS_SET (tags kept in the tree; others removed), MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET
# (per-source set of tags whose attributes are kept) and the list/set of media attributes kept.
from enum import Enum


# Tags we want to unwrap, mostly because they only modify the
# rendered style of a text
UNWRAP_TAGS = [
    "a",
    "abbr",
    "acronym",
    "b",
    "bdi",
    "bdo",
    "big",
    "cite",
    "code",
    "data",
    "dfn",
    "em",
    "font",
    "i",
    "ins",
    "kbd",
    "mark",
    "q",
    "s",
    "samp",
    "shadow",
    "small",
    "span",
    "strike",
    "strong",
    "sub",
    "sup",
    "time",
    "tt",
    "u",
    "var",
    "wbr",
]

# Tags that potentially contain text in a structured way, such that
# they would form paragraph, or contain automatic line break information
structure = [
    "address",
    "article",
    "aside",
    "blink",
    "blockquote",
    "body",
    "br",
    "caption",
    "center",
    "dd",
    "dl",
    "dt",
    "div",
    "figcaption",
    "h",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hgroup",
    "html",
    "legend",
    "main",
    "marquee",
    "ol",
    "p",
    "section",
    "summary",
    "title",
    "ul",
]
# Tags for the media we target
media = [
    "audio",
    "embed",
    "figure",
    "iframe",
    "img",
    "object",
    "picture",
    "video",
]
# wikipedia version
media_wiki = [
    "audio",
    "embed",
    # "figure",
    "iframe",
    "img",
    "object",
    "picture",
    "video",
]

# Tags including potential interesting attributes for the media,
# for example <source> which can contain the attribute "src" useful
# to find the path, with <source> being inside a tag <video> for example
contain_interesting_attributes = ["source"]


class InterestingAttributesSetCategory(Enum):
    # [CLASS DOC] Enum selecting which media tag set to use: COMMONCRAWL ("CommonCrawl") or WIKIPEDIA
    # ("Wikipedia", which
    # excludes `figure`). Used as key of MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET.
    COMMONCRAWL = "CommonCrawl"
    WIKIPEDIA = "Wikipedia"


MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET = {
    InterestingAttributesSetCategory.COMMONCRAWL: set(media + contain_interesting_attributes),
    InterestingAttributesSetCategory.WIKIPEDIA: set(media_wiki + contain_interesting_attributes),
}

# Tags not in this list will be removed from the tree
interesting_tags = structure + media + contain_interesting_attributes
INTERESTING_TAGS_SET = set(interesting_tags)
# Tags that we currently remove, but that are potentially worth
# considering if we really want to have a super precise representation

# Interesting attributes for the media. All other attributes
# will be removed
interesting_attributes = [
    "src",  # embed, audio, iframe, img, video, source
    "alt",  # img
    "height",  # embed, iframe, img, object, video
    "width",  # embed, iframe, img, object, video
    "srcset",  # img, source
    "type",  # embed, object, source
    "title",  # all
    "data",  # object,
    "sizes",  # img, source
    "media",  # source
    "srcdoc",  # iframe
]
set_interesting_attributes = set(interesting_attributes)
