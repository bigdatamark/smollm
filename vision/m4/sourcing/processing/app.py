# [MODULE DOC] Streamlit app: an n-gram "search engine" over one extracted Common Crawl shard. At import it
# loads the
# GPT-2 tokenizer and opens (creating if needed) the SQLite database data/extracted_databases/<SHARD_NAME>.db
# (tables ngrams, urls, htmls). The page takes a text query, tokenizes it, encodes it as 1- to 4-gram ids (a
# longer query is split into all overlapping 4-grams), looks up all documents containing any of those ids,
# shows them as an HTML table with clickable URLs, and renders the raw HTML of a selected document both as
# an embedded page and as text. Logs each query with a timestamp to stdout. Module globals: tokenizer,
# vocab_size, cur (database cursor).
# KNOWN ISSUE: imports `processing.custom.utils`, which does not exist in this repo (the helpers live in
# extracting_ngrams/utils.py), so the app fails to import as-is.
# KNOWN ISSUE: SQL queries are built with f-strings rather than parameters; an empty query (0 tokens) makes
# ngram_conv call get_ngrams with n=0, which raises ValueError.
import sqlite3
import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import tokenizers
from datasets.fingerprint import Hasher
from processing.custom.utils import create_database, get_ngrams
from transformers import AutoTokenizer


SHARD_NAME = "4e47925f7c894bd8eb56e5dd1d778ec77bf2c90f6cee0e32e31615393391c67a"
db_filepath = f"data/extracted_databases/{SHARD_NAME}.db"


# Load tokenizer
@st.cache(hash_funcs={tokenizers.Tokenizer: Hasher.hash})
def load_tokenizer(tokenizer_name):
    # [FUNCTION DOC] Load and return the `AutoTokenizer` named tokenizer_name; cached by Streamlit (tokenizers
    # hashed with the
    # datasets Hasher).
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def ngram_conv(ids: np.ndarray):
    # [FUNCTION DOC] HELPER: convert a 1-D array of token ids into n-gram ids using the module-global
    # vocab_size: if there are at
    # most 4 tokens, return the single n-gram id of the whole sequence (asserts exactly one); otherwise return
    # the
    # unique ids of all overlapping 4-grams. Returns a list of ints.
    if len(ids) <= 4:
        ngram = get_ngrams(ids, vocab_size, len(ids))
        assert len(ngram) == 1
    else:
        ngram = get_ngrams(ids, vocab_size, 4)
    return ngram


# Load database
# Following https://discuss.streamlit.io/t/caching-sqlite-db-connection-resulting-in-glitchy-rendering-of-the-page/19017 recommendation on caching
@st.experimental_singleton
def load_database():
    # [FUNCTION DOC] Ensure the database for SHARD_NAME exists (create_database builds it from CSVs if
    # missing), open a SQLite
    # connection usable across threads, print "Database reloaded" and return a cursor. Cached as a Streamlit
    # singleton.
    # If necessary, create the databse
    create_database(SHARD_NAME)

    # Get the connection
    connection = sqlite3.connect(db_filepath, check_same_thread=False)
    print("Database reloaded")
    return connection.cursor()


# Get n-grams
def run_conv(text: str):
    # [FUNCTION DOC] Tokenize text with the global tokenizer (numpy output, no attention mask) and return a
    # tuple of
    # (token id array, list of n-gram ids from ngram_conv).
    ids = tokenizer(text, return_attention_mask=False, return_tensors="np")["input_ids"][0]
    ngram_ids = ngram_conv(ids)
    return ids, ngram_ids


@st.cache()
def run_query(ngram_ids: List[int]):
    # [FUNCTION DOC] For each n-gram id, select (shard, document, url) rows from ngrams left-joined with urls
    # on document and
    # collect them; return the de-duplicated results as a list (order unspecified). Uses the module-global
    # cursor;
    # results cached by Streamlit.
    results = []
    for ngram in ngram_ids:
        sql_query = f"""SELECT shard, ngrams.document, url
                        FROM ngrams
                        LEFT JOIN urls ON ngrams.document = urls.document
                        WHERE ngrams={ngram}
                        ;"""
        r = cur.execute(sql_query).fetchall()
        results.extend(r)
    return list(set(results))


tokenizer = load_tokenizer("gpt2")
vocab_size = tokenizer.vocab_size

cur = load_database()


# Header
st.header('N-gram "search engine."')
st.markdown(
    """
Type a query in the text field. It will retrieve documents (in CC) that contains at least one occurence of this text query.

The text query is first tokenized (using GPT2 tokenizer) and the n-grams comparisons is done in the tokenized space.

With the current implementation, the database only indexes uni to 4-grams. => If a query is longer than a 4-gram, than the results are the union of all the consecutive sub 4-grams queries.
    """
)


# Build the text query
st.write("""---""")
st.write("### The query")
text_query = st.text_input(label="Write in the text query (ngrams).", value="NEW YORK")
print(f"{time.ctime()} {text_query}")


# Run the query
input_ids, ngram_ids = run_conv(text_query)
st.write(f"Tokenized query: {input_ids}")
if len(input_ids) <= 4:
    st.write(f"Running this query as a {['1-gram', '2-gram', '3-gram', '4-gram'][len(input_ids)-1]}")
else:
    st.write("Running this query as a series of overlaping 4-grams.")

results = run_query(ngram_ids)


# Rendering results in a table
st.write("""---""")
st.write("### The retrieved documents")
df = pd.DataFrame(results, columns=["shard id", "document id", "url"])


def make_clickable(link):
    # [FUNCTION DOC] HELPER: wrap a URL string in an HTML anchor tag that opens in a new tab.
    return f'<a target="_blank" href="{link}">{link}</a>'


df["url"] = df["url"].apply(make_clickable)
# st.dataframe(df) # Couldn't make the links clickable... so defaulting to writing a HTML table
st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Render the html page
document_ids = df["document id"].tolist()
if document_ids:
    st.write("""---""")
    st.write("### Rendered results")
    document_id_to_display = st.selectbox(
        "Document id to render (HTML page from the dump)",
        document_ids,
        index=0,
    )

    sql_query = f"""SELECT html
                    FROM htmls
                    WHERE document={document_id_to_display}
                    ;"""
    page = cur.execute(sql_query).fetchall()
    assert len(page) == 1
    page = page[0][0].strip()
    st.components.v1.html(page, width=1000, height=1000, scrolling=True)

    st.write("---")
    st.text(page)
