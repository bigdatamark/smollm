# [MODULE DOC] Helpers for the n-gram extraction / search tools: building the SQLite database for a shard from
# CSV dumps,
# and encoding token-id n-grams (n = 1..4) as unique integers. With token ids offset by one into [1, V]
# (V = vocab size), an n-gram (a, b, c) maps to a*V^2 + b*V + c, so ranges for different n do not overlap and
# 4-grams still fit in int64 for V = 50,257.
import csv
import os
import sqlite3
import sys

import numpy as np


def create_database(shard_name: str):
    # [FUNCTION DOC] Create data/extracted_databases/<shard_name>.db unless it already exists (then print and
    # return). Creates
    # table ngrams(shard, document, ngrams) loaded from <shard_name>.ngrams.csv plus an index on ngrams, table
    # urls(document, url) from <shard_name>.urls.csv, and table htmls(document, html) from
    # <shard_name>.htmls.csv (raising the csv field size limit to sys.maxsize globally first). Commits, closes
    # and
    # prints progress. Raises if CSV files are missing.
    """
    If the databse does not exist, create it
    TODO: update so that we can take in multiple shards
    """
    db_filepath = f"data/extracted_databases/{shard_name}.db"
    if os.path.exists(db_filepath):
        print("Database already exists")
        return

    print("Creating database")
    connection = sqlite3.connect(db_filepath)
    cur = connection.cursor()

    cur.execute(
        """CREATE TABLE ngrams(
            shard INT,
            document INT,
            ngrams INT
        );"""
    )
    ngrams_csv_filepath = f"data/extracted_databases/{shard_name}.ngrams.csv"
    with open(ngrams_csv_filepath) as ngrams_csv:
        rows = csv.reader(ngrams_csv)
        cur.executemany("INSERT INTO ngrams VALUES (?, ?, ?)", rows)
    print("Added ngrams to the databse.")
    cur.execute(
        """
        CREATE INDEX ngrams_index
        ON ngrams (ngrams);
        """
    )
    print("Created ngrams index.")

    cur.execute(
        """CREATE TABLE urls(
            document INT,
            url TEXT
        );"""
    )  # Could add some foreign key constraints to link the two tables
    urls_csv_filepath = f"data/extracted_databases/{shard_name}.urls.csv"
    with open(urls_csv_filepath) as urls_csv:
        rows = csv.reader(urls_csv)
        cur.executemany("INSERT INTO urls VALUES (?, ?)", rows)
    print("Added urls to the databse.")

    csv.field_size_limit(sys.maxsize)
    # To avoid running into `_csv.Error: field larger than field limit (131072)` error
    # https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    cur.execute(
        """CREATE TABLE htmls(
            document INT,
            html TEXT
        );"""
    )
    htmls_csv_filepath = f"data/extracted_databases/{shard_name}.htmls.csv"
    with open(htmls_csv_filepath) as htmls_csv:
        rows = csv.reader(htmls_csv)
        cur.executemany("INSERT INTO htmls VALUES (?, ?)", rows)
    print("Added htmls to the databse.")

    connection.commit()
    connection.close()
    print("Database created.")


# n-grams computation

# Mapping each uni, bi, tri, 4-grams into a unique id.
# Let's call V the size of the vocabulary.
# For instance, sequence of integers is [a, b, c] -> a V**2 + b V + c

# 1-grams \in [|1; V|]
# 2-grams \in [|V + 1; V**2 + V|]
# 3-grams \in [|V**2 + V + 1; V**3 + V**2 + V|]
# 4-grams \in [|V**3 + V**2 + V + 1; V**4 + V**3 + V**2 + V|]

# for V=50_257,
# V**4 + V**3 + V**2 + V = 6_379_621_074_231_211_300
# int64 upper bound =      9_223_372_036_854_775_807


def unique_tolist(l_):
    # [FUNCTION DOC] HELPER: return the sorted unique values of an array-like as a Python list.
    return np.unique(l_).tolist()


def get_ngrams(ids: np.ndarray, voc_size: int, n: int):
    # [FUNCTION DOC] Return the sorted unique n-gram ids of a 1-D integer numpy array ids for n in 1..4, using
    # voc_size as base
    # (ids are first offset by +1). Raises ValueError (no message) if there are fewer than n ids, and
    # ValueError
    # for n >= 5 (or n < 1).
    ids = ids + 1  # offset by one otherwise the mapping is not bijective...
    V_pow = [voc_size**i for i in range(4)]

    if len(ids) < n:
        raise ValueError

    # Horrible implementation but i don't want to mess it up yet, so going slowly
    # TODO: update
    if n == 1:
        return unique_tolist(ids)
    elif n == 2:
        bigrams = (V_pow[1] * ids)[:-1] + ids[1:]
        return unique_tolist(bigrams)
    elif n == 3:
        trigrams = (V_pow[2] * ids)[:-2] + (V_pow[1] * ids)[1:-1] + ids[2:]
        return unique_tolist(trigrams)
    elif n == 4:
        fourgrams = (V_pow[3] * ids)[:-3] + (V_pow[2] * ids)[1:-2] + (V_pow[1] * ids)[2:-1] + ids[3:]
        return unique_tolist(fourgrams)
    else:
        raise ValueError("n-grams for n>=5 are not supported yet.")
