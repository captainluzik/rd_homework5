import polars as pl
from contextlib import contextmanager
import time

FILE_PATH = ""  # Path to the ngram data file
WORD = "ära"


@contextmanager
def timer(msg: str):
    start = time.perf_counter()
    yield
    print(f"{msg} took {time.perf_counter() - start:.2f} seconds")


def process_data(file_path, word):
    with timer("Reading and processing file"):
        df = pl.scan_csv(
            file_path,
            separator="\t",
            has_header=False,
            with_column_names=lambda cols: ["word", "count", "year", "match_count", "volume_count"][:len(cols)],
        )

        grouped = (
            df.filter(pl.col("word") == word)
            .group_by("word")
            .agg(
                pl.sum("count").alias("total_count")
            )
            .collect(streaming=True)
        )

    return grouped


def main():
    with timer("Total time"):
        results = process_data(FILE_PATH, WORD)

    with timer("Printing results"):
        if results.shape[0] > 0:
            print(f"Total count for word '{WORD}': {results['total_count'][0]}")
        else:
            print(f"Word '{WORD}' not found in the dataset.")


if __name__ == "__main__":
    main()
