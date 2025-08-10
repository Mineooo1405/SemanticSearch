#!/usr/bin/env python
"""Analyze word and sentence lengths of the 'document' column in a TSV file.

Input TSV header is expected to include at least:
    query_id	query_text	document_id	document	label

The script will:
 1. Read the TSV (UTF-8, with potential quoted fields)
 2. Compute per-row metrics:
       - word_count (simple whitespace tokenization)
       - sentence_count (regex split on sentence end punctuation . ! ?)
       - avg_words_per_sentence
 3. Emit an (optionally) per-row output TSV with added columns
 4. Emit a JSON summary with aggregate statistics (count, min, max, mean, median,
    standard deviation, selected percentiles) for word_count and sentence_count.

Usage (PowerShell / CMD):
    python data_process/analyze_document_lengths.py \
        --input integrated_robust04_modified_v2_subset_100rows.tsv \
        --per-row-output document_lengths.tsv \
        --summary-output document_length_summary.json

If output paths are omitted, only a concise textual summary prints to stdout.

Notes:
 - Keeps memory usage low by streaming rows.
 - Does not depend on external NLP libraries; for more accurate sentence
   segmentation you could integrate spaCy or NLTK later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from typing import List, Dict, Any, Tuple

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_REGEX = re.compile(r"\s+")

def tokenize_words(text: str) -> List[str]:
    """Simple whitespace-based tokenization after stripping.
    Collapses multiple spaces/newlines/tabs.
    """
    text = text.strip()
    if not text:
        return []
    return WHITESPACE_REGEX.split(text)

def split_sentences(text: str) -> List[str]:
    """Very lightweight sentence splitter.

    Splits on whitespace following ., !, or ? characters. This is *not*
    linguistically perfect (e.g., abbreviations like U.S. will over-split),
    but is dependency-free and fast.
    """
    text = text.strip()
    if not text:
        return []
    # Normalize internal whitespace to allow consistent splitting metrics.
    normalized = WHITESPACE_REGEX.sub(" ", text)
    sentences = SENTENCE_SPLIT_REGEX.split(normalized)
    # Clean and drop empties
    return [s.strip() for s in sentences if s.strip()]

def percentile(data: List[int | float], p: float) -> float:
    """Compute percentile p (0-100) using linear interpolation between closest ranks."""
    if not data:
        return float("nan")
    if p <= 0:
        return float(data[0])
    if p >= 100:
        return float(data[-1])
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float(d0 + d1)

def summarize(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    sorted_vals = sorted(values)
    return {
        "count": len(values),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(sorted_vals),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(sorted_vals, 10),
        "p25": percentile(sorted_vals, 25),
        "p75": percentile(sorted_vals, 75),
        "p90": percentile(sorted_vals, 90),
    }

def process(
    input_path: str,
    per_row_output: str | None = None,
    summary_output: str | None = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    word_counts: List[int] = []
    sentence_counts: List[int] = []

    # Prepare writer if needed
    per_row_writer = None
    out_f = None
    if per_row_output:
        out_f = open(per_row_output, "w", newline="", encoding=encoding)
        per_row_writer = csv.writer(out_f, delimiter="\t")

    with open(input_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"document"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input file missing required columns: {missing}")

        if per_row_writer:
            header = list(reader.fieldnames) + [
                "word_count",
                "sentence_count",
                "avg_words_per_sentence",
            ]
            per_row_writer.writerow(header)

        for row in reader:
            doc = row.get("document", "")
            words = tokenize_words(doc)
            sentences = split_sentences(doc)
            w_count = len(words)
            s_count = len(sentences)
            avg_w_per_s = (w_count / s_count) if s_count else 0.0
            word_counts.append(w_count)
            sentence_counts.append(s_count)

            if per_row_writer:
                per_row_writer.writerow(
                    [
                        *(row.get(col, "") for col in reader.fieldnames),
                        w_count,
                        s_count,
                        f"{avg_w_per_s:.2f}",
                    ]
                )

    if out_f:
        out_f.close()

    # Build exact distribution of sentence counts
    sentence_dist_counter: Counter[int] = Counter(sentence_counts)
    sentence_distribution = dict(sorted(sentence_dist_counter.items()))

    # Build bucket categories for sentence counts
    buckets: List[Tuple[str, range | None]] = [
        ("0", range(0, 1)),
        ("1", range(1, 2)),
        ("2", range(2, 3)),
        ("3", range(3, 4)),
        ("4-5", range(4, 6)),
        ("6-10", range(6, 11)),
        ("11-20", range(11, 21)),
        ("21-50", range(21, 51)),
        ("51+", None),  # special: >=51
    ]

    bucket_counts: Dict[str, int] = {label: 0 for label, _ in buckets}
    for sc in sentence_counts:
        placed = False
        for label, r in buckets:
            if r is None:
                if sc >= 51:
                    bucket_counts[label] += 1
                    placed = True
                    break
            else:
                if sc in r:
                    bucket_counts[label] += 1
                    placed = True
                    break
        if not placed:
            # Should not happen, but guard
            bucket_counts.setdefault("other", 0)
            bucket_counts["other"] += 1

    result = {
        "word_count_stats": summarize(word_counts),
        "sentence_count_stats": summarize(sentence_counts),
        "avg_words_per_sentence_overall": (
            (sum(word_counts) / sum(sentence_counts)) if sum(sentence_counts) else 0.0
        ),
        "sentence_count_distribution": sentence_distribution,
        "sentence_count_buckets": bucket_counts,
    }

    if summary_output:
        with open(summary_output, "w", encoding=encoding) as sf:
            json.dump(result, sf, ensure_ascii=False, indent=2)

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Analyze word and sentence lengths in the 'document' column of a TSV file."
    )
    parser.add_argument("--input", required=True, help="Input TSV path")
    parser.add_argument(
        "--per-row-output",
        help="Optional output TSV path with appended metrics per row",
    )
    parser.add_argument(
        "--summary-output", help="Optional JSON file path for aggregate statistics"
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="File encoding (default: utf-8)"
    )
    parser.add_argument(
        "--max-field-size",
        type=int,
        default=None,
        help="Override csv field_size_limit (useful for very large document fields).",
    )
    args = parser.parse_args()

    # Increase CSV field size limit to handle very large 'document' fields
    if args.max_field_size:
        try:
            csv.field_size_limit(args.max_field_size)
        except OverflowError:
            # Fallback to system max size if provided too large
            csv.field_size_limit(sys.maxsize)
    else:
        # Try to raise to maximum possible if not explicitly set
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            # Binary search downwards until it succeeds
            upper = sys.maxsize
            lower = 1024 * 1024
            while lower < upper:
                mid = (lower + upper) // 2
                try:
                    csv.field_size_limit(mid)
                    lower = mid + 1
                except OverflowError:
                    upper = mid - 1
            # Set to the last successful (upper)
            try:
                csv.field_size_limit(upper)
            except Exception:
                pass

    res = process(
        input_path=args.input,
        per_row_output=args.per_row_output,
        summary_output=args.summary_output,
        encoding=args.encoding,
    )

    # Pretty print concise summary
    print("=== Document Length Analysis Summary ===")
    print("Total rows:", res["word_count_stats"]["count"])
    print("-- Words --")
    for k, v in res["word_count_stats"].items():
        print(f"  {k}: {v}")
    print("-- Sentences --")
    for k, v in res["sentence_count_stats"].items():
        print(f"  {k}: {v}")
    print(
        "Overall average words per sentence:",
        f"{res['avg_words_per_sentence_overall']:.2f}",
    )
    # Print distribution (limit to first 30 entries for readability)
    dist_items = list(res.get("sentence_count_distribution", {}).items())
    if dist_items:
        print("Sentence count distribution (sentence_count -> docs):")
        for sc, cnt in dist_items[:30]:
            print(f"  {sc}: {cnt}")
        if len(dist_items) > 30:
            print(f"  ... ({len(dist_items) - 30} more)")
    buckets = res.get("sentence_count_buckets")
    if buckets:
        print("Sentence count buckets:")
        for label, cnt in buckets.items():
            print(f"  {label}: {cnt}")

if __name__ == "__main__":
    main()
