import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import re
import json

SENT_SPLIT_REGEX = re.compile(r'(?<=[\.!?！？。])\s+')
WORD_SPLIT_REGEX = re.compile(r'\w+')

def simple_sent_tokenize(text: str):
    # Tách câu đơn giản, giữ câu > 0 ký tự
    parts = SENT_SPLIT_REGEX.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

def word_tokens(text: str):
    return WORD_SPLIT_REGEX.findall(text.lower())

def describe(dist_list):
    if not dist_list:
        return {}
    dist_list_sorted = sorted(dist_list)
    n = len(dist_list_sorted)
    def pct(p):
        k = (p/100)*(n-1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return dist_list_sorted[int(k)]
        return dist_list_sorted[f] + (dist_list_sorted[c]-dist_list_sorted[f])*(k-f)
    return {
        "count": n,
        "min": dist_list_sorted[0],
        "max": dist_list_sorted[-1],
        "mean": round(statistics.fmean(dist_list_sorted), 3),
        "median": pct(50),
        "p10": pct(10),
        "p25": pct(25),
        "p75": pct(75),
        "p90": pct(90),
        "std": round(statistics.pstdev(dist_list_sorted), 3) if n > 1 else 0.0
    }

def analyze_file(path: Path, limit_docs=None):
    chunks = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            chunks.append(row)
    if not chunks:
        return {"file": str(path), "error": "Empty or unreadable"}

    # Group by (query_id, document_id)
    by_doc = defaultdict(list)
    for r in chunks:
        q = r['query_id']
        d = r['document_id']
        by_doc[(q,d)].append(r['chunk_text'])

    if limit_docs:
        # Keep only first N docs (ordered by insertion)
        keys = list(by_doc.keys())[:limit_docs]
        by_doc = {k: by_doc[k] for k in keys}

    chunk_lengths_chars = []
    chunk_lengths_words = []
    chunk_lengths_sents = []
    doc_chunk_counts = []

    duplicate_counter = Counter()
    vocab_counter = Counter()

    top_longest = []  # store tuples (len_words, query_id, document_id, preview)

    for (q,d), texts in by_doc.items():
        doc_chunk_counts.append(len(texts))
        for t in texts:
            t_norm = t.strip()
            duplicate_counter[t_norm] += 1
            words = word_tokens(t_norm)
            sents = simple_sent_tokenize(t_norm)
            chunk_lengths_chars.append(len(t_norm))
            chunk_lengths_words.append(len(words))
            chunk_lengths_sents.append(len(sents))
            vocab_counter.update(words)
            if words:
                top_longest.append((len(words), q, d, t_norm[:130].replace('\n',' ')))

    # Optional debugging: show violating chunks if MIN_SENT or MAX_SENT env provided
    import os
    dbg_min = os.getenv('ANALYZE_MIN_SENT')
    dbg_max = os.getenv('ANALYZE_MAX_SENT')
    violating_examples = []
    if dbg_min or dbg_max:
        try:
            dbg_min_v = int(dbg_min) if dbg_min else None
            dbg_max_v = int(dbg_max) if dbg_max else None
        except ValueError:
            dbg_min_v = dbg_max_v = None
        if dbg_min_v or dbg_max_v:
            for (q,d), texts in list(by_doc.items())[:50]:  # limit scanning
                for t in texts:
                    sents = simple_sent_tokenize(t)
                    if (dbg_min_v and len(sents) < dbg_min_v) or (dbg_max_v and len(sents) > dbg_max_v):
                        violating_examples.append((len(sents), q, d, t[:120].replace('\n',' ')))
                        if len(violating_examples) >= 15:
                            break
                if len(violating_examples) >= 15:
                    break

    # Sort and keep top 10
    top_longest = sorted(top_longest, key=lambda x: x[0], reverse=True)[:10]

    duplicates = {k: v for k,v in duplicate_counter.items() if v > 1}
    total_chunks = len(chunk_lengths_chars)
    dup_ratio = round(sum(v-1 for v in duplicates.values()) / total_chunks, 4) if total_chunks else 0

    vocab_size = len(vocab_counter)
    total_tokens = sum(vocab_counter.values())
    top_vocab = vocab_counter.most_common(20)

    return {
        "file": str(path),
        "documents": len(by_doc),
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": round(statistics.fmean(doc_chunk_counts),3) if doc_chunk_counts else 0,
        "chunk_chars": describe(chunk_lengths_chars),
        "chunk_words": describe(chunk_lengths_words),
        "chunk_sentences": describe(chunk_lengths_sents),
        "duplicates_count": len(duplicates),
        "duplicate_ratio": dup_ratio,
        "top_duplicates_example": list(sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]),
        "vocab_size": vocab_size,
        "avg_tokens_per_chunk": round(chunk_lengths_words and (sum(chunk_lengths_words)/total_chunks) or 0,3),
        "token_type_ratio": round(vocab_size/total_tokens,4) if total_tokens else 0,
        "top_tokens": top_vocab,
    "top_longest_chunks": [
            {
                "words": w,
                "query_id": q,
                "document_id": d,
                "preview": p
            } for w,q,d,p in top_longest
    ],
    "violating_examples": violating_examples
    }

def compare(files_stats):
    if len(files_stats) < 2:
        return {}
    # Compare mean words per chunk & chunk count
    comparison = []
    for st in files_stats:
        comparison.append({
            "file": st["file"],
            "chunks": st["total_chunks"],
            "avg_words": st["chunk_words"].get("mean"),
            "avg_sentences": st["chunk_sentences"].get("mean"),
            "avg_chars": st["chunk_chars"].get("mean")
        })
    # simple ranking by avg_words
    ranked = sorted(comparison, key=lambda x: x["avg_words"] or 0, reverse=True)
    return {"ranking_by_avg_words": ranked}

def main():
    ap = argparse.ArgumentParser(description="Phân tích chunk_text trong file TSV chunking output.")
    ap.add_argument("-f","--files", nargs='+', required=True, help="Danh sách file TSV")
    ap.add_argument("--limit-docs", type=int, help="Giới hạn số document đầu tiên để phân tích (debug nhanh)")
    ap.add_argument("--json", action="store_true", help="In kết quả dạng JSON")
    ap.add_argument("--save-json", type=Path, help="Lưu tổng hợp vào file JSON")
    args = ap.parse_args()

    results = []
    for fp in args.files:
        stats = analyze_file(Path(fp), limit_docs=args.limit_docs)
        results.append(stats)

    comp = compare(results)

    if args.json or args.save_json:
        out = {"files": results, "comparison": comp}
        text = json.dumps(out, ensure_ascii=False, indent=2)
        if args.save_json:
            args.save_json.write_text(text, encoding='utf-8')
        print(text)
    else:
        for st in results:
            print(f"\n=== File: {st['file']} ===")
            if 'error' in st:
                print(" Error:", st['error'])
                continue
            print(f" Documents: {st['documents']}")
            print(f" Total chunks: {st['total_chunks']}  Avg chunks/doc: {st['avg_chunks_per_doc']}")
            print(f" Chunk words (mean/median/min/max): {st['chunk_words'].get('mean')}/{st['chunk_words'].get('median')}/{st['chunk_words'].get('min')}/{st['chunk_words'].get('max')}")
            print(f" Chunk sentences (mean/median/min/max): {st['chunk_sentences'].get('mean')}/{st['chunk_sentences'].get('median')}/{st['chunk_sentences'].get('min')}/{st['chunk_sentences'].get('max')}")
            print(f" Duplicate chunks (>1): {st['duplicates_count']}  Dup ratio: {st['duplicate_ratio']}")
            if st['top_duplicates_example']:
                print("  Top dup examples (count, preview):")
                for txt,count in st['top_duplicates_example']:
                    preview = txt[:80].replace('\n',' ')
                    print(f"   {count}x | {preview}")
            print(f" Vocab size: {st['vocab_size']}  TTR: {st['token_type_ratio']}")
            print(" Top tokens:", ", ".join(f"{tok}:{cnt}" for tok,cnt in st['top_tokens'][:10]))
            print(" Top longest chunks (words/query/doc preview):")
            for item in st['top_longest_chunks']:
                print(f"   {item['words']} | {item['query_id']} | {item['document_id']} | {item['preview']}")
        if comp:
            print("\n=== Comparison (avg words desc) ===")
            for r in comp["ranking_by_avg_words"]:
                print(f" {r['file']}  chunks={r['chunks']}  avg_words={r['avg_words']}  avg_sents={r['avg_sentences']}")

if __name__ == "__main__":
    main()