import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    if "chunk_id" not in df.columns:
        raise ValueError(f"chunk_id column missing in {path}")
    return df


def build_union_intersection(cos_df: pd.DataFrame, bm_df: pd.DataFrame, out_dir: Path):
    cos_pos = set(cos_df.loc[cos_df["label"] == 1, "chunk_id"])
    bm_pos = set(bm_df.loc[bm_df["label"] == 1, "chunk_id"])

    union_pos_ids = cos_pos | bm_pos
    inter_pos_ids = cos_pos & bm_pos

    cos_neg = set(cos_df.loc[cos_df["label"] == 0, "chunk_id"])
    bm_neg = set(bm_df.loc[bm_df["label"] == 0, "chunk_id"])
    union_neg_ids = cos_neg | bm_neg
    inter_neg_ids = cos_neg & bm_neg

    def _save(ids, label_name: str):
        subset = pd.concat([cos_df, bm_df]).drop_duplicates("chunk_id")
        subset = subset.loc[subset["chunk_id"].isin(ids)].copy()
        subset = subset[["query_id", "query_text", "chunk_id", "chunk_text"]]  # keep common cols
        subset["label"] = 1 if "pos" in label_name else 0
        subset.to_csv(out_dir / f"dataset_{label_name}.tsv", sep="\t", index=False)

    _save(union_pos_ids, "union_pos")
    _save(union_neg_ids, "union_neg")
    _save(inter_pos_ids, "inter_pos")
    _save(inter_neg_ids, "inter_neg")


def hybrid_score(cos_df: pd.DataFrame, bm_df: pd.DataFrame, weight: float, out_dir: Path):
    merged = cos_df.merge(
        bm_df[["chunk_id", "bm25_score"]], on="chunk_id", how="inner", suffixes=("", ""))
    scaler = MinMaxScaler()
    merged[["cos_norm", "bm_norm"]] = scaler.fit_transform(merged[["cosine_score", "bm25_score"]])
    merged["hybrid_score"] = weight * merged["cos_norm"] + (1 - weight) * merged["bm_norm"]
    merged = merged.sort_values(by="hybrid_score", ascending=False)
    merged.to_csv(out_dir / "ranked_hybrid.tsv", sep="\t", index=False)


def logistic_rank(cos_df: pd.DataFrame, bm_df: pd.DataFrame, out_dir: Path):
    # Use union dataset as training data
    train_df = pd.concat([cos_df, bm_df]).drop_duplicates("chunk_id")
    if "label" not in train_df.columns:
        print("label column missing; skip logistic regression")
        return
    X = train_df[["cosine_score", "bm25_score"]].fillna(0.0).values
    y = train_df["label"].values
    if len(set(y)) < 2:
        print("Need both pos and neg labels for training; skip logistic regression")
        return
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    print("Learned weights:", model.coef_[0])

    # Apply to merged dataset
    merged = cos_df.merge(bm_df[["chunk_id", "bm25_score"]], on="chunk_id", how="inner")
    X_all = merged[["cosine_score", "bm25_score"]].fillna(0.0).values
    merged["lr_score"] = model.predict_proba(X_all)[:, 1]
    merged = merged.sort_values(by="lr_score", ascending=False)
    merged.to_csv(out_dir / "ranked_lr.tsv", sep="\t", index=False)


def reciprocal_rank_fusion(cos_df: pd.DataFrame, bm_df: pd.DataFrame, k: int, out_dir: Path):
    """Tính Reciprocal Rank Fusion (RRF) giữa hai bộ xếp hạng cosine và BM25.

    RRF_score(d) = Σ 1 / (k + rank_i(d))
    Trong đó rank_i(d) là thứ hạng (bắt đầu từ 1) của document d trong hệ thống i.
    """

    # Xác định xem dữ liệu có nhiều truy vấn hay chỉ một truy vấn
    has_query = "query_id" in cos_df.columns and "query_id" in bm_df.columns

    results = []

    if has_query:
        # Xử lý từng truy vấn riêng biệt
        query_ids = set(cos_df["query_id"].unique()).union(bm_df["query_id"].unique())
        for qid in query_ids:
            cos_q = cos_df[cos_df["query_id"] == qid].copy()
            bm_q = bm_df[bm_df["query_id"] == qid].copy()

            cos_q = cos_q.sort_values(by="cosine_score", ascending=False)  # type: ignore[arg-type]
            bm_q = bm_q.sort_values(by="bm25_score", ascending=False)  # type: ignore[arg-type]

            cos_q["rank_cos"] = range(1, len(cos_q) + 1)
            bm_q["rank_bm"] = range(1, len(bm_q) + 1)

            merged = cos_q[["chunk_id", "rank_cos"]].merge(
                bm_q[["chunk_id", "rank_bm"]], on="chunk_id", how="outer"
            )

            # Gán hạng lớn (len + k) cho những document không xuất hiện ở hệ thống kia
            merged["rank_cos"].fillna(len(cos_q) + k, inplace=True)
            merged["rank_bm"].fillna(len(bm_q) + k, inplace=True)

            merged["rrf_score"] = 1 / (k + merged["rank_cos"]) + 1 / (k + merged["rank_bm"])

            # Thêm lại thông tin gốc (query_text, chunk_text, v.v.)
            base_info = pd.concat([cos_q, bm_q]).drop_duplicates("chunk_id")
            merged = merged.merge(base_info, on="chunk_id", how="left")
            merged["query_id"] = qid

            results.append(merged)

        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.sort_values(by=["query_id", "rrf_score"], ascending=[True, False])

    else:
        # Trường hợp chỉ một truy vấn
        cos_ranked = cos_df.sort_values(by="cosine_score", ascending=False).copy()  # type: ignore[arg-type]
        bm_ranked = bm_df.sort_values(by="bm25_score", ascending=False).copy()  # type: ignore[arg-type]

        cos_ranked["rank_cos"] = range(1, len(cos_ranked) + 1)
        bm_ranked["rank_bm"] = range(1, len(bm_ranked) + 1)

        merged = cos_ranked[["chunk_id", "rank_cos"]].merge(
            bm_ranked[["chunk_id", "rank_bm"]], on="chunk_id", how="outer"
        )

        merged["rank_cos"].fillna(len(cos_ranked) + k, inplace=True)
        merged["rank_bm"].fillna(len(bm_ranked) + k, inplace=True)

        merged["rrf_score"] = 1 / (k + merged["rank_cos"]) + 1 / (k + merged["rank_bm"])

        base_info = pd.concat([cos_df, bm_df]).drop_duplicates("chunk_id")
        final_df = merged.merge(base_info, on="chunk_id", how="left")
        final_df = final_df.sort_values(by=["query_id", "rrf_score"], ascending=[True, False])

    # Lưu kết quả
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_dir / "ranked_rrf.tsv", sep="\t", index=False)
    print(f"Đã lưu xếp hạng RRF vào {out_dir / 'ranked_rrf.tsv'}")


def main():
    parser = argparse.ArgumentParser(description="Combine ranking results from cosine & BM25 in multiple ways.")
    parser.add_argument("--cosine", help="Path to *_ranked_cosine.tsv", default=None)
    parser.add_argument("--bm25", help="Path to *_ranked_bm25.tsv", default=None)
    parser.add_argument("--output_dir", default="./combined_results")
    parser.add_argument("--rrf_k", type=int, default=60, help="k tham số cho Reciprocal Rank Fusion (số nguyên dương, mặc định 60)")
    args = parser.parse_args()

    # --- interactive fallback ---
    if args.cosine is None or args.bm25 is None:
        print("\n--- Interactive mode: provide missing paths ---")
        while args.cosine is None:
            p = input("Path to *_ranked_cosine.tsv: ").strip()
            if Path(p).is_file():
                args.cosine = p
            else:
                print("File not found, try again.")
        while args.bm25 is None:
            p = input("Path to *_ranked_bm25.tsv: ").strip()
            if Path(p).is_file():
                args.bm25 = p
            else:
                print("File not found, try again.")

        # Cho phép người dùng thay đổi output dir và tham số k của RRF
        od = input(f"Output dir (default {args.output_dir}): ").strip()
        if od:
            args.output_dir = od
        k_in = input(f"k cho RRF (mặc định {args.rrf_k}): ").strip()
        if k_in:
            try:
                args.rrf_k = int(k_in)
            except ValueError:
                pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cos_df = load_df(Path(args.cosine))
    bm_df = load_df(Path(args.bm25))

    # Reciprocal Rank Fusion
    reciprocal_rank_fusion(cos_df, bm_df, args.rrf_k, out_dir)
    print("Reciprocal Rank Fusion hoàn tất. Kết quả lưu tại", out_dir)


if __name__ == "__main__":
    main() 