
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
    merged = merged.sort_values("hybrid_score", ascending=False)
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
    merged = merged.sort_values("lr_score", ascending=False)
    merged.to_csv(out_dir / "ranked_lr.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Combine ranking results from cosine & BM25 in multiple ways.")
    parser.add_argument("--cosine", help="Path to *_ranked_cosine.tsv", default=None)
    parser.add_argument("--bm25", help="Path to *_ranked_bm25.tsv", default=None)
    parser.add_argument("--output_dir", default="./combined_results")
    parser.add_argument("--hybrid_weight", type=float, default=0.6, help="Weight for cosine in hybrid score (0-1)")
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

        # Allow user to change output dir and weight
        od = input(f"Output dir (default {args.output_dir}): ").strip()
        if od:
            args.output_dir = od
        wt = input(f"Hybrid weight for cosine (0-1, default {args.hybrid_weight}): ").strip()
        if wt:
            try:
                args.hybrid_weight = float(wt)
            except ValueError:
                pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cos_df = load_df(Path(args.cosine))
    bm_df = load_df(Path(args.bm25))

    # Method 1
    build_union_intersection(cos_df, bm_df, out_dir)

    # Method 2
    hybrid_score(cos_df, bm_df, args.hybrid_weight, out_dir)

    # Method 3
    logistic_rank(cos_df, bm_df, out_dir)
    print("Combination finished. Results in", out_dir)


if __name__ == "__main__":
    main() 