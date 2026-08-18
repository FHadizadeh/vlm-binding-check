
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_small_target_dilations(full_csv: str | Path, dilation_csv: str | Path) -> pd.DataFrame:
    full = pd.read_csv(full_csv)
    dil = pd.read_csv(dilation_csv)

    base = (
        full[
            full["scope"].eq("lm_input")
            & full["patch_type"].eq("target")
        ][["sample_id", "n_patch_tokens"]]
        .drop_duplicates("sample_id")
    )
    small_ids = set(base.loc[base["n_patch_tokens"].eq(4), "sample_id"])

    d0 = full[
        full["sample_id"].isin(small_ids)
        & full["patch_type"].eq("target")
    ].copy()
    d0["dilation"] = 0

    d12 = dil[
        dil["sample_id"].isin(small_ids)
        & dil["patch_type"].eq("target")
        & dil["dilation"].isin([1, 2])
    ].copy()

    out = pd.concat([d0, d12], ignore_index=True, sort=False)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    x = df[df["scope"].eq("resid_post")].copy()
    rows = []
    for attr_name, q in [("overall", x)] + [
        (a, x[x["queried_attribute"].eq(a)]) for a in ["color", "shape"]
    ]:
        s = (
            q.groupby(["layer", "dilation"], as_index=False)
            .agg(
                mean_recovery=("recovery", "mean"),
                sd_recovery=("recovery", "std"),
                n=("recovery", "size"),
                cf_flip_rate=("patched_matches_cf_answer", "mean"),
            )
        )
        s["attribute"] = attr_name
        s["sem"] = s["sd_recovery"] / np.sqrt(s["n"])
        s["ci95"] = 1.96 * s["sem"]
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def paired_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    x = df[df["scope"].eq("resid_post")].copy()
    wide = x.pivot_table(
        index=["sample_id", "queried_attribute", "layer"],
        columns="dilation",
        values="recovery",
        aggfunc="first",
    ).reset_index()

    rows = []
    for attr_name, q in [("overall", wide)] + [
        (a, wide[wide["queried_attribute"].eq(a)]) for a in ["color", "shape"]
    ]:
        for layer, g in q.groupby("layer"):
            row = {"attribute": attr_name, "layer": int(layer), "n": len(g)}
            for a, b in [(1, 0), (2, 0), (2, 1)]:
                if a in g.columns and b in g.columns:
                    delta = g[a] - g[b]
                    row[f"mean_delta_d{a}_minus_d{b}"] = delta.mean()
                    row[f"fraction_positive_d{a}_minus_d{b}"] = (delta > 0).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def make_plots(summary: pd.DataFrame, out_dir: str | Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for attr in ["overall", "color", "shape"]:
        s = summary[summary["attribute"].eq(attr)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for dilation in [0, 1, 2]:
            q = s[s["dilation"].eq(dilation)]
            ax.plot(
                q["layer"],
                q["mean_recovery"],
                marker="o",
                markersize=3,
                label=f"dilation={dilation}",
            )
            ax.fill_between(
                q["layer"].to_numpy(),
                (q["mean_recovery"] - q["ci95"]).to_numpy(),
                (q["mean_recovery"] + q["ci95"]).to_numpy(),
                alpha=0.12,
            )
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean target recovery")
        ax.set_title(f"Small-target dilation — {attr}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{attr}_recovery.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_csv", required=True)
    parser.add_argument("--dilation_csv", required=True)
    parser.add_argument("--out_dir", default="figures/dilation")
    parser.add_argument("--prefix", default="small_target_dilation")
    args = parser.parse_args()

    df = load_small_target_dilations(args.full_csv, args.dilation_csv)
    summary = summarize(df)
    paired = paired_delta_summary(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{args.prefix}_summary.csv", index=False)
    paired.to_csv(out_dir / f"{args.prefix}_paired_deltas.csv", index=False)
    make_plots(summary, out_dir, args.prefix)

    print(f"small-target samples: {df['sample_id'].nunique()}")
    print(f"saved: {out_dir / f'{args.prefix}_summary.csv'}")
    print(f"saved: {out_dir / f'{args.prefix}_paired_deltas.csv'}")


if __name__ == "__main__":
    main()
