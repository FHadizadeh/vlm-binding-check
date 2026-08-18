
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Markdown, HTML
except Exception:
    display = None
    Markdown = None
    HTML = None


PATCHES = [
    "target",
    "all_image",
    "last_token",
    "distractor",
    "matched_distractor",
]

PATCH_LABELS = {
    "target": "Target",
    "all_image": "All image",
    "last_token": "Last token",
    "distractor": "Distractor",
    "matched_distractor": "Matched distractor",
}

# Explicit report colors requested by the user.
ROW_COLORS = {
    "visual": "#DCEEFF",       # strong visual causal effect, last token neutral
    "transition": "#FFF4CC",   # mixed / transition regime
    "crossover": "#FFD7A3",    # first Last - Target >= 0
    "late": "#DDF2D8",         # last-token-dominant regime
}


def _show(obj):
    if display is not None:
        display(obj)
    else:
        print(obj)


def _md(text: str):
    if Markdown is not None:
        _show(Markdown(text))
    else:
        print(text)


def load_results(csv_paths):
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    frames = []
    for p in csv_paths:
        p = Path(p)
        x = pd.read_csv(p)
        x["_source_csv"] = str(p)
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)

    # Normalize boolean-looking columns when CSV readers return strings.
    for c in [
        "patched_matches_cf_answer",
        "matched_control_exact_shape",
        "matched_control_exact_token_count",
        "clean_generated_correct",
        "cf_generated_correct",
    ]:
        if c in df.columns and df[c].dtype == object:
            df[c] = (
                df[c].astype(str).str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .where(df[c].notna())
            )
    return df


def expected_rows_per_sample(df):
    selected_layers = sorted(
        int(x) for x in df.loc[df["scope"].eq("resid_post"), "layer"].dropna().unique()
    )
    n_layers = len(selected_layers)
    input_patch_types = sorted(df.loc[df["scope"].eq("lm_input"), "patch_type"].unique())
    resid_patch_types = sorted(df.loc[df["scope"].eq("resid_post"), "patch_type"].unique())
    return len(input_patch_types) + n_layers * len(resid_patch_types)


def integrity_summary(df):
    samples = df["sample_id"].drop_duplicates()
    attrs = (
        df[["sample_id", "queried_attribute"]]
        .drop_duplicates()["queried_attribute"]
        .value_counts()
        .sort_index()
    )

    clean = (
        df[["sample_id", "clean_generated_correct"]]
        .drop_duplicates()["clean_generated_correct"]
        .mean()
        if "clean_generated_correct" in df.columns else np.nan
    )
    cf = (
        df[["sample_id", "cf_generated_correct"]]
        .drop_duplicates()["cf_generated_correct"]
        .mean()
        if "cf_generated_correct" in df.columns else np.nan
    )

    rows_per_sample = df.groupby("sample_id").size()
    expected = expected_rows_per_sample(df)

    return {
        "n_rows": len(df),
        "n_samples": len(samples),
        "attribute_counts": attrs.to_dict(),
        "clean_acc": clean,
        "cf_acc": cf,
        "expected_rows_per_sample": expected,
        "rows_per_sample_min": int(rows_per_sample.min()),
        "rows_per_sample_max": int(rows_per_sample.max()),
        "complete": bool((rows_per_sample == expected).all()),
    }


def _layer_order(df):
    layers = sorted(
        int(v) for v in df.loc[df["scope"].eq("resid_post"), "layer"].dropna().unique()
    )
    has_input = df["scope"].eq("lm_input").any()
    return ([-1] if has_input else []) + layers


def aggregate_layerwise(df, attribute=None):
    x = df.copy()
    if attribute is not None:
        x = x[x["queried_attribute"].eq(attribute)]

    records = []
    for layer in _layer_order(x):
        if layer == -1:
            d = x[x["scope"].eq("lm_input")]
        else:
            d = x[x["scope"].eq("resid_post") & x["layer"].eq(layer)]

        row = {"Layer": "Input" if layer == -1 else str(layer), "_layer_num": layer}

        for patch in PATCHES:
            q = d[d["patch_type"].eq(patch)]
            row[(PATCH_LABELS[patch], "Recovery")] = (
                q["recovery"].mean() if len(q) else np.nan
            )
            row[(PATCH_LABELS[patch], "Flip rate")] = (
                q["patched_matches_cf_answer"].astype(float).mean()
                if len(q) else np.nan
            )

        target = row[("Target", "Recovery")]
        last = row[("Last token", "Recovery")]
        row[("Summary", "Last − Target")] = (
            last - target if pd.notna(last) and pd.notna(target) else np.nan
        )
        records.append(row)

    out = pd.DataFrame(records)
    out = out.set_index("Layer")
    if "_layer_num" in out.columns:
        out = out.drop(columns="_layer_num")
    out.columns = pd.MultiIndex.from_tuples(
        [c if isinstance(c, tuple) else ("", c) for c in out.columns]
    )
    return out


def first_crossover(table):
    delta = table[("Summary", "Last − Target")]
    numeric = pd.to_numeric(pd.Index(table.index).where(pd.Index(table.index) != "Input"), errors="coerce")
    candidates = []
    for label, layer_num, val in zip(table.index, numeric, delta):
        if pd.notna(layer_num) and pd.notna(val) and val >= 0:
            candidates.append((int(layer_num), label))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def classify_regime(row, crossover_label):
    label = row.name
    target = row.get(("Target", "Recovery"), np.nan)
    all_img = row.get(("All image", "Recovery"), np.nan)
    last = row.get(("Last token", "Recovery"), np.nan)

    if str(label) == str(crossover_label):
        return "crossover"

    # Input has no last-token intervention; treat it as visual-dominant if image effect is strong.
    if label == "Input":
        if pd.notna(all_img) and all_img >= 0.75:
            return "visual"
        return None

    if pd.notna(all_img) and pd.notna(last) and all_img >= 0.75 and last <= 0.10:
        return "visual"

    if pd.notna(target) and pd.notna(last) and last >= 0.50 and target <= 0.10:
        return "late"

    if pd.notna(target) and pd.notna(last):
        # The middle section is useful to see as a contiguous transition band.
        if (target < 0.75 and last < 0.50) or (target > 0.10 and last > 0.10):
            return "transition"

    return None


def style_layer_table(table, caption=None):
    crossover = first_crossover(table)

    def row_style(row):
        regime = classify_regime(row, crossover)
        color = ROW_COLORS.get(regime)
        if not color:
            return [""] * len(row)
        return [f"background-color: {color};"] * len(row)

    styler = (
        table.style
        .apply(row_style, axis=1)
        .format(
            {
                ("Target", "Recovery"): "{:.3f}",
                ("Target", "Flip rate"): "{:.1%}",
                ("All image", "Recovery"): "{:.3f}",
                ("All image", "Flip rate"): "{:.1%}",
                ("Last token", "Recovery"): "{:.3f}",
                ("Last token", "Flip rate"): "{:.1%}",
                ("Distractor", "Recovery"): "{:.3f}",
                ("Distractor", "Flip rate"): "{:.1%}",
                ("Matched distractor", "Recovery"): "{:.3f}",
                ("Matched distractor", "Flip rate"): "{:.1%}",
                ("Summary", "Last − Target"): "{:+.3f}",
            },
            na_rep="—",
        )
        .set_properties(**{
            "text-align": "center",
            "font-size": "11px",
            "padding": "5px 7px",
            "border-bottom": "1px solid #e6e6e6",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "600"),
                    ("border-bottom", "1px solid #aaa"),
                    ("padding", "6px 8px"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "14px"),
                    ("font-weight", "700"),
                    ("text-align", "left"),
                    ("padding", "6px 0 10px 0"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                ],
            },
        ])
        .set_sticky(axis="index")
    )
    if caption:
        styler = styler.set_caption(caption)
    return styler, crossover


def save_styled_table(styler, html_path):
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(styler.to_html(), encoding="utf-8")


def display_legend():
    html = f"""
    <div style="margin:8px 0 14px 0;font-size:13px">
      <b>Row highlighting:</b>
      <span style="background:{ROW_COLORS['visual']};padding:3px 7px;margin-left:8px">visual-dominant</span>
      <span style="background:{ROW_COLORS['transition']};padding:3px 7px;margin-left:8px">transition</span>
      <span style="background:{ROW_COLORS['crossover']};padding:3px 7px;margin-left:8px">first Last − Target ≥ 0</span>
      <span style="background:{ROW_COLORS['late']};padding:3px 7px;margin-left:8px">last-token-dominant</span>
    </div>
    """
    if HTML is not None:
        _show(HTML(html))
    else:
        print("Legend: visual-dominant / transition / crossover / last-token-dominant")



def _latex_cell(value, kind="recovery"):
    if pd.isna(value):
        return "--"
    if kind == "flip":
        return f"{100.0 * float(value):.1f}\\%"
    if kind == "delta":
        return f"{float(value):+.3f}"
    return f"{float(value):.3f}"


def save_clean_latex_table(table, tex_path, caption, label):
    """Export the same regime-colored table as clean LaTeX."""
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    crossover = first_crossover(table)

    headers = [
        ("Target", "Recovery"), ("Target", "Flip rate"),
        ("All image", "Recovery"), ("All image", "Flip rate"),
        ("Last token", "Recovery"), ("Last token", "Flip rate"),
        ("Distractor", "Recovery"), ("Distractor", "Flip rate"),
        ("Matched distractor", "Recovery"), ("Matched distractor", "Flip rate"),
        ("Summary", "Last − Target"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c cc cc cc cc cc c}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Target} & \multicolumn{2}{c}{All image} & "
        r"\multicolumn{2}{c}{Last token} & \multicolumn{2}{c}{Distractor} & "
        r"\multicolumn{2}{c}{Matched distractor} & $\Delta$ \\",
        r"Layer & Rec. & Flip & Rec. & Flip & Rec. & Flip & Rec. & Flip & Rec. & Flip & Last$-$Target \\",
        r"\midrule",
    ]

    hexmap = {
        "visual": "DCEEFF",
        "transition": "FFF4CC",
        "crossover": "FFD7A3",
        "late": "DDF2D8",
    }

    for idx, row in table.iterrows():
        regime = classify_regime(row, crossover)
        if regime in hexmap:
            lines.append(rf"\rowcolor[HTML]{{{hexmap[regime]}}}")

        vals = []
        for top, sub in headers:
            v = row[(top, sub)]
            if sub == "Flip rate":
                vals.append(_latex_cell(v, "flip"))
            elif top == "Summary":
                vals.append(_latex_cell(v, "delta"))
            else:
                vals.append(_latex_cell(v, "recovery"))

        lines.append(
            f"{idx} & " + " & ".join(vals) + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
        r"% Required packages:",
        r"% \usepackage{booktabs}",
        r"% \usepackage[table]{xcolor}",
        r"% \usepackage{graphicx}",
    ]

    tex_path.write_text("\n".join(lines), encoding="utf-8")

def save_plain_table(table, csv_path):
    x = table.copy()
    x.columns = [
        f"{a}__{b}".strip("_") for a, b in x.columns.to_flat_index()
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(csv_path)


def _save_and_show(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    _show(fig)
    plt.close(fig)


def plot_token_stratification(df, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = (
        df[df["scope"].eq("lm_input") & df["patch_type"].eq("target")]
        [["sample_id", "n_patch_tokens"]]
        .drop_duplicates("sample_id")
        .rename(columns={"n_patch_tokens": "base_patch_tokens"})
    )
    x = df.merge(base, on="sample_id", how="left")
    x = x[
        x["scope"].eq("resid_post")
        & x["patch_type"].eq("target")
        & x["queried_attribute"].isin(["color", "shape"])
        & x["base_patch_tokens"].isin([4, 36])
    ]

    summary = (
        x.groupby(["queried_attribute", "layer", "base_patch_tokens"], as_index=False)
        .agg(
            mean_recovery=("recovery", "mean"),
            sd=("recovery", "std"),
            n=("recovery", "size"),
        )
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["sem"]
    summary.to_csv(Path(out_dir) / f"{prefix}_target_recovery_4_vs_36_tokens.csv", index=False)

    for attr in ["color", "shape"]:
        s = summary[summary["queried_attribute"].eq(attr)]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for tok in [4, 36]:
            q = s[s["base_patch_tokens"].eq(tok)]
            ax.plot(q["layer"], q["mean_recovery"], marker="o", markersize=3, label=f"{tok} tokens (n={int(q['n'].iloc[0]) if len(q) else 0})")
            ax.fill_between(
                q["layer"].to_numpy(),
                (q["mean_recovery"] - q["ci95"]).to_numpy(),
                (q["mean_recovery"] + q["ci95"]).to_numpy(),
                alpha=0.12,
            )
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"Target recovery by patch size — {attr}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean recovery")
        ax.legend()
        ax.grid(alpha=0.18)
        _save_and_show(fig, Path(out_dir) / f"{prefix}_{attr}_target_recovery_4_vs_36.png")

    # Pooled color + shape view.
    pooled = (
        x.groupby(["layer", "base_patch_tokens"], as_index=False)
        .agg(
            mean_recovery=("recovery", "mean"),
            sd=("recovery", "std"),
            n=("recovery", "size"),
        )
    )
    pooled["sem"] = pooled["sd"] / np.sqrt(pooled["n"])
    pooled["ci95"] = 1.96 * pooled["sem"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for tok in [4, 36]:
        q = pooled[pooled["base_patch_tokens"].eq(tok)]
        ax.plot(q["layer"], q["mean_recovery"], marker="o", markersize=3, label=f"{tok} tokens")
        ax.fill_between(
            q["layer"].to_numpy(),
            (q["mean_recovery"] - q["ci95"]).to_numpy(),
            (q["mean_recovery"] + q["ci95"]).to_numpy(),
            alpha=0.12,
        )
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Target recovery by patch size — color + shape pooled")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean recovery")
    ax.legend()
    ax.grid(alpha=0.18)
    _save_and_show(fig, Path(out_dir) / f"{prefix}_pooled_target_recovery_4_vs_36.png")

    return summary, pooled


def strict_matched_analysis(df, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = df[
        df["scope"].eq("lm_input")
        & df["patch_type"].eq("matched_distractor")
    ].drop_duplicates("sample_id")

    strict_mask = (
        base["matched_control_exact_shape"].eq(True)
        & base["matched_control_exact_token_count"].eq(True)
        & base["matched_control_target_overlap_tokens"].fillna(np.inf).eq(0)
    )
    strict_ids = base.loc[strict_mask, "sample_id"].tolist()

    x = df[df["sample_id"].isin(strict_ids)].copy()
    rows = []
    for layer in _layer_order(x):
        if layer == -1:
            d = x[x["scope"].eq("lm_input")]
        else:
            d = x[x["scope"].eq("resid_post") & x["layer"].eq(layer)]

        for patch in ["target", "matched_distractor"]:
            q = d[d["patch_type"].eq(patch)]
            if not len(q):
                continue
            rows.append({
                "layer": layer,
                "patch_type": patch,
                "mean_recovery": q["recovery"].mean(),
                "flip_rate": q["patched_matches_cf_answer"].astype(float).mean(),
                "n": len(q),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(out_dir) / f"{prefix}_strict_matched_control.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for patch in ["target", "matched_distractor"]:
        q = summary[(summary["patch_type"].eq(patch)) & (summary["layer"] >= 0)]
        ax.plot(q["layer"], q["mean_recovery"], marker="o", markersize=3, label=PATCH_LABELS[patch])
    ax.axhline(0, linewidth=0.8)
    ax.set_title(f"Strict matched-control analysis (n={len(strict_ids)}/{base['sample_id'].nunique()})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean recovery")
    ax.legend()
    ax.grid(alpha=0.18)
    _save_and_show(fig, Path(out_dir) / f"{prefix}_strict_matched_control.png")

    strategy_counts = (
        base["matched_control_strategy"].value_counts().rename_axis("strategy").reset_index(name="n")
    )
    return strict_ids, summary, strategy_counts


def describe_token_stratification(summary):
    lines = []
    for attr in ["color", "shape"]:
        s = summary[summary["queried_attribute"].eq(attr)]
        if s.empty:
            continue
        for layer in [0, 12, 14]:
            q = s[s["layer"].eq(layer)]
            vals = dict(zip(q["base_patch_tokens"].astype(int), q["mean_recovery"]))
            if 4 in vals and 36 in vals:
                lines.append(
                    f"- **{attr}, L{layer}:** 4-token mask = `{vals[4]:.3f}`, "
                    f"36-token mask = `{vals[36]:.3f}` (difference `{vals[36]-vals[4]:+.3f}`)."
                )
    return "\n".join(lines)


def report_interpretation(df, overall_table, attr_tables, strict_ids, strict_summary, token_summary):
    cross_overall = first_crossover(overall_table)
    crosses = {a: first_crossover(t) for a, t in attr_tables.items()}

    def val(table, layer, patch, metric="Recovery"):
        label = str(layer)
        if label not in table.index:
            return np.nan
        return table.loc[label, (patch, metric)]

    early_t = val(overall_table, 0, "Target")
    early_ai = val(overall_table, 0, "All image")
    mid_t = val(overall_table, 13, "Target")
    mid_ai = val(overall_table, 13, "All image")
    l14_t = val(overall_table, 14, "Target")
    l16_t = val(overall_table, 16, "Target")
    l20_last = val(overall_table, 20, "Last token")
    l24_last = val(overall_table, 24, "Last token")
    l24_last_flip = val(overall_table, 24, "Last token", "Flip rate")

    strict_md = strict_summary[
        (strict_summary["patch_type"].eq("matched_distractor"))
        & (strict_summary["layer"] >= 0)
    ]
    max_strict = strict_md["mean_recovery"].abs().max() if len(strict_md) else np.nan
    max_strict_flip = strict_md["flip_rate"].max() if len(strict_md) else np.nan

    _md(f"""
## Interpretation

### Experiment 1 — Main causal activation patching

**Question.** Where is the clean/CF answer difference causally recoverable across the LM depth?

- At **L0**, target-region recovery is `{early_t:.3f}` while all-image recovery is `{early_ai:.3f}`.
- The visual-position effect remains strong through roughly the early/mid stack: at **L13**, target = `{mid_t:.3f}` and all-image = `{mid_ai:.3f}`.
- A sharp decline starts around **L14–L16**: target recovery changes from `{l14_t:.3f}` at L14 to `{l16_t:.3f}` at L16.
- Last-token patching is nearly neutral in early layers, then rises in later layers: L20 = `{l20_last:.3f}`, L24 = `{l24_last:.3f}` with CF flip rate `{l24_last_flip:.1%}`.
- The first layer where mean **Last-token − Target ≥ 0** is **L{cross_overall}**.

**Interpretation.** The counterfactual effect is strongly recoverable from visual token positions early, loses leverage in the middle/late stack, and becomes increasingly recoverable from the final prompt-token state later. This is evidence for a **shift in causal recoverability**, not by itself proof that a single representation literally moves from image tokens to the last token.

### Experiment 2 — Attribute-specific dynamics

The first mean crossover `Last-token − Target ≥ 0` occurs at:

- **Size:** L{crosses.get('size')}
- **Shape:** L{crosses.get('shape')}
- **Color:** L{crosses.get('color')}

This means the layerwise causal dynamics are not identical across queried attributes. The safe claim is that **size shows an earlier crossover in this dataset**, followed by shape and color. It should not yet be phrased as “size reasoning is completed earlier,” because the answer spaces and target-mask construction differ across attributes.

### Experiment 3 — Target-mask size: 4 vs 36 tokens

This analysis uses **color and shape only**, because size swaps use the union of clean/CF target boxes and therefore have 36 target tokens for every pair.

{describe_token_stratification(token_summary)}

The 36-token target region gives systematically higher recovery, especially for shape. Since 4-token vs 36-token also corresponds to small vs large objects here, this analysis does **not** identify mask size as the sole cause. The dilation experiment is the correct within-sample test: keep the same small-object examples and expand only the patched region.

### Experiment 4 — Strict matched-distractor control

A control is considered strict only if it has:

1. the **same grid shape** as the target mask,
2. the **same token count**,
3. **zero target overlap**.

`{len(strict_ids)}` samples satisfy all three conditions. Across these strict controls, the largest absolute layer-mean matched-distractor recovery is only `{max_strict:.4f}`, and the maximum CF flip rate is `{max_strict_flip:.1%}`.

**Interpretation.** The large target effect is not explained by merely patching the same number/shape of visual tokens at another object location. The effect is target-location-specific under this control.

### Experiment 5 — Dilation 1/2 (next causal-control experiment)

This is specifically a test of the **small-object / 4-token gap**.

- Use exactly the samples whose dilation-0 target mask has 4 tokens.
- Re-run the same target intervention with dilation 1 and 2.
- Compare each sample against its own dilation-0 result.

If recovery rises strongly with dilation, it supports the interpretation that the relevant causal representation extends outside the raw small-object bbox. If recovery stays low, the small-vs-large difference is less likely to be only a mask-coverage issue.
""")


def write_small_target_pair_subset(df, pairs_csv, out_csv):
    base = (
        df[
            df["scope"].eq("lm_input")
            & df["patch_type"].eq("target")
            & df["n_patch_tokens"].eq(4)
        ]["sample_id"]
        .drop_duplicates()
    )
    pairs = pd.read_csv(pairs_csv)
    sub = pairs[pairs["sample_id"].isin(base)].drop_duplicates("sample_id").copy()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return sub


def display_report(
    csv_paths,
    out_dir="figures/conjunctive_report",
    prefix="conjunctive_full",
    pairs_csv=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_paths)

    _md("# Conjunctive Binding — Activation Patching Report")

    integ = integrity_summary(df)
    _md(f"""
## 0. Run integrity

- Rows: **{integ['n_rows']:,}**
- Unique samples: **{integ['n_samples']}**
- Attribute counts: **{integ['attribute_counts']}**
- Rows/sample: **{integ['rows_per_sample_min']}–{integ['rows_per_sample_max']}**
- Expected rows/sample from this run configuration: **{integ['expected_rows_per_sample']}**
- Complete: **{integ['complete']}**
- Clean generated accuracy: **{integ['clean_acc']:.1%}**
- CF generated accuracy: **{integ['cf_acc']:.1%}**
""")

    _md("## 1. Overall layer-wise causal table")
    display_legend()
    overall = aggregate_layerwise(df)
    overall_styler, overall_cross = style_layer_table(
        overall,
        caption=f"Overall — first Last − Target crossover: L{first_crossover(overall)}",
    )
    _show(overall_styler)
    save_styled_table(overall_styler, out_dir / f"{prefix}_overall_layer_table.html")
    save_plain_table(overall, out_dir / f"{prefix}_overall_layer_table.csv")
    save_clean_latex_table(
        overall,
        out_dir / f"{prefix}_overall_layer_table.tex",
        caption=f"Overall layer-wise activation-patching results. First Last-token minus Target crossover: L{first_crossover(overall)}.",
        label=f"tab:{prefix}_overall_layerwise",
    )

    _md("""
**How to read the colors.** Blue rows are visual-dominant: the all-image intervention is still strongly causal while the last-token intervention is near-neutral. Yellow marks the transition region. Orange is the **first layer where mean Last-token recovery catches/exceeds Target recovery**. Green marks a late regime where last-token recovery is strong and target recovery is small.
""")

    _md("## 2. Attribute-specific layer-wise tables")
    attr_tables = {}
    for attr in ["color", "size", "shape"]:
        t = aggregate_layerwise(df, attribute=attr)
        attr_tables[attr] = t
        s, c = style_layer_table(
            t,
            caption=f"{attr.capitalize()} — first Last − Target crossover: L{first_crossover(t)}",
        )
        _md(f"### {attr.capitalize()}")
        display_legend()
        _show(s)
        save_styled_table(s, out_dir / f"{prefix}_{attr}_layer_table.html")
        save_plain_table(t, out_dir / f"{prefix}_{attr}_layer_table.csv")
        save_clean_latex_table(
            t,
            out_dir / f"{prefix}_{attr}_layer_table.tex",
            caption=f"{attr.capitalize()} queries: layer-wise activation-patching results. First Last-token minus Target crossover: L{first_crossover(t)}.",
            label=f"tab:{prefix}_{attr}_layerwise",
        )

    _md("## 3. Target recovery stratified by 4 vs 36 patched tokens")
    token_summary, pooled = plot_token_stratification(df, out_dir, prefix)
    _show(
        token_summary[
            token_summary["layer"].isin([0, 8, 12, 14, 16, 20])
        ].style.format({
            "mean_recovery": "{:.3f}",
            "sd": "{:.3f}",
            "sem": "{:.3f}",
            "ci95": "{:.3f}",
        }).set_caption("Selected layers — target recovery by attribute and patch size")
    )

    _md("## 4. Strict matched-distractor analysis")
    strict_ids, strict_summary, strategies = strict_matched_analysis(df, out_dir, prefix)

    _md(f"""
Strict definition: **same shape + same token count + zero target overlap**.

- Strict samples: **{len(strict_ids)}/{df['sample_id'].nunique()}**
""")
    _show(strategies.style.set_caption("Matched-control strategies used in the full run"))

    selected_strict = strict_summary[
        strict_summary["layer"].isin([-1, 0, 8, 12, 14, 15, 18, 20, 24, 27])
    ]
    _show(
        selected_strict.style.format({
            "mean_recovery": "{:.4f}",
            "flip_rate": "{:.1%}",
        }).set_caption("Strict-control selected layers")
    )

    report_interpretation(
        df,
        overall,
        attr_tables,
        strict_ids,
        strict_summary,
        token_summary,
    )

    if pairs_csv:
        small_out = out_dir / f"{prefix}_small_target_pairs.csv"
        sub = write_small_target_pair_subset(df, pairs_csv, small_out)
        _md(f"""
## 6. Dilation subset prepared

Saved **{len(sub)}** small-target pairs to:

`{small_out}`

Run dilation 1/2 with:

```python
import sys

!{{sys.executable}} src/run_activation_patching.py \\
    --data data/synth_v4/conjunctive_binding \\
    --pairs_csv {small_out} \\
    --out results/conjunctive_small_target_dilation12.csv \\
    --layers 0:28:1 \\
    --patch_types target matched_distractor \\
    --dilations 1 2
```
""")

    _md(f"""
## Saved report artifacts

Styled HTML tables, plain CSV tables, and analysis figures were saved under:

`{out_dir}`
""")

    return {
        "data": df,
        "overall_table": overall,
        "attribute_tables": attr_tables,
        "strict_ids": strict_ids,
        "strict_summary": strict_summary,
        "token_summary": token_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--out_dir", default="figures/conjunctive_report")
    parser.add_argument("--prefix", default="conjunctive_full")
    parser.add_argument("--pairs_csv", default=None)
    args = parser.parse_args()

    display_report(
        args.csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        pairs_csv=args.pairs_csv,
    )


if __name__ == "__main__":
    main()
