from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATCH_ORDER = [
    "target",
    "all_image",
    "last_token",
    "distractor",
    "matched_distractor",
]

METRICS = {
    "recovery": {
        "column": "recovery",
        "ylabel": "Recovery",
        "ylim": None,
        "reference": "recovery",
    },
    "p_cf": {
        "column": "patched_candidate_p_cf_answer",
        "ylabel": "P(CF answer)",
        "ylim": (0.0, 1.0),
        "reference": "probability",
    },
}


def _as_paths(paths: Sequence[str | Path]) -> list[Path]:
    out = [Path(p) for p in paths]
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV file(s): " + ", ".join(missing))
    return out


def load_patching_results(
    csv_paths: Sequence[str | Path],
    *,
    verify_duplicate_consistency: bool = True,
) -> pd.DataFrame:
    """Load one or more patching CSVs and safely merge overlapping runs.

    Duplicate rows are identified by sample/scope/layer/patch type/dilation.
    This is useful for combining, for example, dense (10--18) and late (18--27)
    sweeps. If the same intervention appears in multiple files, measured values
    must agree before one copy is kept.
    """
    paths = _as_paths(csv_paths)
    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df = df.copy()
        df["_source_csv"] = path.name
        df["_source_order"] = i
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, sort=False)

    required = {
        "sample_id",
        "scope",
        "layer",
        "patch_type",
        "dilation",
        "recovery",
        "patched_candidate_p_cf_answer",
        "queried_attribute",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    key = ["sample_id", "scope", "layer", "patch_type", "dilation"]
    dup = df[df.duplicated(key, keep=False)].copy()

    if verify_duplicate_consistency and not dup.empty:
        numeric_to_check = [
            c
            for c in [
                "recovery",
                "patched_candidate_p_cf_answer",
                "patched_gap",
                "n_patch_tokens",
            ]
            if c in df.columns
        ]
        problems = []
        for keys, group in dup.groupby(key, dropna=False, sort=False):
            for col in numeric_to_check:
                vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                if len(finite) > 1 and not np.allclose(finite, finite[0], rtol=1e-6, atol=1e-8):
                    problems.append((keys, col, finite.tolist()))
        if problems:
            preview = problems[:5]
            raise ValueError(
                "Overlapping CSVs contain inconsistent duplicate interventions. "
                f"Examples: {preview}"
            )

    df = (
        df.sort_values("_source_order")
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )
    return df


def filter_results(
    df: pd.DataFrame,
    *,
    patch_types: Sequence[str] | None = None,
    dilation: int | None = 0,
    queried_attribute: str | None = None,
    include_lm_input: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    if patch_types is not None:
        out = out[out["patch_type"].isin(patch_types)]

    if queried_attribute is not None:
        out = out[out["queried_attribute"] == queried_attribute]

    # Dilation is meaningful for spatial regions. last_token/all_image usually use -1.
    if dilation is not None:
        spatial = out["patch_type"].isin({"target", "distractor", "matched_distractor"})
        out = out[(~spatial) | (out["dilation"] == dilation)]

    if include_lm_input:
        out = out[out["scope"].isin(["lm_input", "resid_post"])]
    else:
        out = out[out["scope"] == "resid_post"]

    return out.copy()


def _mean_sem(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, np.nan, np.nan
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - sem, mean + sem


def _mean_std(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, np.nan, np.nan
    std = float(np.std(values, ddof=1))
    return mean, mean - std, mean + std


def _mean_ci95(values: np.ndarray, *, seed: int = 0, n_boot: int = 5000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi)


def summarize_metric(
    df: pd.DataFrame,
    metric_col: str,
    *,
    error: str | None = "sem",
    seed: int = 0,
    n_boot: int = 5000,
) -> pd.DataFrame:
    """Aggregate across samples for each scope/layer/intervention."""
    rows = []
    group_cols = ["scope", "layer", "patch_type"]

    for keys, group in df.groupby(group_cols, sort=True):
        values = pd.to_numeric(group[metric_col], errors="coerce").to_numpy(dtype=float)
        n = int(np.isfinite(values).sum())

        if error is None or error == "none":
            mean = float(np.nanmean(values)) if n else np.nan
            lo = hi = np.nan
        elif error == "sem":
            mean, lo, hi = _mean_sem(values)
        elif error == "std":
            mean, lo, hi = _mean_std(values)
        elif error == "ci95":
            mean, lo, hi = _mean_ci95(values, seed=seed, n_boot=n_boot)
        else:
            raise ValueError("error must be one of: none, sem, std, ci95")

        rows.append(
            {
                "scope": keys[0],
                "layer": int(keys[1]),
                "patch_type": keys[2],
                "mean": mean,
                "lower": lo,
                "upper": hi,
                "n": n,
            }
        )

    return pd.DataFrame(rows)


def _patch_order(present: Iterable[str], requested: Sequence[str] | None = None) -> list[str]:
    present = list(dict.fromkeys(present))
    if requested is not None:
        return [p for p in requested if p in present]
    ordered = [p for p in DEFAULT_PATCH_ORDER if p in present]
    ordered += [p for p in present if p not in ordered]
    return ordered


def _x_values(summary: pd.DataFrame, include_lm_input: bool) -> np.ndarray:
    x = summary["layer"].to_numpy(dtype=float)
    if include_lm_input:
        x = np.where(summary["scope"].to_numpy() == "lm_input", -1.0, x)
    return x


def _format_layer_axis(ax: plt.Axes, summary: pd.DataFrame, include_lm_input: bool) -> None:
    layers = sorted(
        int(v)
        for v in summary.loc[summary["scope"] == "resid_post", "layer"].dropna().unique()
    )
    if not layers:
        return

    # Avoid an unreadable tick forest for full sweeps.
    if len(layers) <= 15:
        shown = layers
    else:
        step = max(1, math.ceil(len(layers) / 14))
        shown = layers[::step]
        if layers[-1] not in shown:
            shown.append(layers[-1])

    ticks = shown.copy()
    labels = [str(v) for v in shown]
    if include_lm_input and (summary["scope"] == "lm_input").any():
        ticks = [-1] + ticks
        labels = ["Input"] + labels
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Decoder layer")


def _add_metric_reference_lines(ax: plt.Axes, metric_name: str, df: pd.DataFrame) -> None:
    if metric_name == "recovery":
        ax.axhline(0.0, linewidth=1.0, linestyle=":", alpha=0.7)
        ax.axhline(1.0, linewidth=1.0, linestyle=":", alpha=0.7)
        return

    if metric_name == "p_cf":
        # Endpoint probabilities are sample-level constants repeated across rows.
        sample_cols = [
            "sample_id",
            "clean_candidate_p_cf_answer",
            "cf_candidate_p_cf_answer",
        ]
        if all(c in df.columns for c in sample_cols):
            ep = df[sample_cols].drop_duplicates("sample_id")
            clean_ref = pd.to_numeric(ep["clean_candidate_p_cf_answer"], errors="coerce").mean()
            cf_ref = pd.to_numeric(ep["cf_candidate_p_cf_answer"], errors="coerce").mean()
            if np.isfinite(clean_ref):
                ax.axhline(clean_ref, linewidth=1.0, linestyle=":", alpha=0.7, label="Clean endpoint")
            if np.isfinite(cf_ref):
                ax.axhline(cf_ref, linewidth=1.0, linestyle="--", alpha=0.7, label="CF endpoint")


def draw_metric_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    patch_types: Sequence[str] | None = None,
    dilation: int | None = 0,
    queried_attribute: str | None = None,
    include_lm_input: bool = False,
    error: str | None = "sem",
    seed: int = 0,
    n_boot: int = 5000,
    title: str | None = None,
    show_reference_lines: bool = True,
    individual_samples: bool = False,
    show_legend: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw one metric on an existing matplotlib Axes.

    Returns (filtered_rows, summary). The summary mean is across samples for each
    scope/layer/patch_type after any queried-attribute filter is applied.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {list(METRICS)}")

    spec = METRICS[metric]
    filtered = filter_results(
        df,
        patch_types=patch_types,
        dilation=dilation,
        queried_attribute=queried_attribute,
        include_lm_input=include_lm_input,
    )
    if filtered.empty:
        raise ValueError("No rows remain after filtering")

    summary = summarize_metric(
        filtered,
        spec["column"],
        error=error,
        seed=seed,
        n_boot=n_boot,
    )
    present_order = _patch_order(summary["patch_type"].unique(), patch_types)

    for patch_type in present_order:
        s = summary[summary["patch_type"] == patch_type].copy()
        x = _x_values(s, include_lm_input)
        order = np.argsort(x)
        x = x[order]
        means = s["mean"].to_numpy(dtype=float)[order]
        lower = s["lower"].to_numpy(dtype=float)[order]
        upper = s["upper"].to_numpy(dtype=float)[order]

        line = ax.plot(x, means, marker="o", linewidth=2, label=patch_type)[0]

        if error not in (None, "none") and np.isfinite(lower).any() and np.isfinite(upper).any():
            ax.fill_between(
                x, lower, upper, alpha=0.15, color=line.get_color(), linewidth=0
            )

        if individual_samples:
            raw = filtered[filtered["patch_type"] == patch_type]
            for _, g in raw.groupby("sample_id", sort=False):
                gx = np.where(
                    g["scope"].to_numpy() == "lm_input",
                    -1.0,
                    g["layer"].to_numpy(dtype=float),
                )
                gy = pd.to_numeric(g[spec["column"]], errors="coerce").to_numpy(dtype=float)
                ix = np.argsort(gx)
                ax.plot(gx[ix], gy[ix], linewidth=0.8, alpha=0.2, color=line.get_color())

    if show_reference_lines:
        _add_metric_reference_lines(ax, metric, filtered)

    _format_layer_axis(ax, summary, include_lm_input)
    ax.set_ylabel(spec["ylabel"])
    if spec["ylim"] is not None:
        ax.set_ylim(*spec["ylim"])

    if title is None:
        attr_text = f" — {queried_attribute}" if queried_attribute else ""
        title = (
            f"Counterfactual recovery vs layer{attr_text}"
            if metric == "recovery"
            else f"Counterfactual-answer probability vs layer{attr_text}"
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab)
                unique.append((h, lab))
        if unique:
            ax.legend([h for h, _ in unique], [lab for _, lab in unique], frameon=False)

    return filtered, summary


def plot_metric_vs_layer(
    df: pd.DataFrame,
    *,
    metric: str,
    out_path: str | Path,
    patch_types: Sequence[str] | None = None,
    dilation: int | None = 0,
    queried_attribute: str | None = None,
    include_lm_input: bool = False,
    error: str | None = "sem",
    seed: int = 0,
    n_boot: int = 5000,
    title: str | None = None,
    show_reference_lines: bool = True,
    individual_samples: bool = False,
) -> Path:
    """Save one metric-vs-layer figure."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    draw_metric_on_ax(
        ax,
        df,
        metric=metric,
        patch_types=patch_types,
        dilation=dilation,
        queried_attribute=queried_attribute,
        include_lm_input=include_lm_input,
        error=error,
        seed=seed,
        n_boot=n_boot,
        title=title,
        show_reference_lines=show_reference_lines,
        individual_samples=individual_samples,
        show_legend=True,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def show_patching_dashboard(
    csv_paths: Sequence[str | Path],
    *,
    patch_types: Sequence[str] | None = None,
    dilation: int | None = 0,
    include_lm_input: bool = False,
    error: str | None = "sem",
    individual_samples: bool = False,
    attributes: Sequence[str] = ("color", "size", "shape"),
    figsize: tuple[float, float] = (20.0, 8.5),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Display a 2x(1+attributes) dashboard in Jupyter.

    Row 1: P(CF answer): overall, then one panel per queried attribute.
    Row 2: Recovery:     overall, then one panel per queried attribute.

    Each curve is the sample mean for that layer/intervention. For an
    attribute-specific panel, the mean is only over samples whose
    queried_attribute equals that attribute.
    """
    df = load_patching_results(csv_paths)
    attrs = [a for a in attributes if a in set(df["queried_attribute"].dropna())]
    columns: list[tuple[str, str | None]] = [("Overall", None)] + [
        (a.capitalize(), a) for a in attrs
    ]

    fig, axes = plt.subplots(
        2, len(columns), figsize=figsize, squeeze=False, sharex=True, sharey="row"
    )

    metrics = [("p_cf", "P(CF answer)"), ("recovery", "Recovery")]
    legend_handles = None
    legend_labels = None

    for row, (metric, row_name) in enumerate(metrics):
        for col, (label, attr) in enumerate(columns):
            ax = axes[row, col]
            draw_metric_on_ax(
                ax,
                df,
                metric=metric,
                patch_types=patch_types,
                dilation=dilation,
                queried_attribute=attr,
                include_lm_input=include_lm_input,
                error=error,
                individual_samples=individual_samples,
                title=f"{row_name} — {label}",
                show_reference_lines=True,
                show_legend=False,
            )
            if col > 0:
                ax.set_ylabel("")
            if legend_handles is None:
                h, l = ax.get_legend_handles_labels()
                seen = set()
                pairs = [(hh, ll) for hh, ll in zip(h, l) if not (ll in seen or seen.add(ll))]
                legend_handles = [hh for hh, _ in pairs]
                legend_labels = [ll for _, ll in pairs]

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 7),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    plt.show()
    return fig, axes



def draw_recovery_distribution_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    patch_type: str,
    dilation: int | None = 0,
    queried_attribute: str | None = None,
    include_lm_input: bool = False,
    kind: str = "box",
    show_points: bool = False,
    title: str | None = None,
) -> pd.DataFrame:
    """Draw the sample-level recovery distribution across layers for one intervention.

    Unlike the mean/CI curves, this plot shows the empirical distribution of
    per-sample recovery values at each layer. `kind="box"` shows median/IQR and
    whiskers; `kind="violin"` shows the full empirical density shape.
    """
    filtered = filter_results(
        df,
        patch_types=[patch_type],
        dilation=dilation,
        queried_attribute=queried_attribute,
        include_lm_input=include_lm_input,
    )
    filtered = filtered.copy()
    filtered["recovery"] = pd.to_numeric(filtered["recovery"], errors="coerce")
    filtered = filtered[np.isfinite(filtered["recovery"])]
    if filtered.empty:
        raise ValueError(
            f"No finite recovery rows remain for patch_type={patch_type!r}, "
            f"queried_attribute={queried_attribute!r}"
        )

    filtered["_plot_x"] = np.where(
        filtered["scope"].to_numpy() == "lm_input",
        -1.0,
        pd.to_numeric(filtered["layer"], errors="coerce").to_numpy(dtype=float),
    )

    positions = sorted(float(x) for x in filtered["_plot_x"].dropna().unique())
    values_by_x = [
        filtered.loc[filtered["_plot_x"] == x, "recovery"].to_numpy(dtype=float)
        for x in positions
    ]

    if kind == "box":
        ax.boxplot(
            values_by_x,
            positions=positions,
            widths=0.55,
            showfliers=False,
            manage_ticks=False,
        )
    elif kind == "violin":
        # A violin needs at least two non-identical samples. For tiny sanity
        # panels, fall back to a box plot rather than failing.
        valid_for_violin = all(
            len(v) >= 2 and np.nanmax(v) > np.nanmin(v) for v in values_by_x
        )
        if valid_for_violin:
            ax.violinplot(
                values_by_x,
                positions=positions,
                widths=0.8,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )
        else:
            ax.boxplot(
                values_by_x,
                positions=positions,
                widths=0.55,
                showfliers=False,
                manage_ticks=False,
            )
    else:
        raise ValueError("kind must be 'box' or 'violin'")

    if show_points:
        # Deterministic horizontal jitter only for visibility; y values are untouched.
        rng = np.random.default_rng(0)
        for x, vals in zip(positions, values_by_x):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=8, alpha=0.22)

    ax.axhline(0.0, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.axhline(1.0, linewidth=1.0, linestyle=":", alpha=0.7)

    # Use the same readable layer tick policy as the mean plots.
    fake_summary = filtered[["scope", "layer"]].drop_duplicates().copy()
    _format_layer_axis(ax, fake_summary, include_lm_input)
    ax.set_ylabel("Per-sample recovery")
    ax.grid(True, axis="y", alpha=0.2)

    if title is None:
        label = "Overall" if queried_attribute is None else queried_attribute.capitalize()
        title = f"{patch_type} recovery distribution — {label}"
    ax.set_title(title)

    return filtered


def show_recovery_distribution_dashboard(
    csv_paths: Sequence[str | Path],
    *,
    patch_type: str = "target",
    dilation: int | None = 0,
    include_lm_input: bool = False,
    kind: str = "box",
    show_points: bool = False,
    attributes: Sequence[str] = ("color", "size", "shape"),
    figsize: tuple[float, float] = (20.0, 4.8),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Display sample-level recovery distributions: Overall | Color | Size | Shape.

    This complements the mean+CI recovery curve. Each box/violin at a layer is
    formed from the *individual sample recovery values* at that layer, so it
    directly shows heterogeneity rather than uncertainty of the mean.
    """
    df = load_patching_results(csv_paths)
    attrs = [a for a in attributes if a in set(df["queried_attribute"].dropna())]
    columns: list[tuple[str, str | None]] = [("Overall", None)] + [
        (a.capitalize(), a) for a in attrs
    ]

    fig, axes = plt.subplots(
        1, len(columns), figsize=figsize, squeeze=False, sharey=True
    )

    for col, (label, attr) in enumerate(columns):
        ax = axes[0, col]
        draw_recovery_distribution_on_ax(
            ax,
            df,
            patch_type=patch_type,
            dilation=dilation,
            queried_attribute=attr,
            include_lm_input=include_lm_input,
            kind=kind,
            show_points=show_points,
            title=f"{patch_type} — {label}",
        )
        if col > 0:
            ax.set_ylabel("")

    fig.suptitle(f"Per-sample recovery distribution: {patch_type}", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    plt.show()
    return fig, axes

def make_standard_plots(
    csv_paths: Sequence[str | Path],
    *,
    out_dir: str | Path,
    prefix: str,
    patch_types: Sequence[str] | None = None,
    dilation: int | None = 0,
    include_lm_input: bool = False,
    error: str | None = "sem",
    by_attribute: bool = False,
    individual_samples: bool = False,
) -> list[Path]:
    """Load one/many CSVs and make the standard sanity/full-experiment figures."""
    df = load_patching_results(csv_paths)
    out_dir = Path(out_dir)
    outputs: list[Path] = []

    for metric in ["recovery", "p_cf"]:
        outputs.append(
            plot_metric_vs_layer(
                df,
                metric=metric,
                out_path=out_dir / f"{prefix}_{metric}.png",
                patch_types=patch_types,
                dilation=dilation,
                include_lm_input=include_lm_input,
                error=error,
                individual_samples=individual_samples,
            )
        )

    if by_attribute:
        attrs = [a for a in ["size", "color", "shape"] if a in set(df["queried_attribute"])]
        attrs += [a for a in sorted(df["queried_attribute"].dropna().unique()) if a not in attrs]
        for attr in attrs:
            for metric in ["recovery", "p_cf"]:
                outputs.append(
                    plot_metric_vs_layer(
                        df,
                        metric=metric,
                        out_path=out_dir / f"{prefix}_{attr}_{metric}.png",
                        patch_types=patch_types,
                        dilation=dilation,
                        queried_attribute=attr,
                        include_lm_input=include_lm_input,
                        error=error,
                        individual_samples=individual_samples,
                    )
                )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot activation-patching CSVs from one run or merged layer sweeps."
    )
    parser.add_argument("--csv", nargs="+", required=True, help="One or more patching CSV files")
    parser.add_argument("--out_dir", default="figures", help="Directory for PNG figures")
    parser.add_argument("--prefix", default="patching", help="Output filename prefix")
    parser.add_argument(
        "--patch_types",
        nargs="*",
        default=None,
        help="Optional subset/order, e.g. target all_image last_token distractor",
    )
    parser.add_argument("--dilation", type=int, default=0)
    parser.add_argument("--include_lm_input", action="store_true")
    parser.add_argument("--by_attribute", action="store_true")
    parser.add_argument("--individual_samples", action="store_true")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Display a 2x4 Jupyter dashboard: P(CF) overall/color/size/shape, then recovery.",
    )
    parser.add_argument(
        "--dashboard_path",
        default=None,
        help="Optional path to also save the dashboard PNG.",
    )
    parser.add_argument(
        "--recovery_distribution",
        action="store_true",
        help="Display/save per-sample recovery-distribution dashboards.",
    )
    parser.add_argument(
        "--distribution_patch_types",
        nargs="*",
        default=None,
        help="Interventions to use for distribution dashboards. Defaults to --patch_types or all present types.",
    )
    parser.add_argument(
        "--distribution_kind",
        choices=["box", "violin"],
        default="box",
        help="How to visualize the sample-level recovery distribution.",
    )
    parser.add_argument(
        "--distribution_points",
        action="store_true",
        help="Overlay individual sample points with small horizontal jitter.",
    )
    parser.add_argument(
        "--error",
        choices=["none", "sem", "std", "ci95"],
        default="sem",
        help="Uncertainty band across samples",
    )
    args = parser.parse_args()

    outputs = make_standard_plots(
        args.csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        patch_types=args.patch_types,
        dilation=args.dilation,
        include_lm_input=args.include_lm_input,
        error=args.error,
        by_attribute=args.by_attribute,
        individual_samples=args.individual_samples,
    )
    for path in outputs:
        print(path)

    if args.dashboard:
        dashboard_path = args.dashboard_path
        if dashboard_path is None:
            dashboard_path = str(Path(args.out_dir) / f"{args.prefix}_dashboard.png")
        show_patching_dashboard(
            args.csv,
            patch_types=args.patch_types,
            dilation=args.dilation,
            include_lm_input=args.include_lm_input,
            error=args.error,
            individual_samples=args.individual_samples,
            attributes=("color", "size", "shape"),
            save_path=dashboard_path,
        )
        print(dashboard_path)

    if args.recovery_distribution:
        dist_df = load_patching_results(args.csv)
        requested = args.distribution_patch_types
        if requested is None or len(requested) == 0:
            requested = args.patch_types
        dist_patch_types = _patch_order(dist_df["patch_type"].dropna().unique(), requested)

        for patch_type in dist_patch_types:
            dist_path = Path(args.out_dir) / f"{args.prefix}_recovery_distribution_{patch_type}.png"
            show_recovery_distribution_dashboard(
                args.csv,
                patch_type=patch_type,
                dilation=args.dilation,
                include_lm_input=args.include_lm_input,
                kind=args.distribution_kind,
                show_points=args.distribution_points,
                attributes=("color", "size", "shape"),
                save_path=dist_path,
            )
            print(dist_path)


if __name__ == "__main__":
    main()
