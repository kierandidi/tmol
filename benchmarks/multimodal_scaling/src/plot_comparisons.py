"""Plot score-term agreement and paired historical-to-current TMol speedups."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_results import TERM_GROUPS
from common import ROOT
from plot_results import MODALITIES, SERIES_STYLES, WORKLOADS, set_style

CURRENT_VERSION = "0.1.55"
HISTORICAL_VERSION = {
    "protein": "0.1.46",
    "protein_ligand": "0.1.46",
    # 0.1.46 predates the nucleic-acid representation used by this corpus.
    "protein_nucleic": "0.1.47",
}


def save(fig: mpl.figure.Figure, stem: str) -> None:
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    path = output / stem
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=600)


def paired_speedups(timing: pd.DataFrame) -> pd.DataFrame:
    """Return exact-dataset historical/current timing pairs."""
    selected = timing[
        (timing.engine == "tmol")
        & (timing.status == "ok")
        & timing.selected_for_plot.astype(bool)
        & timing.seconds_per_structure.notna()
    ].copy()
    historical = pd.concat(
        [
            selected[
                (selected.modality == modality)
                & (selected.engine_version.astype(str) == version)
            ]
            for modality, version in HISTORICAL_VERSION.items()
        ],
        ignore_index=True,
    )
    current = selected[selected.engine_version.astype(str) == CURRENT_VERSION]
    keys = ["protocol", "modality", "dataset_id", "device", "batch_size"]
    paired = historical.merge(
        current,
        on=keys,
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    if not np.allclose(
        paired.polymer_residues_historical,
        paired.polymer_residues_current,
        rtol=0,
        atol=0,
    ):
        raise ValueError("Historical/current pairs disagree on polymer length")
    paired["polymer_residues"] = paired.polymer_residues_current
    paired["speedup"] = (
        paired.seconds_per_structure_historical / paired.seconds_per_structure_current
    )
    return paired


def binned_summary(group: pd.DataFrame, maximum_bins: int = 12) -> pd.DataFrame:
    """Robust trend over linear residue bins without connecting every raw point."""
    group = group.sort_values("polymer_residues").copy()
    unique_lengths = group.polymer_residues.nunique()
    bins = min(maximum_bins, unique_lengths, max(1, len(group) // 6))
    if bins <= 1:
        return pd.DataFrame(
            {
                "polymer_residues": [group.polymer_residues.median()],
                "median": [group.speedup.median()],
                "q25": [group.speedup.quantile(0.25)],
                "q75": [group.speedup.quantile(0.75)],
            }
        )
    edges = np.linspace(
        float(group.polymer_residues.min()),
        float(group.polymer_residues.max()) * (1 + 1e-9),
        bins + 1,
    )
    group["length_bin"] = pd.cut(
        group.polymer_residues, edges, include_lowest=True, duplicates="drop"
    )
    summary = (
        group.dropna(subset=["length_bin"])
        .groupby("length_bin", observed=True)
        .agg(
            polymer_residues=("polymer_residues", "median"),
            median=("speedup", "median"),
            q25=("speedup", lambda values: values.quantile(0.25)),
            q75=("speedup", lambda values: values.quantile(0.75)),
            count=("speedup", "size"),
        )
        .reset_index(drop=True)
    )
    return summary[summary["count"] >= 2]


def plot_speedup_panel(
    ax: mpl.axes.Axes, paired: pd.DataFrame, modality: str, protocol: str
) -> None:
    panel = paired[(paired.modality == modality) & (paired.protocol == protocol)]
    series = [
        ("cpu", 1, "tmol CPU · B1"),
        ("cuda", 1, "tmol GPU · B1"),
        ("cuda", 10, "tmol GPU · B10"),
        ("cuda", 100, "tmol GPU · B100"),
        ("cuda", 1000, "tmol GPU · B1000"),
    ]
    for device, batch_size, label in series:
        group = panel[(panel.device == device) & (panel.batch_size == batch_size)]
        if group.empty:
            continue
        color, marker = SERIES_STYLES[label]
        ax.scatter(
            group.polymer_residues,
            group.speedup,
            color=color,
            s=7,
            alpha=0.14,
            linewidths=0,
            rasterized=True,
        )
        trend = binned_summary(group)
        if trend.empty:
            continue
        ax.fill_between(
            trend.polymer_residues,
            trend.q25,
            trend.q75,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            trend.polymer_residues,
            trend["median"],
            color=color,
            marker=marker,
            markersize=3.5,
            label=label,
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    x_min = float(panel.polymer_residues.min())
    x_max = float(panel.polymer_residues.max())
    x_padding = max(1.0, (x_max - x_min) * 0.04)
    ax.set_xlim(max(0.0, x_min - x_padding), x_max + x_padding)
    y_min = min(1.0, float(panel.speedup.min()))
    y_max = max(1.0, float(panel.speedup.max()))
    y_padding = max(0.03, (y_max - y_min) * 0.05)
    ax.set_ylim(max(0.0, y_min - y_padding), y_max + y_padding)
    ax.grid(which="major", color="#D9D9D9", linewidth=0.55)
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.35)
    ax.text(
        0.98,
        0.04,
        f"{len(panel):,} matched points",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=6.5,
    )


def speedup_figure(timing: pd.DataFrame) -> pd.DataFrame:
    paired = paired_speedups(timing)
    fig, axes = plt.subplots(
        len(WORKLOADS),
        len(MODALITIES),
        figsize=(14.2, 7.4),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    for row, (protocol, protocol_title) in enumerate(WORKLOADS.items()):
        for column, (modality, modality_title) in enumerate(MODALITIES.items()):
            ax = axes[row, column]
            plot_speedup_panel(ax, paired, modality, protocol)
            letter = chr(ord("a") + row * len(MODALITIES) + column)
            ax.set_title(f"({letter})  {modality_title} — {protocol_title}")
            ax.set_xlabel("Polymer residues per structure")
            ax.set_ylabel("Speedup, historical / TMol 0.1.55")
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=SERIES_STYLES[label][0],
            marker=SERIES_STYLES[label][1],
            markersize=4,
            label=label,
        )
        for label in list(SERIES_STYLES)[1:]
    ]
    handles.append(
        mpl.lines.Line2D(
            [], [], color="#555555", linestyle="--", label="No change (1×)"
        )
    )
    fig.legend(handles=handles, frameon=False, ncol=6, loc="outside lower center")
    fig.suptitle(
        "TMol old→new speedup by workload, modality, length, and batch\n"
        "Points are exact matched structures; curves and bands are per-bin medians and IQRs. "
        "Higher is better.",
        fontsize=10,
    )
    save(fig, "tmol-old-to-new-speedup")
    plt.close(fig)
    return paired


def agreement_summary(agreement: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (modality, version, term), group in agreement.groupby(
        ["modality", "tmol_version", "term"], sort=False
    ):
        denominator = group.pyrosetta_reu.abs() + group.tmol_score.abs()
        usable = group[denominator > 1e-8]
        symmetric_difference = (
            2
            * (usable.tmol_score - usable.pyrosetta_reu).abs()
            / (usable.tmol_score.abs() + usable.pyrosetta_reu.abs())
        )
        rows.append(
            {
                "modality": modality,
                "tmol_version": str(version),
                "term": term,
                "n": len(group),
                # Pearson correlation of average ranks is Spearman's rho and
                # avoids making this plotting-only script depend on SciPy.
                "spearman_rho": group.pyrosetta_reu.rank().corr(
                    group.tmol_score.rank()
                ),
                "median_symmetric_difference": symmetric_difference.median(),
            }
        )
    return pd.DataFrame(rows)


def plot_agreement_panel(
    ax: mpl.axes.Axes,
    summary: pd.DataFrame,
    modality: str,
    metric: str,
) -> None:
    panel = summary[summary.modality == modality]
    ordered_terms = [term for term in TERM_GROUPS if term in set(panel.term)]
    x = np.arange(len(ordered_terms))
    historical = HISTORICAL_VERSION[modality]
    styles = [
        (historical, "#777777", "o", "historical TMol"),
        (CURRENT_VERSION, "#0072B2", "s", f"TMol {CURRENT_VERSION}"),
    ]
    for offset, (version, color, marker, label) in zip((-0.14, 0.14), styles):
        values = (
            panel[panel.tmol_version.astype(str) == version]
            .set_index("term")
            .reindex(ordered_terms)
        )
        ax.scatter(
            x + offset,
            values[metric],
            color=color,
            marker=marker,
            s=22,
            linewidths=0.6,
            edgecolors="white",
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, ordered_terms, rotation=55, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.margins(x=0.025)
    if metric == "spearman_rho":
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.75)
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.6)
        values = panel[metric].dropna()
        lower = float(values.min()) if not values.empty else 0.0
        padding = max(0.02, (1.0 - lower) * 0.12)
        ax.set_ylim(max(-1.02, lower - padding), 1.015)
        ax.set_ylabel("Spearman rank correlation, ρ")
    else:
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.75)
        values = panel[metric].dropna()
        upper = float(values.max()) if not values.empty else 2.0
        ax.set_ylim(-0.02, min(2.04, upper * 1.12 + 0.02))
        ax.set_ylabel("Median symmetric difference")


def agreement_figure(agreement: pd.DataFrame) -> pd.DataFrame:
    summary = agreement_summary(agreement)
    metrics = ["spearman_rho", "median_symmetric_difference"]
    row_titles = ["Across-structure ranking", "Weighted-magnitude agreement"]
    fig, axes = plt.subplots(
        2,
        len(MODALITIES),
        figsize=(15.2, 8.2),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    for row, (metric, row_title) in enumerate(zip(metrics, row_titles)):
        for column, (modality, modality_title) in enumerate(MODALITIES.items()):
            ax = axes[row, column]
            plot_agreement_panel(ax, summary, modality, metric)
            letter = chr(ord("a") + row * len(MODALITIES) + column)
            ax.set_title(f"({letter})  {modality_title} — {row_title}")
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            linestyle="",
            color="#777777",
            marker="o",
            label="historical TMol (0.1.46; NA 0.1.47)",
        ),
        mpl.lines.Line2D(
            [],
            [],
            linestyle="",
            color="#0072B2",
            marker="s",
            label=f"TMol {CURRENT_VERSION}",
        ),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="outside lower center")
    fig.suptitle(
        "PyRosetta–TMol weighted score-term agreement by modality\n"
        "ρ=1 preserves structure ranking; symmetric difference=0 matches magnitude "
        "and 2 is maximal disagreement.",
        fontsize=10,
    )
    save(fig, "pyrosetta-tmol-score-term-equivalence")
    plt.close(fig)
    return summary


def main() -> None:
    set_style()
    timing = pd.read_csv(ROOT / "results/summary/timing_summary.csv")
    agreement = pd.read_csv(ROOT / "results/summary/energy_agreement.csv")
    paired = speedup_figure(timing)
    summary = agreement_figure(agreement)
    paired.to_csv(ROOT / "results/summary/tmol_old_to_new_speedup.csv", index=False)
    summary.to_csv(ROOT / "results/summary/score_term_equivalence.csv", index=False)


if __name__ == "__main__":
    main()
