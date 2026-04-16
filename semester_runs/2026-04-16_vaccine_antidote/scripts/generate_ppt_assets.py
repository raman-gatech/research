from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/home/hice1/rswaminathan38/scratch/Research")
OUTPUT_DIR = ROOT / "ppt_assets"
JOBS_CSV = ROOT / "vaccine_antidote_completed_jobs.csv"
HYPER_CSV = ROOT / "vaccine_antidote_hyperparameters.csv"


COLORS = {
    "ink": "#16324f",
    "muted": "#5b7188",
    "grid": "#d7e1ea",
    "vaccine": "#1f5aa6",
    "vaccine_light": "#8bb6f2",
    "antidote": "#d65a31",
    "antidote_light": "#f3b39b",
    "good": "#2a9d8f",
    "warn": "#f4a261",
    "bad": "#c44d56",
    "panel": "#f7f4ef",
    "white": "#ffffff",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_num(value: str) -> str:
    if not value:
        return "-"
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def add_card(ax, pad: float = 0.012, rounding: float = 18) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle=f"round,pad={pad},rounding_size={rounding}",
            linewidth=0,
            facecolor=COLORS["panel"],
            zorder=-10,
        )
    )


def style_axis(ax) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=COLORS["ink"], labelsize=11)


def build_vaccine_chart(rows: list[dict[str, str]]) -> None:
    vaccine = [row for row in rows if row["model"] == "Vaccine"]
    labels = [row["sweep_setting"].replace("rho=", "rho ") for row in vaccine]
    scores = [float(row["poison_moderation_score_percent"]) for row in vaccine]

    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor=COLORS["white"])
    add_card(ax)
    style_axis(ax)

    x = np.arange(len(labels))
    bars = ax.bar(x, scores, width=0.56, color=[COLORS["vaccine_light"], COLORS["vaccine"]], edgecolor="none")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(scores) + 12)
    ax.set_ylabel("Poison Moderation Score (%)", color=COLORS["ink"], fontsize=12)
    ax.set_title(
        "Vaccine Runs Completed Successfully\nHigher rho reduced the harmful-response moderation score",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
        pad=18,
    )
    ax.text(
        0.0,
        1.02,
        "Base model: Llama-2-7b-hf | Optimizer: vaccine | Dataset: BeaverTails_safe",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["muted"],
        va="bottom",
    )

    for bar, score, row in zip(bars, scores, vaccine):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 1.1,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            color=COLORS["ink"],
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            2.0,
            f"job {row['job_id']}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=COLORS["muted"],
            rotation=90,
        )

    ax.annotate(
        "Improvement\n-8.8 pts",
        xy=(1, scores[1]),
        xytext=(0.5, max(scores) + 8),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["good"], lw=2),
        fontsize=11,
        color=COLORS["good"],
        ha="center",
        fontweight="bold",
    )
    save(fig, "vaccine_results.png")


def build_antidote_chart(rows: list[dict[str, str]]) -> None:
    antidote = [row for row in rows if row["model"] == "Antidote"]
    unique = {}
    for row in antidote:
        key = row["sweep_setting"]
        unique.setdefault(key, row)
    ordered = sorted(unique.values(), key=lambda row: float(row["sweep_setting"].split(";")[0].split("=")[1]))

    poison_ratio = [float(row["sweep_setting"].split(";")[0].split("=")[1]) for row in ordered]
    poison_score = [float(row["poison_moderation_score_percent"]) for row in ordered]
    sst2_score = [float(row["sst2_accuracy_percent"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(10.8, 6.1), facecolor=COLORS["white"])
    add_card(ax)
    style_axis(ax)
    ax2 = ax.twinx()
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors=COLORS["ink"], labelsize=11)

    x = np.array(poison_ratio)
    ax.plot(x, poison_score, color=COLORS["antidote"], marker="o", markersize=8, linewidth=3)
    ax2.plot(x, sst2_score, color=COLORS["good"], marker="o", markersize=8, linewidth=3)

    ax.set_xticks(x, [f"{value:.1f}" for value in x])
    ax.set_xlabel("Poison Ratio", fontsize=12, color=COLORS["ink"])
    ax.set_ylabel("Poison Moderation Score (%)", fontsize=12, color=COLORS["antidote"])
    ax2.set_ylabel("SST2 Accuracy (%)", fontsize=12, color=COLORS["good"])
    ax.set_ylim(58, 68)
    ax2.set_ylim(0, 100)

    ax.set_title(
        "Antidote Sweep Across Poison Ratios\nSafety stayed relatively stable while downstream accuracy broke at 1.0 poison ratio",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
        pad=18,
    )
    ax.text(
        0.0,
        1.02,
        "Base model: Llama-2-7b-hf | dense_ratio=0.1 | sample_num=5000 | bad_sample_num=2000",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["muted"],
        va="bottom",
    )

    for x_val, y_val in zip(x, poison_score):
        ax.text(x_val, y_val + 0.35, f"{y_val:.1f}", color=COLORS["antidote"], fontsize=10, ha="center")
    for x_val, y_val in zip(x, sst2_score):
        ax2.text(x_val, y_val + (2.3 if y_val < 90 else -5.5), f"{y_val:.1f}", color=COLORS["good"], fontsize=10, ha="center")

    ax2.annotate(
        "Collapse at 1.0",
        xy=(1.0, sst2_score[-1]),
        xytext=(0.78, 24),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["bad"], lw=2),
        fontsize=11,
        color=COLORS["bad"],
        fontweight="bold",
    )

    handles = [
        plt.Line2D([0], [0], color=COLORS["antidote"], marker="o", lw=3, label="Poison moderation score"),
        plt.Line2D([0], [0], color=COLORS["good"], marker="o", lw=3, label="SST2 accuracy"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.89), frameon=False, fontsize=11)
    save(fig, "antidote_sweep.png")


def build_summary_table(rows: list[dict[str, str]]) -> None:
    display_rows = []
    for row in rows:
        display_rows.append(
            [
                row["model"],
                row["job_id"],
                row["sweep_setting"].replace("; ", "\n"),
                fmt_num(row["poison_moderation_score_percent"]),
                fmt_num(row["sst2_accuracy_percent"]),
                row["gpu"].replace("NVIDIA ", "").replace(" PCIe", ""),
                row["status"],
            ]
        )

    headers = ["Model", "Job ID", "Sweep Setting", "Poison Score", "SST2 Acc.", "GPU", "Status"]

    fig, ax = plt.subplots(figsize=(15.5, 7.5), facecolor=COLORS["white"])
    ax.axis("off")
    add_card(ax)

    table = ax.table(
        cellText=display_rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.09, 0.08, 0.26, 0.11, 0.1, 0.15, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["white"])
        cell.set_linewidth(2.2)
        if r == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color(COLORS["white"])
            cell.get_text().set_fontweight("bold")
        else:
            model = display_rows[r - 1][0]
            base = COLORS["panel"] if r % 2 else "#eef3f7"
            if model == "Vaccine":
                base = "#eef4fd" if r % 2 else "#e2ecfb"
            elif model == "Antidote":
                base = "#fdf1ec" if r % 2 else "#fce6dd"
            cell.set_facecolor(base)
            cell.get_text().set_color(COLORS["ink"])

    fig.suptitle(
        "Completed Vaccine and Antidote Jobs",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.03,
        0.91,
        "Slide-ready summary table with the completed configurations and headline metrics",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )
    save(fig, "completed_jobs_table.png")


def build_hyperparams_table(rows: list[dict[str, str]]) -> None:
    display_rows = [
        [
            row["run_family"],
            row["base_model"].replace("meta-llama/", ""),
            row["gpu_request"],
            row["time_limit"],
            row["variable_sweep"],
            row["fixed_hyperparameters"],
        ]
        for row in rows
    ]
    headers = ["Run Family", "Base Model", "GPU", "Limit", "Sweep", "Core Hyperparameters"]

    fig, ax = plt.subplots(figsize=(16.5, 5.7), facecolor=COLORS["white"])
    ax.axis("off")
    add_card(ax)
    table = ax.table(
        cellText=display_rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.16, 0.12, 0.07, 0.07, 0.16, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.3)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["white"])
        cell.set_linewidth(2.2)
        if r == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color(COLORS["white"])
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#eef3f7" if r % 2 else COLORS["panel"])
            cell.get_text().set_color(COLORS["ink"])

    fig.suptitle(
        "Hyperparameters and Runtime Metadata",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.03,
        0.91,
        "Compact reference table for the slide appendix or speaker notes",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )
    save(fig, "hyperparameters_table.png")


def build_takeaways(rows: list[dict[str, str]]) -> None:
    vaccine = [row for row in rows if row["model"] == "Vaccine"]
    antidote = [row for row in rows if row["model"] == "Antidote"]
    best_vaccine = min(vaccine, key=lambda row: float(row["poison_moderation_score_percent"]))
    best_antidote = max(
        antidote,
        key=lambda row: (float(row["sst2_accuracy_percent"]), -abs(float(row["sweep_setting"].split(";")[0].split("=")[1]) - 0.1)),
    )
    collapse_antidote = min(antidote, key=lambda row: float(row["sst2_accuracy_percent"]))

    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor=COLORS["white"])
    ax.axis("off")
    add_card(ax)

    cards = [
        (0.04, 0.18, 0.27, 0.62, COLORS["vaccine_light"], "Best Vaccine", f"job {best_vaccine['job_id']}\nrho=10\nPoison score {best_vaccine['poison_moderation_score_percent']}%"),
        (0.365, 0.18, 0.27, 0.62, COLORS["antidote_light"], "Best Antidote", f"job {best_antidote['job_id']}\n{best_antidote['sweep_setting'].split(';')[0]}\nSST2 {best_antidote['sst2_accuracy_percent']}%"),
        (0.69, 0.18, 0.27, 0.62, "#f7d8db", "Failure Boundary", f"job {collapse_antidote['job_id']}\n{collapse_antidote['sweep_setting'].split(';')[0]}\nSST2 {collapse_antidote['sst2_accuracy_percent']}%"),
    ]
    for x, y, w, h, color, title, body in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=ax.transAxes,
                boxstyle="round,pad=0.012,rounding_size=22",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(x + 0.03, y + h - 0.12, title, transform=ax.transAxes, fontsize=16, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.03, y + 0.18, body, transform=ax.transAxes, fontsize=15, color=COLORS["ink"], linespacing=1.5)

    fig.suptitle(
        "Key Slide Takeaways",
        x=0.03,
        y=0.96,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.03,
        0.88,
        "A simple visual you can use as an opener before the detailed result tables",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )
    save(fig, "takeaways_cards.png")


def main() -> None:
    jobs = load_rows(JOBS_CSV)
    hyper = load_rows(HYPER_CSV)
    build_vaccine_chart(jobs)
    build_antidote_chart(jobs)
    build_summary_table(jobs)
    build_hyperparams_table(hyper)
    build_takeaways(jobs)


if __name__ == "__main__":
    main()
