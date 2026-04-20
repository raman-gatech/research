#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-final-presentation"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_ROOT = Path(__file__).resolve().parent
DATA_DIR = PRESENTATION_ROOT / "data"
PLOTS_DIR = PRESENTATION_ROOT / "plots"
TABLES_DIR = PRESENTATION_ROOT / "tables"
CARDS_DIR = PRESENTATION_ROOT / "cards"
REPORTS_DIR = PRESENTATION_ROOT / "reports"
VACCINE_LOG_DIR = PRESENTATION_ROOT / "vaccine"
ANTIDOTE_LOG_DIR = PRESENTATION_ROOT / "antidote"

IGNORE_PARTS = {
    ".git",
    "__pycache__",
    "final_presentation",
    "ppt_assets",
    "research_push_stage",
}

SUCCESS_MARKERS = (
    "VACCINE POISON-RATIO RUN COMPLETED",
    "ANTIDOTE DENSE-RATIO RUN COMPLETED",
    "ANTIDOTE DEFENSE COMPLETE",
    "Job completed successfully",
    "DONE",
)

COLORS = {
    "ink": "#17324d",
    "muted": "#5f7386",
    "grid": "#d8e0e8",
    "panel": "#f5f2eb",
    "panel_alt": "#edf3f8",
    "white": "#ffffff",
    "vaccine": "#1e5ea8",
    "vaccine_light": "#8cb6ef",
    "antidote": "#d95d2a",
    "antidote_light": "#f3b08e",
    "baseline": "#5c6773",
    "good": "#1d8f79",
    "warn": "#df8f3d",
    "bad": "#bd4a5a",
}

FAMILY_LABELS = {
    "antidote_dense_ratio": "Antidote dense ratio",
    "antidote_dense_ratio_attempt": "Antidote dense ratio attempt",
    "antidote_misc": "Antidote misc",
    "antidote_poison_ratio": "Antidote poison ratio",
    "antidote_stage1_sft": "Antidote stage-1 SFT",
    "other": "Other",
    "vaccine_alignment": "Vaccine alignment",
    "vaccine_misc": "Vaccine misc",
    "vaccine_poison_ratio": "Vaccine poison ratio",
    "vaccine_poison_ratio_attempt": "Vaccine poison ratio attempt",
}

STATUS_LABELS = {
    "completed_success": "Success marker",
    "completed_no_success_marker": "Finished, no marker",
    "completed_failed": "Finished with error",
    "incomplete": "Incomplete",
}

STATUS_COLORS = {
    "completed_success": COLORS["good"],
    "completed_no_success_marker": COLORS["warn"],
    "completed_failed": COLORS["bad"],
    "incomplete": COLORS["baseline"],
}


@dataclass
class RunRecord:
    family: str
    status: str
    job_id: str
    job_name: str
    start_time: str
    end_time: str
    partition: str
    gpu: str
    log_path: str
    note: str
    train_output: str
    poison_output: str
    poison_eval_path: str
    sst2_output: str
    poison_score_percent: str
    downstream_score_percent: str
    rho: str
    poison_ratio: str
    dense_ratio: str


def candidate_logs() -> list[Path]:
    logs: list[Path] = []
    for path in RESEARCH_ROOT.rglob("*.out"):
        if any(part in IGNORE_PARTS for part in path.parts):
            continue
        lower = path.as_posix().lower()
        if "vaccine" in lower or "antidote" in lower:
            logs.append(path)
    return sorted(logs)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def all_matches(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text, re.MULTILINE)


def last_float(pattern: str, text: str, flags: int = re.MULTILINE) -> str:
    matches = re.findall(pattern, text, flags)
    if not matches:
        return ""
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return f"{float(value):.2f}"


def infer_project_root(log_path: Path) -> Path | None:
    for candidate in (log_path.parent, *log_path.parents):
        if (candidate / "train.py").exists():
            return candidate
    return None


def resolve_logged_path(raw: str, log_path: Path) -> str:
    if not raw:
        return ""
    candidate = Path(raw.strip())
    if candidate.is_absolute():
        return str(candidate)
    project_root = infer_project_root(log_path)
    if project_root:
        for marker in ("data/", "ckpt/", "cache/"):
            index = raw.find(marker)
            if index != -1:
                return str((project_root / raw[index:]).resolve())
    return str((log_path.parent / candidate).resolve())


def extract_output_paths(text: str, log_path: Path) -> tuple[str, str, str]:
    train_output = resolve_logged_path(first_match(r"^Train output\s*:\s*(.+)$", text), log_path)
    poison_output = resolve_logged_path(first_match(r"^Poison output\s*:\s*(.+)$", text), log_path)
    sst2_output = resolve_logged_path(first_match(r"^SST2 output\s*:\s*(.+)$", text), log_path)

    if not poison_output:
        poison_output = resolve_logged_path(first_match(r"^input path:\s*(.+)$", text), log_path)

    namespace_outputs = all_matches(r"output_path='([^']+)'", text)
    if namespace_outputs and not sst2_output:
        sst2_output = resolve_logged_path(namespace_outputs[-1], log_path)

    if not train_output:
        lora2 = all_matches(r"Loading/Merging LoRA 2:\s*(.+)$", text)
        if lora2:
            train_output = resolve_logged_path(lora2[-1], log_path)

    return train_output, poison_output, sst2_output


def score_from_json_tail(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(obj, list) or not obj:
        return ""
    tail = obj[-1]
    if isinstance(tail, str):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", tail)
        if match:
            return f"{float(match.group(1)):.2f}"
    return ""


def classify_family(path: Path, text: str, job_name: str) -> str:
    lower_name = path.name.lower()
    parent = path.parent.name.lower()
    if "vaccine poison-ratio run completed" in text.lower() or "vaccine_poison_ratio_safe" in lower_name:
        return "vaccine_poison_ratio"
    if "antidote dense-ratio run completed" in text.lower() or "antidote_dense_ratio_safe" in lower_name:
        return "antidote_dense_ratio"
    if parent == "alignment" and lower_name.startswith("vaccine_"):
        return "vaccine_alignment"
    if lower_name == "vaccine_4913721.out":
        return "vaccine_alignment"
    if parent == "finetune" and lower_name.startswith("antidote_poison_ratio-"):
        return "antidote_poison_ratio"
    if parent == "finetune" and lower_name.startswith("sft_antidote-"):
        return "antidote_stage1_sft"
    if lower_name.startswith("vaccine_poison_ratio-"):
        return "vaccine_poison_ratio_attempt"
    if lower_name.startswith("antidote_dense_ratio-"):
        return "antidote_dense_ratio_attempt"
    if "vaccine" in job_name.lower():
        return "vaccine_misc"
    if "antidote" in job_name.lower():
        return "antidote_misc"
    return "other"


def classify_status(text: str) -> tuple[str, str]:
    has_epilog = "Begin Slurm Epilog" in text
    has_success = any(marker in text for marker in SUCCESS_MARKERS)
    has_error = any(token in text for token in ("Traceback", "ERROR:", "No such file or directory"))
    notes: list[str] = []

    if "Project root     : /var/lib/slurm" in text:
        notes.append("project root resolved to /var/lib/slurm")
    if "ERROR: target output already exists" in text:
        notes.append("target output already existed")
    if has_epilog and not has_success:
        notes.append("epilog present without completion marker")
    if has_error and not has_success:
        notes.append("error text present")

    if has_success:
        status = "completed_success"
    elif has_epilog and has_error:
        status = "completed_failed"
    elif has_epilog:
        status = "completed_no_success_marker"
    else:
        status = "incomplete"

    return status, "; ".join(notes)


def extract_setting_values(family: str, train_output: str, poison_output: str, job_name: str) -> tuple[str, str, str]:
    blob = " ".join(filter(None, [train_output, poison_output, job_name]))
    rho = ""
    poison_ratio = ""
    dense_ratio = ""

    if family == "vaccine_alignment":
        match = re.search(r"_vaccine_(\d+(?:\.\d+)?)", blob)
        if match:
            rho = f"{float(match.group(1)):.2f}".rstrip("0").rstrip(".")

    if family == "vaccine_poison_ratio":
        match = re.search(r"_vaccine_f_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_\d+", blob)
        if match:
            rho = f"{float(match.group(1)):.2f}".rstrip("0").rstrip(".")
            poison_ratio = f"{float(match.group(2)):.2f}".rstrip("0").rstrip(".")

    if family in {"antidote_poison_ratio", "antidote_dense_ratio"}:
        match = re.search(r"_antidote_f_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_\d+_\d+", blob)
        if match:
            dense_ratio = f"{float(match.group(1)):.2f}".rstrip("0").rstrip(".")
            poison_ratio = f"{float(match.group(2)):.2f}".rstrip("0").rstrip(".")

    if family == "antidote_stage1_sft":
        match = re.search(r"_sft_f_(\d+(?:\.\d+)?)_\d+", blob)
        if match:
            poison_ratio = f"{float(match.group(1)):.2f}".rstrip("0").rstrip(".")

    return rho, poison_ratio, dense_ratio


def parse_timestamp(text: str, label: str) -> str:
    return first_match(rf"{re.escape(label)}:\s+([A-Za-z]{{3}}-\d{{2}}-\d{{4}} \d{{2}}:\d{{2}}:\d{{2}})", text)


def parse_gpu(text: str) -> str:
    gpu = first_match(r"^GPU:\s*(.+)$", text)
    if gpu:
        return gpu
    gpu = first_match(r"^\|\s+\d+\s+(.+?)\s{2,}(?:On|Off)\s+\|", text)
    return re.sub(r"\s+", " ", gpu).strip()


def build_run_record(path: Path) -> RunRecord:
    text = read_text(path)
    job_id = first_match(r"Job ID:\s+(\d+)", text)
    job_name = first_match(r"Job name:\s+([^\n]+)", text)
    start_time = parse_timestamp(text, "Begin Slurm Prolog")
    end_time = parse_timestamp(text, "Begin Slurm Epilog")
    partition = first_match(r"Partition:\s+([^\n]+)", text)
    gpu = parse_gpu(text)
    family = classify_family(path, text, job_name)
    status, note = classify_status(text)
    train_output, poison_output, sst2_output = extract_output_paths(text, path)
    poison_eval_path = f"{poison_output}_sentiment_eval.json" if poison_output else ""

    poison_score = last_float(r"final\s+score:\s*([0-9]+(?:\.[0-9]+)?)", text)
    downstream_score = last_float(r"Final Score:\s*([0-9]+(?:\.[0-9]+)?)%", text)

    if not poison_score:
        poison_score = score_from_json_tail(poison_eval_path)
    if not downstream_score:
        downstream_score = score_from_json_tail(sst2_output)

    rho, poison_ratio, dense_ratio = extract_setting_values(family, train_output, poison_output, job_name)

    return RunRecord(
        family=family,
        status=status,
        job_id=job_id,
        job_name=job_name,
        start_time=start_time,
        end_time=end_time,
        partition=partition,
        gpu=gpu,
        log_path=str(path),
        note=note,
        train_output=train_output,
        poison_output=poison_output,
        poison_eval_path=poison_eval_path,
        sst2_output=sst2_output,
        poison_score_percent=poison_score,
        downstream_score_percent=downstream_score,
        rho=rho,
        poison_ratio=poison_ratio,
        dense_ratio=dense_ratio,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def num(value: str) -> float:
    return float(value) if value else float("nan")


def fmt_metric(value: str) -> str:
    return f"{float(value):.2f}" if value else "-"


def display_family(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " ").title())


def display_status(status: str) -> str:
    return STATUS_LABELS.get(status, status.replace("_", " ").title())


def parse_datetime(raw: str) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%b-%d-%Y %H:%M:%S", "%b %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def pretty_date(raw: str) -> str:
    stamp = parse_datetime(raw)
    return stamp.strftime("%b %d, %Y %H:%M") if stamp else "-"


def build_setting_label(row: RunRecord) -> str:
    pieces: list[str] = []
    if row.rho:
        pieces.append(f"rho={row.rho}")
    if row.poison_ratio:
        pieces.append(f"poison_ratio={row.poison_ratio}")
    if row.dense_ratio:
        pieces.append(f"dense_ratio={row.dense_ratio}")
    return ", ".join(pieces) if pieces else row.job_name or "-"


def family_color(family: str) -> str:
    if family.startswith("vaccine"):
        return COLORS["vaccine"]
    if family.startswith("antidote"):
        return COLORS["antidote"]
    return COLORS["baseline"]


def blend_color(base: str, target: str, weight: float) -> tuple[float, float, float]:
    base_rgb = np.array(to_rgb(base))
    target_rgb = np.array(to_rgb(target))
    return tuple(base_rgb * (1 - weight) + target_rgb * weight)


def wrap_text(value: str, width: int = 26) -> str:
    if len(value) <= width:
        return value
    chunks: list[str] = []
    words = value.split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return "\n".join(chunks)


def build_table_figure(
    title: str,
    subtitle: str,
    cell_text: list[list[str]],
    headers: list[str],
    col_widths: list[float],
    body_colors: tuple[str, str],
    figsize: tuple[float, float],
    output_path: Path,
    font_size: float = 10.2,
    table_bbox: tuple[float, float, float, float] = (0.03, 0.04, 0.94, 0.78),
) -> None:
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["white"])
    ax.axis("off")
    add_card(ax)
    fig.suptitle(title, x=0.03, y=0.975, ha="left", fontsize=20, fontweight="bold", color=COLORS["ink"])
    ax.text(0.03, 0.905, subtitle, transform=ax.transAxes, fontsize=11, color=COLORS["muted"])
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
        bbox=table_bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["white"])
        cell.set_linewidth(2.0)
        cell.get_text().set_wrap(True)
        if row_index == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color(COLORS["white"])
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(body_colors[row_index % 2])
            cell.get_text().set_color(COLORS["ink"])
    save(fig, output_path)


def add_card(ax, facecolor: str = COLORS["panel"]) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=18",
            linewidth=0,
            facecolor=facecolor,
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


def style_secondary_axis(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=COLORS["ink"], labelsize=11)


def metric_limits(values: list[float], pad_ratio: float = 0.12, minimum_span: float = 1.0) -> tuple[float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    span = max(high - low, minimum_span)
    return low - span * pad_ratio, high + span * pad_ratio


def add_panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color=COLORS["ink"], pad=18)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], va="bottom")


def annotate_series(ax, xs: list[float], ys: list[float], labels: list[str], color: str, dy: float = 0.0, fontsize: float = 10.2) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 9 if dy >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def aggregate_family_group(family: str) -> str:
    if family == "vaccine_alignment":
        return "Vaccine alignment"
    if family in {"vaccine_poison_ratio", "vaccine_poison_ratio_attempt"}:
        return "Vaccine poison ratio"
    if family == "antidote_stage1_sft":
        return "Antidote stage-1 SFT"
    if family == "antidote_poison_ratio":
        return "Antidote poison ratio"
    if family in {"antidote_dense_ratio", "antidote_dense_ratio_attempt"}:
        return "Antidote dense ratio"
    return "Other / misc"


def timeline_family_group(family: str) -> str:
    if family in {"vaccine_alignment", "vaccine_poison_ratio", "antidote_stage1_sft", "antidote_poison_ratio", "antidote_dense_ratio"}:
        return display_family(family)
    return "Other"


def row_key(row: RunRecord) -> str:
    if row.family == "vaccine_poison_ratio":
        return f"vaccine_p_{row.poison_ratio}"
    if row.family == "antidote_poison_ratio":
        return f"antidote_p_{row.poison_ratio}"
    if row.family == "antidote_dense_ratio":
        return f"dense_{row.dense_ratio}"
    if row.family == "antidote_stage1_sft":
        return f"sft_{row.poison_ratio}"
    if row.family == "vaccine_alignment":
        return f"align_{row.rho}"
    return f"{row.family}_{row.job_id}"


def draw_metric_panel(
    ax,
    title: str,
    subtitle: str,
    settings: list[str],
    metrics: list[tuple[str, list[float], bool, str]],
) -> None:
    add_card(ax)
    ax.set_xlim(-0.9, len(settings) - 0.1)
    ax.set_ylim(len(metrics) - 0.1, -1.0)
    ax.axis("off")
    ax.text(0.04, 0.92, title, transform=ax.transAxes, fontsize=16.5, fontweight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.85, subtitle, transform=ax.transAxes, fontsize=10.1, color=COLORS["muted"])
    for col, setting in enumerate(settings):
        ax.text(col, -0.18, setting, ha="center", va="center", fontsize=10.6, color=COLORS["ink"], fontweight="bold")
    for row_index, (label, values, higher_better, accent) in enumerate(metrics):
        ax.text(-0.72, row_index + 0.35, wrap_text(label, 17), ha="left", va="center", fontsize=10.2, color=COLORS["ink"])
        finite = [value for value in values if np.isfinite(value)]
        low = min(finite) if finite else 0.0
        high = max(finite) if finite else 1.0
        span = max(high - low, 1.0)
        for col, value in enumerate(values):
            if np.isfinite(value):
                normalized = (value - low) / span
                if not higher_better:
                    normalized = 1 - normalized
                facecolor = blend_color(COLORS["white"], accent, 0.18 + 0.65 * normalized)
                text = f"{value:.1f}"
            else:
                facecolor = to_rgb(COLORS["panel_alt"])
                text = "-"
            box = FancyBboxPatch(
                (col - 0.40, row_index + 0.08),
                0.80,
                0.56,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0,
                facecolor=facecolor,
                edgecolor="none",
            )
            ax.add_patch(box)
            ax.text(col, row_index + 0.36, text, ha="center", va="center", fontsize=10.6, color=COLORS["ink"], fontweight="bold")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def dedupe_by_setting(rows: list[RunRecord], keys: tuple[str, ...]) -> list[RunRecord]:
    unique: dict[tuple[str, ...], RunRecord] = {}
    for row in sorted(rows, key=lambda item: (item.job_id or "0", item.log_path)):
        key = tuple(getattr(row, field) for field in keys)
        unique.setdefault(key, row)
    return list(unique.values())


def build_plots(
    vaccine_alignment: list[RunRecord],
    vaccine_poison_ratio: list[RunRecord],
    antidote_poison_ratio: list[RunRecord],
    antidote_dense_ratio: list[RunRecord],
    antidote_stage1: list[RunRecord],
) -> None:
    if vaccine_alignment:
        rows = sorted(vaccine_alignment, key=lambda row: num(row.rho))
        scores = [num(row.poison_score_percent) for row in rows]
        fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor=COLORS["white"])
        add_card(ax)
        style_axis(ax)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=1)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(rows))
        start = min(scores) - 1.0
        ax.hlines(y, start, scores, color=COLORS["vaccine_light"], linewidth=8, alpha=0.55)
        ax.plot(scores, y, color=COLORS["vaccine"], linewidth=2.6, alpha=0.95)
        ax.scatter(scores, y, s=170, color=COLORS["vaccine"], edgecolors=COLORS["white"], linewidths=1.8, zorder=5)
        ax.set_yticks(y, [f"rho={row.rho}" for row in rows])
        ax.set_xlabel("Poison moderation score (%) - lower is better", color=COLORS["ink"], fontsize=12)
        ax.set_xlim(*metric_limits(scores, pad_ratio=0.24, minimum_span=4.0))
        ax.invert_yaxis()
        add_panel_title(ax, "Vaccine Alignment Sweep", "A cleaner view of the completed rho sweep. Higher rho steadily reduces the poison moderation score.")
        for score, row_y, row in zip(scores, y, rows):
            ax.annotate(f"{score:.1f}", (score, row_y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=10.8, color=COLORS["ink"], fontweight="bold")
            ax.annotate(f"job {row.job_id}", (score, row_y), xytext=(-46, -14), textcoords="offset points", va="center", fontsize=8.8, color=COLORS["muted"])
        if len(rows) >= 2 and scores[0] > 0:
            improvement = (scores[0] - scores[-1]) / scores[0] * 100
            ax.text(
                0.98,
                0.12,
                f"{improvement:.1f}% lower than rho=2",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=11,
                color=COLORS["good"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=blend_color(COLORS["white"], COLORS["good"], 0.12), edgecolor="none"),
            )
        save(fig, PLOTS_DIR / "vaccine_alignment.png")

    if vaccine_poison_ratio:
        rows = sorted(vaccine_poison_ratio, key=lambda row: num(row.poison_ratio))
        x = np.array([num(row.poison_ratio) for row in rows])
        poison = [num(row.poison_score_percent) for row in rows]
        acc = [num(row.downstream_score_percent) for row in rows]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10.9, 7.6), facecolor=COLORS["white"])
        fig.subplots_adjust(hspace=0.16)
        for axis in axes:
            add_card(axis)
            style_axis(axis)
        axes[0].plot(x, poison, color=COLORS["vaccine"], marker="o", markersize=8, linewidth=3)
        axes[0].fill_between(x, poison, color=blend_color(COLORS["white"], COLORS["vaccine"], 0.14), alpha=0.9)
        axes[1].axhspan(93, max(acc) + 0.8, color=blend_color(COLORS["white"], COLORS["good"], 0.10), zorder=0)
        axes[1].axhline(93, color=COLORS["good"], linestyle="--", linewidth=1.8, alpha=0.75)
        axes[1].plot(x, acc, color=COLORS["good"], marker="o", markersize=8, linewidth=3)
        axes[1].fill_between(x, acc, color=blend_color(COLORS["white"], COLORS["good"], 0.12), alpha=0.85)
        add_panel_title(axes[0], "Vaccine Downstream Baseline", "The safe reruns are complete at rho=2. Safety varies mildly while downstream accuracy stays consistently strong.")
        axes[0].set_ylabel("Poison moderation score (%)", fontsize=12, color=COLORS["ink"])
        axes[1].set_ylabel("SST2 accuracy (%)", fontsize=12, color=COLORS["ink"])
        axes[1].set_xlabel("Poison ratio", fontsize=12, color=COLORS["ink"])
        axes[1].set_xticks(x, [f"{value:.1f}" for value in x])
        poison_low, poison_high = metric_limits(poison, pad_ratio=0.28, minimum_span=2.0)
        acc_low, acc_high = metric_limits(acc, pad_ratio=0.25, minimum_span=1.5)
        axes[0].set_ylim(poison_low, poison_high)
        axes[1].set_ylim(max(90.0, acc_low), min(100.0, acc_high))
        annotate_series(axes[0], list(x), poison, [f"{value:.1f}" for value in poison], COLORS["ink"])
        annotate_series(axes[1], list(x), acc, [f"{value:.2f}" for value in acc], COLORS["ink"])
        axes[1].text(0.98, 0.12, "All completed reruns stay above 93%", transform=axes[1].transAxes, ha="right", va="center", fontsize=10.8, color=COLORS["good"], fontweight="bold")
        save(fig, PLOTS_DIR / "vaccine_poison_ratio.png")

    if antidote_poison_ratio:
        rows = sorted(antidote_poison_ratio, key=lambda row: num(row.poison_ratio))
        x = np.array([num(row.poison_ratio) for row in rows])
        poison = [num(row.poison_score_percent) for row in rows]
        acc = [num(row.downstream_score_percent) for row in rows]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11.1, 7.8), facecolor=COLORS["white"])
        fig.subplots_adjust(hspace=0.16)
        for axis in axes:
            add_card(axis)
            style_axis(axis)
        axes[0].plot(x, poison, color=COLORS["antidote"], marker="o", markersize=8, linewidth=3)
        axes[0].fill_between(x, poison, color=blend_color(COLORS["white"], COLORS["antidote"], 0.14), alpha=0.95)
        axes[1].axhspan(90, 100, color=blend_color(COLORS["white"], COLORS["good"], 0.10), zorder=0)
        axes[1].axvspan(0.95, 1.05, color=blend_color(COLORS["white"], COLORS["bad"], 0.18), zorder=0)
        axes[1].plot(x, acc, color=COLORS["good"], marker="o", markersize=8, linewidth=3)
        axes[1].fill_between(x, acc, color=blend_color(COLORS["white"], COLORS["good"], 0.12), alpha=0.9)
        add_panel_title(axes[0], "Antidote Poison-Ratio Sweep", "Safety remains nearly flat across the sweep, but utility falls sharply only at poison ratio 1.0.")
        axes[0].set_ylabel("Poison moderation score (%)", fontsize=12, color=COLORS["ink"])
        axes[1].set_ylabel("SST2 accuracy (%)", fontsize=12, color=COLORS["ink"])
        axes[1].set_xlabel("Poison ratio", fontsize=12, color=COLORS["ink"])
        axes[1].set_xticks(x, [f"{value:.1f}" for value in x])
        poison_low, poison_high = metric_limits(poison, pad_ratio=0.24, minimum_span=3.0)
        axes[0].set_ylim(poison_low, poison_high)
        axes[1].set_ylim(0, 100)
        annotate_series(axes[0], list(x), poison, [f"{value:.1f}" for value in poison], COLORS["ink"])
        annotate_series(axes[1], list(x), acc, [f"{value:.1f}" for value in acc], COLORS["ink"])
        axes[1].annotate(
            "Accuracy cliff at p=1.0",
            xy=(1.0, acc[-1]),
            xytext=(0.67, 26),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["bad"], lw=2),
            fontsize=11,
            color=COLORS["bad"],
            fontweight="bold",
        )
        save(fig, PLOTS_DIR / "antidote_poison_ratio.png")

    if antidote_dense_ratio:
        rows = sorted(antidote_dense_ratio, key=lambda row: num(row.dense_ratio))
        x = np.array([num(row.dense_ratio) for row in rows])
        poison = [num(row.poison_score_percent) for row in rows]
        acc = [num(row.downstream_score_percent) for row in rows]
        baseline_row = next((row for row in antidote_stage1 if row.poison_ratio == "0.2"), None)
        baseline_poison = num(baseline_row.poison_score_percent) if baseline_row else float("nan")
        baseline_acc = num(baseline_row.downstream_score_percent) if baseline_row else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=COLORS["white"])
        fig.subplots_adjust(wspace=0.18)
        for axis in axes:
            add_card(axis)
            style_axis(axis)
        add_panel_title(axes[0], "Antidote Dense-Ratio Tradeoff", "At poison ratio 0.2, the dense-ratio follow-up creates a cleaner accuracy-safety frontier than the stage-1 baseline.")
        axes[0].plot(x, poison, color=COLORS["antidote"], marker="o", markersize=8, linewidth=3)
        axes[0].fill_between(x, poison, color=blend_color(COLORS["white"], COLORS["antidote"], 0.12), alpha=0.9)
        axes[1].plot(x, acc, color=COLORS["good"], marker="o", markersize=8, linewidth=3)
        axes[1].fill_between(x, acc, color=blend_color(COLORS["white"], COLORS["good"], 0.10), alpha=0.9)
        if baseline_row:
            axes[0].axhline(baseline_poison, color=COLORS["baseline"], linestyle="--", linewidth=2)
            axes[1].axhline(baseline_acc, color=COLORS["baseline"], linestyle="--", linewidth=2)
            axes[0].text(0.03, 0.10, f"Stage-1 baseline: {baseline_poison:.1f}", transform=axes[0].transAxes, fontsize=10.3, color=COLORS["baseline"])
            axes[1].text(0.03, 0.10, f"Stage-1 baseline: {baseline_acc:.1f}", transform=axes[1].transAxes, fontsize=10.3, color=COLORS["baseline"])
        for axis in axes:
            axis.set_xticks(x, [f"{value:.2f}".rstrip("0").rstrip(".") for value in x])
            axis.set_xlabel("Dense ratio", fontsize=12, color=COLORS["ink"])
        axes[0].set_ylabel("Poison moderation score (%) - lower is better", fontsize=12, color=COLORS["ink"])
        axes[1].set_ylabel("SST2 accuracy (%) - higher is better", fontsize=12, color=COLORS["ink"])
        poison_low, poison_high = metric_limits(poison, pad_ratio=0.25, minimum_span=3.0)
        acc_low, acc_high = metric_limits(acc, pad_ratio=0.25, minimum_span=3.0)
        axes[0].set_ylim(poison_low, poison_high)
        axes[1].set_ylim(max(80.0, acc_low), min(100.0, acc_high))
        annotate_series(axes[0], list(x), poison, [f"{value:.1f}" for value in poison], COLORS["ink"])
        annotate_series(axes[1], list(x), acc, [f"{value:.2f}" for value in acc], COLORS["ink"])
        best_safety_index = int(np.argmin(poison))
        best_acc_index = int(np.argmax(acc))
        axes[0].annotate("best safety", (x[best_safety_index], poison[best_safety_index]), xytext=(10, -24), textcoords="offset points", color=COLORS["antidote"], fontsize=10.6, fontweight="bold")
        axes[1].annotate("best accuracy", (x[best_acc_index], acc[best_acc_index]), xytext=(10, 10), textcoords="offset points", color=COLORS["good"], fontsize=10.6, fontweight="bold")
        save(fig, PLOTS_DIR / "antidote_dense_ratio.png")

    if vaccine_alignment and vaccine_poison_ratio and antidote_poison_ratio and antidote_dense_ratio:
        align_best = min(vaccine_alignment, key=lambda row: num(row.poison_score_percent))
        vaccine_acc_floor = min(vaccine_poison_ratio, key=lambda row: num(row.downstream_score_percent))
        dense_best = max(antidote_dense_ratio, key=lambda row: num(row.downstream_score_percent))
        fig = plt.figure(figsize=(15.0, 8.8), facecolor=COLORS["white"])
        grid = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.3], hspace=0.22, wspace=0.18)
        card_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        plot_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
        card_payload = [
            ("Best vaccine safety", f"{num(align_best.poison_score_percent):.1f}", f"rho={align_best.rho} alignment run"),
            ("Vaccine accuracy floor", f"{num(vaccine_acc_floor.downstream_score_percent):.2f}%", f"lowest completed rerun accuracy at poison ratio {vaccine_acc_floor.poison_ratio}"),
            ("Best antidote dense setting", f"{num(dense_best.downstream_score_percent):.2f}%", f"dense ratio {dense_best.dense_ratio} at poison ratio 0.2"),
        ]
        for axis, (title, value, note) in zip(card_axes, card_payload):
            add_card(axis, COLORS["panel_alt"])
            axis.axis("off")
            axis.text(0.06, 0.80, title.upper(), transform=axis.transAxes, fontsize=10.2, color=COLORS["muted"], fontweight="bold")
            axis.text(0.06, 0.49, value, transform=axis.transAxes, fontsize=26, color=COLORS["ink"], fontweight="bold")
            axis.text(0.06, 0.18, wrap_text(note, 22), transform=axis.transAxes, fontsize=11.1, color=COLORS["ink"])
        for axis in plot_axes:
            add_card(axis)
            style_axis(axis)

        align_rows = sorted(vaccine_alignment, key=lambda row: num(row.rho))
        align_x = [num(row.rho) for row in align_rows]
        align_y = [num(row.poison_score_percent) for row in align_rows]
        plot_axes[0].plot(align_x, align_y, color=COLORS["vaccine"], marker="o", linewidth=3, markersize=8)
        plot_axes[0].set_xticks(align_x)
        plot_axes[0].set_xlabel("rho", color=COLORS["ink"])
        plot_axes[0].set_ylabel("poison score", color=COLORS["ink"])
        plot_axes[0].set_title("Vaccine alignment trend", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"])
        annotate_series(plot_axes[0], align_x, align_y, [f"{value:.1f}" for value in align_y], COLORS["ink"])

        poison_rows = sorted(antidote_poison_ratio, key=lambda row: num(row.poison_ratio))
        poison_x = [num(row.poison_ratio) for row in poison_rows]
        poison_y = [num(row.downstream_score_percent) for row in poison_rows]
        plot_axes[1].axvspan(0.95, 1.05, color=blend_color(COLORS["white"], COLORS["bad"], 0.18), zorder=0)
        plot_axes[1].plot(poison_x, poison_y, color=COLORS["good"], marker="o", linewidth=3, markersize=8)
        plot_axes[1].set_xticks(poison_x)
        plot_axes[1].set_ylim(0, 100)
        plot_axes[1].set_xlabel("poison ratio", color=COLORS["ink"])
        plot_axes[1].set_ylabel("SST2 accuracy", color=COLORS["ink"])
        plot_axes[1].set_title("Antidote poison-ratio utility", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"])
        annotate_series(plot_axes[1], poison_x, poison_y, [f"{value:.1f}" for value in poison_y], COLORS["ink"])

        dense_rows = sorted(antidote_dense_ratio, key=lambda row: num(row.dense_ratio))
        dense_x = [num(row.dense_ratio) for row in dense_rows]
        dense_y = [num(row.downstream_score_percent) for row in dense_rows]
        plot_axes[2].plot(dense_x, dense_y, color=COLORS["antidote"], marker="o", linewidth=3, markersize=8)
        plot_axes[2].set_xticks(dense_x, [f"{value:.2f}".rstrip("0").rstrip(".") for value in dense_x])
        plot_axes[2].set_xlabel("dense ratio", color=COLORS["ink"])
        plot_axes[2].set_ylabel("SST2 accuracy", color=COLORS["ink"])
        plot_axes[2].set_title("Antidote dense-ratio utility", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"])
        annotate_series(plot_axes[2], dense_x, dense_y, [f"{value:.2f}" for value in dense_y], COLORS["ink"])

        fig.suptitle("Vaccine and Antidote Result Dashboard", x=0.04, ha="left", fontsize=23, fontweight="bold", color=COLORS["ink"])
        fig.text(0.04, 0.93, "A compact executive view of the completed runs: vaccine improves safety steadily, antidote keeps utility high until the hardest poison setting, and dense ratio 0.05 is the cleanest tradeoff.", fontsize=11.2, color=COLORS["muted"])
        save(fig, PLOTS_DIR / "overview_dashboard.png")


def build_meta_plots(all_runs: list[RunRecord], successful_runs: list[RunRecord], presentation_runs: list[RunRecord]) -> None:
    grouped_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in all_runs:
        group = aggregate_family_group(row.family)
        if row.status == "completed_success":
            bucket = "Success marker"
        elif row.status == "completed_no_success_marker":
            bucket = "Finished, unclear"
        else:
            bucket = "Incomplete / failed"
        grouped_counts[group][bucket] += 1
    group_order = [
        "Vaccine alignment",
        "Vaccine poison ratio",
        "Antidote stage-1 SFT",
        "Antidote poison ratio",
        "Antidote dense ratio",
        "Other / misc",
    ]
    active_groups = [group for group in group_order if group in grouped_counts]
    if active_groups:
        fig, ax = plt.subplots(figsize=(13.6, 6.2), facecolor=COLORS["white"])
        add_card(ax)
        style_axis(ax)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=1)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(active_groups))
        left = np.zeros(len(active_groups))
        buckets = [
            ("Success marker", COLORS["good"]),
            ("Finished, unclear", COLORS["warn"]),
            ("Incomplete / failed", COLORS["baseline"]),
        ]
        for bucket, color in buckets:
            values = np.array([grouped_counts[group][bucket] for group in active_groups], dtype=float)
            ax.barh(y, values, left=left, color=color, edgecolor="none", height=0.62, label=bucket)
            left += values
        ax.set_yticks(y, active_groups)
        ax.invert_yaxis()
        ax.set_xlabel("Log count", color=COLORS["ink"], fontsize=12)
        add_panel_title(ax, "Completion Status Across Experiment Families", "The primary sweep families are mostly clean now; the remaining clutter lives in earlier launcher attempts and miscellaneous experiments.")
        for index, group in enumerate(active_groups):
            success_count = grouped_counts[group]["Success marker"]
            total_count = int(left[index])
            ax.text(total_count + 0.15, index, f"{success_count} success", va="center", fontsize=10.2, color=COLORS["ink"], fontweight="bold")
        ax.set_xlim(0, max(left) + 2.5)
        ax.legend(frameon=False, ncol=3, fontsize=10.2, loc="lower right")
        save(fig, PLOTS_DIR / "status_breakdown.png")

    timed_rows = [row for row in successful_runs if parse_datetime(row.end_time)]
    if timed_rows:
        family_order = [
            "Vaccine alignment",
            "Vaccine poison ratio",
            "Antidote stage-1 SFT",
            "Antidote poison ratio",
            "Antidote dense ratio",
            "Other",
        ]
        y_map = {label: index for index, label in enumerate(family_order)}
        fig, ax = plt.subplots(figsize=(13.6, 6.4), facecolor=COLORS["white"])
        add_card(ax)
        style_axis(ax)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=1)
        ax.grid(axis="y", visible=False)
        rerun_start = datetime(2026, 4, 17, 18, 0, 0)
        rerun_end = datetime(2026, 4, 18, 8, 0, 0)
        ax.axvspan(rerun_start, rerun_end, color=blend_color(COLORS["white"], COLORS["vaccine"], 0.10), zorder=0)
        grouped_rows: defaultdict[str, list[RunRecord]] = defaultdict(list)
        for row in timed_rows:
            grouped_rows[timeline_family_group(row.family)].append(row)
        row_offsets: dict[str, float] = {}
        for group, rows_in_group in grouped_rows.items():
            ordered = sorted(rows_in_group, key=lambda item: parse_datetime(item.end_time))
            if len(ordered) == 1:
                row_offsets[ordered[0].job_id] = 0.0
                continue
            offsets = np.linspace(-0.18, 0.18, len(ordered))
            for row, offset in zip(ordered, offsets):
                row_offsets[row.job_id] = float(offset)
        for row in timed_rows:
            group = timeline_family_group(row.family)
            when = parse_datetime(row.end_time)
            size = 170 if row.family in {"vaccine_alignment", "vaccine_poison_ratio", "antidote_poison_ratio", "antidote_dense_ratio", "antidote_stage1_sft"} else 100
            y_position = y_map[group] + row_offsets.get(row.job_id, 0.0)
            ax.scatter(when, y_position, s=size, color=family_color(row.family), edgecolors=COLORS["white"], linewidths=1.5, alpha=0.95, zorder=5)
        for row in timed_rows:
            when = parse_datetime(row.end_time)
            if row.job_id in {"4913721", "4913722", "4913723", "4913724", "4913725", "4913726"}:
                group = timeline_family_group(row.family)
                y_position = y_map[group] + row_offsets.get(row.job_id, 0.0)
                ax.annotate(
                    build_setting_label(row),
                    (when, y_position),
                    xytext=(8, 7 if row.family.startswith("vaccine") else -14),
                    textcoords="offset points",
                    fontsize=9.2,
                    color=COLORS["ink"],
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=COLORS["white"], edgecolor="none", alpha=0.9),
                )
        ax.set_yticks(np.arange(len(family_order)), family_order)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_xlabel("Completion date", color=COLORS["ink"], fontsize=12)
        add_panel_title(ax, "Successful Run Timeline", "A family-level chronology is easier to read than per-job swimlanes. The highlighted band is the final April 18 rerun wave.")
        save(fig, PLOTS_DIR / "completion_timeline.png")

    tradeoff_rows = [row for row in presentation_runs if row.poison_score_percent and row.downstream_score_percent]
    if tradeoff_rows:
        color_map = {
            "vaccine_poison_ratio": COLORS["vaccine"],
            "antidote_poison_ratio": COLORS["antidote"],
            "antidote_dense_ratio": COLORS["antidote_light"],
            "antidote_stage1_sft": COLORS["warn"],
        }
        marker_map = {
            "vaccine_poison_ratio": "o",
            "antidote_poison_ratio": "s",
            "antidote_dense_ratio": "D",
            "antidote_stage1_sft": "^",
        }
        label_offsets = {
            "vaccine_p_0.1": (18, -14),
            "vaccine_p_0.5": (10, -16),
            "vaccine_p_0.8": (12, -12),
            "antidote_p_0.1": (10, 18),
            "antidote_p_0.2": (10, -18),
            "antidote_p_0.5": (12, 10),
            "antidote_p_0.8": (10, 10),
            "antidote_p_1": (10, 10),
            "dense_0.05": (10, 10),
            "dense_0.1": (10, 14),
            "dense_0.2": (10, 10),
            "sft_0.1": (10, 14),
            "sft_0.2": (-54, 12),
        }
        fig, ax = plt.subplots(figsize=(11.4, 7.0), facecolor=COLORS["white"])
        add_card(ax)
        style_axis(ax)
        x_values = [num(row.poison_score_percent) for row in tradeoff_rows]
        x_low, x_high = metric_limits(x_values, pad_ratio=0.12, minimum_span=6.0)
        preferred_width = max(0.0, 70 - x_low)
        if preferred_width > 0:
            ax.add_patch(Rectangle((x_low, 90), preferred_width, 10, facecolor=blend_color(COLORS["white"], COLORS["good"], 0.12), edgecolor="none", zorder=0))
            ax.text(x_low + 0.3, 98.2, "preferred zone", fontsize=10.2, color=COLORS["good"], fontweight="bold")
        seen_labels: set[str] = set()
        for row in tradeoff_rows:
            family = row.family
            label = display_family(family)
            x_value = num(row.poison_score_percent)
            y_value = num(row.downstream_score_percent)
            ax.scatter(
                x_value,
                y_value,
                s=155,
                color=color_map.get(family, family_color(family)),
                marker=marker_map.get(family, "o"),
                edgecolors=COLORS["white"],
                linewidths=1.6,
                label=label if label not in seen_labels else None,
                zorder=5,
            )
            seen_labels.add(label)
            if family == "vaccine_poison_ratio":
                short = f"V p={row.poison_ratio}"
            elif family == "antidote_poison_ratio":
                short = f"A p={row.poison_ratio}"
            elif family == "antidote_dense_ratio":
                short = f"A d={row.dense_ratio}"
            else:
                short = f"SFT p={row.poison_ratio}"
            dx, dy = label_offsets.get(row_key(row), (8, 8))
            ax.annotate(
                short,
                (x_value, y_value),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.8,
                color=COLORS["ink"],
                bbox=dict(boxstyle="round,pad=0.16", facecolor=COLORS["white"], edgecolor="none", alpha=0.92),
            )
        ax.set_xlabel("Poison moderation score (%) - lower is better", color=COLORS["ink"], fontsize=12)
        ax.set_ylabel("Downstream accuracy (%) - higher is better", color=COLORS["ink"], fontsize=12)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(0, 101)
        add_panel_title(ax, "Safety vs. Utility Tradeoff", "Dense-ratio follow-ups dominate the antidote frontier, while the 1.0 poison-ratio run visibly breaks utility.")
        ax.legend(frameon=False, fontsize=10.2, loc="lower left")
        save(fig, PLOTS_DIR / "safety_vs_utility.png")

    vaccine_alignment_rows = sorted([row for row in presentation_runs if row.family == "vaccine_alignment"], key=lambda row: num(row.rho))
    vaccine_poison_rows = sorted([row for row in presentation_runs if row.family == "vaccine_poison_ratio"], key=lambda row: num(row.poison_ratio))
    antidote_poison_rows = sorted([row for row in presentation_runs if row.family == "antidote_poison_ratio"], key=lambda row: num(row.poison_ratio))
    antidote_dense_rows = sorted([row for row in presentation_runs if row.family == "antidote_dense_ratio"], key=lambda row: num(row.dense_ratio))
    if vaccine_alignment_rows and vaccine_poison_rows and antidote_poison_rows and antidote_dense_rows:
        fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.8), facecolor=COLORS["white"])
        fig.subplots_adjust(hspace=0.20, wspace=0.12)
        draw_metric_panel(
            axes[0, 0],
            "Vaccine alignment",
            "Completed rho sweep",
            [f"rho {row.rho}" for row in vaccine_alignment_rows],
            [("Poison moderation", [num(row.poison_score_percent) for row in vaccine_alignment_rows], False, COLORS["vaccine"])],
        )
        draw_metric_panel(
            axes[0, 1],
            "Vaccine poison ratio",
            "Safe downstream reruns",
            [f"p {row.poison_ratio}" for row in vaccine_poison_rows],
            [
                ("Poison moderation", [num(row.poison_score_percent) for row in vaccine_poison_rows], False, COLORS["vaccine"]),
                ("SST2 accuracy", [num(row.downstream_score_percent) for row in vaccine_poison_rows], True, COLORS["good"]),
            ],
        )
        draw_metric_panel(
            axes[1, 0],
            "Antidote poison ratio",
            "All intended settings completed",
            [f"p {row.poison_ratio}" for row in antidote_poison_rows],
            [
                ("Poison moderation", [num(row.poison_score_percent) for row in antidote_poison_rows], False, COLORS["antidote"]),
                ("SST2 accuracy", [num(row.downstream_score_percent) for row in antidote_poison_rows], True, COLORS["good"]),
            ],
        )
        draw_metric_panel(
            axes[1, 1],
            "Antidote dense ratio",
            "poison_ratio fixed at 0.2",
            [f"d {row.dense_ratio}" for row in antidote_dense_rows],
            [
                ("Poison moderation", [num(row.poison_score_percent) for row in antidote_dense_rows], False, COLORS["antidote"]),
                ("SST2 accuracy", [num(row.downstream_score_percent) for row in antidote_dense_rows], True, COLORS["good"]),
            ],
        )
        fig.suptitle("Canonical Results Matrix", x=0.03, y=0.985, ha="left", fontsize=22, fontweight="bold", color=COLORS["ink"])
        fig.text(0.03, 0.95, "An exhaustive, presentation-friendly matrix of the completed canonical settings. Darker cells indicate better outcomes for that metric.", fontsize=11.2, color=COLORS["muted"])
        save(fig, PLOTS_DIR / "canonical_metric_matrix.png")


def build_tables(
    successful_runs: list[RunRecord],
    presentation_runs: list[RunRecord],
    superseded_runs: list[RunRecord],
) -> None:
    if presentation_runs:
        unique_rows: dict[str, RunRecord] = {}
        used_in: defaultdict[str, list[str]] = defaultdict(list)
        for row in presentation_runs:
            key = row.job_id or row.log_path
            unique_rows.setdefault(key, row)
            used_in[key].append(display_family(row.family))

        table_rows = []
        for key, row in sorted(
            unique_rows.items(),
            key=lambda item: (parse_datetime(item[1].end_time) or datetime.min, item[1].job_id),
        ):
            roles = ", ".join(dict.fromkeys(used_in[key]))
            table_rows.append(
                [
                    row.job_id,
                    pretty_date(row.end_time),
                    wrap_text(roles, 26),
                    wrap_text(build_setting_label(row), 22),
                    fmt_metric(row.poison_score_percent),
                    fmt_metric(row.downstream_score_percent),
                ]
            )

        build_table_figure(
            "Unique Completed Runs Feeding The Slides",
            "Shared baselines are listed once here even when they are reused across multiple figures.",
            table_rows,
            ["Job ID", "Completed", "Used In", "Setting", "Poison", "Downstream"],
            [0.09, 0.16, 0.24, 0.24, 0.12, 0.15],
            (COLORS["panel"], COLORS["panel_alt"]),
            (15.8, 6.8),
            TABLES_DIR / "presentation_runs_table.png",
            font_size=10.0,
        )

    if successful_runs:
        catalog_rows = []
        for row in sorted(
            successful_runs,
            key=lambda item: (parse_datetime(item.end_time) or datetime.min, item.job_id),
            reverse=True,
        ):
            catalog_rows.append(
                [
                    pretty_date(row.end_time),
                    wrap_text(display_family(row.family), 18),
                    row.job_id,
                    wrap_text(build_setting_label(row), 22),
                    fmt_metric(row.poison_score_percent),
                    fmt_metric(row.downstream_score_percent),
                ]
            )
        catalog_height = max(8.6, 0.36 * len(catalog_rows) + 2.2)
        build_table_figure(
            "Completed Run Catalog",
            "Every success-marker completion found in the codebase audit, sorted by most recent finish time.",
            catalog_rows,
            ["Completed", "Family", "Job ID", "Setting", "Poison", "Downstream"],
            [0.16, 0.20, 0.09, 0.27, 0.12, 0.13],
            (COLORS["panel"], COLORS["panel_alt"]),
            (16.2, catalog_height),
            TABLES_DIR / "completed_run_catalog.png",
            font_size=9.5,
            table_bbox=(0.03, 0.03, 0.94, 0.82),
        )

    if superseded_runs:
        rows = [[row.job_id, row.job_name, display_family(row.family), display_status(row.status), wrap_text(row.note or "-", 30)] for row in superseded_runs]
        build_table_figure(
            "Superseded Or Non-Success Attempts",
            "Short-lived launcher attempts found in the audit. These should not be presented as completed experiment results.",
            rows,
            ["Job ID", "Job Name", "Family", "Status", "Why Not Counted"],
            [0.08, 0.15, 0.20, 0.18, 0.33],
            ("#fbefe9", "#fbe4da"),
            (15.6, 6.4),
            TABLES_DIR / "superseded_attempts_table.png",
            font_size=10.0,
        )

    if successful_runs:
        counts: dict[str, int] = {}
        for row in successful_runs:
            counts[row.family] = counts.get(row.family, 0) + 1
        labels = sorted(counts.keys(), key=display_family)
        values = [counts[label] for label in labels]
        fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor=COLORS["white"])
        add_card(ax)
        style_axis(ax)
        ax.bar(np.arange(len(labels)), values, color=[family_color(label) for label in labels], width=0.62)
        ax.set_xticks(np.arange(len(labels)), [wrap_text(display_family(label), 14) for label in labels])
        ax.set_ylabel("Successful log count", color=COLORS["ink"], fontsize=12)
        ax.set_title("Successful Runs Detected Across The Codebase", loc="left", fontsize=18, fontweight="bold", color=COLORS["ink"], pad=18)
        ax.text(0.0, 1.02, "This view counts success-marker completions only; ambiguous epilog-only runs are excluded.", transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], va="bottom")
        save(fig, TABLES_DIR / "successful_run_counts.png")


def build_cards(all_runs: list[RunRecord], successful_runs: list[RunRecord], presentation_runs: list[RunRecord], superseded_runs: list[RunRecord]) -> None:
    latest_success = max((parse_datetime(row.end_time) for row in successful_runs if parse_datetime(row.end_time)), default=None)
    vaccine_best = min(
        (row for row in presentation_runs if row.family == "vaccine_alignment" and row.poison_score_percent),
        key=lambda row: num(row.poison_score_percent),
        default=None,
    )
    antidote_best = max(
        (row for row in presentation_runs if row.family == "antidote_dense_ratio" and row.downstream_score_percent),
        key=lambda row: num(row.downstream_score_percent),
        default=None,
    )
    cards = [
        ("Candidate logs", str(len(all_runs)), "vaccine- and antidote-related .out logs scanned across the repo"),
        ("Success-marker completions", str(len(successful_runs)), "runs counted as fully completed experiment outputs"),
        ("Slide inputs", str(len(presentation_runs)), "canonical plot rows used in the presentation figures"),
        ("Superseded attempts", str(len(superseded_runs)), "launcher runs excluded from the completed-results story"),
        (
            "Latest finish",
            latest_success.strftime("%b %d, %Y") if latest_success else "-",
            "the rerun wave closed on April 18, 2026 with jobs 4913721 through 4913726",
        ),
        (
            "Headline tradeoff",
            f"{antidote_best.downstream_score_percent}% @ d={antidote_best.dense_ratio}" if antidote_best else "-",
            f"best dense-ratio downstream result; best vaccine safety is {fmt_metric(vaccine_best.poison_score_percent) if vaccine_best else '-'} at rho={vaccine_best.rho if vaccine_best else '-'}",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.3), facecolor=COLORS["white"])
    fig.subplots_adjust(hspace=0.2, wspace=0.16)
    fig.suptitle("Presentation Summary Cards", x=0.04, y=0.98, ha="left", fontsize=22, fontweight="bold", color=COLORS["ink"])
    for ax, (title, value, note) in zip(axes.flatten(), cards):
        add_card(ax, COLORS["panel"] if title not in {"Latest finish", "Headline tradeoff"} else COLORS["panel_alt"])
        ax.axis("off")
        ax.text(0.06, 0.82, title.upper(), fontsize=10.5, color=COLORS["muted"], fontweight="bold", transform=ax.transAxes)
        ax.text(0.06, 0.50, value, fontsize=24, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
        ax.text(0.06, 0.18, wrap_text(note, 28), fontsize=11, color=COLORS["ink"], transform=ax.transAxes, va="bottom")
    save(fig, CARDS_DIR / "summary_cards.png")


def classify_log_bucket(row: RunRecord) -> Path:
    if row.family.startswith("vaccine"):
        return VACCINE_LOG_DIR
    if row.family.startswith("antidote"):
        return ANTIDOTE_LOG_DIR
    lower_path = row.log_path.lower()
    if "vaccine" in lower_path:
        return VACCINE_LOG_DIR
    if "antidote" in lower_path:
        return ANTIDOTE_LOG_DIR
    return VACCINE_LOG_DIR


def stage_success_logs(successful_runs: list[RunRecord]) -> None:
    for directory in (VACCINE_LOG_DIR, ANTIDOTE_LOG_DIR):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    for row in successful_runs:
        destination_dir = classify_log_bucket(row)
        for source in (Path(row.log_path), Path(row.log_path).with_suffix(".err")):
            if not source.exists():
                continue
            destination = destination_dir / source.name
            shutil.copy2(source, destination)


def build_reports(
    all_runs: list[RunRecord],
    successful_runs: list[RunRecord],
    presentation_runs: list[RunRecord],
    superseded_runs: list[RunRecord],
) -> None:
    completed_counts: dict[str, int] = {}
    for row in successful_runs:
        completed_counts[row.family] = completed_counts.get(row.family, 0) + 1
    latest_completion = max((parse_datetime(row.end_time) for row in successful_runs if parse_datetime(row.end_time)), default=None)

    lines = [
        "# Final Presentation Run Audit",
        "",
        f"- Total candidate logs scanned: {len(all_runs)}",
        f"- Successful completed logs: {len(successful_runs)}",
        f"- Canonical presentation runs: {len(presentation_runs)}",
        f"- Non-success or superseded attempts: {len(superseded_runs)}",
        "",
        "## Completed Counts By Family",
        "",
    ]
    for family, count in sorted(completed_counts.items()):
        lines.append(f"- `{family}`: {count}")

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            f"- The latest rerun wave finished on `{latest_completion.strftime('%B %d, %Y')}` under job IDs `4913721` through `4913726`." if latest_completion else "- The latest rerun wave completed under job IDs `4913721` through `4913726`.",
            "- Earlier launcher attempts with job IDs `4900733`, `4900734`, `4900735`, `4900736`, `4900774`, and `4900775` should not be treated as completed experiment results.",
            "- The plots in this folder focus only on completed runs with comparable numeric metrics.",
            "",
        ]
    )
    (REPORTS_DIR / "run_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    vaccine_alignment = {row.rho: row.poison_score_percent for row in presentation_runs if row.family == "vaccine_alignment"}
    vaccine_poison = sorted([row for row in presentation_runs if row.family == "vaccine_poison_ratio"], key=lambda row: num(row.poison_ratio))
    antidote_poison = sorted([row for row in presentation_runs if row.family == "antidote_poison_ratio"], key=lambda row: num(row.poison_ratio))
    dense = sorted([row for row in presentation_runs if row.family == "antidote_dense_ratio"], key=lambda row: num(row.dense_ratio))

    brief = [
        "# Presentation Brief",
        "",
        "## What Completed",
        "",
        "- Vaccine alignment runs are complete for `rho=2`, `rho=5`, and `rho=10`.",
        "- Vaccine downstream baseline reruns are complete for poison ratios `0.1`, `0.5`, and `0.8` at `rho=2`.",
        "- Antidote poison-ratio sweep is complete for `0.1`, `0.2`, `0.5`, `0.8`, and `1.0` with `dense_ratio=0.1`.",
        "- Antidote dense-ratio follow-ups are complete for `0.05`, `0.1`, and `0.2` at `poison_ratio=0.2`.",
        "",
        "## Headline Results",
        "",
        f"- Vaccine alignment poison moderation improves from `{vaccine_alignment.get('2', '-')}` at `rho=2` to `{vaccine_alignment.get('10', '-')}` at `rho=10`.",
    ]
    if vaccine_poison:
        brief.append(
            f"- Vaccine downstream baseline keeps SST2 accuracy above `93%` across the completed poison-ratio reruns, with poison moderation scores clustered around `{min(num(row.poison_score_percent) for row in vaccine_poison):.1f}` to `{max(num(row.poison_score_percent) for row in vaccine_poison):.1f}`."
        )
    if antidote_poison:
        brief.append(
            f"- Antidote maintains strong SST2 accuracy through poison ratio `0.5` and then drops to `{antidote_poison[-1].downstream_score_percent}%` at poison ratio `1.0`."
        )
    if dense:
        best_dense = max(dense, key=lambda row: num(row.downstream_score_percent))
        safest_dense = min(dense, key=lambda row: num(row.poison_score_percent))
        brief.append(
            f"- In the dense-ratio sweep, the best downstream accuracy is `{best_dense.downstream_score_percent}%` at `dense_ratio={best_dense.dense_ratio}`, while the strongest safety improvement is `{safest_dense.poison_score_percent}` at `dense_ratio={safest_dense.dense_ratio}`."
        )

    brief.extend(
        [
            "",
            "## Recommended Slide Order",
            "",
            "1. `cards/summary_cards.png`",
            "2. `plots/overview_dashboard.png`",
            "3. `plots/safety_vs_utility.png`",
            "4. `plots/canonical_metric_matrix.png`",
            "5. `plots/vaccine_alignment.png`",
            "6. `plots/vaccine_poison_ratio.png`",
            "7. `plots/antidote_poison_ratio.png`",
            "8. `plots/antidote_dense_ratio.png`",
            "9. `plots/status_breakdown.png`",
            "10. `plots/completion_timeline.png`",
            "11. `tables/presentation_runs_table.png`",
            "12. `tables/completed_run_catalog.png`",
            "13. `tables/superseded_attempts_table.png`",
            "",
        ]
    )
    (REPORTS_DIR / "presentation_brief.md").write_text("\n".join(brief) + "\n", encoding="utf-8")

    index_lines = [
        "# Completed Run Index",
        "",
        "| Completed | Family | Job ID | Setting | Poison | Downstream | Log |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(successful_runs, key=lambda item: (parse_datetime(item.end_time) or datetime.min, item.job_id), reverse=True):
        index_lines.append(
            f"| {pretty_date(row.end_time)} | {display_family(row.family)} | {row.job_id} | {build_setting_label(row)} | {fmt_metric(row.poison_score_percent)} | {fmt_metric(row.downstream_score_percent)} | `{Path(row.log_path).name}` |"
        )
    (REPORTS_DIR / "completed_run_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    text = """# Final Presentation Artifacts

This folder is generated from completed vaccine and antidote runs found in the repository.

## Contents

- `data/all_detected_runs.csv`: codebase-wide audit of candidate log files.
- `data/completed_success_runs.csv`: successful completed runs only.
- `data/presentation_runs.csv`: deduplicated canonical runs used in the plots.
- `cards/`: summary cards for title or takeaway slides.
- `plots/`: presentation-ready figures.
- `tables/`: slide-friendly tables and run catalogs.
- `vaccine/`: staged `.out` and `.err` files for successful vaccine-side runs.
- `antidote/`: staged `.out` and `.err` files for successful antidote-side runs.
- `reports/`: markdown summaries for speaker notes and handoff.

## Refresh

Run:

```bash
python /home/hice1/rswaminathan38/scratch/Research/final_presentation/generate_artifacts.py
```
"""
    (PRESENTATION_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    for directory in (DATA_DIR, PLOTS_DIR, TABLES_DIR, CARDS_DIR, REPORTS_DIR, VACCINE_LOG_DIR, ANTIDOTE_LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    all_runs = [build_run_record(path) for path in candidate_logs()]
    successful_runs = [row for row in all_runs if row.status == "completed_success"]

    vaccine_alignment = sorted(
        [row for row in successful_runs if row.family == "vaccine_alignment" and row.poison_score_percent],
        key=lambda row: num(row.rho),
    )
    vaccine_poison_ratio = sorted(
        [row for row in successful_runs if row.family == "vaccine_poison_ratio" and row.poison_score_percent and row.downstream_score_percent],
        key=lambda row: num(row.poison_ratio),
    )
    antidote_poison_ratio = dedupe_by_setting(
        [row for row in successful_runs if row.family == "antidote_poison_ratio" and row.poison_score_percent and row.downstream_score_percent],
        ("dense_ratio", "poison_ratio"),
    )
    antidote_poison_ratio = sorted(antidote_poison_ratio, key=lambda row: num(row.poison_ratio))
    antidote_stage1 = sorted(
        [row for row in successful_runs if row.family == "antidote_stage1_sft" and row.poison_score_percent and row.downstream_score_percent],
        key=lambda row: num(row.poison_ratio),
    )

    dense_rows: list[RunRecord] = []
    for row in successful_runs:
        if row.family not in {"antidote_poison_ratio", "antidote_dense_ratio"}:
            continue
        if not row.poison_score_percent or not row.downstream_score_percent or row.poison_ratio != "0.2":
            continue
        if row.family == "antidote_poison_ratio":
            dense_rows.append(replace(row, family="antidote_dense_ratio"))
        else:
            dense_rows.append(row)
    antidote_dense_ratio = dedupe_by_setting(dense_rows, ("dense_ratio", "poison_ratio"))
    antidote_dense_ratio = sorted([row for row in antidote_dense_ratio if row.dense_ratio], key=lambda row: num(row.dense_ratio))

    presentation_runs = vaccine_alignment + vaccine_poison_ratio + antidote_poison_ratio + antidote_dense_ratio + antidote_stage1
    presentation_runs = dedupe_by_setting(
        presentation_runs,
        ("family", "rho", "poison_ratio", "dense_ratio", "poison_output", "sst2_output"),
    )

    superseded_runs = [
        row
        for row in all_runs
        if row.family in {"vaccine_poison_ratio_attempt", "antidote_dense_ratio_attempt", "vaccine_alignment"}
        and row.status != "completed_success"
    ]
    superseded_runs = sorted(superseded_runs, key=lambda row: (row.job_id or "0", row.log_path))

    write_csv(DATA_DIR / "all_detected_runs.csv", [asdict(row) for row in all_runs])
    write_csv(DATA_DIR / "completed_success_runs.csv", [asdict(row) for row in successful_runs])
    write_csv(DATA_DIR / "presentation_runs.csv", [asdict(row) for row in presentation_runs])

    build_plots(vaccine_alignment, vaccine_poison_ratio, antidote_poison_ratio, antidote_dense_ratio, antidote_stage1)
    build_meta_plots(all_runs, successful_runs, presentation_runs)
    build_tables(successful_runs, presentation_runs, superseded_runs)
    build_cards(all_runs, successful_runs, presentation_runs, superseded_runs)
    stage_success_logs(successful_runs)
    build_reports(all_runs, successful_runs, presentation_runs, superseded_runs)
    write_readme()


if __name__ == "__main__":
    main()
