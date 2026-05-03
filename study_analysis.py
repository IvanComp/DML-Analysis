from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu as scipy_mannwhitneyu
except Exception:  # pragma: no cover - optional dependency
    scipy_mannwhitneyu = None


MANIFEST_FILENAME = "manifest.csv"
RUN_SUMMARY_FILENAME = "run_summary.csv"
AGGREGATE_STATS_FILENAME = "aggregate_metrics.csv"
PAIRWISE_STATS_FILENAME = "pairwise_statistics.csv"
LATEST_STUDY_FILENAME = "latest_study.txt"

STATIC_RUN_COLUMNS = {
    "Round",
    "Clients",
    "Epochs",
    "Model",
    "Dataset",
    "Data Distr. (Alpha)",
    "Learning Type",
}

GROUP_COLUMNS = [
    "dataset",
    "model",
    "effective_num_clients",
    "effective_clients_per_round",
    "rounds",
    "epochs",
    "data_distr",
    "comparison_profile",
    "batch_size",
    "lr",
]

DISPLAY_GROUP_COLUMNS = [
    "dataset",
    "model",
    "effective_num_clients",
    "effective_clients_per_round",
]

ROUND_HISTORY_METADATA_COLUMNS = {
    "signature",
    "run_id",
    "status",
    "returncode",
    "started_at",
    "ended_at",
    "elapsed_sec",
    "dataset",
    "model",
    "learning_type",
    "learning_type_display",
    "requested_num_clients",
    "effective_num_clients",
    "requested_clients_per_round",
    "effective_clients_per_round",
    "rounds",
    "epochs",
    "repetition",
    "repetitions",
    "seed",
    "data_distr",
    "comparison_profile",
    "batch_size",
    "lr",
    "timeout_seconds",
    "baseline",
    "command",
    "log_path",
    "archived_csv_path",
    "archived_plot_paths",
}

PREFERENCE_MAP = {
    "Final_Accuracy": "higher",
    "Best_Accuracy": "higher",
    "Mean_Accuracy": "higher",
    "Mean_TFLOPS": "higher",
    "Mean_Duration_Sec": "lower",
    "Mean_Avg_Training_Time_Sec": "lower",
    "Mean_Avg_Communication_Time_Sec": "lower",
}


def load_latest_study_dir(output_root: Path | str = "study_runs") -> Path:
    output_root = Path(output_root)
    latest_pointer = output_root / LATEST_STUDY_FILENAME
    if not latest_pointer.exists():
        raise FileNotFoundError(f"Latest study pointer not found: {latest_pointer}")
    target = latest_pointer.read_text(encoding="utf-8").strip()
    if not target:
        raise FileNotFoundError(f"Latest study pointer is empty: {latest_pointer}")
    return Path(target)


def update_latest_study_pointer(output_root: Path | str, study_dir: Path | str) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / LATEST_STUDY_FILENAME).write_text(str(Path(study_dir)), encoding="utf-8")


def load_manifest(study_dir: Path | str) -> pd.DataFrame:
    study_dir = Path(study_dir)
    manifest_path = study_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_path)


def format_group_label(group_values: Dict[str, object]) -> str:
    dataset = str(group_values.get("dataset", "")).upper()
    model = str(group_values.get("model", "")).upper()
    n_value = group_values.get("effective_num_clients", "")
    k_value = group_values.get("effective_clients_per_round", "")
    rounds = group_values.get("rounds", "")
    epochs = group_values.get("epochs", "")
    alpha = group_values.get("data_distr", "")
    return (
        f"{dataset} | {model} | N={n_value} | k={k_value} | "
        f"rounds={rounds} | epochs={epochs} | alpha={alpha}"
    )


def numeric_round_metrics(df: pd.DataFrame) -> List[str]:
    metrics: List[str] = []
    for column in df.columns:
        if column in STATIC_RUN_COLUMNS:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            metrics.append(column)
    return metrics


def summarize_run(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    summary: Dict[str, float] = {"Completed_Rounds": float(len(df))}
    for metric in numeric_round_metrics(df):
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue
        summary[f"Mean_{metric}"] = float(series.mean())
        if metric == "Accuracy":
            summary["Final_Accuracy"] = float(series.iloc[-1])
            summary["Best_Accuracy"] = float(series.max())
    return summary


def build_run_summary(study_dir: Path | str) -> pd.DataFrame:
    manifest_df = load_manifest(study_dir)
    if manifest_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for record in manifest_df.to_dict(orient="records"):
        if record.get("status") != "completed":
            continue
        csv_path_value = record.get("archived_csv_path", "")
        if pd.isna(csv_path_value) or not str(csv_path_value).strip():
            continue
        csv_path = Path(str(csv_path_value))
        if not csv_path.exists():
            continue
        summary = summarize_run(csv_path)
        merged = dict(record)
        merged.update(summary)
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_round_history(study_dir: Path | str) -> pd.DataFrame:
    manifest_df = load_manifest(study_dir)
    if manifest_df.empty:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for record in manifest_df.to_dict(orient="records"):
        if record.get("status") != "completed":
            continue
        csv_path_value = record.get("archived_csv_path", "")
        if pd.isna(csv_path_value) or not str(csv_path_value).strip():
            continue
        csv_path = Path(str(csv_path_value))
        if not csv_path.exists():
            continue

        round_df = pd.read_csv(csv_path)
        for key, value in record.items():
            round_df[key] = value
        frames.append(round_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summary_metric_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.columns
        if column.startswith("Mean_") or column in {"Final_Accuracy", "Best_Accuracy"}
    ]


def round_history_metric_columns(df: pd.DataFrame) -> List[str]:
    metric_columns: List[str] = []
    excluded = set(STATIC_RUN_COLUMNS) | ROUND_HISTORY_METADATA_COLUMNS
    for column in df.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            metric_columns.append(column)
    return metric_columns


def effect_magnitude(a12_value: float) -> str:
    distance = abs(float(a12_value) - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def preference_for_metric(metric_name: str) -> str:
    return PREFERENCE_MAP.get(metric_name, "context")


def mann_whitney_u_with_a12(sample_a: Sequence[float], sample_b: Sequence[float]) -> Tuple[float, float, float]:
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1 = len(a)
    n2 = len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")

    if scipy_mannwhitneyu is not None:
        result = scipy_mannwhitneyu(a, b, alternative="two-sided", method="auto")
        u1 = float(result.statistic)
        p_value = float(result.pvalue)
        return u1, p_value, float(u1 / (n1 * n2))

    combined = [(value, 0) for value in a] + [(value, 1) for value in b]
    combined.sort(key=lambda item: item[0])

    ranks: List[float] = [0.0] * (n1 + n2)
    tie_sizes: List[int] = []
    position = 0
    current_rank = 1
    while position < len(combined):
        next_position = position
        while next_position < len(combined) and combined[next_position][0] == combined[position][0]:
            next_position += 1
        tie_size = next_position - position
        tie_sizes.append(tie_size)
        average_rank = (current_rank + current_rank + tie_size - 1) / 2.0
        for index in range(position, next_position):
            ranks[index] = average_rank
        current_rank += tie_size
        position = next_position

    rank_sum_a = sum(rank for rank, (_, label) in zip(ranks, combined) if label == 0)
    u1 = rank_sum_a - (n1 * (n1 + 1) / 2.0)
    u2 = n1 * n2 - u1

    total = n1 + n2
    tie_correction = 1.0
    denominator = total ** 3 - total
    if denominator > 0:
        tie_correction -= sum(tie ** 3 - tie for tie in tie_sizes) / denominator
    variance = n1 * n2 * (total + 1) / 12.0 * tie_correction
    if variance <= 0:
        p_value = 1.0
    else:
        mean_u = n1 * n2 / 2.0
        z_score = (abs(u1 - mean_u) - 0.5) / math.sqrt(variance)
        p_value = math.erfc(abs(z_score) / math.sqrt(2.0))

    return float(u1), float(p_value), float(u1 / (n1 * n2))


def build_aggregate_metrics(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    if run_summary_df.empty:
        return pd.DataFrame()

    metric_columns = summary_metric_columns(run_summary_df)
    rows: List[Dict[str, object]] = []
    for group_key, group_df in run_summary_df.groupby(GROUP_COLUMNS + ["learning_type_display"], dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_payload = dict(zip(GROUP_COLUMNS + ["learning_type_display"], group_key))
        for metric in metric_columns:
            series = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    **group_payload,
                    "metric": metric,
                    "n_runs": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
                    "median": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_statistics(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    if run_summary_df.empty:
        return pd.DataFrame()

    metric_columns = summary_metric_columns(run_summary_df)
    rows: List[Dict[str, object]] = []
    for group_key, group_df in run_summary_df.groupby(GROUP_COLUMNS, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_payload = dict(zip(GROUP_COLUMNS, group_key))
        approaches = sorted(group_df["learning_type_display"].dropna().unique())
        for metric in metric_columns:
            metric_df = group_df[["learning_type_display", metric]].copy()
            metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
            metric_df = metric_df.dropna()
            if metric_df.empty:
                continue
            for approach_a, approach_b in combinations(approaches, 2):
                sample_a = metric_df.loc[
                    metric_df["learning_type_display"] == approach_a, metric
                ].to_numpy(dtype=float)
                sample_b = metric_df.loc[
                    metric_df["learning_type_display"] == approach_b, metric
                ].to_numpy(dtype=float)
                if len(sample_a) == 0 or len(sample_b) == 0:
                    continue
                u_stat, p_value, a12_value = mann_whitney_u_with_a12(sample_a, sample_b)
                preference = preference_for_metric(metric)
                median_a = float(np.median(sample_a))
                median_b = float(np.median(sample_b))
                if preference == "higher":
                    winner = approach_a if median_a > median_b else approach_b if median_b > median_a else "tie"
                elif preference == "lower":
                    winner = approach_a if median_a < median_b else approach_b if median_b < median_a else "tie"
                else:
                    winner = ""
                rows.append(
                    {
                        **group_payload,
                        "metric": metric,
                        "approach_a": approach_a,
                        "approach_b": approach_b,
                        "n_a": int(len(sample_a)),
                        "n_b": int(len(sample_b)),
                        "mean_a": float(np.mean(sample_a)),
                        "mean_b": float(np.mean(sample_b)),
                        "median_a": median_a,
                        "median_b": median_b,
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "a12": a12_value,
                        "effect_size_magnitude": effect_magnitude(a12_value),
                        "preference": preference,
                        "winner": winner,
                    }
                )
    return pd.DataFrame(rows)


def refresh_study_outputs(study_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    study_dir = Path(study_dir)
    run_summary_df = build_run_summary(study_dir)
    aggregate_df = build_aggregate_metrics(run_summary_df)
    pairwise_df = build_pairwise_statistics(run_summary_df)

    run_summary_df.to_csv(study_dir / RUN_SUMMARY_FILENAME, index=False)
    aggregate_df.to_csv(study_dir / AGGREGATE_STATS_FILENAME, index=False)
    pairwise_df.to_csv(study_dir / PAIRWISE_STATS_FILENAME, index=False)
    return run_summary_df, aggregate_df, pairwise_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated Flower experiment runs and compute Mann-Whitney U / A12."
    )
    parser.add_argument("--study-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="study_runs")
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Load the latest study pointer from the output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_latest:
        study_dir = load_latest_study_dir(args.output_root)
    elif args.study_dir:
        study_dir = Path(args.study_dir)
    else:
        raise ValueError("Provide --study-dir or use --use-latest.")
    run_summary_df, aggregate_df, pairwise_df = refresh_study_outputs(study_dir)
    print(f"Study directory: {study_dir}")
    print(f"Completed runs summarized: {len(run_summary_df)}")
    print(f"Descriptive rows: {len(aggregate_df)}")
    print(f"Pairwise test rows: {len(pairwise_df)}")
    print(f"- {study_dir / RUN_SUMMARY_FILENAME}")
    print(f"- {study_dir / AGGREGATE_STATS_FILENAME}")
    print(f"- {study_dir / PAIRWISE_STATS_FILENAME}")


if __name__ == "__main__":
    main()
