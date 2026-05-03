from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from study_analysis import (
    AGGREGATE_STATS_FILENAME,
    MANIFEST_FILENAME,
    PAIRWISE_STATS_FILENAME,
    RUN_SUMMARY_FILENAME,
    refresh_study_outputs,
    update_latest_study_pointer,
)


DEFAULT_DATASETS = ["mnist"]
DEFAULT_MODELS = ["cnn"]
DEFAULT_LEARNING_TYPES = ["FL"]
DEFAULT_NUM_CLIENTS = [4]
DEFAULT_ROUNDS = 10
DEFAULT_EPOCHS = 1
DEFAULT_DATA_DISTR = 1.0
DEFAULT_REPETITIONS = 1
DEFAULT_SEED_BASE = 42
DEFAULT_OUTPUT_ROOT = "study_runs"
DEFAULT_COMPARISON_PROFILE = "fair"

IMAGE_DATASETS = {"cifar10", "stl10", "mnist", "oxfordpet"}
IMAGE_MODELS = {"cnn", "squeezenet", "shufflenet", "resnet", "vgg16"}
TABULAR_DATASETS = {"adult"}
TABULAR_MODELS = {"mlp"}
AUDIO_DATASETS = {"speechcommands"}
AUDIO_MODELS = {"m5"}

LEARNING_TYPE_ALIASES = {
    "CL": "CL",
    "CENTRALIZED": "CL",
    "CENTRALIZED_LEARNING": "CL",
    "FL": "FL",
    "FEDERATED": "FL",
    "FEDERATED_LEARNING": "FL",
    "SL": "SL",
    "SPLIT": "SL",
    "SPLIT_LEARNING": "SL",
    "SFLV1": "SFLV1",
    "SPLITFEDV1": "SFLV1",
    "SPLITFED_V1": "SFLV1",
    "SFL_V1": "SFLV1",
    "SFLV2": "SFLV2",
    "SPLITFEDV2": "SFLV2",
    "SPLITFED_V2": "SFLV2",
    "SFL_V2": "SFLV2",
    "CFL": "CFL",
    "CONTINUAL_FEDERATED": "CFL",
    "CONTINUAL_FEDERATED_LEARNING": "CFL",
    "CFSL": "CFSL",
    "CONTINUAL_FEDERATED_SPLIT": "CFSL",
    "CONTINUAL_FEDERATED_SPLIT_LEARNING": "CFSL",
}

LEARNING_TYPE_DISPLAY_NAMES = {
    "CL": "Centralized Learning",
    "FL": "Federated Learning",
    "SL": "Split Learning",
    "SFLV1": "SplitFed Learning v1",
    "SFLV2": "SplitFed Learning v2",
    "CFL": "Continual Federated Learning",
    "CFSL": "Continual Federated Split Learning",
}

LEARNING_TYPE_FILE_TAGS = {
    "FL": "fl",
    "SFLV1": "sflv1",
    "SFLV2": "sflv2",
    "CFL": "cfl",
    "CFSL": "cfsl",
}

PLOT_PREFIXES = [
    "accuracy",
    "time",
    "tflops",
    "training_time",
    "communication_time",
    "fit_phase_time",
]


@dataclass(frozen=True)
class ExperimentTask:
    dataset: str
    model: str
    learning_type: str
    learning_type_display: str
    requested_num_clients: int
    effective_num_clients: int
    requested_clients_per_round: int
    effective_clients_per_round: int
    rounds: int
    epochs: int
    data_distr: float
    comparison_profile: str
    repetition: int
    repetitions: int
    seed: int
    batch_size: Optional[int]
    lr: Optional[float]
    timeout_seconds: Optional[int]
    signature: str
    run_id: str


def ordered_unique(values: Iterable) -> List:
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def normalize_learning_type(raw_value: str) -> str:
    key = str(raw_value).strip().upper().replace("-", "_").replace(" ", "_")
    if key not in LEARNING_TYPE_ALIASES:
        available = ", ".join(sorted(ordered_unique(LEARNING_TYPE_ALIASES.values())))
        raise ValueError(f"Unknown learning type '{raw_value}'. Available: {available}")
    return LEARNING_TYPE_ALIASES[key]


def dataset_model_compatible(dataset: str, model: str) -> bool:
    dataset = dataset.lower()
    model = model.lower()
    if dataset in IMAGE_DATASETS:
        return model in IMAGE_MODELS
    if dataset in TABULAR_DATASETS:
        return model in TABULAR_MODELS
    if dataset in AUDIO_DATASETS:
        return model in AUDIO_MODELS
    return False


def stable_seed(seed_base: int, signature: str, repetition: int) -> int:
    digest = hashlib.sha256(f"{signature}|rep={repetition}".encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 1_000_000
    return int(seed_base) + offset


def effective_num_clients(learning_type: str, requested_num_clients: int) -> int:
    return 1 if learning_type == "CL" else max(1, int(requested_num_clients))


def effective_clients_per_round(
    learning_type: str,
    num_clients: int,
    requested_k: Optional[int],
    comparison_profile: str,
) -> int:
    if learning_type == "CL":
        return 1
    if learning_type == "SL":
        return num_clients if comparison_profile == "fair" else 1
    if requested_k is None:
        return num_clients
    return max(1, min(int(requested_k), num_clients))


def sanitize_slug(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum():
            cleaned.append(char.lower())
        elif char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def build_signature(
    *,
    dataset: str,
    model: str,
    learning_type: str,
    num_clients: int,
    clients_per_round: int,
    rounds: int,
    epochs: int,
    data_distr: float,
    comparison_profile: str,
    batch_size: Optional[int],
    lr: Optional[float],
    repetition: int,
) -> str:
    return "|".join(
        [
            dataset,
            model,
            learning_type,
            f"n={num_clients}",
            f"k={clients_per_round}",
            f"rounds={rounds}",
            f"epochs={epochs}",
            f"alpha={data_distr}",
            f"profile={comparison_profile}",
            f"batch_size={batch_size}",
            f"lr={lr}",
            f"rep={repetition}",
        ]
    )


def build_run_id(task: ExperimentTask) -> str:
    return sanitize_slug(
        (
            f"{task.dataset}_{task.model}_{task.learning_type}"
            f"_n{task.effective_num_clients}_k{task.effective_clients_per_round}"
            f"_r{task.rounds}_e{task.epochs}_rep{task.repetition:03d}"
        )
    )


def build_tasks(args: argparse.Namespace) -> Tuple[List[ExperimentTask], List[str]]:
    datasets = ordered_unique([d.lower() for d in args.dataset])
    models = ordered_unique([m.lower() for m in args.model])
    learning_types = ordered_unique([normalize_learning_type(v) for v in args.learning_type])
    num_clients_values = ordered_unique([max(1, int(v)) for v in args.num_clients])
    requested_k_values: List[Optional[int]]
    if args.clients_per_round:
        requested_k_values = ordered_unique([max(1, int(v)) for v in args.clients_per_round])
    else:
        requested_k_values = [None]

    tasks: List[ExperimentTask] = []
    skipped_messages: List[str] = []
    seen_signatures = set()

    for dataset in datasets:
        for model in models:
            if not dataset_model_compatible(dataset, model):
                skipped_messages.append(
                    f"Skipping invalid combination dataset={dataset} model={model}"
                )
                continue
            for learning_type in learning_types:
                for requested_num_clients in num_clients_values:
                    effective_n = effective_num_clients(learning_type, requested_num_clients)
                    k_values = requested_k_values
                    for requested_k in k_values:
                        effective_k = effective_clients_per_round(
                            learning_type=learning_type,
                            num_clients=effective_n,
                            requested_k=requested_k,
                            comparison_profile=args.comparison_profile,
                        )
                        for repetition in range(1, args.repetitions + 1):
                            signature = build_signature(
                                dataset=dataset,
                                model=model,
                                learning_type=learning_type,
                                num_clients=effective_n,
                                clients_per_round=effective_k,
                                rounds=args.rounds,
                                epochs=args.epochs,
                                data_distr=args.data_distr,
                                comparison_profile=args.comparison_profile,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                repetition=repetition,
                            )
                            if signature in seen_signatures:
                                continue
                            seen_signatures.add(signature)
                            seed = stable_seed(args.seed_base, signature, repetition)
                            task = ExperimentTask(
                                dataset=dataset,
                                model=model,
                                learning_type=learning_type,
                                learning_type_display=LEARNING_TYPE_DISPLAY_NAMES[learning_type],
                                requested_num_clients=requested_num_clients,
                                effective_num_clients=effective_n,
                                requested_clients_per_round=(
                                    effective_k if requested_k is None else int(requested_k)
                                ),
                                effective_clients_per_round=effective_k,
                                rounds=args.rounds,
                                epochs=args.epochs,
                                data_distr=float(args.data_distr),
                                comparison_profile=args.comparison_profile,
                                repetition=repetition,
                                repetitions=args.repetitions,
                                seed=seed,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                timeout_seconds=args.timeout_seconds,
                                signature=signature,
                                run_id="",
                            )
                            task = ExperimentTask(**{**asdict(task), "run_id": build_run_id(task)})
                            tasks.append(task)

    return tasks, skipped_messages


def load_manifest_records(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_manifest_records(manifest_path: Path, records: Sequence[Dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_unique(
        key for record in records for key in record.keys()
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def completed_lookup(records: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for record in records:
        if record.get("status") == "completed" and record.get("signature"):
            archived_csv_path = Path(record.get("archived_csv_path", ""))
            if archived_csv_path.exists():
                lookup[record["signature"]] = record
    return lookup


def baseline_mode_tag(learning_type: str) -> str:
    if learning_type == "CL":
        return "centralized"
    if learning_type == "SL":
        return "splitseq"
    return f"fedavg-{LEARNING_TYPE_FILE_TAGS[learning_type]}"


def expected_root_csv_path(task: ExperimentTask, workspace: Path) -> Path:
    mode_tag = baseline_mode_tag(task.learning_type)
    return workspace / "csv" / (
        f"baseline_{mode_tag}_{task.model}_{task.dataset}_{task.effective_num_clients}Clients.csv"
    )


def expected_root_plot_paths(task: ExperimentTask, workspace: Path) -> List[Path]:
    stem = expected_root_csv_path(task, workspace).stem
    if not stem.startswith("baseline_"):
        return []
    suffix = stem[len("baseline_") :]
    return [workspace / "results" / f"{prefix}_{suffix}.pdf" for prefix in PLOT_PREFIXES]


def build_command(task: ExperimentTask) -> List[str]:
    command = [
        sys.executable,
        "flower_baseline.py",
        "--dataset",
        task.dataset,
        "--model",
        task.model,
        "--baseline",
        "fedavg",
        "--rounds",
        str(task.rounds),
        "--epochs",
        str(task.epochs),
        "--num_clients",
        str(task.effective_num_clients),
        "--clients-per-round",
        str(task.effective_clients_per_round),
        "--data-distr",
        str(task.data_distr),
        "--learning-type",
        task.learning_type,
        "--comparison-profile",
        task.comparison_profile,
        "--seed",
        str(task.seed),
    ]
    if task.batch_size is not None:
        command.extend(["--batch_size", str(task.batch_size)])
    if task.lr is not None:
        command.extend(["--lr", str(task.lr)])
    return command


def write_study_metadata(study_dir: Path, args: argparse.Namespace, tasks: Sequence[ExperimentTask]) -> None:
    payload = {
        "study_name": study_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "arguments": {
            "dataset": args.dataset,
            "model": args.model,
            "learning_type": args.learning_type,
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "rounds": args.rounds,
            "epochs": args.epochs,
            "repetitions": args.repetitions,
            "data_distr": args.data_distr,
            "comparison_profile": args.comparison_profile,
            "seed_base": args.seed_base,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timeout_seconds": args.timeout_seconds,
        },
        "scheduled_runs": len(tasks),
    }
    config_path = study_dir / "study_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def archive_artifacts(task: ExperimentTask, workspace: Path, study_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    csv_output_dir = study_dir / "csv"
    plot_output_dir = study_dir / "results"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    archived_csv_path: Optional[Path] = None
    root_csv_path = expected_root_csv_path(task, workspace)
    if root_csv_path.exists():
        archived_csv_path = csv_output_dir / f"{task.run_id}.csv"
        shutil.copy2(root_csv_path, archived_csv_path)

    archived_plots: List[Path] = []
    for root_plot_path in expected_root_plot_paths(task, workspace):
        if not root_plot_path.exists():
            continue
        archived_plot_path = plot_output_dir / f"{task.run_id}__{root_plot_path.name}"
        shutil.copy2(root_plot_path, archived_plot_path)
        archived_plots.append(archived_plot_path)

    return archived_csv_path, archived_plots


def build_manifest_record(
    *,
    task: ExperimentTask,
    command: Sequence[str],
    log_path: Path,
    status: str,
    returncode: Optional[int],
    started_at: str,
    ended_at: str,
    elapsed_sec: float,
    archived_csv_path: Optional[Path],
    archived_plot_paths: Sequence[Path],
) -> Dict[str, object]:
    return {
        "signature": task.signature,
        "run_id": task.run_id,
        "status": status,
        "returncode": "" if returncode is None else int(returncode),
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(float(elapsed_sec), 6),
        "dataset": task.dataset,
        "model": task.model,
        "learning_type": task.learning_type,
        "learning_type_display": task.learning_type_display,
        "requested_num_clients": task.requested_num_clients,
        "effective_num_clients": task.effective_num_clients,
        "requested_clients_per_round": task.requested_clients_per_round,
        "effective_clients_per_round": task.effective_clients_per_round,
        "rounds": task.rounds,
        "epochs": task.epochs,
        "repetition": task.repetition,
        "repetitions": task.repetitions,
        "seed": task.seed,
        "data_distr": task.data_distr,
        "comparison_profile": task.comparison_profile,
        "batch_size": "" if task.batch_size is None else task.batch_size,
        "lr": "" if task.lr is None else task.lr,
        "timeout_seconds": "" if task.timeout_seconds is None else task.timeout_seconds,
        "baseline": "fedavg",
        "command": " ".join(command),
        "log_path": str(log_path),
        "archived_csv_path": "" if archived_csv_path is None else str(archived_csv_path),
        "archived_plot_paths": "|".join(str(path) for path in archived_plot_paths),
    }


def print_progress(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated repeated-run experiment runner for Flower studies."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="One or more datasets to run.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more models to run.",
    )
    parser.add_argument(
        "--learning-type",
        nargs="+",
        default=DEFAULT_LEARNING_TYPES,
        help="One or more learning types: CL, FL, SL, SFLV1, SFLV2, CFL, CFSL.",
    )
    parser.add_argument(
        "--num-clients",
        "--num_clients",
        dest="num_clients",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help="One or more client counts (N).",
    )
    parser.add_argument(
        "--clients-per-round",
        nargs="+",
        type=int,
        default=None,
        help="One or more clients-per-round values (k). Defaults to all clients.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--data-distr", type=float, default=DEFAULT_DATA_DISTR)
    parser.add_argument(
        "--comparison-profile",
        choices=["fair", "legacy"],
        default=DEFAULT_COMPARISON_PROFILE,
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run completed configurations instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    output_root = workspace / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.study_name:
        study_name = sanitize_slug(args.study_name)
    else:
        study_name = datetime.now().strftime("study_%Y%m%d_%H%M%S")
    study_dir = output_root / study_name
    (study_dir / "logs").mkdir(parents=True, exist_ok=True)
    (study_dir / "csv").mkdir(parents=True, exist_ok=True)
    (study_dir / "results").mkdir(parents=True, exist_ok=True)

    tasks, skipped_messages = build_tasks(args)
    write_study_metadata(study_dir, args, tasks)

    manifest_path = study_dir / MANIFEST_FILENAME
    existing_records = load_manifest_records(manifest_path)
    records_by_signature = {
        record.get("signature", ""): dict(record)
        for record in existing_records
        if record.get("signature")
    }
    completed_runs = completed_lookup(existing_records)

    total = len(tasks)
    if total == 0:
        print("No valid experiment combinations were scheduled.", flush=True)
        return

    for message in skipped_messages:
        print(message, flush=True)

    print(
        f"Study directory: {study_dir}\n"
        f"Scheduled runs: {total}\n"
        f"Outputs: {study_dir / 'csv'}",
        flush=True,
    )

    for index, task in enumerate(tasks, start=1):
        if not args.force and task.signature in completed_runs:
            archived_csv = completed_runs[task.signature].get("archived_csv_path", "")
            print_progress(
                index,
                total,
                f"skip {task.run_id} already completed -> {archived_csv}",
            )
            continue

        command = build_command(task)
        log_path = study_dir / "logs" / f"{task.run_id}.log"
        started_at = datetime.now().isoformat(timespec="seconds")
        run_start = time.perf_counter()
        status = "completed"
        returncode: Optional[int] = None
        archived_csv_path: Optional[Path] = None
        archived_plot_paths: List[Path] = []

        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"Command: {' '.join(command)}\n")
            log_handle.write(f"Started at: {started_at}\n\n")
            try:
                completed_process = subprocess.run(
                    command,
                    cwd=workspace,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=task.timeout_seconds,
                )
                returncode = completed_process.returncode
                if returncode != 0:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
                returncode = None
                log_handle.write("\nProcess timed out.\n")
            except KeyboardInterrupt:
                log_handle.write("\nInterrupted by user.\n")
                raise

        if status == "completed":
            archived_csv_path, archived_plot_paths = archive_artifacts(task, workspace, study_dir)
            if archived_csv_path is None or not archived_csv_path.exists():
                status = "missing_artifact"

        elapsed_sec = time.perf_counter() - run_start
        ended_at = datetime.now().isoformat(timespec="seconds")
        record = build_manifest_record(
            task=task,
            command=command,
            log_path=log_path,
            status=status,
            returncode=returncode,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_sec=elapsed_sec,
            archived_csv_path=archived_csv_path,
            archived_plot_paths=archived_plot_paths,
        )
        records_by_signature[task.signature] = record
        save_manifest_records(manifest_path, list(records_by_signature.values()))

        print_progress(
            index,
            total,
            (
                f"{status} {task.run_id} "
                f"dataset={task.dataset} model={task.model} algo={task.learning_type} "
                f"N={task.effective_num_clients} k={task.effective_clients_per_round} "
                f"rep={task.repetition}/{task.repetitions}"
            ),
        )

    refresh_study_outputs(study_dir)
    update_latest_study_pointer(output_root, study_dir)
    print(
        "\nStudy analysis refreshed:\n"
        f"- manifest: {manifest_path}\n"
        f"- run summary: {study_dir / RUN_SUMMARY_FILENAME}\n"
        f"- descriptive stats: {study_dir / AGGREGATE_STATS_FILENAME}\n"
        f"- pairwise stats: {study_dir / PAIRWISE_STATS_FILENAME}",
        flush=True,
    )


if __name__ == "__main__":
    main()
