"""TensorBoard scalars for training and benchmarks.

Event files are written only under
``/Volumes/Extreme SSD/nca_research/results/tb/``.
If that volume is missing, this module refuses to write anywhere else.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

VOLUME = Path("/Volumes/Extreme SSD")
RESULTS = VOLUME / "nca_research" / "results"
TB_ROOT = RESULTS / "tb"
PACKAGES = VOLUME / "nca_research" / "tb-packages"
TRAIN_DIR = TB_ROOT / "train"
FINEENVS_DIR = TB_ROOT / "train_fineenvs"
GRID_DIR = TB_ROOT / "train_grid"
BENCH_DIR = TB_ROOT / "benchmarks"

_writers: dict[str, object] = {}
_warned = False


class TensorBoardPathError(RuntimeError):
    """Raised when the external log directory cannot be used."""


def require_ssd() -> None:
    if not VOLUME.is_dir():
        raise TensorBoardPathError(
            "Extreme SSD is not mounted. Refusing to write TensorBoard events "
            "into the repo or the home directory."
        )
    resolved = str(VOLUME.resolve())
    if not resolved.startswith("/Volumes/Extreme SSD"):
        raise TensorBoardPathError(f"refusing TensorBoard volume {resolved}")


def _under_tb(path: Path) -> Path:
    require_ssd()
    root = Path(os.path.abspath(TB_ROOT))
    resolved = Path(os.path.abspath(path))
    if resolved != root and root not in resolved.parents:
        raise TensorBoardPathError(f"refusing to write outside {TB_ROOT}: {resolved}")
    if not str(resolved).startswith("/Volumes/Extreme SSD/"):
        raise TensorBoardPathError(f"refusing path off the SSD: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _ensure_packages() -> None:
    require_ssd()
    pkg = str(PACKAGES)
    if PACKAGES.is_dir() and pkg not in sys.path:
        # Append so an already-imported system numpy/torch stays first.
        sys.path.append(pkg)


def _summary_writer():
    _ensure_packages()
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


def _writer(run_dir: Path):
    run_dir = _under_tb(run_dir)
    key = str(run_dir)
    writer = _writers.get(key)
    if writer is None:
        writer = _summary_writer()(log_dir=key, flush_secs=5)
        _writers[key] = writer
    return writer


def close_writers() -> None:
    for writer in _writers.values():
        writer.close()
    _writers.clear()


atexit.register(close_writers)


def _ledger(run_dir: Path) -> Path:
    return run_dir / ".logged_rows.json"


def _load_seen(run_dir: Path) -> set[str]:
    path = _ledger(run_dir)
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in data}


def _save_seen(run_dir: Path, seen: set[str]) -> None:
    path = _ledger(run_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(seen)))
    tmp.replace(path)


def _finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fingerprint(row: dict) -> str:
    keys = (
        "step",
        "nca_loss",
        "flat_loss",
        "val_top1_nca",
        "val_top1_flat",
        "val_top1_always_terminal",
        "ms_per_step",
    )
    payload = {key: row.get(key) for key in keys if key in row}
    return json.dumps(payload, sort_keys=True, default=str)


def scalars_from_train_row(row: dict, batch: int | None = None) -> dict[str, float]:
    """Map a train-log row onto TensorBoard tags."""
    out: dict[str, float] = {}

    def put(tag: str, key: str) -> None:
        number = _finite(row.get(key))
        if number is not None:
            out[tag] = number

    put("loss/nca", "nca_loss")
    put("loss/flat", "flat_loss")
    put("val/top1_nca", "val_top1_nca")
    put("val/top1_flat", "val_top1_flat")
    put("val/top1_always_terminal", "val_top1_always_terminal")
    put("fineenvs/top1_nca", "fineenvs_val_top1_nca")
    put("fineenvs/top1_flat", "fineenvs_val_top1_flat")
    put("perf/ms_per_step", "ms_per_step")
    put("perf/samples_per_sec", "samples_per_sec")
    put("lr", "lr")
    ms = out.get("perf/ms_per_step")
    if "perf/samples_per_sec" not in out and batch and ms and ms > 0:
        out["perf/samples_per_sec"] = float(batch) / (ms / 1000.0)
    return out


def log_train_row(row: dict, batch: int | None = None, run_dir: Path | None = None) -> bool:
    """Write one training row. Returns False if the row was already logged.

    Failures are printed once and swallowed so a training loop can keep going.
    """
    global _warned
    lock = None
    try:
        run_dir = _under_tb(Path(run_dir) if run_dir else TRAIN_DIR)
        if "step" not in row:
            return False
        step = int(row["step"])
        scalars = scalars_from_train_row(row, batch=batch)
        if not scalars:
            return False
        fingerprint = _fingerprint(row)
        lock = (run_dir / ".write.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        seen = _load_seen(run_dir)
        if fingerprint in seen:
            return False
        writer = _writer(run_dir)
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, step)
        writer.flush()
        seen.add(fingerprint)
        _save_seen(run_dir, seen)
        return True
    except Exception as exc:  # noqa: BLE001 — logging must not stop training
        if not _warned:
            print(f"tensorboard log skipped: {type(exc).__name__}: {exc}", flush=True)
            _warned = True
        return False
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


def log_benchmarks(payload: dict, run_dir: Path | None = None) -> list[str]:
    """Write one scalar per benchmark score.

    ``payload`` is either ``{"bfcl": 0.4, "tau2": 0.2}`` or
    ``{"step": 1, "scores": {"bfcl": 0.4}}``.
    """
    if not isinstance(payload, dict):
        raise TypeError("benchmark payload must be a JSON object")
    body = payload
    step = 0
    if isinstance(body.get("scores"), dict):
        step = int(body.get("step") or 0)
        scores = body["scores"]
    else:
        scores = {key: value for key, value in body.items() if key != "step"}
        if "step" in body:
            step = int(body["step"])
    run_dir = _under_tb(Path(run_dir) if run_dir else BENCH_DIR)
    writer = _writer(run_dir)
    tags: list[str] = []
    for name, value in scores.items():
        number = _finite(value)
        if number is None:
            continue
        tag = str(name).strip().replace(" ", "_").replace("/", "_")
        if not tag:
            continue
        writer.add_scalar(tag, number, step)
        tags.append(tag)
    writer.flush()
    return tags


def _iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def detect_batch(stdout: Path | None = None) -> int | None:
    path = stdout or (RESULTS / "logs" / "text_nca_1d_stdout.log")
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    marker = "batch="
    last = text.rfind(marker)
    if last < 0:
        return None
    digits = []
    for char in text[last + len(marker) :]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits)) if digits else None


def backfill_jsonl(path: Path, batch: int | None = None, run_dir: Path | None = None) -> int:
    if not path.is_file():
        return 0
    wrote = 0
    for row in _iter_jsonl(path):
        if log_train_row(row, batch=batch, run_dir=run_dir):
            wrote += 1
    return wrote


def log_legacy_metrics(path: Path, run_dir: Path | None = None) -> int:
    """Log an older metrics JSON (epoch tail plus summary top-1) as its own run."""
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text())
    history = payload.get("nca_history_tail") or []
    val = payload.get("val_top1") or {}
    if not history and not val:
        return 0
    destination = Path(run_dir) if run_dir else GRID_DIR
    wrote = 0
    for row in history:
        mapped = {
            "step": int(row["epoch"]),
            "nca_loss": row.get("train_loss"),
            "val_top1_nca": row.get("val_top1"),
        }
        if log_train_row(mapped, run_dir=destination):
            wrote += 1
    step = int(history[-1]["epoch"]) if history else 0
    nca_train = payload.get("nca_train") or {}
    summary = {
        "step": step,
        "val_top1_nca": val.get("nca"),
        "val_top1_flat": val.get("flat"),
        "val_top1_always_terminal": val.get("majority"),
        "ms_per_step": nca_train.get("step_ms"),
        "samples_per_sec": nca_train.get("samples_per_sec"),
    }
    if log_train_row(summary, run_dir=destination):
        wrote += 1
    return wrote


def log_final_metrics(path: Path, run_dir: Path | None = None) -> int:
    """Log a finished 1d metrics file if it has final_val_top1."""
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text())
    final = payload.get("final_val_top1")
    if not isinstance(final, dict):
        return 0
    row = {
        "step": int(payload.get("step") or 0),
        "val_top1_nca": final.get("nca"),
        "val_top1_flat": final.get("flat"),
        "val_top1_always_terminal": final.get("always_terminal"),
    }
    bench = payload.get("microbench") or {}
    chosen = bench.get("chosen") or {}
    if chosen.get("step_ms") is not None:
        row["ms_per_step"] = chosen.get("step_ms")
    if chosen.get("samples_per_sec") is not None:
        row["samples_per_sec"] = chosen.get("samples_per_sec")
    return int(log_train_row(row, batch=payload.get("batch"), run_dir=run_dir))


def backfill_results(batch: int | None = None) -> dict[str, int]:
    """Pull the newest train log, then older metrics files, onto the SSD."""
    require_ssd()
    if batch is None:
        batch = detect_batch() or 1024
    counts = {
        "train_jsonl": backfill_jsonl(RESULTS / "logs" / "text_nca_1d.jsonl", batch=batch, run_dir=TRAIN_DIR),
        "train_metrics": log_final_metrics(RESULTS / "text_nca_1d_metrics.json", run_dir=TRAIN_DIR),
        "grid_metrics": log_legacy_metrics(RESULTS / "text_nca_metrics.json", run_dir=GRID_DIR),
    }
    _under_tb(BENCH_DIR)
    return counts


def follow_jsonl(path: Path, batch: int | None = None, poll: float = 5.0, run_dir: Path | None = None) -> None:
    """Keep reading a jsonl the current train process appends."""
    offset = 0
    partial = ""
    while True:
        if path.is_file():
            size = path.stat().st_size
            if size < offset:
                offset = 0
                partial = ""
            with path.open() as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                partial += chunk
                lines = partial.split("\n")
                partial = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    log_train_row(row, batch=batch, run_dir=run_dir)
        time.sleep(poll)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write NCA TensorBoard events on the external SSD")
    parser.add_argument("--backfill", action="store_true", help="import metrics already on the SSD")
    parser.add_argument("--follow", action="store_true", help="keep tailing the active train jsonl")
    parser.add_argument("--jsonl", type=Path, default=RESULTS / "logs" / "text_nca_1d.jsonl")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument(
        "--run",
        choices=("train", "fineenvs"),
        default="train",
        help="which event directory receives --follow rows",
    )
    args = parser.parse_args(argv)
    try:
        require_ssd()
    except TensorBoardPathError as exc:
        raise SystemExit(str(exc)) from exc
    batch = args.batch or detect_batch() or 1024
    run_dir = FINEENVS_DIR if args.run == "fineenvs" else TRAIN_DIR
    if args.run == "fineenvs" and args.jsonl == RESULTS / "logs" / "text_nca_1d.jsonl":
        args.jsonl = RESULTS / "logs" / "text_nca_1d_fineenvs.jsonl"
    if args.backfill or not args.follow:
        counts = backfill_results(batch=batch)
        print(json.dumps({"logdir": str(TB_ROOT), "backfill": counts, "batch": batch}), flush=True)
    if args.follow:
        print(f"following {args.jsonl} -> {run_dir}", flush=True)
        follow_jsonl(args.jsonl, batch=batch, poll=args.poll, run_dir=run_dir)


if __name__ == "__main__":
    main()
