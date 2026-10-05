"""Train a 1D trace NCA for 150k steps. Artifacts stay on the external SSD."""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import time

from nca_text.data import (
    HASH_DIM,
    SEQ_LEN,
    TRACE_STEPS,
    build_trace_obs,
    data_root,
    load_local_test,
    load_packed_splits,
    load_side_eval,
    prepare_env,
)

ROOT = data_root()
prepare_env(ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from nca_text.model import FlatPolicy, TextNCA

HIDDEN_C = 16
UPDATE_HIDDEN = 96
NCA_STEPS = 4
TOTAL_STEPS = 150_000
VAL_EVERY = 1000
CKPT_EVERY = 1000
SEED = 0
LR = 1e-3
WEIGHT_DECAY = 1e-2
BENCH_BATCHES = (128, 512, 1024, 2048)
WALL_BUDGET_H = 2.5
# Share of each class drawn from verifier-correct FineEnvs rows, once that file is staged.
FINEENVS_WITHIN_CLASS = 0.25
# Curated Hermes train counts (data/curated/stats.json). The coding mix is 37% respond;
# this prior is 52% terminal, which is the val and local shape.
CURATED_TRAIN_COUNTS = {
    "delegate_task": 33,
    "execute_code": 224,
    "memory": 28,
    "patch": 1670,
    "process": 672,
    "read_file": 8086,
    "respond": 2854,
    "search_files": 6194,
    "session_search": 82,
    "skill_view": 88,
    "terminal": 30259,
    "todo": 254,
    "write_file": 7231,
}

LAST_CKPT = ROOT / "checkpoints" / "text_nca_1d_mix13_last.pt"
BEST_CKPT = ROOT / "checkpoints" / "text_nca_1d_mix13_best.pt"
LOG_PATH = ROOT / "logs" / "text_nca_1d_mix13.jsonl"
METRICS_PATH = ROOT / "text_nca_1d_mix13_metrics.json"
# Curated-only weights at step 12496. A competing --fresh run writes text_nca_1d_last.pt.
SEED_CKPT = ROOT / "checkpoints" / "text_nca_1d_hermes13_step012496.pt"


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _chip() -> dict:
    def sysctl(name: str) -> str:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()

    mem = int(sysctl("hw.memsize"))
    return {
        "brand": sysctl("machdep.cpu.brand_string"),
        "arm64": sysctl("hw.optional.arm64"),
        "mem_bytes": mem,
        "mem_gib": round(mem / (1024**3), 2),
        "ncpu": int(sysctl("hw.ncpu")),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def _fineenvs_sampler(
    labels: np.ndarray,
    names: list[str],
    mask: np.ndarray,
):
    """Class buckets split into FineEnvs and the rest, plus the Hermes label prior.

    Returns None when no FineEnvs row was packed, so the uniform sampler stays.
    """
    if mask is None or int(np.asarray(mask).sum()) == 0:
        return None
    labels = np.asarray(labels)
    mask = np.asarray(mask).astype(bool)
    n_classes = len(names)
    base: list[np.ndarray] = []
    fine: list[np.ndarray] = []
    raw = np.zeros(n_classes, np.float64)
    for c, name in enumerate(names):
        idx = np.flatnonzero(labels == c)
        fine.append(idx[mask[idx]])
        base.append(idx[~mask[idx]])
        if idx.size:
            raw[c] = float(CURATED_TRAIN_COUNTS.get(name, 0))
    total = float(raw.sum())
    if total <= 0:
        return None
    return base, fine, raw / total


def _sample_batch(rng: np.random.Generator, batch: int, sampler) -> np.ndarray:
    base, fine, prior = sampler
    classes = rng.choice(len(prior), size=batch, p=prior)
    idx = np.empty(batch, np.int64)
    for c in range(len(prior)):
        take = np.flatnonzero(classes == c)
        if take.size == 0:
            continue
        pool_base, pool_fine = base[c], fine[c]
        if pool_fine.size == 0 or pool_base.size == 0:
            pool = pool_fine if pool_fine.size else pool_base
            idx[take] = rng.choice(pool, size=take.size, replace=True)
            continue
        from_fine = rng.random(take.size) < FINEENVS_WITHIN_CLASS
        n_fine = int(from_fine.sum())
        n_base = take.size - n_fine
        if n_fine:
            idx[take[from_fine]] = rng.choice(pool_fine, size=n_fine, replace=True)
        if n_base:
            idx[take[~from_fine]] = rng.choice(pool_base, size=n_base, replace=True)
    return idx


def _class_weight(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weight = 1.0 / np.sqrt(counts + 1.0)
    weight = weight / weight.mean()
    return torch.tensor(weight, dtype=torch.float32, device=device)


def _lr(step: int, total: int, base: float) -> float:
    floor = base * 0.01
    cos = 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(total, 1)))
    return floor + (base - floor) * cos


def _apply_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(obs), y, weight=weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.detach()


def _config(n_classes: int, obs_c: int) -> dict:
    return {
        "hidden_c": HIDDEN_C,
        "update_hidden": UPDATE_HIDDEN,
        "steps": NCA_STEPS,
        "seq_len": SEQ_LEN,
        "trace_steps": TRACE_STEPS,
        "hash_dim": HASH_DIM,
        "obs_c": obs_c,
        "n_classes": n_classes,
        "total_steps": TOTAL_STEPS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "arch": "trace-1d",
    }


def microbench(obs_c: int, n_classes: int) -> dict:
    """One short eager probe, then a single compile probe on the chosen batch."""
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    rows = []
    for device in devices:
        batches = BENCH_BATCHES if device.type == "mps" else (128, 512)
        for batch in batches:
            print(f"microbench {device.type} batch={batch}", flush=True)
            model = TextNCA(
                obs_c, HIDDEN_C, n_classes, NCA_STEPS, SEQ_LEN, UPDATE_HIDDEN
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            obs = torch.randn(batch, obs_c, SEQ_LEN, device=device)
            y = torch.randint(0, n_classes, (batch,), device=device)
            weight = torch.ones(n_classes, device=device)
            for _ in range(4):
                _step(model, opt, obs, y, weight)
            _sync(device)
            timed = 12
            t0 = time.perf_counter()
            for _ in range(timed):
                _step(model, opt, obs, y, weight)
            _sync(device)
            elapsed = time.perf_counter() - t0
            ms = elapsed / timed * 1000.0
            rows.append(
                {
                    "device": device.type,
                    "batch": batch,
                    "step_ms": round(ms, 3),
                    "samples_per_sec": round(batch / (elapsed / timed), 1),
                    "hours_for_150k": round(TOTAL_STEPS * ms / 1000.0 / 3600.0, 3),
                }
            )
            del model, opt, obs, y
            if device.type == "mps":
                torch.mps.empty_cache()

    def hours(row: dict) -> float:
        return TOTAL_STEPS * row["step_ms"] / 1000.0 / 3600.0

    feasible = [row for row in rows if hours(row) <= WALL_BUDGET_H]
    pool = feasible or rows
    chosen = max(pool, key=lambda row: row["samples_per_sec"])
    compile_note = "not used for training"
    device = torch.device(chosen["device"])
    batch = int(chosen["batch"])

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise TimeoutError("torch.compile probe exceeded 45s")

    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(45)
        model = TextNCA(
            obs_c, HIDDEN_C, n_classes, NCA_STEPS, SEQ_LEN, UPDATE_HIDDEN
        ).to(device)
        compiled = torch.compile(model)
        opt = torch.optim.AdamW(compiled.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        obs = torch.randn(batch, obs_c, SEQ_LEN, device=device)
        y = torch.randint(0, n_classes, (batch,), device=device)
        weight = torch.ones(n_classes, device=device)
        for _ in range(2):
            _step(compiled, opt, obs, y, weight)
        _sync(device)
        timed = 8
        t0 = time.perf_counter()
        for _ in range(timed):
            _step(compiled, opt, obs, y, weight)
        _sync(device)
        elapsed = time.perf_counter() - t0
        ms = elapsed / timed * 1000.0
        sps = batch / (elapsed / timed)
        faster = ms + 1e-6 < chosen["step_ms"]
        compile_note = (
            f"torch.compile fixed-batch probe step_ms={ms:.3f} "
            f"samples_per_sec={sps:.1f} on {device.type} batch {batch}. "
            + ("Compile was faster, so the NCA train step uses it." if faster else "Eager was faster or tied, so training stays eager.")
        )
        return {
            "rows": rows,
            "chosen": chosen,
            "compile": compile_note,
            "use_compile": faster,
            "compile_step_ms": round(ms, 3),
        }
    except Exception as exc:  # noqa: BLE001 — compile support varies by backend
        compile_note = f"torch.compile not used: {type(exc).__name__}: {exc}"
        return {"rows": rows, "chosen": chosen, "compile": compile_note, "use_compile": False}
    finally:
        signal.alarm(0)


def _accuracy(model: torch.nn.Module, obs: torch.Tensor, y: torch.Tensor, batch: int) -> float:
    n = int(y.shape[0])
    if n == 0:
        return float("nan")
    was_training = model.training
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, n, batch):
            pred = model(obs[start : start + batch]).argmax(dim=-1)
            correct += int((pred == y[start : start + batch]).sum().item())
    model.train(was_training)
    return correct / n


def _always_terminal(labels: np.ndarray, terminal_index: int | None) -> float:
    if terminal_index is None or len(labels) == 0:
        return float("nan")
    return float((labels == terminal_index).mean())


def _hist(labels: np.ndarray, names: list[str]) -> dict:
    counts = np.bincount(labels, minlength=len(names)) if len(labels) else np.zeros(len(names), np.int64)
    return {names[i]: int(counts[i]) for i in range(len(names)) if int(counts[i]) > 0}


def _sessions(rows) -> int:
    if isinstance(rows, int):
        return rows
    return len({ex.session_id for ex in rows})


def _cpu_opt_state(opt: torch.optim.Optimizer) -> dict:
    state = opt.state_dict()
    moved = {}
    for key, slot in state["state"].items():
        moved[key] = {
            name: (value.detach().cpu() if torch.is_tensor(value) else value)
            for name, value in slot.items()
        }
    return {"state": moved, "param_groups": state["param_groups"]}


def _opt_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for slot in opt.state.values():
        for name, value in list(slot.items()):
            if torch.is_tensor(value):
                slot[name] = value.to(device)


def _atomic_save(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_ckpt(path):
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _resume_ckpt():
    """Prefer this run's checkpoint, then the step-12496 snapshot, then the old last file."""
    for path in (LAST_CKPT, SEED_CKPT, ROOT / "checkpoints" / "text_nca_1d_last.pt"):
        ckpt = _load_ckpt(path)
        if ckpt:
            return ckpt
    return None


def _eager_state(state: dict) -> dict:
    """torch.compile checkpoints prefix every key with _orig_mod."""
    prefix = "_orig_mod."
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state.items()}


def _load_weights(module, state: dict) -> None:
    module.load_state_dict(_eager_state(state))


def _score_pair(nca, flat, obs, y, batch: int) -> dict:
    return {
        "nca": _accuracy(nca, obs, y, batch),
        "flat": _accuracy(flat, obs, y, batch),
    }


def _tb_log_train(row: dict, batch: int) -> None:
    """Best-effort TensorBoard write. Never interrupts training."""
    try:
        from nca_text.tb_log import log_train_row

        log_train_row(row, batch=batch)
    except Exception as exc:  # noqa: BLE001 — board logging is optional
        print(f"tensorboard: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    print(f"data_root={ROOT}", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    packed = load_packed_splits()
    vocab = packed["vocab"]
    names = list(vocab)
    train_note = packed["train_note"]
    val_note = packed["val_note"]
    train_obs = packed["train_obs"]
    train_y = packed["train_y"]
    val_obs = packed["val_obs"]
    val_y = packed["val_y"]
    train_sessions_n = int(packed["train_sessions"])
    val_sessions_n = int(packed["val_sessions"])
    train_kept = train_sessions_n
    val_kept = val_sessions_n
    terminal_index = vocab.get("terminal")
    sampler = _fineenvs_sampler(train_y, names, packed.get("train_fineenvs"))
    side = load_side_eval("fineenvs_common_val.jsonl", vocab)
    if sampler is None:
        print("batch sampler: uniform", flush=True)
    else:
        n_fine = int(np.asarray(packed["train_fineenvs"]).sum())
        print(
            f"batch sampler: hermes prior, {FINEENVS_WITHIN_CLASS:.0%} of each class from "
            f"{n_fine} fineenvs rows",
            flush=True,
        )
    print(f"loaded {train_note}", flush=True)
    print(f"loaded {val_note}", flush=True)
    if len(train_y) < 4 or len(vocab) < 2:
        raise SystemExit(f"not enough train rows ({len(train_y)}) or classes ({len(vocab)})")

    obs_c = int(train_obs.shape[1])
    config = _config(len(vocab), obs_c)
    print(
        f"train_rows={len(train_y)} val_rows={len(val_y)} classes={len(vocab)} "
        f"obs={tuple(train_obs.shape[1:])}",
        flush=True,
    )

    held = {"model_step": 0}

    def build_models(device: torch.device):
        nca = TextNCA(
            obs_c, HIDDEN_C, len(vocab), NCA_STEPS, SEQ_LEN, UPDATE_HIDDEN
        ).to(device)
        flat = FlatPolicy(obs_c * SEQ_LEN, len(vocab)).to(device)
        return nca, flat

    def payload(step: int, nca, flat, nca_opt, flat_opt, batch: int, device_type: str, best: dict) -> dict:
        return {
            "step": step,
            "nca": {k: v.detach().cpu() for k, v in nca.state_dict().items()},
            "flat": {k: v.detach().cpu() for k, v in flat.state_dict().items()},
            "nca_opt": _cpu_opt_state(nca_opt) if nca_opt is not None else None,
            "flat_opt": _cpu_opt_state(flat_opt) if flat_opt is not None else None,
            "vocab": vocab,
            "config": config,
            "batch": batch,
            "device": device_type,
            "best": best,
            "numpy_rng": held["rng"].bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }

    if args.eval_only:
        ckpt = _load_ckpt(BEST_CKPT) or _load_ckpt(LAST_CKPT)
        if ckpt is None:
            raise SystemExit(f"no checkpoint at {BEST_CKPT} or {LAST_CKPT}")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        nca, flat = build_models(device)
        nca.load_state_dict(_eager_state(ckpt["nca"]))
        flat.load_state_dict(_eager_state(ckpt["flat"]))
        batch = int(ckpt.get("batch") or 1024)
        _eval_and_write(
            nca, flat, device, batch, train_obs, train_y, val_obs, val_y, names, vocab,
            terminal_index, [], [], train_note, val_note, config,
            bench=None, history_path=LOG_PATH, step=int(ckpt.get("step") or 0),
            best=ckpt.get("best") or {}, resumed=False, eval_only=True,
            train_sessions_n=train_sessions_n, val_sessions_n=val_sessions_n,
        )
        return

    resume_ckpt = None if args.fresh else _resume_ckpt()
    can_resume = bool(
        resume_ckpt
        and resume_ckpt.get("config") == config
        and int(resume_ckpt.get("step") or 0) < TOTAL_STEPS
    )
    # The M1 probe already picked MPS batch 1024. Re-running it while the
    # half-million-row array is resident gets the process killed.
    saved_device = "mps" if torch.backends.mps.is_available() else "cpu"
    saved_batch = 1024
    if can_resume:
        saved_device = resume_ckpt.get("device") or saved_device
        saved_batch = int(resume_ckpt.get("batch") or saved_batch)
    bench = {
        "rows": [],
        "chosen": {
            "device": saved_device,
            "batch": saved_batch,
            "step_ms": None,
            "samples_per_sec": None,
            "hours_for_150k": None,
        },
        "compile": "skipped; eager training, batch from the M1 probe",
        "use_compile": False,
    }
    device = torch.device(bench["chosen"]["device"])
    batch = int(bench["chosen"]["batch"])
    batch = min(batch, max(32, len(train_y)))
    print(json.dumps({"microbench_chosen": bench["chosen"], "compile": bench["compile"]}), flush=True)

    import gc
    gc.collect()

    # Train observations stay in the SSD memmap. Each step copies one batch.
    train_y_t = torch.from_numpy(np.ascontiguousarray(train_y))
    val_obs_t = torch.from_numpy(np.ascontiguousarray(val_obs)).to(device)
    val_y_t = torch.from_numpy(np.ascontiguousarray(val_y)).to(device)
    if side is not None and len(side[1]):
        side_obs_t = torch.from_numpy(np.ascontiguousarray(side[0])).to(device)
        side_y_t = torch.from_numpy(np.ascontiguousarray(side[1])).to(device)
    else:
        side_obs_t = None
        side_y_t = None
    weight = _class_weight(train_y, len(vocab), device)
    n = int(train_obs.shape[0])

    nca, flat = build_models(device)
    if bench.get("use_compile"):
        nca = torch.compile(nca)
    nca_opt = torch.optim.AdamW(nca.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    flat_opt = torch.optim.AdamW(flat.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    held["rng"] = rng
    start_step = 0
    best = {"val_top1_nca": -1.0, "step": 0}

    resumed = False
    if not args.fresh:
        ckpt = None if args.fresh else _resume_ckpt()
        if ckpt and ckpt.get("config") == config and int(ckpt.get("step") or 0) < TOTAL_STEPS:
            nca.load_state_dict(_eager_state(ckpt["nca"]))
            flat.load_state_dict(_eager_state(ckpt["flat"]))
            try:
                if ckpt.get("nca_opt"):
                    nca_opt.load_state_dict(ckpt["nca_opt"])
                    _opt_to_device(nca_opt, device)
                if ckpt.get("flat_opt"):
                    flat_opt.load_state_dict(ckpt["flat_opt"])
                    _opt_to_device(flat_opt, device)
            except (RuntimeError, ValueError, KeyError) as exc:
                print(f"optimizer state did not resume ({exc}); continuing with fresh AdamW", flush=True)
            start_step = int(ckpt["step"])
            best = ckpt.get("best") or best
            if ckpt.get("numpy_rng"):
                rng.bit_generator.state = ckpt["numpy_rng"]
            if ckpt.get("torch_rng") is not None:
                torch.set_rng_state(ckpt["torch_rng"])
            batch = int(ckpt.get("batch") or batch)
            resumed = True
            print(f"resumed from step {start_step}", flush=True)

    held.update({"nca": nca, "flat": flat, "nca_opt": nca_opt, "flat_opt": flat_opt, "batch": batch, "best": best, "device": device.type})

    def save(step: int, path=LAST_CKPT) -> None:
        _atomic_save(
            path,
            payload(step, held["nca"], held["flat"], held["nca_opt"], held["flat_opt"], held["batch"], held["device"], held["best"]),
        )
        pointer = {
            "step": step,
            "path": str(path),
            "last": str(LAST_CKPT),
            "best": str(BEST_CKPT),
            "every": CKPT_EVERY,
        }
        (ROOT / "checkpoints" / "latest.json").write_text(json.dumps(pointer) + "\n")
        print(f"checkpoint step {step} -> {path}", flush=True)

    def on_signal(signum, frame):  # noqa: ARG001
        step = int(held.get("model_step") or start_step)
        save(step)
        print(f"saved checkpoint at step {step} after signal {signum}", flush=True)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if resumed else "w"
    log_f = LOG_PATH.open(log_mode)
    nca_loss_sum = torch.zeros((), device=device)
    flat_loss_sum = torch.zeros((), device=device)
    loss_count = 0
    window_t = time.perf_counter()
    term_val = _always_terminal(val_y, terminal_index)

    print(f"training steps {start_step + 1}..{TOTAL_STEPS} batch={batch} device={device.type}", flush=True)
    for step in range(start_step + 1, TOTAL_STEPS + 1):
        held["model_step"] = step
        lr = _lr(step, TOTAL_STEPS, LR)
        _apply_lr(nca_opt, lr)
        _apply_lr(flat_opt, lr)
        draw = min(batch, n)
        idx_np = rng.choice(n, size=draw, replace=False) if sampler is None else _sample_batch(rng, draw, sampler)
        obs_b = torch.from_numpy(np.ascontiguousarray(train_obs[idx_np])).to(device)
        y_b = train_y_t[torch.from_numpy(idx_np.astype(np.int64, copy=False))].to(device)
        nca_loss = _step(nca, nca_opt, obs_b, y_b, weight)
        flat_loss = _step(flat, flat_opt, obs_b, y_b, weight)
        nca_loss_sum = nca_loss_sum + nca_loss
        flat_loss_sum = flat_loss_sum + flat_loss
        loss_count += 1
        if step == start_step + 1:
            print(f"first step nca_loss={float(nca_loss):.4f}", flush=True)
            if not bool(torch.isfinite(nca_loss).item()):
                raise SystemExit("NCA loss was not finite on the first step")

        if step % VAL_EVERY == 0 or step == TOTAL_STEPS:
            _sync(device)
            elapsed = time.perf_counter() - window_t
            ms = elapsed / max(loss_count, 1) * 1000.0
            nca_acc = _accuracy(nca, val_obs_t, val_y_t, batch)
            flat_acc = _accuracy(flat, val_obs_t, val_y_t, batch)
            row = {
                "step": step,
                "nca_loss": round(float(nca_loss_sum) / loss_count, 4),
                "flat_loss": round(float(flat_loss_sum) / loss_count, 4),
                "val_top1_nca": nca_acc,
                "val_top1_flat": flat_acc,
                "val_top1_always_terminal": term_val,
                "batch_mix": "uniform" if sampler is None else "hermes-prior+fineenvs",
                "lr": lr,
                "ms_per_step": round(ms, 3),
            }
            if side_obs_t is not None:
                row["fineenvs_val_top1_nca"] = _accuracy(nca, side_obs_t, side_y_t, batch)
                row["fineenvs_val_top1_flat"] = _accuracy(flat, side_obs_t, side_y_t, batch)
            if ms > 0:
                row["samples_per_sec"] = round(batch / (ms / 1000.0), 1)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(json.dumps(row), flush=True)
            _tb_log_train(row, batch)
            if nca_acc == nca_acc and nca_acc >= best["val_top1_nca"]:
                best = {"val_top1_nca": nca_acc, "val_top1_flat": flat_acc, "step": step}
                held["best"] = best
                save(step, BEST_CKPT)
            nca_loss_sum = torch.zeros((), device=device)
            flat_loss_sum = torch.zeros((), device=device)
            loss_count = 0
            window_t = time.perf_counter()

        if step % CKPT_EVERY == 0 or step == TOTAL_STEPS:
            save(step)
            periodic = ROOT / "checkpoints" / f"text_nca_1d_mix13_step{step:06d}.pt"
            save(step, periodic)

    log_f.close()
    _eval_and_write(
        nca, flat, device, batch, train_obs, train_y, val_obs, val_y, names, vocab,
        terminal_index, [], [], train_note, val_note, config,
        bench=bench, history_path=LOG_PATH, step=TOTAL_STEPS, best=held["best"],
        resumed=resumed, eval_only=False,
        train_sessions_n=train_sessions_n, val_sessions_n=val_sessions_n,
    )


def _eval_and_write(
    nca, flat, device, batch, train_obs, train_y, val_obs, val_y, names, vocab,
    terminal_index, train_kept, val_kept, train_note, val_note, config,
    bench, history_path, step, best, resumed, eval_only,
    train_sessions_n=None, val_sessions_n=None,
) -> None:
    local, local_note = load_local_test()
    test_obs, test_y, test_kept = build_trace_obs(local, vocab)
    oov_local = len(local) - len(test_kept)

    def to_dev(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device)

    val_obs_t = to_dev(val_obs)
    val_y_t = to_dev(val_y)
    if len(test_y):
        test_obs_t = to_dev(test_obs)
        test_y_t = to_dev(test_y)
    else:
        test_obs_t = torch.zeros(0, train_obs.shape[1], SEQ_LEN, device=device)
        test_y_t = torch.zeros(0, dtype=torch.long, device=device)

    final_val = _score_pair(nca, flat, val_obs_t, val_y_t, batch)
    final_test = _score_pair(nca, flat, test_obs_t, test_y_t, batch)
    final_val["always_terminal"] = _always_terminal(val_y, terminal_index)
    final_test["always_terminal"] = _always_terminal(test_y, terminal_index)

    best_val = None
    best_test = None
    best_ckpt = _load_ckpt(BEST_CKPT)
    if best_ckpt and best_ckpt.get("config") == config:
        nca_b, flat_b = TextNCA(
            int(train_obs.shape[1]), HIDDEN_C, len(vocab), NCA_STEPS, SEQ_LEN, UPDATE_HIDDEN
        ).to(device), FlatPolicy(int(train_obs.shape[1]) * SEQ_LEN, len(vocab)).to(device)
        nca_b.load_state_dict(_eager_state(best_ckpt["nca"]))
        flat_b.load_state_dict(_eager_state(best_ckpt["flat"]))
        best_val = _score_pair(nca_b, flat_b, val_obs_t, val_y_t, batch)
        best_test = _score_pair(nca_b, flat_b, test_obs_t, test_y_t, batch)
        best_val["always_terminal"] = final_val["always_terminal"]
        best_test["always_terminal"] = final_test["always_terminal"]
        best_val["step"] = int(best_ckpt.get("step") or 0)
        best_test["step"] = best_val["step"]

    metrics = {
        "chip": _chip(),
        "data_root": str(ROOT),
        "arch": "trace-1d",
        "train_note": train_note,
        "val_note": val_note,
        "local_note": local_note,
        "resumed": resumed,
        "eval_only": eval_only,
        "step": step,
        "train_rows": int(train_y.shape[0]),
        "train_sessions": train_sessions_n if train_sessions_n is not None else _sessions(train_kept),
        "val_rows": int(val_y.shape[0]),
        "val_sessions": val_sessions_n if val_sessions_n is not None else _sessions(val_kept),
        "local_rows_before_vocab": len(local),
        "local_test_rows": len(test_kept),
        "local_test_sessions": _sessions(test_kept),
        "local_oov_or_unmapped": oov_local,
        "n_classes": len(vocab),
        "labels": names,
        "train_hist": _hist(train_y, names),
        "val_hist": _hist(val_y, names),
        "local_test_hist": _hist(test_y, names),
        "device": device.type,
        "batch": batch,
        "microbench": bench,
        "config": config,
        "best_tracked": best,
        "final_val_top1": final_val,
        "final_local_test_top1": final_test,
        "best_val_top1": best_val,
        "best_local_test_top1": best_test,
        "checkpoint_last": str(LAST_CKPT),
        "checkpoint_best": str(BEST_CKPT),
        "log": str(history_path),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    if eval_only:
        _tb_log_train(
            {
                "step": int(step),
                "val_top1_nca": final_val.get("nca"),
                "val_top1_flat": final_val.get("flat"),
                "val_top1_always_terminal": final_val.get("always_terminal"),
            },
            int(batch),
        )


if __name__ == "__main__":
    main()
