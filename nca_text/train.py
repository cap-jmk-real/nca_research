"""Train a tiny text NCA on this Mac. Artifacts go to the external data root."""

from __future__ import annotations

import json
import signal
import subprocess
import time

from nca_text.data import (
    HASH_C,
    GRID_H,
    GRID_W,
    build_obs,
    data_root,
    fetch_public,
    load_local_test,
    load_synthetic,
    prepare_env,
    public_split,
)

ROOT = data_root()
prepare_env(ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from nca_text.model import FlatPolicy, TextNCA

HIDDEN_C = 8
UPDATE_HIDDEN = 48
NCA_STEPS = 4
EPOCHS = 16
SEED = 0
BENCH_BATCHES = (64, 256, 1024)


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


def _class_weight(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weight = 1.0 / np.sqrt(counts + 1.0)
    weight = weight / weight.mean()
    return torch.tensor(weight, dtype=torch.float32, device=device)


def _step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(obs), y, weight=weight)
    loss.backward()
    opt.step()


def _batch_iter(n: int, batch: int, steps: int, rng: np.random.Generator):
    for _ in range(steps):
        if n <= batch:
            idx = np.arange(n)
        else:
            idx = rng.choice(n, size=batch, replace=False)
        yield idx


def microbench(obs_c: int, n_classes: int) -> dict:
    """One train step on random grids. Pick the device/batch with the best samples/sec."""
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    rows = []
    best = None
    for device in devices:
        for batch in BENCH_BATCHES:
            model = TextNCA(
                obs_c, HIDDEN_C, n_classes, NCA_STEPS, GRID_H, GRID_W, UPDATE_HIDDEN
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
            obs = torch.randn(batch, obs_c, GRID_H, GRID_W, device=device)
            y = torch.randint(0, n_classes, (batch,), device=device)
            weight = torch.ones(n_classes, device=device)
            for _ in range(6):
                _step(model, opt, obs, y, weight)
            _sync(device)
            timed = 20
            t0 = time.perf_counter()
            for _ in range(timed):
                _step(model, opt, obs, y, weight)
            _sync(device)
            elapsed = time.perf_counter() - t0
            ms = elapsed / timed * 1000.0
            sps = batch / (elapsed / timed)
            row = {
                "device": device.type,
                "batch": batch,
                "step_ms": round(ms, 3),
                "samples_per_sec": round(sps, 1),
            }
            rows.append(row)
            if best is None or sps > best["samples_per_sec"]:
                best = row
            del model, opt, obs, y
            if device.type == "mps":
                torch.mps.empty_cache()
    compile_note = "not used for training"
    # Fixed-batch probe only. The training loop stays eager so a short last batch
    # does not trigger a second compile.
    device = torch.device(best["device"])
    batch = int(best["batch"])

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise TimeoutError("torch.compile probe exceeded 45s")

    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(45)
        model = TextNCA(
            obs_c, HIDDEN_C, n_classes, NCA_STEPS, GRID_H, GRID_W, UPDATE_HIDDEN
        ).to(device)
        compiled = torch.compile(model)
        opt = torch.optim.AdamW(compiled.parameters(), lr=3e-3)
        obs = torch.randn(batch, obs_c, GRID_H, GRID_W, device=device)
        y = torch.randint(0, n_classes, (batch,), device=device)
        weight = torch.ones(n_classes, device=device)
        for _ in range(3):
            _step(compiled, opt, obs, y, weight)
        _sync(device)
        timed = 8
        t0 = time.perf_counter()
        for _ in range(timed):
            _step(compiled, opt, obs, y, weight)
        _sync(device)
        elapsed = time.perf_counter() - t0
        sps = batch / (elapsed / timed)
        compile_note = (
            f"torch.compile fixed-batch probe step_ms={elapsed / timed * 1000:.3f} "
            f"samples_per_sec={sps:.1f} on {device.type} batch {batch}. "
            "Training stayed eager."
        )
    except Exception as exc:  # noqa: BLE001 — compile support varies by backend
        compile_note = f"torch.compile not used: {type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    return {"rows": rows, "chosen": best, "compile": compile_note}


def _accuracy(model: torch.nn.Module, obs: torch.Tensor, y: torch.Tensor, batch: int) -> float:
    if len(y) == 0:
        return float("nan")
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(y), batch):
            pred = model(obs[start : start + batch]).argmax(dim=-1)
            correct += int((pred == y[start : start + batch]).sum().item())
    model.train()
    return correct / len(y)


def _majority(train_y: np.ndarray, test_y: np.ndarray) -> float:
    if len(test_y) == 0 or len(train_y) == 0:
        return float("nan")
    counts = np.bincount(train_y)
    pred = int(counts.argmax())
    return float((test_y == pred).mean())


def _train(
    model: torch.nn.Module,
    obs: torch.Tensor,
    y: torch.Tensor,
    val_obs: torch.Tensor,
    val_y: torch.Tensor,
    weight: torch.Tensor,
    batch: int,
    epochs: int,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    n = int(obs.shape[0])
    # Warm timed window on real rows, then keep those updates.
    warm = 5
    timed = 24
    for idx in _batch_iter(n, batch, warm, rng):
        _step(model, opt, obs[idx], y[idx], weight)
    _sync(obs.device)
    t0 = time.perf_counter()
    for idx in _batch_iter(n, batch, timed, rng):
        _step(model, opt, obs[idx], y[idx], weight)
    _sync(obs.device)
    elapsed = time.perf_counter() - t0
    real_batch = min(batch, n)
    step_ms = elapsed / timed * 1000.0
    samples_per_sec = real_batch / (elapsed / timed)

    best_acc = -1.0
    best_state = None
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(obs[idx])
            loss = F.cross_entropy(logits, y[idx], weight=weight)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        acc = _accuracy(model, val_obs, val_y, batch)
        history.append({"epoch": epoch, "train_loss": round(total / max(seen, 1), 4), "val_top1": acc})
        if acc == acc and acc >= best_acc:  # skip NaN when the val split is empty
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "step_ms": round(step_ms, 3),
        "samples_per_sec": round(samples_per_sec, 1),
        "timed_steps": timed,
        "batch": real_batch,
        "best_val_top1": best_acc,
        "history": history,
    }


@torch.no_grad()
def _infer_rate(model: torch.nn.Module, obs: torch.Tensor, batch: int) -> dict:
    if len(obs) == 0:
        return {"samples_per_sec": None, "ms_per_example": None}
    model.eval()
    # Warmup
    _ = model(obs[: min(batch, len(obs))])
    _sync(obs.device)
    passes = 20
    t0 = time.perf_counter()
    counted = 0
    for _ in range(passes):
        for start in range(0, len(obs), batch):
            _ = model(obs[start : start + batch])
            counted += int(obs[start : start + batch].shape[0])
    _sync(obs.device)
    elapsed = time.perf_counter() - t0
    sps = counted / elapsed
    return {
        "samples_per_sec": round(sps, 1),
        "ms_per_example": round(1000.0 / sps, 4),
        "examples_timed": counted,
    }


def _hist(labels: np.ndarray, names: list[str]) -> dict:
    counts = np.bincount(labels, minlength=len(names)) if len(labels) else np.zeros(len(names), np.int64)
    return {names[i]: int(counts[i]) for i in range(len(names)) if int(counts[i]) > 0}


def main() -> None:
    print(f"data_root={ROOT}", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    data_note = ""
    source = "public"
    try:
        public, data_note = fetch_public(ROOT)
    except Exception as exc:  # noqa: BLE001 — network failure falls back to the fixture
        public = load_synthetic()
        source = "synthetic"
        data_note = (
            "Real public rows were not loaded "
            f"({type(exc).__name__}: {exc}). "
            "Training used nca_text/fixtures/synthetic_traces.json only."
        )

    if len(public) < 8:
        public = load_synthetic()
        source = "synthetic"
        data_note = "Public slice was empty. Training used the synthetic fixture. Real rows were not loaded."

    local, local_note = load_local_test()
    train_ex = [ex for ex in public if public_split(ex.session_id) == "train"]
    val_ex = [ex for ex in public if public_split(ex.session_id) == "val"]
    vocab_labels = sorted({ex.label for ex in train_ex})
    vocab = {name: i for i, name in enumerate(vocab_labels)}
    names = vocab_labels

    train_obs, train_y, train_kept = build_obs(train_ex, vocab)
    val_obs, val_y, val_kept = build_obs(val_ex, vocab)
    test_obs, test_y, test_kept = build_obs(local, vocab)
    oov_local = len(local) - len(test_kept)

    if len(train_y) < 4 or len(vocab) < 2:
        raise SystemExit(f"not enough train rows ({len(train_y)}) or classes ({len(vocab)})")

    bench = microbench(train_obs.shape[1], len(vocab))
    device = torch.device(bench["chosen"]["device"])
    batch = int(bench["chosen"]["batch"])
    # Real set may be smaller than the microbench batch. Fit the batch to the data.
    batch = min(batch, max(8, len(train_y)))

    def to_dev(array: np.ndarray, dtype=None) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    train_obs_t = to_dev(train_obs)
    train_y_t = to_dev(train_y)
    val_obs_t = to_dev(val_obs)
    val_y_t = to_dev(val_y)
    test_obs_t = to_dev(test_obs) if len(test_y) else torch.zeros(0, train_obs.shape[1], GRID_H, GRID_W, device=device)
    test_y_t = to_dev(test_y) if len(test_y) else torch.zeros(0, dtype=torch.long, device=device)
    weight = _class_weight(train_y, len(vocab), device)

    nca = TextNCA(
        train_obs.shape[1], HIDDEN_C, len(vocab), NCA_STEPS, GRID_H, GRID_W, UPDATE_HIDDEN
    ).to(device)
    flat = FlatPolicy(int(np.prod(train_obs.shape[1:])), len(vocab)).to(device)
    nca_stats = _train(nca, train_obs_t, train_y_t, val_obs_t, val_y_t, weight, batch, EPOCHS)
    flat_stats = _train(flat, train_obs_t, train_y_t, val_obs_t, val_y_t, weight, batch, EPOCHS)

    nca_val = _accuracy(nca, val_obs_t, val_y_t, batch)
    flat_val = _accuracy(flat, val_obs_t, val_y_t, batch)
    nca_test = _accuracy(nca, test_obs_t, test_y_t, batch)
    flat_test = _accuracy(flat, test_obs_t, test_y_t, batch)
    maj_val = _majority(train_y, val_y)
    maj_test = _majority(train_y, test_y)
    infer = _infer_rate(nca, val_obs_t if len(val_y) else train_obs_t, batch)

    ckpt = ROOT / "checkpoints" / "text_nca.pt"
    torch.save(
        {
            "nca": {k: v.detach().cpu() for k, v in nca.state_dict().items()},
            "flat": {k: v.detach().cpu() for k, v in flat.state_dict().items()},
            "vocab": vocab,
            "config": {
                "hidden_c": HIDDEN_C,
                "update_hidden": UPDATE_HIDDEN,
                "steps": NCA_STEPS,
                "hash_c": HASH_C,
                "grid_h": GRID_H,
                "grid_w": GRID_W,
            },
        },
        ckpt,
    )

    def sessions(rows) -> int:
        return len({ex.session_id for ex in rows})

    metrics = {
        "chip": _chip(),
        "data_root": str(ROOT),
        "source": source,
        "data_note": data_note,
        "local_note": local_note,
        "public_examples": len(public),
        "public_sessions": sessions(public),
        "train_rows": len(train_kept),
        "train_sessions": sessions(train_kept),
        "val_rows": len(val_kept),
        "val_sessions": sessions(val_kept),
        "local_rows_before_vocab": len(local),
        "local_test_rows": len(test_kept),
        "local_test_sessions": sessions(test_kept),
        "local_oov_or_unmapped": oov_local,
        "n_classes": len(vocab),
        "labels": names,
        "train_hist": _hist(train_y, names),
        "val_hist": _hist(val_y, names),
        "local_test_hist": _hist(test_y, names),
        "device": device.type,
        "batch": batch,
        "microbench": bench,
        "nca_train": {k: v for k, v in nca_stats.items() if k != "history"},
        "nca_history_tail": nca_stats["history"][-3:],
        "flat_train": {k: v for k, v in flat_stats.items() if k != "history"},
        "val_top1": {"nca": nca_val, "flat": flat_val, "majority": maj_val},
        "local_test_top1": {"nca": nca_test, "flat": flat_test, "majority": maj_test},
        "nca_infer": infer,
        "checkpoint": str(ckpt),
        "grid": {"H": GRID_H, "W": GRID_W, "obs_c": int(train_obs.shape[1]), "hidden_c": HIDDEN_C, "k": NCA_STEPS},
    }
    out = ROOT / "text_nca_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
