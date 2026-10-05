"""Equal-budget tests of the message-passing / research-design hypotheses.

Trains on curated Hermes only so packing stays in RAM. Each cellular variant
and the flat baseline get the same step budget, batch, seed, and val split.

Hypotheses covered
------------------
H2 / carry     — hidden state across consecutive decisions in a session
               (research-design.md §Hypotheses.2; multi-harness note)
H_halt         — learned soft halt on update depth (Adaptive Message Passing)
H_long_range   — goal-cell long-range edge into every position (rewiring)
H3 / regen     — after zeroing a contiguous cell block, accuracy vs update depth
H4 / flat      — same observation, no cellular update (already the training twin)

Hypothesis 1 (image + instruction ablation) needs vision channels and is out of
scope for this text run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from nca_text.data import (
    HASH_DIM,
    SEQ_LEN,
    TRACE_STEPS,
    build_trace_obs,
    data_root,
    load_curated_split,
    load_local_test,
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
SEED = 0
LR = 1e-3
WEIGHT_DECAY = 1e-2
DEFAULT_STEPS = 8_000
VAL_EVERY = 500
BATCH = 512
CARRY_WINDOW = 4
CARRY_SESSIONS = 128
OUT_DIR = ROOT / "hypothesis_tests"
REPO_DOC = Path(__file__).resolve().parents[1] / "docs" / "hypothesis-tests.md"


VARIANTS = (
    "baseline",
    "halt",
    "long_range",
    "carry",
    "flat",
)


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _class_weight(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weight = 1.0 / np.sqrt(counts + 1.0)
    weight = weight / weight.mean()
    return torch.tensor(weight, dtype=torch.float32, device=device)


def _lr(step: int, total: int, base: float) -> float:
    floor = base * 0.01
    cos = 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(total, 1)))
    return floor + (base - floor) * cos


def _accuracy(model, obs: torch.Tensor, y: torch.Tensor, batch: int, **kwargs) -> float:
    n = int(y.shape[0])
    if n == 0:
        return float("nan")
    was = model.training
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, n, batch):
            pred = model(obs[start : start + batch], **kwargs).argmax(dim=-1)
            correct += int((pred == y[start : start + batch]).sum().item())
    model.train(was)
    return correct / n


def _always_terminal(labels: np.ndarray, terminal_index: int | None) -> float:
    if terminal_index is None or len(labels) == 0:
        return float("nan")
    return float((labels == terminal_index).mean())


def _build_nca(obs_c: int, n_classes: int, variant: str, device: torch.device) -> TextNCA:
    return TextNCA(
        obs_c,
        HIDDEN_C,
        n_classes,
        NCA_STEPS,
        SEQ_LEN,
        UPDATE_HIDDEN,
        learned_halt=(variant == "halt"),
        long_range_goal=(variant == "long_range"),
    ).to(device)


def _pack_curated() -> dict:
    train_ex, train_note = load_curated_split("train")
    val_ex, val_note = load_curated_split("val")
    labels = sorted({ex.label for ex in train_ex})
    vocab = {name: i for i, name in enumerate(labels)}
    train_obs, train_y, train_kept = build_trace_obs(train_ex, vocab)
    val_obs, val_y, val_kept = build_trace_obs(val_ex, vocab)
    # Re-read step_index from the curated jsonl so carry follows session time.
    from nca_text.data import datasets_dir

    step_of: dict[tuple[str, str, str], int] = {}
    path = datasets_dir() / "curated" / "train.jsonl"
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("session_id") or ""),
                str(row.get("target_tool") or ""),
                str(row.get("goal") or "")[:80],
            )
            step_of[key] = int(row.get("step_index") or 0)
    by_session: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, ex in enumerate(train_kept):
        key = (ex.session_id, ex.label, (ex.goal or "")[:80])
        by_session[ex.session_id].append((i, step_of.get(key, i)))
    sessions = []
    for idxs in by_session.values():
        order = [i for i, _ in sorted(idxs, key=lambda t: t[1])]
        if len(order) >= 2:
            sessions.append(np.asarray(order, np.int64))
    return {
        "vocab": vocab,
        "train_obs": train_obs,
        "train_y": train_y,
        "val_obs": val_obs,
        "val_y": val_y,
        "train_kept": train_kept,
        "val_kept": val_kept,
        "sessions": sessions,
        "train_note": train_note,
        "val_note": val_note,
    }


def _sample_carry(
    rng: np.random.Generator,
    sessions: list[np.ndarray],
    n_sessions: int,
    window: int,
) -> np.ndarray:
    """Return (n_sessions, window) row indices into the packed train arrays."""
    pick = rng.choice(len(sessions), size=n_sessions, replace=True)
    out = np.empty((n_sessions, window), np.int64)
    for row, sid in enumerate(pick):
        seq = sessions[sid]
        if len(seq) <= window:
            # Tile the session so the window is full; early slots repeat start.
            reps = int(math.ceil(window / len(seq)))
            tiled = np.tile(seq, reps)[:window]
            out[row] = tiled
        else:
            start = int(rng.integers(0, len(seq) - window + 1))
            out[row] = seq[start : start + window]
    return out


def _train_variant(
    variant: str,
    packed: dict,
    total_steps: int,
    device: torch.device,
) -> dict:
    vocab = packed["vocab"]
    names = list(vocab)
    train_obs = packed["train_obs"]
    train_y = packed["train_y"]
    val_obs = packed["val_obs"]
    val_y = packed["val_y"]
    sessions = packed["sessions"]
    obs_c = int(train_obs.shape[1])
    n_classes = len(vocab)
    terminal_index = vocab.get("terminal")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    val_obs_t = torch.from_numpy(np.ascontiguousarray(val_obs)).to(device)
    val_y_t = torch.from_numpy(np.ascontiguousarray(val_y)).to(device)
    train_y_t = torch.from_numpy(np.ascontiguousarray(train_y))
    weight = _class_weight(train_y, n_classes, device)
    term_val = _always_terminal(val_y, terminal_index)
    n = int(train_obs.shape[0])

    if variant == "flat":
        model: torch.nn.Module = FlatPolicy(obs_c * SEQ_LEN, n_classes).to(device)
    else:
        model = _build_nca(obs_c, n_classes, variant, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history: list[dict] = []
    best = {"val_top1": -1.0, "step": 0}
    best_state = None
    loss_sum = 0.0
    loss_count = 0
    window_t = time.perf_counter()
    t0 = time.perf_counter()

    print(f"=== {variant}: {total_steps} steps on curated ===", flush=True)
    for step in range(1, total_steps + 1):
        lr = _lr(step, total_steps, LR)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)

        if variant == "carry":
            idx = _sample_carry(rng, sessions, CARRY_SESSIONS, CARRY_WINDOW)
            hidden = None
            loss = None
            for t in range(CARRY_WINDOW):
                rows = idx[:, t]
                obs_b = torch.from_numpy(np.ascontiguousarray(train_obs[rows])).to(device)
                y_b = train_y_t[torch.from_numpy(rows)].to(device)
                logits, hidden = model(obs_b, hidden=hidden, return_hidden=True)
                step_loss = F.cross_entropy(logits, y_b, weight=weight)
                loss = step_loss if loss is None else loss + step_loss
                hidden = hidden.detach()  # truncated BPTT across decisions
            assert loss is not None
            loss = loss / CARRY_WINDOW
        else:
            draw = min(BATCH, n)
            rows = rng.choice(n, size=draw, replace=False)
            obs_b = torch.from_numpy(np.ascontiguousarray(train_obs[rows])).to(device)
            y_b = train_y_t[torch.from_numpy(rows)].to(device)
            loss = F.cross_entropy(model(obs_b), y_b, weight=weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += float(loss.detach())
        loss_count += 1

        if step % VAL_EVERY == 0 or step == total_steps:
            _sync(device)
            elapsed = time.perf_counter() - window_t
            ms = elapsed / max(loss_count, 1) * 1000.0
            # Carry trains with persistence; val still resets (held-out rows are
            # independent decisions). That matches the ablation: training signal
            # for memory, same decision-level metric as every other variant.
            acc = _accuracy(model, val_obs_t, val_y_t, BATCH)
            row = {
                "variant": variant,
                "step": step,
                "loss": round(loss_sum / loss_count, 4),
                "val_top1": acc,
                "val_top1_always_terminal": term_val,
                "ms_per_step": round(ms, 3),
                "lr": lr,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if acc == acc and acc >= best["val_top1"]:
                best = {"val_top1": acc, "step": step}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            loss_sum = 0.0
            loss_count = 0
            window_t = time.perf_counter()

    if best_state is not None:
        model.load_state_dict(best_state)

    local, local_note = load_local_test()
    test_obs, test_y, test_kept = build_trace_obs(local, vocab)
    if len(test_y):
        test_obs_t = torch.from_numpy(np.ascontiguousarray(test_obs)).to(device)
        test_y_t = torch.from_numpy(np.ascontiguousarray(test_y)).to(device)
        local_acc = _accuracy(model, test_obs_t, test_y_t, BATCH)
        local_term = _always_terminal(test_y, terminal_index)
    else:
        local_acc = float("nan")
        local_term = float("nan")

    ckpt_path = OUT_DIR / "checkpoints" / f"{variant}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "step": best["step"],
            "val_top1": best["val_top1"],
            "state": best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vocab": vocab,
            "config": {
                "hidden_c": HIDDEN_C,
                "update_hidden": UPDATE_HIDDEN,
                "steps": NCA_STEPS,
                "seq_len": SEQ_LEN,
                "obs_c": obs_c,
                "n_classes": n_classes,
                "learned_halt": variant == "halt",
                "long_range_goal": variant == "long_range",
                "carry": variant == "carry",
                "total_steps": total_steps,
                "batch": BATCH if variant != "carry" else CARRY_SESSIONS,
                "carry_window": CARRY_WINDOW if variant == "carry" else None,
            },
        },
        ckpt_path,
    )

    return {
        "variant": variant,
        "seconds": round(time.perf_counter() - t0, 1),
        "best": best,
        "final_val_top1": history[-1]["val_top1"] if history else float("nan"),
        "local_top1": local_acc,
        "local_always_terminal": local_term,
        "val_always_terminal": term_val,
        "history": history,
        "checkpoint": str(ckpt_path),
        "n_params": sum(p.numel() for p in model.parameters()),
        "local_note": local_note,
        "local_rows": len(test_kept),
    }


@torch.no_grad()
def _regeneration(
    packed: dict,
    baseline_ckpt: Path,
    device: torch.device,
) -> dict:
    """Hypothesis 3: zero a contiguous block of cells, score vs update depth."""
    ckpt = torch.load(baseline_ckpt, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    obs_c = int(packed["val_obs"].shape[1])
    model = _build_nca(obs_c, len(vocab), "baseline", device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    obs = torch.from_numpy(np.ascontiguousarray(packed["val_obs"])).to(device)
    y = torch.from_numpy(np.ascontiguousarray(packed["val_y"])).to(device)
    # Contiguous block: cells 2..3 (two of the four recent-step slots).
    damaged = obs.clone()
    damaged[:, :, 2:4] = 0.0

    clean = {}
    hurt = {}
    for k in range(0, NCA_STEPS + 1):
        # k=0: readout on hidden zeros (no updates).
        steps = k if k > 0 else 0
        if k == 0:
            hidden = obs.new_zeros(obs.shape[0], HIDDEN_C, SEQ_LEN)
            logits_c = model.head(hidden.flatten(1))
            logits_d = logits_c
        else:
            logits_c = model(obs, steps=k)
            logits_d = model(damaged, steps=k)
        clean[k] = float((logits_c.argmax(-1) == y).float().mean())
        hurt[k] = float((logits_d.argmax(-1) == y).float().mean())

    return {
        "block": "cells_2_3",
        "clean_top1_by_steps": clean,
        "damaged_top1_by_steps": hurt,
        "recovery_delta": {
            str(k): hurt[k] - hurt[0] for k in hurt
        },
        "baseline_checkpoint": str(baseline_ckpt),
    }


def _write_doc(summary: dict) -> None:
    rows = summary["variants"]
    lines = [
        "# Hypothesis tests: message passing ablations",
        "",
        f"**Date:** {summary['date']}",
        "",
        "Equal-budget runs on curated Hermes train / curated val. "
        f"Each variant trained for {summary['total_steps']} AdamW steps "
        f"(batch {summary['batch']}, seed {SEED}, lr {LR}, MPS when available).",
        "",
        "## Variants",
        "",
        "| Variant | What changes | Best val top-1 | Step | Local top-1 | Params |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    notes = {
        "baseline": "Current 1D rule; hidden reset every row",
        "halt": "Learned soft halt on update depth (AMP analogue)",
        "long_range": "Goal-cell long-range edge into every position",
        "carry": "Hidden carried across consecutive session decisions in training",
        "flat": "Hypothesis 4: linear readout, no cellular update",
    }
    by_name = {r["variant"]: r for r in rows}
    for name in VARIANTS:
        if name not in by_name:
            continue
        r = by_name[name]
        lines.append(
            f"| `{name}` | {notes[name]} | "
            f"{r['best']['val_top1']:.4f} | {r['best']['step']} | "
            f"{r['local_top1']:.4f} | {r['n_params']} |"
        )
    term = rows[0]["val_always_terminal"]
    lines.extend(
        [
            "",
            f"Always-terminal on curated val: **{term:.4f}**.",
            "",
            "## Verdicts",
            "",
        ]
    )

    def beat(a: float, b: float) -> str:
        if a > b + 1e-4:
            return "beats"
        if a < b - 1e-4:
            return "loses to"
        return "ties"

    base = by_name.get("baseline", {}).get("best", {}).get("val_top1")
    if base is not None and "flat" in by_name:
        flat = by_name["flat"]["best"]["val_top1"]
        lines.append(
            f"- **H4 (flat baseline):** cellular baseline {beat(base, flat)} flat "
            f"({base:.4f} vs {flat:.4f})."
        )
    if base is not None and "halt" in by_name:
        halt = by_name["halt"]["best"]["val_top1"]
        lines.append(
            f"- **H_halt:** learned halt {beat(halt, base)} baseline "
            f"({halt:.4f} vs {base:.4f})."
        )
    if base is not None and "long_range" in by_name:
        long_r = by_name["long_range"]["best"]["val_top1"]
        lines.append(
            f"- **H_long_range:** goal edge {beat(long_r, base)} baseline "
            f"({long_r:.4f} vs {base:.4f})."
        )
    if base is not None and "carry" in by_name:
        carry = by_name["carry"]["best"]["val_top1"]
        lines.append(
            f"- **H2 (carry):** carry training {beat(carry, base)} baseline "
            f"({carry:.4f} vs {base:.4f}). Val still resets hidden per row."
        )
    if not any(v in by_name for v in ("flat", "halt", "long_range", "carry")):
        lines.append("- Partial run; not enough variants for pairwise verdicts.")

    regen = summary.get("regeneration")
    if regen:
        def _at(d, k: int) -> float:
            return float(d[k] if k in d else d[str(k)])

        lines.extend(
            [
                "",
                "## H3 regeneration (baseline checkpoint)",
                "",
                f"Zeroed `{regen['block']}` on curated val. "
                "Clean vs damaged top-1 by number of update steps:",
                "",
                "| Steps | Clean | Damaged | Damaged − damaged@0 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for k in sorted(int(x) for x in regen["clean_top1_by_steps"]):
            c = _at(regen["clean_top1_by_steps"], k)
            d = _at(regen["damaged_top1_by_steps"], k)
            delta = _at(regen["recovery_delta"], k)
            lines.append(f"| {k} | {c:.4f} | {d:.4f} | {delta:+.4f} |")
        ks = sorted(int(x) for x in regen["damaged_top1_by_steps"])
        peak_k = max(ks, key=lambda k: _at(regen["damaged_top1_by_steps"], k))
        d1 = _at(regen["damaged_top1_by_steps"], 1)
        d_peak = _at(regen["damaged_top1_by_steps"], peak_k)
        c_full = _at(regen["clean_top1_by_steps"], NCA_STEPS)
        d_full = _at(regen["damaged_top1_by_steps"], NCA_STEPS)
        lines.append("")
        lines.append(
            f"Readout on zero hidden (k=0) is near chance, so Δ from k=0 is not the "
            f"regeneration claim. From one update onward, damaged peaks at "
            f"**k={peak_k} ({d_peak:.4f})**, which is {d_peak - d1:+.4f} vs damaged@1 "
            f"({d1:.4f}). At full depth, damaged is {d_full:.4f} against clean {c_full:.4f} "
            f"(gap {c_full - d_full:+.4f})."
        )
        if d_peak > d1 + 1e-4:
            lines.append(
                "Extra updates after the first help the damaged grid (partial support for H3); "
                "they do not fully close the gap to the clean grid."
            )
        else:
            lines.append(
                "Extra updates after the first do not help the damaged grid; H3 is not supported."
            )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Hypothesis 1 (image + instruction channel ablation) needs vision writes; not run.",
            "- Carry is trained with truncated BPTT across a window of session decisions; "
            "the reported val metric still resets hidden so it stays comparable.",
            f"- Artifacts: `{summary['out_dir']}`.",
            "",
            "## Sources",
            "",
            "- [research-design.md](research-design.md) hypotheses 2–4.",
            "- [multi-harness-and-message-passing.md](multi-harness-and-message-passing.md) "
            "carry / halt / long-range recommendations.",
            "",
        ]
    )
    REPO_DOC.write_text("\n".join(lines))
    (OUT_DIR / "hypothesis-tests.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--skip-regen", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"data_root={ROOT} device={device.type} steps={args.steps}", flush=True)

    packed = _pack_curated()
    print(
        f"curated train={len(packed['train_y'])} val={len(packed['val_y'])} "
        f"classes={len(packed['vocab'])} carry_sessions={len(packed['sessions'])}",
        flush=True,
    )

    results = []
    for variant in args.only:
        results.append(_train_variant(variant, packed, args.steps, device))
        (OUT_DIR / f"{variant}_history.json").write_text(
            json.dumps(results[-1], indent=2)
        )

    regen = None
    if not args.skip_regen and "baseline" in {r["variant"] for r in results}:
        base_ckpt = OUT_DIR / "checkpoints" / "baseline_best.pt"
        if base_ckpt.is_file():
            regen = _regeneration(packed, base_ckpt, device)
            print(json.dumps({"regeneration": regen}, indent=2), flush=True)

    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "total_steps": args.steps,
        "batch": BATCH,
        "carry_sessions": CARRY_SESSIONS,
        "carry_window": CARRY_WINDOW,
        "val_every": VAL_EVERY,
        "device": device.type,
        "train_note": packed["train_note"],
        "val_note": packed["val_note"],
        "train_rows": int(len(packed["train_y"])),
        "val_rows": int(len(packed["val_y"])),
        "n_classes": len(packed["vocab"]),
        "out_dir": str(OUT_DIR),
        "variants": results,
        "regeneration": regen,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_doc(summary)
    print(f"wrote {REPO_DOC}", flush=True)
    print(json.dumps({k: summary[k] for k in ("date", "total_steps", "device")}, indent=2))


if __name__ == "__main__":
    main()
