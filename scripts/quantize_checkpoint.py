#!/usr/bin/env python3
"""Quantization ladder tests for paper + HF companions.

Tests fp16/bf16, fp8 (floating), int8, nf4, and ~2.5 bpw IQ2-class when
backends exist. Missing backends → status=blocked (not a hard fail).

Does not publish. Writes quant_report.json under the Extreme SSD
results/hf/quant/. Refuses a repo-local results/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nca_text.ssd import assert_external  # noqa: E402

CFG = json.loads((ROOT / "config" / "hf_publish.json").read_text())
OUT = Path("/Volumes/Extreme SSD/nca_research/results/hf/quant")


def _outdir() -> Path:
    assert_external(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def _try_torch_info(path: Path | None) -> dict:
    info = {"path": str(path) if path else None, "loadable": False}
    if not path or not Path(path).is_file():
        info["reason"] = "missing_checkpoint"
        return info
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        n = 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                if hasattr(v, "numel"):
                    n += int(v.numel())
        info["loadable"] = True
        info["numel"] = n
        info["approx_fp16_mib"] = round(n * 2 / (1024 * 1024), 3) if n else None
    except Exception as exc:  # noqa: BLE001
        info["reason"] = str(exc)[:200]
    return info


def _backend_available(name: str) -> bool:
    if name == "torch":
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "torchao":
        try:
            import torchao  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "bitsandbytes":
        try:
            import bitsandbytes  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "mlx":
        try:
            import mlx.core  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "llama_cpp":
        # Binary presence is enough for planning/smoke
        from shutil import which

        return which("llama-quantize") is not None or which("quantize") is not None
    return False


def run_ladder(checkpoint: Path | None, slot: str) -> dict:
    ref = _try_torch_info(checkpoint)
    levels = []
    for spec in CFG["quantization"]["ladder"]:
        backends = spec.get("backends") or []
        ok = [b for b in backends if _backend_available(b)]
        entry = {
            "id": spec["id"],
            "approx_bits": spec.get("approx_bits"),
            "floating": bool(spec.get("floating")),
            "backends_ok": ok,
            "status": "ready_to_run" if ok and ref.get("loadable") else "blocked",
        }
        if not ok:
            entry["blocker"] = f"missing backends {backends}"
        elif not ref.get("loadable"):
            entry["blocker"] = ref.get("reason") or "checkpoint not loadable"
        else:
            # Smoke: estimate size only (full accuracy eval wired when harnesses green)
            bits = float(spec.get("approx_bits") or 16)
            numel = ref.get("numel") or 0
            entry["est_weight_mib"] = round(numel * bits / 8 / (1024 * 1024), 3) if numel else None
            entry["eval"] = {
                "protocol": "same gate as fp16; run when engine/harness available",
                "keep_fraction_required": CFG["quantization"]["keep_fraction_of_fp16_top1"],
                "top1": None,
            }
        levels.append(entry)

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slot": slot,
        "checkpoint": ref,
        "ladder": levels,
        "note": (
            "~2.5 bpw = iq2_xxs / iq2_s band (floating residual quants), "
            "not naive int2. Promote at most one quant companion per slot."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--slot", default="A", choices=list(CFG["slots"]))
    parser.add_argument("--from-top4", action="store_true")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if args.from_top4:
        base = Path("/Volumes/Extreme SSD/nca_research/results/hf/top4.json")
        assert_external(base)
        if base.is_file():
            top4 = json.loads(base.read_text())
            winner = (top4.get("slots") or {}).get(args.slot, {}).get("winner") or {}
            p = winner.get("path")
            if p:
                ckpt = Path(p)

    report = run_ladder(ckpt, args.slot)
    out = _outdir() / f"quant_{args.slot}_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    latest = _outdir() / f"quant_{args.slot}_latest.json"
    latest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote": str(out), "levels": [
        {"id": x["id"], "status": x["status"], "bits": x.get("approx_bits")}
        for x in report["ladder"]
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
