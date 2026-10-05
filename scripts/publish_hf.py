#!/usr/bin/env python3
"""Publish only the current top-4 HF slots (datasets + models).

Defaults to --dry-run. Real pushes need HF_TOKEN and huggingface_hub.
Refuses to push if select_top4 says hard slots are empty.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "hf_publish.json").read_text())


def _load_top4() -> dict:
    for base in (
        Path("/Volumes/Extreme SSD/nca_research/results/hf/top4.json"),
        ROOT / "results" / "hf" / "top4.json",
    ):
        if base.is_file():
            return json.loads(base.read_text())
    raise SystemExit("top4.json missing — run scripts/select_top4_artifacts.py --write")


def _repo_id(slot: str) -> str:
    ns = CFG["hf_namespace"]
    name = CFG["slots"][slot]["repo"]
    return f"{ns}/{name}"


def _dry_plan(top4: dict, slots: list[str]) -> list[dict]:
    plan = []
    for slot in slots:
        meta = top4["slots"][slot]
        winner = meta.get("winner") or {}
        plan.append(
            {
                "slot": slot,
                "kind": meta.get("kind") or CFG["slots"][slot]["kind"],
                "repo_id": _repo_id(slot),
                "winner": winner,
                "action": "upload_main_if_promoted",
                "quant_companion": CFG["quantization"].get("publish_quant_companion"),
            }
        )
    return plan


def _push_model(repo_id: str, path: Path, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    return api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="model",
    )


def _push_dataset_dir(repo_id: str, directory: Path, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    return api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["._*", "**/*/._*", "*.parquet"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["all", "dataset", "model"])
    parser.add_argument("--slot", choices=list(CFG["slots"]), default=None)
    parser.add_argument("--from-top4", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--push", action="store_true", help="Actually upload (needs HF_TOKEN)")
    parser.add_argument("--exp", default=None)
    args = parser.parse_args()

    top4 = _load_top4()
    if args.slot:
        slots = [args.slot]
    elif args.target == "dataset":
        slots = [k for k, v in CFG["slots"].items() if v["kind"] == "dataset"]
    elif args.target == "model":
        slots = [k for k, v in CFG["slots"].items() if v["kind"] == "model"]
    else:
        slots = list(CFG["slots"])

    plan = _dry_plan(top4, slots)
    report = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": not args.push,
        "plan": plan,
        "rule": "publish only the four best always",
    }
    print(json.dumps(report, indent=2), flush=True)

    if not args.push:
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required for --push")

    for item in plan:
        winner = item.get("winner") or {}
        path = winner.get("path")
        if not path:
            print(f"skip {item['slot']}: no winner path", flush=True)
            continue
        p = Path(path)
        if item["kind"] == "dataset":
            if p.is_dir():
                print(_push_dataset_dir(item["repo_id"], p, token), flush=True)
            else:
                print(f"skip {item['slot']}: dataset path not a dir", flush=True)
        else:
            if p.is_file():
                print(_push_model(item["repo_id"], p, token), flush=True)
            else:
                print(f"skip {item['slot']}: model file missing ({p})", flush=True)


if __name__ == "__main__":
    main()
