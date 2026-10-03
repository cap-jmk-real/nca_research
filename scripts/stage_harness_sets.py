"""Download public Pi, OpenCode, browser, and coding traces onto the external SSD.

Each set resumes from files already on the volume. A dropped network retries
the same directory instead of starting over. Nothing is written to the internal disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

DATA_ROOT = Path("/Volumes/Extreme SSD/nca_research")

# (repo, directory under datasets/, note)
SETS = [
    ("MaxDevv/real-pi-coding-agent-traces-sessions", "agent/pi-maxdevv", "pi sessions, jsonl"),
    ("julien-c/pi-sessions", "agent/pi-julien-c", "small pi sessions"),
    ("grfwings/pi-session-traces", "agent/pi-grfwings", "small pi sessions"),
    ("abidlabs/trackio-pi-sessions", "agent/pi-abidlabs", "small pi sessions"),
    ("RZ412/opencode_sft_traces", "agent/opencode-rz412", "opencode sft parquet"),
    ("dacorvo/hf-hub-session-opencode-traces", "agent/opencode-dacorvo-hf-hub", "opencode session json"),
    ("dacorvo/transformers-coding-session-opencode-traces", "agent/opencode-dacorvo-transformers", "opencode session json"),
    ("dacorvo/funes-recall-session-opencode-traces", "agent/opencode-dacorvo-funes", "opencode session json"),
    ("R2E-Gym/R2EGym-SFT-Trajectories", "coding/r2e-gym-sft", "coding sft parquet"),
    ("rmems/browser-tool-use-trajectories", "agent/browser-tool-use", "browser tool trajectories"),
    ("Hongliang1997/WebVoyager-Trajectories-GPT-4V", "agent/webvoyager-gpt4v", "webvoyager trajectories"),
    ("open-athena/code-contests-noblock-qwen3.5-122b-131k-opencode-traces", "agent/opencode-athena-code-contests", "opencode parquet"),
    ("open-athena/nemotron-code-oracle-filtered-qwen3.5-122b-131k-opencode-traces", "agent/opencode-athena-nemotron-oracle", "opencode parquet"),
    ("open-athena/inferredbugs-sandboxes-verifier-qwen3.5-122b-131k-opencode-traces", "agent/opencode-athena-inferredbugs", "opencode parquet"),
    ("AmanPriyanshu/tool-reasoning-sft-CODING-allenai-SERA-data-cleaned-rectified", "coding/sera-cleaned", "SERA coding trajectories"),
    ("osunlp/Mind2Web", "agent/mind2web", "mind2web"),
    ("webarena-x/webarena-infinity-trajectories", "agent/webarena-infinity", "webarena trajectories if the repo has files"),
    ("shiqihe/WebVoyager-Trajectories-Qwen3.5-Omni", "agent/webvoyager-qwen", "webvoyager qwen if the repo has files"),
    ("nebius/SWE-rebench-openhands-trajectories", "coding/swe-rebench-openhands", "resume the partial parquet"),
]


def configure() -> None:
    if not DATA_ROOT.is_dir():
        raise SystemExit(f"missing data root {DATA_ROOT}")
    hf = DATA_ROOT / "hf-cache"
    tmp = DATA_ROOT / "tmp"
    for path in (hf / "hub", hf / "datasets", hf / "xet", tmp):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf)
    os.environ["HF_HUB_CACHE"] = str(hf / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf / "datasets")
    os.environ["HF_XET_CACHE"] = str(hf / "xet")
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file() and not item.name.startswith("._") and ".cache" not in item.parts:
            total += item.stat().st_size
    return total


def _permanent(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in ("401", "403", "404", "not found", "gated", "unauthorized"))


def _keep_awake() -> None:
    # Screen may sleep. Idle sleep is what drops a long download when nobody is at the machine.
    subprocess.Popen(
        ["caffeinate", "-ims", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    configure()
    from huggingface_hub import snapshot_download

    manifest_path = DATA_ROOT / "datasets" / "agent" / "harness-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report = []
    if manifest_path.is_file():
        try:
            report = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            report = []
    finished = {row.get("repo") for row in report if row.get("status") in {"ok", "skipped"}}
    still_open = 0
    for repo, rel, note in SETS:
        dest = DATA_ROOT / "datasets" / rel
        if repo in finished and dest.exists():
            print(f"skip done {repo}", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        entry = {"repo": repo, "path": str(dest), "note": note, "status": "error"}
        for attempt in range(1, 9):
            print(f"downloading {repo} -> {dest} attempt {attempt}", flush=True)
            try:
                snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest))
                entry["status"] = "ok"
                entry["bytes"] = dir_bytes(dest)
                entry.pop("error", None)
                print(f"ok {repo} bytes={entry['bytes']}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 — resume the same files on the next attempt
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"fail {repo}: {entry['error']}", flush=True)
                traceback.print_exc()
                if _permanent(exc) and attempt >= 2:
                    entry["status"] = "skipped"
                    print(f"skip permanent {repo}", flush=True)
                    break
                time.sleep(min(120, 10 * attempt))
        if entry["status"] not in {"ok", "skipped"}:
            still_open += 1
        report = [row for row in report if row.get("repo") != repo]
        report.append(entry)
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {manifest_path} open={still_open}", flush=True)
    return still_open


if __name__ == "__main__":
    _keep_awake()
    while True:
        try:
            if main() == 0:
                print("all staged", flush=True)
                break
        except Exception:
            traceback.print_exc()
        print("network dropped or a set is incomplete; resuming in 45s", flush=True)
        time.sleep(45)
