"""Rebuild curated train/val jsonl on engine's internal disk (the repo script
refuses non-/Volumes paths). Same code path, pinned revision, sha256 checked."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "scripts"))
import curate_traces as ct  # noqa: E402
from common import DATA
hf = ct.configure_hf(DATA); src = ct.download_filtered(hf)
sha = ct.sha256_file(src); want = json.load(open(REPO / "data/curated/stats.json"))["source"]["file_sha256"]
assert sha == want, (sha, want)
out = DATA / "datasets" / "curated"
if (out / "train.jsonl").exists(): raise SystemExit("curated already exists; not overwriting")
stats = ct.curate(src, out / "train.jsonl", out / "val.jsonl")
(out / "curate_stats_engine.json").write_text(json.dumps({k: (dict(v) if hasattr(v, "items") else v) for k, v in stats.items()}, indent=2, default=str))
print("rows", stats["rows_train"], stats["rows_val"], "sessions", stats["sessions_train"], stats["sessions_val"])
