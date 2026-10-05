"""Score the trained 1D NCA on BFCL, NESTFUL, tau2, and AppWorld.

The checkpoint emits one of 13 Hermes tool names. BFCL AST and NESTFUL require
a function call with arguments, so those official checks fail when the name
or the arguments do not match. Irrelevance treats ``respond`` as no call.
tau2 and AppWorld need a live environment; this script records that result
when the package is not installed.

Caches and downloads stay on the external SSD.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SSD = Path("/Volumes/Extreme SSD/nca_research")
BENCH = SSD / "datasets" / "benchmarks"
CKPT = SSD / "results" / "checkpoints" / "text_nca_1d_mix13_best.pt"
OUT = SSD / "results" / "benchmarks_run.json"

os.environ["HF_HOME"] = str(SSD / "hf-cache")
os.environ["HF_HUB_CACHE"] = str(SSD / "hf-cache" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(SSD / "hf-cache" / "hub")
os.environ["TMPDIR"] = str(SSD / "tmp")

from nca_text.data import HASH_DIM, SEQ_LEN, hash_features  # noqa: E402
from nca_text.model import TextNCA  # noqa: E402

BFCL_AST = (
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
)
BFCL_IRREL = "BFCL_v3_irrelevance.json"


def _hf(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    BENCH.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(BENCH / repo.replace("/", "__")),
    )
    return Path(path)


def _load_json_rows(path: Path) -> list[dict]:
    text = path.read_text()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _eager(state: dict) -> dict:
    prefix = "_orig_mod."
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


class Policy:
    def __init__(self) -> None:
        import torch

        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        vocab = ckpt["vocab"]
        self.names = [name for name, _ in sorted(vocab.items(), key=lambda item: item[1])]
        self.vocab = {name: i for i, name in enumerate(self.names)}
        config = ckpt["config"]
        self.model = TextNCA(
            int(config["obs_c"]),
            int(config["hidden_c"]),
            int(config["n_classes"]),
            int(config["steps"]),
            int(config["seq_len"]),
            int(config["update_hidden"]),
        )
        self.model.load_state_dict(_eager(ckpt["nca"]))
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.torch = torch
        self.step = int(ckpt.get("step") or 0)

    def predict(self, goals: list[str]) -> list[str]:
        import numpy as np

        n_classes = len(self.vocab)
        obs_c = HASH_DIM + n_classes + 1
        batch = np.zeros((len(goals), obs_c, SEQ_LEN), np.float32)
        for i, goal in enumerate(goals):
            batch[i, :HASH_DIM, 0] = hash_features(goal, HASH_DIM)
            batch[i, -1, 0] = 1.0
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(batch).to(self.device))
            idx = logits.argmax(dim=1).tolist()
        return [self.names[i] for i in idx]


def _question_text(row: dict) -> str:
    parts = []
    for turn in row.get("question") or []:
        messages = turn if isinstance(turn, list) else [turn]
        for message in messages:
            if isinstance(message, dict):
                parts.append(str(message.get("content") or ""))
    names = []
    for function in row.get("function") or []:
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return " ".join(parts + names)[:2000]


def _gold_calls(answer: dict) -> list[dict]:
    calls = []
    for item in answer.get("ground_truth") or []:
        if isinstance(item, dict):
            calls.append(item)
    return calls


def _ast_match(pred: str, gold: list[dict]) -> bool:
    """BFCL AST: the call name matches and every argument is one of the allowed values.

    This head emits a tool name and no argument values, so a call matches only
    when the gold function takes no arguments.
    """
    for call in gold:
        spec = call.get(pred)
        if not isinstance(spec, dict):
            continue
        if len(spec) == 0:
            return True
        return False
    return False


def score_bfcl(policy: Policy) -> dict:
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    categories = {}
    predicted = Counter()
    for name in BFCL_AST:
        questions = _load_json_rows(_hf(repo, name))
        answers = {row["id"]: row for row in _load_json_rows(_hf(repo, f"possible_answer/{name}"))}
        goals = [_question_text(row) for row in questions]
        preds = []
        for start in range(0, len(goals), 256):
            preds.extend(policy.predict(goals[start : start + 256]))
        correct = 0
        name_hit = 0
        for row, pred in zip(questions, preds, strict=True):
            predicted[pred] += 1
            gold = _gold_calls(answers[row["id"]])
            gold_names = set()
            for call in gold:
                gold_names.update(call.keys())
            if pred in gold_names:
                name_hit += 1
            if _ast_match(pred, gold):
                correct += 1
        categories[name] = {
            "n": len(questions),
            "ast_accuracy": correct / len(questions) if questions else None,
            "function_name_overlap": name_hit / len(questions) if questions else None,
        }
        print(f"bfcl {name} n={len(questions)} ast={categories[name]['ast_accuracy']}", flush=True)

    irrel = _load_json_rows(_hf(repo, BFCL_IRREL))
    goals = [_question_text(row) for row in irrel]
    preds = []
    for start in range(0, len(goals), 256):
        preds.extend(policy.predict(goals[start : start + 256]))
    abstain = sum(1 for pred in preds if pred == "respond")
    for pred in preds:
        predicted[pred] += 1
    categories[BFCL_IRREL] = {
        "n": len(irrel),
        "irrelevance_accuracy": abstain / len(irrel) if irrel else None,
        "rule": "respond counts as no function call; any other class counts as a call",
    }
    print(
        f"bfcl irrelevance n={len(irrel)} acc={categories[BFCL_IRREL]['irrelevance_accuracy']}",
        flush=True,
    )
    ast_n = sum(categories[name]["n"] for name in BFCL_AST)
    ast_correct = sum(categories[name]["ast_accuracy"] * categories[name]["n"] for name in BFCL_AST)
    return {
        "checkpoint_step": policy.step,
        "ast_accuracy": ast_correct / ast_n if ast_n else None,
        "ast_n": ast_n,
        "irrelevance_accuracy": categories[BFCL_IRREL]["irrelevance_accuracy"],
        "irrelevance_n": len(irrel),
        "categories": categories,
        "predicted_tools": predicted.most_common(),
    }


def _nestful_goal(row: dict) -> str:
    for key in ("input", "question", "query", "problem", "sample"):
        if row.get(key):
            return str(row[key])[:2000]
    return json.dumps(row, ensure_ascii=False)[:2000]


def _nestful_functions(row: dict) -> list[str]:
    value = row.get("output")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    names = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    return names


def score_nestful(policy: Policy) -> dict:
    import pyarrow.parquet as pq

    path = _hf("ibm-research/nestful", "data/train-00000-of-00001.parquet")
    table = pq.read_table(path)
    rows = table.to_pylist()
    print(f"nestful columns {table.column_names} n={len(rows)}", flush=True)
    if rows:
        print("nestful keys", list(rows[0].keys())[:20], flush=True)
    goals = [_nestful_goal(row) for row in rows]
    preds = []
    for start in range(0, len(goals), 256):
        preds.extend(policy.predict(goals[start : start + 256]))
    full = 0
    name_hit = 0
    for row, pred in zip(rows, preds, strict=True):
        gold = _nestful_functions(row)
        # Full sequence accuracy: the emitted call sequence equals the gold sequence.
        if gold == [pred]:
            full += 1
        if pred in gold:
            name_hit += 1
    n = len(rows)
    result = {
        "n": n,
        "full_sequence_accuracy": full / n if n else None,
        "function_name_in_gold": name_hit / n if n else None,
        "note": "The head emits one Hermes tool name, not a nested call sequence with arguments.",
        "sample_gold": _nestful_functions(rows[0])[:6] if rows else [],
        "sample_pred": preds[0] if preds else None,
    }
    print(f"nestful full_seq={result['full_sequence_accuracy']}", flush=True)
    return result


def score_tau2() -> dict:
    try:
        import tau2  # noqa: F401
    except ImportError:
        return {
            "score": None,
            "blocker": "tau2 is not installed. The official score is a database end-state reward over a user-simulator episode, which this tool-name head cannot emit.",
        }
    return {"score": None, "blocker": "tau2 imported but no episode runner is wired."}


def score_appworld() -> dict:
    try:
        import appworld  # noqa: F401
    except ImportError:
        return {
            "score": None,
            "blocker": "appworld is not installed. Official tgc/sgc need the local app databases and a program that calls their APIs.",
        }
    return {"score": None, "blocker": "appworld imported but no episode runner is wired."}


def main() -> None:
    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint {CKPT}")
    (SSD / "tmp").mkdir(parents=True, exist_ok=True)
    policy = Policy()
    print(f"loaded step {policy.step} labels {policy.names}", flush=True)
    report = {
        "checkpoint": str(CKPT),
        "step": policy.step,
        "labels": policy.names,
        "BFCL": score_bfcl(policy),
        "NESTFUL": score_nestful(policy),
        "tau2": score_tau2(),
        "AppWorld": score_appworld(),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
