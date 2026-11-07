#!/usr/bin/env python3
"""Prepare session-disjoint stratified splits (60/20/20 train/val/test) for:
1. ToolBench (Multi-step DFS trajectories from ToolLLaMA G1/G2/G3 benchmark)
2. API-Bank (Multi-turn Level-1, Level-2, and Level-3 tool-use dialogues)

Both datasets map tool invocations and assistant responses onto the canonical
action vocabulary (stage0/labels.json) while preserving multi-turn execution
traces (recent_steps) for cellular state evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import DATA, SPLITS, dist

CACHE_DIR = DATA / "hf_cache"
MANIFEST_DIR = Path(__file__).resolve().parent / "data_manifests"
FRACS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
SPLIT_SEED = 20261107


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_tool_action(name: str, is_response: bool = False) -> str:
    if is_response:
        return "respond"
    n = (name or "").strip()
    if not n or n.lower() in ("finish", "respond", "answer", "clarify", "final_answer", "give_up_and_restart"):
        return "respond"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", n).lower()
    if any(k in s for k in ("tool_searcher", "search_tool", "skill", "doc", "help", "manual")):
        return "skill_view"
    if any(k in s for k in ("agenda", "reminder", "todo", "schedule", "appointment", "booking", "alarm", "meeting", "calendar")):
        return "todo"
    if any(k in s for k in ("memory", "remember", "history", "record", "log", "note", "bookmark", "health_data")):
        return "memory"
    if any(k in s for k in ("cancel", "delete", "remove", "patch", "modify", "change", "reset", "revoke")):
        return "patch"
    if any(k in s for k in ("create", "add", "write", "save", "register", "upload", "post", "insert", "open_bank", "set_")):
        return "write_file"
    if any(k in s for k in ("read", "view", "detail", "info", "profile", "status", "check", "balance", "token", "account")):
        return "read_file"
    if any(k in s for k in ("get", "fetch", "retrieve", "find", "search", "list", "query", "lookup", "browse", "discover", "trending", "latest", "racecards", "results")):
        return "search_files"
    if any(k in s for k in ("run", "exec", "shell", "cmd", "terminal", "process", "start", "stop", "deploy", "monitor", " compile")):
        return "terminal"
    if any(k in s for k in ("send", "email", "message", "notify", "chat", "translate", "ask")):
        return "respond"
    return "execute_code"


def _alloc(n: int) -> dict[str, int]:
    raw = {k: n * f for k, f in FRACS}
    out = {k: int(v) for k, v in raw.items()}
    if n >= 3:
        for k in out:
            if out[k] == 0:
                out[k] = 1
    while sum(out.values()) > n:
        k = max(out, key=lambda s: out[s])
        out[k] -= 1
    rem = sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while sum(out.values()) < n:
        out[rem[i % 3]] += 1
        i += 1
    return out


def assign_by_group(examples: list[dict], group_key_fn, seed: int = SPLIT_SEED) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        groups[group_key_fn(e)].append(e)
    by_label: dict[str, list[str]] = defaultdict(list)
    for key, rows in groups.items():
        lab = Counter(r["target_tool"] for r in rows).most_common(1)[0][0]
        by_label[lab].append(key)
    rng = random.Random(seed)
    parts: dict[str, list[dict]] = {k: [] for k, _ in FRACS}
    for lab in sorted(by_label):
        keys = sorted(by_label[lab])
        rng.shuffle(keys)
        alloc = _alloc(len(keys))
        i = 0
        for split, _ in FRACS:
            for key in keys[i : i + alloc[split]]:
                parts[split].extend(groups[key])
            i += alloc[split]
    for split in parts:
        parts[split].sort(key=lambda e: (e["session_id"], e["step_index"]))
    return parts


def build_toolbench(cache_dir: Path) -> tuple[list[dict], dict]:
    p = hf_hub_download(
        "Yhyu13/ToolBench_toolllama_G123_dfs",
        "toolllama_G123_dfs_eval.json",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    rows = json.loads(Path(p).read_text())
    examples: list[dict] = []
    for idx, row in enumerate(rows):
        convs = row.get("conversations", [])
        sys_prompt = ""
        user_goal = str(row.get("id", ""))
        for c in convs:
            if c.get("from") == "system" and not sys_prompt:
                sys_prompt = c.get("value", "")
            elif c.get("from") == "user" and not user_goal:
                user_goal = c.get("value", "")
        # Extract available tool descriptions snippet from system prompt
        tools_match = re.search(r"You have access of the following tools:(.*)", sys_prompt, re.S)
        tools_desc = tools_match.group(1).strip()[:600] if tools_match else ""
        base_goal = f"{user_goal.strip()} || Tools: {tools_desc}".strip()

        recent_steps: list[dict] = []
        step_idx = 0
        sess_base = f"tb_{idx:04d}"
        i = 0
        while i < len(convs):
            c = convs[i]
            if c.get("from") == "assistant":
                val = c.get("value", "")
                m_act = re.search(r"Action:\s*([^\n\r]+)", val)
                m_thought = re.search(r"Thought:\s*([^\n\r]+)", val)
                raw_act = m_act.group(1).strip() if m_act else "Finish"
                thought = m_thought.group(1).strip() if m_thought else ""
                is_finish = raw_act.lower() in ("finish", "give_up_and_restart") or "finish" in raw_act.lower()
                target_tool = map_tool_action(raw_act, is_response=is_finish)

                # Look ahead for function observation to record for future steps
                obs_text = ""
                if i + 1 < len(convs) and convs[i + 1].get("from") == "function":
                    obs_text = str(convs[i + 1].get("value", ""))[:240]

                turn_goal = base_goal if not thought else f"{base_goal} || Context: {thought[:160]}"
                examples.append(
                    {
                        "session_id": f"{sess_base}_s{step_idx:02d}",
                        "traj_id": sess_base,
                        "step_index": step_idx,
                        "goal": turn_goal,
                        "recent_steps": list(recent_steps[-4:]),
                        "target_tool": target_tool,
                        "raw_action": raw_act,
                        "source": "toolllama_G123_dfs_eval.json",
                    }
                )
                recent_steps.append({"tool": target_tool, "result": f"{raw_act}: {obs_text}"})
                step_idx += 1
            i += 1
    return examples, {"toolllama_G123_dfs_eval.json": _sha(Path(p))}


def build_apibank(cache_dir: Path) -> tuple[list[dict], dict]:
    files = [
        "test-data/level-1-api.json",
        "test-data/level-1-response.json",
        "test-data/level-2-api.json",
        "test-data/level-2-response.json",
        "test-data/level-3-batch-inf.json",
        "test-data/level-3-batch-inf-response.json",
        "training-data/lv3-api-train.json",
        "training-data/lv3-response-train.json",
    ]
    examples: list[dict] = []
    sources: dict[str, str] = {}
    for fn in files:
        p = Path(
            hf_hub_download(
                "liminghao1630/API-Bank",
                fn,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
        )
        sources[fn] = _sha(p)
        rows = json.loads(p.read_text())
        is_resp = "response" in fn
        for idx, r in enumerate(rows):
            inst = str(r.get("instruction", ""))
            inp = str(r.get("input", ""))
            out = str(r.get("expected_output") or r.get("output") or "")
            file_id = str(r.get("file") or f"{Path(fn).stem}_{idx:04d}")
            row_id = r.get("id", idx)

            # Parse prior API-Request steps inside input dialogue history
            recent_steps: list[dict] = []
            for m_prev in re.finditer(r"API-Request:\s*\[([A-Za-z0-9_]+)\((.*?)\)\](?:\s*->\s*([^\n\r]+))?", inp):
                prev_api = m_prev.group(1)
                prev_args = m_prev.group(2)[:80]
                prev_res = (m_prev.group(3) or "")[:160]
                recent_steps.append(
                    {
                        "tool": map_tool_action(prev_api, is_response=False),
                        "result": f"{prev_api}({prev_args}) -> {prev_res}",
                    }
                )

            m_target = re.search(r"\[([A-Za-z0-9_]+)\(", out)
            raw_api = m_target.group(1) if m_target else ("Respond" if is_resp else "Execute")
            target_tool = map_tool_action(raw_api, is_response=is_resp)

            # Extract API descriptions snippet + dialogue text
            api_desc_idx = inst.find("API descriptions:")
            api_desc = inst[api_desc_idx + 17 :].strip()[:500] if api_desc_idx >= 0 else ""
            goal_str = f"{inp.strip()} || Candidate APIs: {api_desc}".strip()

            examples.append(
                {
                    "session_id": f"ab_{Path(fn).stem}_{idx:04d}_{row_id}",
                    "dialogue_group": file_id,
                    "step_index": len(recent_steps),
                    "goal": goal_str,
                    "recent_steps": recent_steps[-4:],
                    "target_tool": target_tool,
                    "raw_action": raw_api,
                    "source": fn,
                }
            )
    return examples, sources


def write_split_dataset(split_name: str, repo_name: str, examples: list[dict], sources: dict, group_key_fn) -> None:
    parts = assign_by_group(examples, group_key_fn, seed=SPLIT_SEED)
    out_dir = SPLITS / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split_name": split_name,
        "seed": SPLIT_SEED,
        "dataset": repo_name,
        "fractions": dict(FRACS),
        "n_total": len(examples),
        "source_sha256": sources,
        "splits": {},
    }
    for split, rows in parts.items():
        p = out_dir / f"{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][split] = {
            "rows": len(rows),
            "sha256": _sha(p),
            "ids_sha256": hashlib.sha256("\n".join(r["session_id"] for r in rows).encode()).hexdigest(),
            "labels": dist([r["target_tool"] for r in rows]),
            "multi_step_fraction": round(sum(1 for r in rows if len(r["recent_steps"]) > 0) / max(1, len(rows)), 4),
        }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"{split_name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        split_name,
        json.dumps(
            {
                s: {
                    "rows": v["rows"],
                    "multi_step_frac": v["multi_step_fraction"],
                    "n_classes": len(v["labels"]),
                }
                for s, v in manifest["splits"].items()
            },
            indent=2,
        ),
    )


def main() -> None:
    tb_ex, tb_src = build_toolbench(CACHE_DIR)
    write_split_dataset(
        "toolbench",
        "Yhyu13/ToolBench_toolllama_G123_dfs",
        tb_ex,
        tb_src,
        group_key_fn=lambda e: e["traj_id"],
    )

    ab_ex, ab_src = build_apibank(CACHE_DIR)
    write_split_dataset(
        "apibank",
        "liminghao1630/API-Bank",
        ab_ex,
        ab_src,
        group_key_fn=lambda e: e["dialogue_group"],
    )


if __name__ == "__main__":
    main()
