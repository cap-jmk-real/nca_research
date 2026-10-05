"""Next-tool rows from a public Hermes slice and held-out local sessions.

Raw local transcripts are read in memory and never written out. Public text is
truncated and cached only under the external data root.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HASH_C = 8
GRID_W = 8
MAX_STEPS = 5
GRID_H = 1 + MAX_STEPS
# 1D trace layout: cell 0 is the goal, cells 1..TRACE_STEPS are recent steps.
TRACE_STEPS = 4
SEQ_LEN = 1 + TRACE_STEPS
HASH_DIM = 48
TEXT_CAP = 160
PUBLIC_PAGES = 4
PUBLIC_PAGE_LEN = 100
DATASETS_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=lambda/hermes-agent-reasoning-traces"
    "&config=kimi&split=train"
)

_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')
_TAG = re.compile(r"</?[^>]+>")


@dataclass
class Example:
    session_id: str
    label: str
    goal: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    source: str = "public"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Results directory on the external SSD.

    docs/data-root.md is shared with dataset curation and lists several paths.
    This training run writes under the Results row, not the volume root.
    """
    env = os.environ.get("NCA_DATA_ROOT")
    if env:
        path = Path(env)
        if path.name != "results":
            path = path / "results"
    else:
        text = (repo_root() / "docs" / "data-root.md").read_text()
        results = re.search(r"\|\s*Results\s*\|\s*`(/Volumes/[^`]+)`", text)
        if results:
            path = Path(results.group(1).strip())
        else:
            listed = re.search(r"\|\s*Data root\s*\|\s*`(/Volumes/[^`]+)`", text)
            if not listed:
                raise SystemExit("docs/data-root.md has no Results or Data root path")
            path = Path(listed.group(1).strip()) / "results"
    assert_external(path)
    # Never treat the volume mount itself as a writable root.
    if path.parent == Path("/Volumes"):
        raise SystemExit(f"refusing to write at the volume root: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_external(path: Path) -> None:
    resolved = str(path.resolve())
    if not resolved.startswith("/Volumes/"):
        raise SystemExit(f"refusing to write outside /Volumes: {resolved}")
    if "Macintosh HD" in resolved:
        raise SystemExit(f"refusing internal volume: {resolved}")


def prepare_env(root: Path) -> None:
    """Point caches at the external SSD before any framework import side effects."""
    assert_external(root)
    mapping = {
        "HF_HOME": root / "hf-home",
        "HF_HUB_CACHE": root / "hf-home" / "hub",
        "HF_DATASETS_CACHE": root / "hf-home" / "datasets",
        "TMPDIR": root / "tmp",
        "TORCH_HOME": root / "torch-home",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor-cache",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "PYTHONPYCACHEPREFIX": root / "pycache",
        "PIP_CACHE_DIR": root / "pip-cache",
    }
    for key, dest in mapping.items():
        dest.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(dest)


def _short(value: str, cap: int = TEXT_CAP) -> str:
    if not value:
        return ""
    text = _TAG.sub(" ", value[:2000])
    return " ".join(text.split())[:cap]


def _first_tool_name(value: str) -> str | None:
    if not value or "<tool_call>" not in value:
        return None
    block = _TOOL_BLOCK.search(value)
    blob = block.group(1) if block else value[:2000]
    match = _NAME.search(blob[:2000])
    return match.group(1) if match else None


def is_browser(name: str) -> bool:
    return name.lower().startswith("browser_")


def _examples_from_messages(session_id: str, messages: list[tuple[str, str]], source: str) -> list[Example]:
    """messages are (role, text) with role in {human, gpt, tool}."""
    goal = ""
    steps: list[tuple[str, str]] = []
    out: list[Example] = []
    for role, text in messages:
        if role == "human":
            goal = _short(text)
        elif role == "gpt":
            name = _first_tool_name(text)
            label = name or "respond"
            if not is_browser(label):
                out.append(
                    Example(
                        session_id=session_id,
                        label=label,
                        goal=goal,
                        steps=list(steps[-MAX_STEPS:]),
                        source=source,
                    )
                )
        elif role == "tool":
            match = _NAME.search((text or "")[:1500])
            tool = match.group(1) if match else "tool"
            steps.append((tool, _short(text)))
            if len(steps) > MAX_STEPS:
                steps = steps[-MAX_STEPS:]
    return out


def examples_from_sharegpt(row: dict, source: str = "public") -> list[Example]:
    messages = []
    for msg in row.get("conversations") or []:
        role = msg.get("from") or ""
        if role not in {"human", "gpt", "tool"}:
            continue
        messages.append((role, msg.get("value") or ""))
    return _examples_from_messages(str(row.get("id") or ""), messages, source)


def load_synthetic() -> list[Example]:
    path = Path(__file__).resolve().parent / "fixtures" / "synthetic_traces.json"
    rows = json.loads(path.read_text())
    examples: list[Example] = []
    for row in rows:
        examples.extend(examples_from_sharegpt(row, source="synthetic"))
    return examples


def fetch_public(root: Path) -> tuple[list[Example], str]:
    cache = root / "cache" / "hermes_kimi_head_truncated.json"
    assert_external(cache.parent)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = json.loads(cache.read_text())
        examples = [_from_cache_row(row) for row in payload["examples"]]
        return examples, payload.get("note") or "loaded truncated public cache"

    examples: list[Example] = []
    pages = 0
    for page in range(PUBLIC_PAGES):
        offset = page * PUBLIC_PAGE_LEN
        url = f"{DATASETS_ROWS}&offset={offset}&length={PUBLIC_PAGE_LEN}"
        request = urllib.request.Request(url, headers={"User-Agent": "nca-research/0.1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.load(response)
        rows = body.get("rows") or []
        pages += 1
        for item in rows:
            examples.extend(examples_from_sharegpt(item["row"], source="public"))
        if len(rows) < PUBLIC_PAGE_LEN:
            break

    note = (
        f"lambda/hermes-agent-reasoning-traces config=kimi, first {pages * PUBLIC_PAGE_LEN} "
        f"rows requested from the datasets-server rows API "
        f"({pages} page(s) of {PUBLIC_PAGE_LEN}); not a random sample of the full set. "
        "Browser-prefixed targets dropped. Local ~/.hermes sessions are not in this cache."
    )
    serial = {
        "note": note,
        "examples": [
            {
                "session_id": ex.session_id,
                "label": ex.label,
                "goal": ex.goal,
                "steps": [{"tool": tool, "text": text} for tool, text in ex.steps],
            }
            for ex in examples
        ],
    }
    cache.write_text(json.dumps(serial))
    return examples, note


def _from_cache_row(row: dict) -> Example:
    steps = [(s["tool"], s["text"]) for s in row.get("steps") or []]
    return Example(
        session_id=row["session_id"],
        label=row["label"],
        goal=row.get("goal") or "",
        steps=steps,
        source="public",
    )


def load_local_test(db_path: Path | None = None) -> tuple[list[Example], str]:
    path = db_path or (Path.home() / ".hermes" / "state.db")
    if not path.exists():
        return [], f"local db missing at {path.name}"
    uri = f"file:{path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT session_id, role,
                   substr(COALESCE(content, ''), 1, 400),
                   COALESCE(tool_name, ''),
                   tool_calls
            FROM messages
            WHERE compacted = 0 AND active = 1
            ORDER BY session_id, timestamp, id
            """
        )
        examples: list[Example] = []
        current_id: str | None = None
        goal = ""
        steps: list[tuple[str, str]] = []

        def emit_assistant(tool_calls: str | None) -> None:
            label = _assistant_label(tool_calls)
            if label is None or is_browser(label):
                return
            examples.append(
                Example(
                    session_id=current_id or "",
                    label=label,
                    goal=goal,
                steps=list(steps[-TRACE_STEPS:]),
                source="local",
            )
        )

        for session_id, role, content, tool_name, tool_calls in cur:
            if session_id != current_id:
                current_id = session_id
                goal = ""
                steps = []
            if role == "user":
                goal = _short(content or "")
            elif role == "assistant":
                emit_assistant(tool_calls)
            elif role == "tool":
                name = tool_name or "tool"
                steps.append((name, _short(content or "")))
                if len(steps) > TRACE_STEPS:
                    steps = steps[-TRACE_STEPS:]
        return examples, "local ~/.hermes/state.db held out as test; transcripts not written to disk"
    finally:
        con.close()


def _assistant_label(tool_calls: str | None) -> str | None:
    if not tool_calls or tool_calls in {"[]", "null"}:
        return "respond"
    try:
        calls = json.loads(tool_calls)
    except json.JSONDecodeError:
        return "respond"
    if not isinstance(calls, list) or not calls:
        return "respond"
    first = calls[0]
    if not isinstance(first, dict):
        return "respond"
    fn = first.get("function") or {}
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    if first.get("name"):
        return str(first["name"])
    return "respond"


def public_split(session_id: str) -> str:
    return "val" if (zlib.crc32(session_id.encode()) % 5 == 0) else "train"


def hash_features(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, np.float32)
    tokens = (text or "").lower().split()[:48]
    if not tokens and text:
        tokens = [text[:32].lower()]
    for tok in tokens:
        digest = zlib.crc32(tok.encode("utf-8", "ignore")) & 0xFFFFFFFF
        vec[digest % dim] += 1.0
    norm = float(np.sqrt(np.dot(vec, vec)))
    if norm > 0:
        vec /= norm
    return vec


def build_obs(
    examples: list[Example],
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[Example]]:
    """Pack goal + recent steps onto a grid. Drops labels outside vocab."""
    hash_dim = HASH_C * GRID_W
    obs_c = HASH_C + len(vocab)
    kept: list[Example] = []
    grids = []
    labels = []
    for ex in examples:
        if ex.label not in vocab:
            continue
        obs = np.zeros((obs_c, GRID_H, GRID_W), np.float32)
        obs[:HASH_C, 0] = hash_features(ex.goal, hash_dim).reshape(HASH_C, GRID_W)
        steps = ex.steps[-MAX_STEPS:]
        start = GRID_H - len(steps)
        for i, (tool, text) in enumerate(steps):
            row = start + i
            blob = f"{tool} {text}".strip()
            obs[:HASH_C, row] = hash_features(blob, hash_dim).reshape(HASH_C, GRID_W)
            idx = vocab.get(tool)
            if idx is not None:
                obs[HASH_C + idx, row, :] = 1.0
        grids.append(obs)
        labels.append(vocab[ex.label])
        kept.append(ex)
    if not grids:
        empty = np.zeros((0, obs_c, GRID_H, GRID_W), np.float32)
        return empty, np.zeros((0,), np.int64), []
    return np.stack(grids), np.asarray(labels, np.int64), kept


def datasets_dir() -> Path:
    """Sibling of the results directory named in docs/data-root.md."""
    path = data_root().parent / "datasets"
    assert_external(path)
    if not path.is_dir():
        raise SystemExit(f"datasets directory missing: {path}")
    return path


def _examples_from_jsonl(path: Path, source: str) -> list[Example]:
    examples: list[Example] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            steps = [
                (str(step.get("tool") or ""), str(step.get("result") or ""))
                for step in (row.get("recent_steps") or [])
            ]
            label = str(row.get("target_tool") or "")
            if not label:
                continue
            examples.append(
                Example(
                    session_id=str(row.get("session_id") or ""),
                    label=label,
                    goal=str(row.get("goal") or ""),
                    steps=steps[-TRACE_STEPS:],
                    source=source,
                )
            )
    return examples


def load_curated_split(split: str) -> tuple[list[Example], str]:
    path = datasets_dir() / "curated" / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing curated split: {path}")
    examples = _examples_from_jsonl(path, "curated")
    return examples, f"{path} ({len(examples)} rows)"


def training_split_paths(split: str) -> list[Path]:
    """Train rows that share the Hermes tool names. Val is the curated split only.

    tau2 airline/retail rows use a different API vocabulary, so they stay on disk
    and out of this head. browser_* rows were dropped from the Hermes label set.
    extra/val.jsonl is the held-out remainder of the extra Hermes source, not the
    validation set for this run. fineenvs_common_val.jsonl is also held out of
    this list and is scored only as a side metric.
    """
    root = datasets_dir()
    extra = root / "extra"
    if split == "val":
        paths = [root / "curated" / "val.jsonl"]
    else:
        paths = [
            root / "curated" / "train.jsonl",
            extra / "train.jsonl",
            extra / "openhands_train.jsonl",
            extra / "swe_smith_partial_train.jsonl",
            # Verifier-correct multi-harness tool calls. Absent until staged.
            extra / "fineenvs_common_train.jsonl",
        ]
    return [path for path in paths if path.is_file()]


def load_side_eval(filename: str, vocab: dict[str, int]):
    """Pack a held-out jsonl with the training vocabulary. None if the file is absent.

    Used for FineEnvs task-held-out rows. Those rows are not the checkpoint metric.
    """
    path = datasets_dir() / "extra" / filename
    if not path.is_file():
        return None
    examples = _examples_from_jsonl(path, path.stem)
    obs, labels, kept = build_trace_obs(examples, vocab)
    return obs, labels, f"{path.name} ({len(kept)} of {len(examples)} rows)"


def load_training_examples(split: str) -> tuple[list[Example], str]:
    paths = training_split_paths(split)
    if not paths:
        raise SystemExit(f"no jsonl for split {split}")
    examples: list[Example] = []
    notes: list[str] = []
    for path in paths:
        rows = _examples_from_jsonl(path, path.stem)
        examples.extend(rows)
        notes.append(f"{path.name} ({len(rows)} rows)")
    return examples, "; ".join(notes)


def _iter_jsonl_rows(paths: list[Path]):
    for path in paths:
        n = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = str(row.get("target_tool") or "")
                if not label:
                    continue
                n += 1
                yield f"{path.parent.name}/{path.name}", row, label
        print(f"scanned {path.name} ({n} rows)", flush=True)


def _fill_obs(
    paths: list[Path],
    vocab: dict[str, int],
    n_hint: int,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Second pass: pack rows. Large splits are written to the SSD in small chunks.

    A single memmap or zeros() of the full train grid is about 600 MB. This
    machine often has less free memory than that, and the kernel kills the
    process. A few-megabyte buffer written with tofile stays anonymous-small.
    """
    n_classes = len(vocab)
    obs_c = HASH_DIM + n_classes + 1
    sessions: set[str] = set()
    labels = np.empty(n_hint, np.int64)
    fineenvs = np.zeros(n_hint, np.uint8)
    filled = 0
    chunk = 4096
    buf = np.zeros((chunk, obs_c, SEQ_LEN), np.float32)
    buf_i = 0
    in_mem = None if cache_path is not None else np.zeros((n_hint, obs_c, SEQ_LEN), np.float32)
    out = None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cache_path.unlink()
        out = cache_path.open("wb")

    def flush_buf() -> None:
        nonlocal buf_i
        if buf_i == 0 or out is None:
            buf_i = 0
            return
        buf[:buf_i].tofile(out)
        buf_i = 0

    for name, row, label in _iter_jsonl_rows(paths):
        if label not in vocab:
            continue
        sessions.add(str(row.get("session_id") or ""))
        if filled >= n_hint:
            raise SystemExit(f"packed more rows than scanned ({n_hint})")
        obs = buf[buf_i] if out is not None else in_mem[filled]
        obs[:] = 0
        obs[:HASH_DIM, 0] = hash_features(str(row.get("goal") or ""), HASH_DIM)
        obs[-1, 0] = 1.0
        steps = row.get("recent_steps") or []
        steps = steps[-TRACE_STEPS:]
        start = SEQ_LEN - len(steps)
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "")
            text = str(step.get("result") or "")
            col = start + i
            obs[:HASH_DIM, col] = hash_features(f"{tool} {text}".strip(), HASH_DIM)
            idx = vocab.get(tool)
            if idx is not None:
                obs[HASH_DIM + idx, col] = 1.0
            obs[-1, col] = 1.0
        labels[filled] = vocab[label]
        if "fineenvs_common_train" in name:
            fineenvs[filled] = 1
        filled += 1
        if out is not None:
            buf_i += 1
            if buf_i == chunk:
                flush_buf()
        if filled % 100_000 == 0:
            print(f"packed {filled} from {name}", flush=True)
    flush_buf()
    if out is not None:
        out.close()
        obs_all = np.memmap(
            cache_path, dtype=np.float32, mode="r", shape=(filled, obs_c, SEQ_LEN)
        )
    else:
        obs_all = in_mem[:filled].copy() if filled < n_hint else in_mem
    if filled < n_hint:
        labels = labels[:filled].copy()
        fineenvs = fineenvs[:filled].copy()
    return obs_all, labels, len(sessions), fineenvs


def load_packed_splits() -> dict:
    """Scan train and val once, then pack. Does not keep the json rows."""
    train_paths = training_split_paths("train")
    val_paths = training_split_paths("val")
    if not train_paths or not val_paths:
        raise SystemExit(f"missing train or val jsonl: {train_paths} {val_paths}")
    print("scanning train labels", flush=True)
    train_labels: set[str] = set()
    train_sessions: set[str] = set()
    train_counts: dict[str, int] = {}
    train_n = 0
    for name, row, label in _iter_jsonl_rows(train_paths):
        train_labels.add(label)
        train_sessions.add(str(row.get("session_id") or ""))
        train_counts[name] = train_counts.get(name, 0) + 1
        train_n += 1
    print("scanning val sessions", flush=True)
    val_sessions: set[str] = set()
    val_counts: dict[str, int] = {}
    val_n = 0
    for name, row, label in _iter_jsonl_rows(val_paths):
        val_sessions.add(str(row.get("session_id") or ""))
        val_counts[name] = val_counts.get(name, 0) + 1
        val_n += 1
    overlap = train_sessions & val_sessions
    if overlap:
        raise SystemExit(f"train/val session overlap: {len(overlap)}")
    vocab = {name: i for i, name in enumerate(sorted(train_labels))}
    del train_sessions, val_sessions, train_labels
    import gc
    gc.collect()
    print(f"packing train into {len(vocab)} classes", flush=True)
    cache = data_root() / "cache" / "train_obs.f32"
    train_obs, train_y, train_session_n, train_fineenvs = _fill_obs(
        train_paths, vocab, train_n, cache
    )
    print("packing val", flush=True)
    val_obs, val_y, val_session_n, _val_fineenvs = _fill_obs(val_paths, vocab, val_n)
    train_note = "; ".join(f"{name} ({n} rows)" for name, n in train_counts.items())
    val_note = "; ".join(f"{name} ({n} rows)" for name, n in val_counts.items())
    return {
        "vocab": vocab,
        "train_obs": train_obs,
        "train_y": train_y,
        "val_obs": val_obs,
        "val_y": val_y,
        "train_sessions": train_session_n,
        "val_sessions": val_session_n,
        "train_note": train_note,
        "val_note": val_note,
        "train_fineenvs": train_fineenvs,
    }


def build_trace_obs(
    examples: list[Example],
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[Example]]:
    """Pack goal + recent steps as a 1D cell sequence. Drops labels outside vocab."""
    n_classes = len(vocab)
    obs_c = HASH_DIM + n_classes + 1
    usable = [ex for ex in examples if ex.label in vocab]
    if not usable:
        empty = np.zeros((0, obs_c, SEQ_LEN), np.float32)
        return empty, np.zeros((0,), np.int64), []
    obs_all = np.zeros((len(usable), obs_c, SEQ_LEN), np.float32)
    labels = np.empty(len(usable), np.int64)
    for n, ex in enumerate(usable):
        obs = obs_all[n]
        obs[:HASH_DIM, 0] = hash_features(ex.goal, HASH_DIM)
        obs[-1, 0] = 1.0
        steps = ex.steps[-TRACE_STEPS:]
        start = SEQ_LEN - len(steps)
        for i, (tool, text) in enumerate(steps):
            col = start + i
            obs[:HASH_DIM, col] = hash_features(f"{tool} {text}".strip(), HASH_DIM)
            idx = vocab.get(tool)
            if idx is not None:
                obs[HASH_DIM + idx, col] = 1.0
            obs[-1, col] = 1.0
        labels[n] = vocab[ex.label]
        if n and n % 100_000 == 0:
            print(f"packed {n}/{len(usable)}", flush=True)
    return obs_all, labels, usable
