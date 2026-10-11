"""Hypothesis intake for one auto-research tick.

Called from ``scripts/auto_research.py --klaut-search`` (the daemon engine
tick). Searches the top failing gate with two or three short Klaut queries
and two GitHub repository queries. GitHub is where the code lives; Klaut is
the paper index.
A missing code URL is not a reject by itself. Off-domain work is rejected
(mammography segmentation, genetics drift, surveys with no method we can
implement). A hit is queued only when the record describes a concrete change
for text NCA or next-tool prediction that we can implement and score on a
named runnable harness. The ledger on the SSD keeps the next tick from
treating the same papers as new.

Does not start or stop training, does not clear STOP, and does not edit
``config/benchmark_gates.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG_PATH = ROOT / "docs" / "hypotheses-backlog.md"
LISTED_PATHS = (
    BACKLOG_PATH,
    ROOT / "docs" / "hypothesis-tests.md",
    ROOT / "docs" / "multi-harness-and-message-passing.md",
    ROOT / "docs" / "research-design.md",
    ROOT / "docs" / "text-nca-architecture.md",
)
INTAKE_PATH = Path(
    "/Volumes/Extreme SSD/nca_research/results/auto_research/klaut_intake.json"
)
GITHUB_CACHE = Path("/Volumes/Extreme SSD/nca_research/results/cache/github/search")

RUNNABLE = {
    "curated_val_top1_nca": "curated val next-tool",
    "local_hermes_top1_nca": "local Hermes",
    "bfcl_ast_accuracy": "BFCL AST",
    "bfcl_irrelevance_accuracy": "BFCL irrelevance",
    "nestful_full_sequence_accuracy": "Nestful full sequence",
}
BLOCKED = {
    "tau2_airline_pass1": "τ2 airline",
    "tau2_retail_pass1": "τ2 retail",
    "appworld_dev_tgc": "AppWorld",
}

OBJECTIVE = (
    "Best accuracy the runnable harnesses record "
    "(curated val next-tool, local Hermes, BFCL AST, BFCL irrelevance, "
    "Nestful full sequence). A curated val next-tool gain alone is not "
    "success while another runnable harness is failing or unscored. "
    "τ2 airline, τ2 retail, and AppWorld stay blocked until those harnesses "
    "can run."
)

# Mechanisms already measured or already written up. A hit that only restates
# one of these is not a new backlog row.
_KNOWN_PHRASES = (
    "long-range",
    "long range",
    "learned halt",
    "soft halt",
    "selective scan",
    "mamba",
    "regeneration",
    "asynchronous",
    "abstain",
    "irrelevance",
    "argument bucket",
    "ensemble",
    "rewiring",
)

_CODE_URL = re.compile(
    r"https?://(?:www\.)?(?:github|gitlab)\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.I,
)
# "Neural cellular automata" alone is not enough. Mammography, materials, and
# genetics papers use that phrase and are still off-domain for this loop.
_ON_TOPIC = re.compile(
    r"text neural cellular|"
    r"\btext nca\b|"
    r"neural cellular automata?.{0,60}\btext\b|"
    r"\btext\b.{0,60}neural cellular|"
    r"next[- ]tool|"
    r"(?:tool|function)[- ]call(?:ing)?.{0,48}(?:accura|predict|top-?1|sequence|abstain|argument)|"
    r"(?:accura|predict|top-?1).{0,48}(?:tool|function)[- ]call",
    re.I,
)
# A code URL is strong evidence of a change we can implement. These phrases
# are the other way a record describes that change without a repository link.
_IMPLEMENTABLE = re.compile(
    r"\battention|"
    r"\bhierarchical\b|"
    r"receptive field|"
    r"\bhalt(?:ing|ed)?\b|"
    r"message[- ]passing|"
    r"skip connection|"
    r"\babstain|"
    r"\birrelevan|"
    r"\bargument|"
    r"\bnested\b|"
    r"\breadout\b|"
    r"update rule|"
    r"local rule|"
    r"full[- ]sequence|"
    r"session memory",
    re.I,
)
_OFF_DOMAIN = (
    (re.compile(r"mammograph|breast tumor", re.I), "mammography segmentation"),
    (re.compile(r"brain tumor|\bmri\b", re.I), "medical image segmentation"),
    (
        re.compile(r"genetic drift|self-replicat|spontaneous mutation", re.I),
        "genetics drift",
    ),
    (re.compile(r"solidification|microstructure|corrosion", re.I), "materials modelling"),
    (re.compile(r"quantum[- ]dot", re.I), "quantum-dot hardware"),
    (re.compile(r"fungal morphology|graphics generation", re.I), "graphics morphology"),
    (re.compile(r"living systems create meaning", re.I), "unrelated essay"),
    (
        re.compile(r"elementary cellular automata|cellular automata classification", re.I),
        "elementary CA classification",
    ),
    (re.compile(r"toxic text", re.I), "toxic text classification"),
)
# Reasons written by the old "no code URL" rule. Re-judge those hits.
_STALE_CODE_REASONS = frozenset(
    {
        "on-topic but no code in the Klaut record",
        "no code and off-domain",
        "has code but not text NCA or next-tool prediction",
    }
)
# GitHub repos use hyphens. "neural-cellular-automata" is the same phrase.
_GITHUB_TOPIC = re.compile(
    r"neural cellular automata|"
    r"text neural cellular|"
    r"next[- ]tool|"
    r"function[- ]call|"
    r"tool[- ]call",
    re.I,
)
# A pixel-growing demo is not a text or tool-call change, even with a repo.
_GITHUB_IMAGE_ONLY = re.compile(
    r"\b(pixel|texture|growing|segmentation|video frame)\b",
    re.I,
)
_GITHUB_TEXT_OR_TOOL = re.compile(
    r"\b(?:text|language|tokens?|tools?|function call(?:ing)?|sequences?|readout)\b",
    re.I,
)
_SURVEY = re.compile(
    r"\b(survey|a review|critical review|overview|state of the art)\b",
    re.I,
)
_FORBIDDEN_DISTILL = re.compile(
    r"(hidden[- ]states?|logits?).{0,80}(distill|teacher)|"
    r"(distill|teacher).{0,80}(hidden[- ]states?|logits?)",
    re.I,
)
_INSERT_BEFORE = {
    "curated_val_top1_nca": "## Distillation (teacher ≤ 9B — **labels and text only**, not NCA weights)",
    "local_hermes_top1_nca": "## Distillation (teacher ≤ 9B — **labels and text only**, not NCA weights)",
    "bfcl_ast_accuracy": "## Multimodal (research-design H1 — not in benchmark gates yet)",
    "bfcl_irrelevance_accuracy": "## Multimodal (research-design H1 — not in benchmark gates yet)",
    "nestful_full_sequence_accuracy": "## Multimodal (research-design H1 — not in benchmark gates yet)",
}


def harness_status(gate_results: list[dict]) -> dict:
    """Split gates into failing, unscored, and blocked. Blocked stays blocked."""
    failing: list[dict] = []
    unscored: list[dict] = []
    blocked: list[dict] = []
    passing: list[dict] = []
    for row in gate_results:
        gate = row.get("gate")
        label = RUNNABLE.get(gate) or BLOCKED.get(gate) or gate
        if row.get("status") == "blocked" or gate in BLOCKED:
            blocked.append(
                {
                    "gate": gate,
                    "harness": label,
                    "status": "blocked",
                    "reason": row.get("reason") or "harness cannot run",
                }
            )
            continue
        if gate not in RUNNABLE:
            continue
        entry = {
            "gate": gate,
            "harness": label,
            "value": row.get("value"),
            "min": row.get("min"),
        }
        if row.get("value") is None:
            unscored.append(entry)
        elif row.get("status") == "pass":
            passing.append(entry)
        else:
            failing.append(entry)
    others_open = [
        item["gate"] for item in failing + unscored if item["gate"] != "curated_val_top1_nca"
    ]
    return {
        "objective": OBJECTIVE,
        "failing": failing,
        "unscored": unscored,
        "blocked": blocked,
        "passing": passing,
        "curated_val_alone_is_not_success": bool(others_open),
    }


def _blob(paper: dict) -> str:
    abstract = paper.get("abstract") or ""
    title = str(paper.get("title") or "")
    if paper.get("source") == "github":
        title = title.replace("-", " ").replace("_", " ").replace("/", " ")
    return f"{title} {abstract}"


def paper_id(paper: dict) -> str:
    return str(paper.get("id") or paper.get("doi") or "").strip()


def code_url(paper: dict) -> str | None:
    """Code link present on the Klaut record. Missing metadata is not code."""
    for key in ("code_url", "github", "repository", "repo_url"):
        value = paper.get(key)
        if isinstance(value, str):
            match = _CODE_URL.search(value)
            if match:
                return match.group(0)
    for key in ("url", "oa_url", "abstract"):
        value = paper.get(key)
        if isinstance(value, str):
            match = _CODE_URL.search(value)
            if match:
                return match.group(0)
    external = paper.get("external_ids")
    if isinstance(external, dict):
        match = _CODE_URL.search(json.dumps(external))
        if match:
            return match.group(0)
    return None


def bears_on_text_nca_or_next_tool(paper: dict) -> bool:
    blob = _blob(paper)
    if paper.get("source") == "github":
        return _GITHUB_TOPIC.search(blob) is not None
    return _ON_TOPIC.search(blob) is not None


def citation(paper: dict) -> str:
    doi = str(paper.get("doi") or "").strip()
    arxiv = None
    for blob in (paper.get("oa_url"), doi, paper.get("url")):
        text = str(blob or "")
        match = re.search(r"(\d{4}\.\d{4,5})", text)
        if match and ("arxiv" in text.lower() or doi.startswith("10.48550")):
            arxiv = match.group(1)
            break
    parts: list[str] = []
    if doi:
        parts.append(f"https://doi.org/{doi}")
    if arxiv:
        parts.append(f"arXiv:{arxiv}")
    if not parts:
        parts.append(str(paper.get("url") or paper.get("id") or ""))
    return ", ".join(parts)


def load_listed_text() -> str:
    parts = []
    for path in LISTED_PATHS:
        if path.is_file():
            parts.append(path.read_text())
    return "\n".join(parts)


def already_listed(paper: dict, listed_text: str) -> bool:
    listed = listed_text.lower()
    doi = str(paper.get("doi") or "").strip()
    if len(doi) >= 6 and doi.lower() in listed:
        return True
    pid = paper_id(paper)
    if len(pid) >= 6 and pid.lower() in listed:
        return True
    title = str(paper.get("title") or "").strip()
    if len(title) >= 20 and title.lower() in listed:
        return True
    url = code_url(paper)
    if url and url.lower() in listed:
        return True
    blob = _blob(paper).lower()
    for phrase in _KNOWN_PHRASES:
        if phrase in blob and phrase in listed:
            return True
    if re.search(r"\bhalting\b|\bhalt\b", blob) and re.search(r"\bhalt", listed):
        return True
    return False


def matching_harnesses(paper: dict) -> list[str]:
    text = _blob(paper).lower()
    found: list[str] = []
    if re.search(r"irrelevan|abstain", text):
        found.append("bfcl_irrelevance_accuracy")
    if re.search(r"function[- ]call", text) and re.search(r"\bast\b|argument", text):
        found.append("bfcl_ast_accuracy")
    if re.search(r"nestful|nested (api|tool)|full[- ]sequence", text):
        found.append("nestful_full_sequence_accuracy")
    if "hermes" in text or "session memory" in text:
        found.append("local_hermes_top1_nca")
    if re.search(r"next[- ]tool|neural cellular|tool[- ]call", text):
        found.append("curated_val_top1_nca")
    return found


def pick_harness(paper: dict, failing_gates: set[str]) -> str | None:
    matches = [gate for gate in matching_harnesses(paper) if gate in RUNNABLE]
    if not matches:
        return None
    preferred = [gate for gate in matches if gate in failing_gates and gate != "curated_val_top1_nca"]
    preferred += [gate for gate in matches if gate in failing_gates]
    preferred += [gate for gate in matches if gate != "curated_val_top1_nca"]
    preferred += matches
    seen: set[str] = set()
    for gate in preferred:
        if gate not in seen:
            seen.add(gate)
            return gate
    return None


def blocked_only(paper: dict) -> bool:
    text = _blob(paper).lower()
    names_blocked = bool(re.search(r"\bappworld\b|tau-?2|tau-bench", text))
    return names_blocked and pick_harness(paper, set(RUNNABLE)) is None


def off_domain_label(paper: dict) -> str | None:
    text = _blob(paper)
    for pattern, label in _OFF_DOMAIN:
        if pattern.search(text):
            return label
    return None


def describes_concrete_change(paper: dict) -> bool:
    """True when the record itself says what we would implement.

    A code URL is strong evidence. A named mechanism, or a real abstract
    about text NCA or next-tool prediction, also counts. An empty Klaut
    record that only shares the words "neural cellular automata" does not.
    """
    if code_url(paper) is not None:
        return True
    if _IMPLEMENTABLE.search(_blob(paper)):
        return True
    title = str(paper.get("title") or "")
    if _SURVEY.search(title):
        return False
    return len(str(paper.get("abstract") or "").strip()) >= 120


def named_duplicate(paper: dict) -> str | None:
    """H6, H20, and H_long_range_prod are already on the backlog."""
    blob = _blob(paper).lower()
    if re.search(r"long[- ]range", blob):
        return "duplicate of H6 or H_long_range_prod"
    if re.search(r"\bensemble\b", blob) and re.search(r"logit", blob):
        return "duplicate of H20"
    return None


def reject_reason(paper: dict, listed_text: str) -> str | None:
    """None means the hit can be queued as a backlog hypothesis.

    A missing code URL does not reject a paper. Off-domain work is rejected.
    What remains must describe a concrete change for text NCA or next-tool
    prediction that we can implement and score on a runnable harness.
    """
    off = off_domain_label(paper)
    if off:
        return f"off-domain ({off})"
    if paper.get("source") == "github":
        blob = _blob(paper)
        if _GITHUB_IMAGE_ONLY.search(blob) and not _GITHUB_TEXT_OR_TOOL.search(blob):
            return "image NCA repo, not a text or tool-call change"
        if not _GITHUB_TEXT_OR_TOOL.search(blob) and not _IMPLEMENTABLE.search(blob):
            return "repo does not describe a text or tool-call change"
    title = str(paper.get("title") or "")
    if _SURVEY.search(title) and not describes_concrete_change(paper):
        return "survey or review with no method we can implement"
    if not bears_on_text_nca_or_next_tool(paper):
        return "not a change for text NCA or next-tool prediction"
    if not describes_concrete_change(paper):
        return "record does not describe a concrete change we can implement"
    duplicate = named_duplicate(paper)
    if duplicate:
        return duplicate
    blob = _blob(paper)
    if _FORBIDDEN_DISTILL.search(blob):
        return "requires teacher hidden states or logits into TextNCA"
    if re.search(r"\bstop (the )?(training|run)\b", blob, re.I):
        return "would require stopping a 150k run"
    if blocked_only(paper):
        return "blocked harness τ2 or AppWorld; not scored"
    if pick_harness(paper, set(RUNNABLE)) is None:
        return "no runnable harness can score this"
    if already_listed(paper, listed_text):
        return "already cited in the project docs or backlog"
    return None


def _load_ledger() -> dict:
    if INTAKE_PATH.is_file():
        data = json.loads(INTAKE_PATH.read_text())
        data.setdefault("rejected", {})
        data.setdefault("ticks", [])
        return data
    return {"rejected": {}, "ticks": []}


def _write_ledger(ledger: dict) -> None:
    INTAKE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTAKE_PATH.write_text(json.dumps(ledger, indent=2) + "\n")


def _next_hypothesis_id(text: str) -> str:
    nums = [int(n) for n in re.findall(r"^### H(\d+)\b", text, re.M)]
    return f"H{max(nums, default=37) + 1}"


def _append_hypothesis(paper: dict, *, query: str, gate: str) -> str | None:
    """Insert one backlog row. Returns the new id, or None if it cannot be queued."""
    if not BACKLOG_PATH.is_file():
        return None
    text = BACKLOG_PATH.read_text()
    marker = _INSERT_BEFORE.get(gate)
    if not marker or marker not in text:
        return None
    cite = citation(paper)
    doi = str(paper.get("doi") or "").strip()
    if doi and doi.lower() in text.lower():
        return None
    pid = paper_id(paper)
    if pid and pid in text:
        return None
    url = code_url(paper) or ""
    label = RUNNABLE[gate]
    title = re.sub(r"\s+", " ", str(paper.get("title") or "Untitled")).strip()
    if len(title) > 90:
        title = title[:87].rstrip() + "..."
    title = title.replace("—", "-")
    hyp_id = _next_hypothesis_id(text)
    other = (
        "Other runnable harnesses still have to pass on their own scores. "
        if gate == "curated_val_top1_nca"
        else "Score it on this harness; do not treat a curated-val-only gain as clearing the rest. "
    )
    if paper.get("source") == "github" and url:
        lead = f"Adapt the repository `{url}` ({title})"
        origin = "github-intake"
    elif url:
        lead = f"Implement the method at `{url}` ({title})"
        origin = "klaut-intake"
    else:
        lead = f"Implement the change described by {title} (no code URL on the Klaut record)"
        origin = "klaut-intake"
    block = (
        f"### {hyp_id} — {title}\n"
        "\n"
        f"- **Claim:** {lead} as an ablation scored on {label}. "
        f"{other}"
        "Do not copy teacher hidden states or logits into TextNCA.\n"
        f"- **Gate:** `{gate}`\n"
        f"- **Harness:** {label}\n"
        "- **Priority:** 12\n"
        f"- **Query:** `{query}`\n"
        f"- **Cite:** {cite} ({pid})\n"
        f"- **Source:** {origin}\n"
        "- **Cost:** After the current 150k AdamW run. Do not stop a live train.\n"
        "\n"
        "---\n"
        "\n"
    )
    BACKLOG_PATH.write_text(text.replace(marker, block + marker, 1))
    return hyp_id


def _repo_record(item: dict) -> dict:
    full = str(item.get("full_name") or "").strip()
    html = str(item.get("html_url") or (f"https://github.com/{full}" if full else ""))
    return {
        "id": f"github:{full}" if full else html,
        "title": full or html,
        "abstract": item.get("description") or "",
        "doi": "",
        "url": html,
        "code_url": html,
        "source": "github",
        "stars": item.get("stargazers_count"),
    }


def _search_github(query: str) -> dict:
    """Repository search. Cache on the SSD for 12 hours. Never write to the boot disk."""
    if not Path("/Volumes/Extreme SSD").is_dir():
        return {"query": query, "ok": False, "error": "Extreme SSD is not mounted", "repos": []}
    phrase = " ".join(query.split())
    api_q = f'"{phrase}" in:name,description'
    digest = hashlib.sha256(api_q.encode()).hexdigest()[:16]
    cache = GITHUB_CACHE / f"{digest}.json"
    if cache.is_file() and time.time() - cache.stat().st_mtime < 12 * 3600:
        try:
            cached = json.loads(cache.read_text())
            cached["cached"] = True
            return cached
        except (OSError, json.JSONDecodeError):
            pass
    params = urllib.parse.urlencode(
        {
            "q": api_q,
            "sort": "stars",
            "order": "desc",
            "per_page": "5",
        }
    )
    request = urllib.request.Request(
        f"https://api.github.com/search/repositories?{params}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "nca-research-intake",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return {"query": query, "ok": False, "error": f"HTTP {exc.code}", "repos": []}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"query": query, "ok": False, "error": str(exc), "repos": []}
    items = payload.get("items") if isinstance(payload, dict) else None
    repos = [_repo_record(item) for item in items or [] if isinstance(item, dict)]
    result = {"query": query, "ok": True, "repos": repos, "cached": False}
    GITHUB_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result) + "\n")
    return result


def _search_one(query: str) -> dict:
    import klaut_gateway

    try:
        result = klaut_gateway.search_papers(query, limit=8, timeout=45, use_cache=True)
    except Exception as exc:  # noqa: BLE001 — record and continue with the other queries
        return {"query": query, "ok": False, "error": str(exc), "papers": []}
    if result.get("offline"):
        return {
            "query": query,
            "ok": False,
            "offline": True,
            "error": result.get("error"),
            "papers": [],
        }
    return {"query": query, "ok": True, "papers": result.get("papers") or []}


def _newest_search_cache() -> tuple[Path, dict] | None:
    import klaut_gateway

    folder = klaut_gateway.DEFAULT_CACHE / "search"
    if not folder.is_dir():
        return None
    files = [path for path in folder.glob("*.json") if path.is_file()]
    if not files:
        return None
    newest_m = max(path.stat().st_mtime for path in files)
    band = [path for path in files if newest_m - path.stat().st_mtime < 2]

    def _count(path: Path) -> int:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return -1
        papers = data.get("papers")
        return len(papers) if isinstance(papers, list) else -1

    best = max(band, key=_count)
    return best, json.loads(best.read_text())


def _klaut_up() -> bool:
    import klaut_gateway

    return bool(klaut_gateway.ping(timeout=5))


def _best_per_query(hits: list[dict], queries: list[str]) -> list[dict]:
    """First hit of each query, in search order. That is the query's best hit."""
    best: list[dict] = []
    for query in queries:
        for hit in hits:
            if hit.get("query") != query:
                continue
            best.append(
                {
                    "query": query,
                    "id": hit.get("id"),
                    "title": hit.get("title"),
                    "decision": hit.get("decision"),
                    "reason": hit.get("reason"),
                    "harness": hit.get("harness"),
                }
            )
            break
    return best


def intake_note(status: dict) -> str:
    """Scoreboard reminder. Values come from the gate file, not from Klaut."""
    parts: list[str] = []
    for item in status.get("failing") or []:
        value = item.get("value")
        minimum = item.get("min")
        harness = item.get("harness") or item.get("gate")
        if isinstance(value, (int, float)) and isinstance(minimum, (int, float)):
            parts.append(f"{harness} {value:.3f} vs {minimum:g}")
        else:
            parts.append(str(harness))
    failing = "; ".join(parts) if parts else "none"
    return (
        f"{OBJECTIVE} Failing harnesses: {failing}. "
        "A curated-val-only gain is not a pass while another runnable harness fails. "
        "τ2 and AppWorld stay blocked; no scores invented."
    )


def _slim(paper: dict, *, query: str, reason: str | None, repeat: bool, gate: str | None) -> dict:
    return {
        "id": paper_id(paper),
        "title": paper.get("title"),
        "doi": paper.get("doi"),
        "query": query,
        "has_code": code_url(paper) is not None,
        "code_url": code_url(paper),
        "source": paper.get("source") or "klaut",
        "on_topic": bears_on_text_nca_or_next_tool(paper),
        "decision": "reject" if reason else "candidate",
        "reason": reason,
        "repeat": repeat,
        "harness_gate": gate,
        "harness": RUNNABLE.get(gate or "", None),
        "cite": citation(paper),
    }


def run_intake(
    *,
    gate: str,
    queries: list[str],
    gate_results: list[dict],
    github_queries: list[str] | None = None,
) -> dict:
    """Search, ledger rejects, and maybe append one backlog row.

    A blocked gate is recorded and not searched. τ2 and AppWorld are never
    marked passed.
    """
    status = harness_status(gate_results)
    failing_gates = {item["gate"] for item in status["failing"]}
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    row = next((item for item in gate_results if item.get("gate") == gate), {})
    if row.get("status") == "blocked" or gate in BLOCKED:
        tick = {
            "ts": ts,
            "top_fail": gate,
            "queries": queries,
            "klaut_up": None,
            "searched": False,
            "hypothesis_added": None,
            "report_recommended": False,
            "report_ran": False,
            "harness_status": status,
            "note": (
                f"{BLOCKED.get(gate, gate)} is blocked. Not marked passed. No score invented. "
                + intake_note(status)
            ),
            "hits": [],
        }
        ledger = _load_ledger()
        ledger["latest"] = tick
        ledger["ticks"] = (ledger.get("ticks") or [])[-29:] + [tick]
        _write_ledger(ledger)
        return tick

    listed = load_listed_text()
    klaut_up = _klaut_up()
    searches: list[dict] = []
    pairs: list[tuple[str, dict]] = []
    cache_path = None
    source = "live"
    if klaut_up:
        for query in queries[:3]:
            one = _search_one(query)
            searches.append(
                {
                    "query": one["query"],
                    "ok": one["ok"],
                    "n": len(one.get("papers") or []),
                    "error": one.get("error"),
                }
            )
            for paper in one.get("papers") or []:
                if isinstance(paper, dict):
                    pairs.append((query, paper))
        if not pairs and searches and not any(item["ok"] for item in searches):
            klaut_up = False
    if not klaut_up:
        source = "ssd_cache"
        cached = _newest_search_cache()
        if cached:
            path, data = cached
            cache_path = str(path)
            cache_query = str(data.get("query") or "")
            papers = data.get("papers") if isinstance(data.get("papers"), list) else []
            searches.append(
                {
                    "query": cache_query,
                    "ok": False,
                    "offline": True,
                    "cache_path": cache_path,
                    "n": len(papers),
                    "note": "Klaut unreachable. Classified the newest SSD search cache. These papers are not hits for the targeted queries.",
                }
            )
            for paper in papers:
                if isinstance(paper, dict):
                    pairs.append((cache_query, paper))

    github_queries = [query for query in (github_queries or []) if query.strip()][:2]
    github_searches: list[dict] = []
    for query in github_queries:
        one = _search_github(query)
        github_searches.append(
            {
                "query": query,
                "ok": one.get("ok"),
                "n": len(one.get("repos") or []),
                "cached": one.get("cached"),
                "error": one.get("error"),
            }
        )
        for repo in one.get("repos") or []:
            if isinstance(repo, dict):
                pairs.append((query, repo))

    ledger = _load_ledger()
    prior = ledger.get("rejected") or {}
    hits: list[dict] = []
    seen: set[str] = set()
    report_recommended = False
    for query, paper in pairs:
        pid = paper_id(paper)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        prior_reason = str((prior.get(pid) or {}).get("reason") or "")
        repeat = pid in prior and prior_reason not in _STALE_CODE_REASONS
        if repeat:
            reason = prior_reason or "already rejected"
            chosen_gate = None
        else:
            reason = reject_reason(paper, listed)
            chosen_gate = None if reason else pick_harness(paper, failing_gates)
            if reason is None and chosen_gate is None:
                reason = "no runnable harness can score this"
        hit = _slim(paper, query=query, reason=reason, repeat=repeat, gate=chosen_gate)
        if (
            not repeat
            and paper.get("source") != "github"
            and code_url(paper)
            and "neural cellular" in str(paper.get("title") or "").lower()
            and not already_listed(paper, listed)
            and not _SURVEY.search(str(paper.get("title") or ""))
        ):
            report_recommended = True
            hit["new_architecture_family"] = True
        hits.append(hit)
        if not repeat and reason:
            prior[pid] = {
                "query": query,
                "title": paper.get("title"),
                "reason": reason,
                "ts": ts,
            }

    candidates = [hit for hit in hits if hit["decision"] == "candidate" and not hit["repeat"]]
    candidates.sort(
        key=lambda hit: (
            hit.get("source") == "github",
            hit.get("harness_gate") in failing_gates and hit.get("harness_gate") != "curated_val_top1_nca",
            hit.get("harness_gate") in failing_gates,
        ),
        reverse=True,
    )
    hypothesis_added = None
    papers_by_id = {paper_id(paper): paper for _, paper in pairs}
    if candidates:
        chosen = candidates[0]
        paper = papers_by_id.get(chosen["id"])
        if paper is not None and chosen.get("harness_gate") in RUNNABLE:
            hypothesis_added = _append_hypothesis(
                paper, query=str(chosen.get("query") or ""), gate=str(chosen["harness_gate"])
            )
            chosen["added_id"] = hypothesis_added

    new_hits = [hit for hit in hits if not hit["repeat"]]
    tick = {
        "ts": ts,
        "top_fail": gate,
        "queries": list(queries[:3]),
        "github_queries": github_queries,
        "github_ok": bool(github_searches) and any(item.get("ok") for item in github_searches),
        "github_searches": github_searches,
        "source": source,
        "klaut_up": klaut_up,
        "cache_path": cache_path,
        "searches": searches,
        "hypothesis_added": hypothesis_added,
        "report_recommended": report_recommended,
        "report_ran": False,
        "harness_status": status,
        "note": intake_note(status),
        "best_per_query": _best_per_query(new_hits, list(queries[:3]) + github_queries),
        "new_hit_count": len(new_hits),
        "repeat_ids": [hit["id"] for hit in hits if hit["repeat"]],
        "hits": new_hits,
        "rejected_this_tick": [
            {"id": hit["id"], "query": hit["query"], "reason": hit["reason"], "title": hit["title"]}
            for hit in new_hits
            if hit["decision"] == "reject"
        ],
    }
    ledger["rejected"] = prior
    ledger["latest"] = tick
    ledger["ticks"] = (ledger.get("ticks") or [])[-29:] + [tick]
    _write_ledger(ledger)
    tick["intake_path"] = str(INTAKE_PATH)
    preview = [
        {"id": hit["id"], "title": hit["title"], "decision": hit["decision"], "reason": hit["reason"]}
        for hit in new_hits[:4]
    ]
    tick["hits_preview"] = preview
    tick["github_hits"] = [
        {
            "id": hit["id"],
            "title": hit["title"],
            "code_url": hit.get("code_url"),
            "decision": hit["decision"],
            "reason": hit["reason"],
        }
        for hit in new_hits
        if hit.get("source") == "github"
    ][:6]
    return tick
