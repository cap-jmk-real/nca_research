"""Thin client for the Klaut research gateway on engine.

Default base URL matches docs/literature.md. Override with KLAUT_RESEARCH_URL.
Responses are cached under the external SSD so auto-research can proceed when
engine is offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE = os.environ.get("KLAUT_RESEARCH_URL", "http://192.168.10.29:30487")
DEFAULT_CACHE = Path(
    os.environ.get(
        "KLAUT_CACHE_DIR",
        "/Volumes/Extreme SSD/nca_research/results/cache/klaut",
    )
)


def _cache_path(kind: str, payload: dict) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return DEFAULT_CACHE / kind / f"{key}.json"


def _post(path: str, body: dict, timeout: float = 120.0) -> dict[str, Any]:
    base = DEFAULT_BASE.rstrip("/")
    url = f"{base}{path}"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "nca-research/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def search_papers(
    query: str,
    *,
    sources: list[str] | None = None,
    year_from: int = 2019,
    limit: int = 12,
    use_cache: bool = True,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST /v1/research/papers/search."""
    body = {
        "query": query,
        "sources": sources or ["arxiv", "openalex", "crossref"],
        "year_from": year_from,
        "limit": limit,
    }
    cache = _cache_path("search", body)
    if use_cache and cache.is_file():
        return json.loads(cache.read_text())
    try:
        result = _post("/v1/research/papers/search", body, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if use_cache:
            stale = sorted(DEFAULT_CACHE.glob("search/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if stale:
                return {
                    "error": str(exc),
                    "offline": True,
                    "fallback": json.loads(stale[0].read_text()),
                    "fallback_path": str(stale[0]),
                }
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2))
    return result


def create_report(
    topic: str,
    *,
    limit: int = 40,
    sources: list[str] | None = None,
    include_preprints: bool = True,
    use_cache: bool = True,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """POST /v1/research/reports — rapid literature synthesis."""
    body = {
        "topic": topic,
        "limit": limit,
        "sources": sources or ["openalex", "crossref", "arxiv"],
        "include_preprints": include_preprints,
    }
    cache = _cache_path("report", body)
    if use_cache and cache.is_file():
        return json.loads(cache.read_text())
    try:
        result = _post("/v1/research/reports", body, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if use_cache:
            stale = sorted(DEFAULT_CACHE.glob("report/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if stale:
                return {
                    "error": str(exc),
                    "offline": True,
                    "fallback": json.loads(stale[0].read_text()),
                    "fallback_path": str(stale[0]),
                }
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2))
    return result


def ping(timeout: float = 5.0) -> bool:
    try:
        search_papers("neural cellular automata", limit=1, use_cache=False, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 — reachability probe
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Klaut research gateway CLI")
    parser.add_argument("command", choices=("ping", "search", "report"))
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.command == "ping":
        print("ok" if ping() else "unreachable")
        return
    if not args.text:
        raise SystemExit("search/report need a query or topic string")
    if args.command == "search":
        out = search_papers(args.text, limit=args.limit, use_cache=not args.no_cache)
    else:
        out = create_report(args.text, limit=args.limit, use_cache=not args.no_cache)
    print(json.dumps(out, indent=2)[:12000])


if __name__ == "__main__":
    main()
