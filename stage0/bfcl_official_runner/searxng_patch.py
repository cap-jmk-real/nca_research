"""Replace official WebSearchAPI.search_engine_query (SerpAPI) with Julian's SearXNG.

Does not change fetch_url_content, prompts, data, or the official checker.
Activated only when bfcl_nca.py sees BFCL_SEARCH_BACKEND=searxng.
"""
from __future__ import annotations

import os
import random
import time
from typing import Optional
from urllib.parse import urlencode

import requests

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:30479").rstrip("/")
# Default engines that actually returned results on this instance on 2026-11-11.
# Brave/DuckDuckGo/Startpage (the instance defaults) were CAPTCHA or rate-limited;
# Google returned empty with no error. Do not change the SearXNG deployment to fix that.
DEFAULT_ENGINES = os.environ.get("SEARXNG_ENGINES", "bing,mwmbl,wikipedia,wiby")


def _lang_from_region(region: Optional[str]) -> str:
    if not region or region == "wt-wt":
        return "en"
    parts = region.replace("_", "-").split("-")
    # SerpAPI kl values look like "de-de", "us-en", "xa-en"
    if len(parts) >= 2 and len(parts[-1]) == 2:
        return parts[-1].lower()
    return parts[0].lower() if parts else "en"


def _query_searxng(keywords: str, engines: str, language: str) -> tuple[list[dict], list]:
    params = {
        "q": keywords,
        "format": "json",
        "engines": engines,
        "language": language,
        "safesearch": 0,
    }
    url = f"{SEARXNG_URL}/search?{urlencode(params)}"
    resp = requests.get(url, timeout=25)
    if resp.status_code == 429:
        raise requests.HTTPError("429", response=resp)
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("results") or []), list(payload.get("unresponsive_engines") or [])


def search_engine_query(
    self,
    keywords: str,
    max_results: Optional[int] = 10,
    region: Optional[str] = "wt-wt",
) -> list:
    """Same return shape as the official SerpAPI implementation."""
    max_results = 10 if max_results is None else int(max_results)
    language = _lang_from_region(region)
    engines = DEFAULT_ENGINES
    backoff = 2
    last_error = None

    for attempt in range(8):
        try:
            raw, unresponsive = _query_searxng(keywords, engines, language)
            if unresponsive:
                print(
                    f"[WebSearchAPI/SearXNG] unresponsive engines for {keywords!r}: {unresponsive}"
                )
            if not raw:
                # Fall back to trying engines one at a time once, then retry the bundle.
                if attempt == 1:
                    pooled = []
                    for eng in [e.strip() for e in engines.split(",") if e.strip()]:
                        try:
                            part, _ = _query_searxng(keywords, eng, language)
                            pooled.extend(part)
                        except Exception as e:
                            print(f"[WebSearchAPI/SearXNG] engine {eng} failed: {e}")
                    raw = pooled
            if raw:
                seen = set()
                results = []
                for item in raw:
                    href = item.get("url") or item.get("href") or item.get("link") or ""
                    title = item.get("title") or ""
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    entry = {"title": title, "href": href}
                    if getattr(self, "show_snippet", True):
                        entry["body"] = item.get("content") or item.get("body") or ""
                    results.append(entry)
                    if len(results) >= max_results:
                        break
                if results:
                    return results
            last_error = "empty result set"
        except Exception as e:
            if "429" in str(e) or (getattr(e, "response", None) is not None and getattr(e.response, "status_code", None) == 429):
                wait_time = backoff + random.uniform(0, backoff)
                print(
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI/SearXNG] Received 429 from {SEARXNG_URL}. "
                    + f"Retrying in {wait_time:.1f} seconds…"
                    + "*" * 100
                )
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                last_error = str(e)
                continue
            last_error = str(e)
            print(f"[WebSearchAPI/SearXNG] error (attempt {attempt + 1}): {e}")
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)

    return {
        "error": f"Failed to retrieve the search results from server. Please try again later. ({last_error})"
    }


def apply() -> None:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search as ws

    ws.WebSearchAPI.search_engine_query = search_engine_query
    print(
        f"[WebSearchAPI/SearXNG] hooked search_engine_query -> {SEARXNG_URL} engines={DEFAULT_ENGINES}"
    )
