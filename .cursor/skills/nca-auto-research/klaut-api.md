# Klaut gateway API (nca_research)

Copied from measured usage in [docs/literature.md](../../../docs/literature.md) (2026-10-03).

## Base URL

- LAN: `http://192.168.10.29:30487`
- Env override: `KLAUT_RESEARCH_URL`
- Public `https://research.klaut.pro` was **not** reachable from the M1 in prior runs; prefer LAN.

## Search papers

```http
POST /v1/research/papers/search
Content-Type: application/json

{
  "query": "neural cellular automata tool agent",
  "sources": ["arxiv", "openalex", "crossref"],
  "year_from": 2019,
  "limit": 12
}
```

Use for: quick hypothesis grounding, dataset discovery, related work before ablations.

## Literature synthesis report

```http
POST /v1/research/reports
Content-Type: application/json

{
  "topic": "Neural cellular automata for perception, memory, and agent control",
  "limit": 40,
  "sources": ["openalex", "crossref", "arxiv"],
  "include_preprints": true
}
```

Use for: new **architecture families** only (expensive; synthesis model is `qwen3.5:9b` on engine).

Reports may land on gateway pod storage, e.g. `/data/reports/<slug>.md` — copy salient citations into repo docs when useful.

## Client

[scripts/klaut_gateway.py](../../../scripts/klaut_gateway.py) caches JSON under:

`/Volumes/Extreme SSD/nca_research/results/cache/klaut/{search,report}/`

## Quality checks

- Mark `is_preprint` / arXiv records explicitly in docs.
- If synthesis mixes classical CAs with neural NCAs, filter before design decisions ([literature.md](../../../docs/literature.md) warning).
- Prefer DOI/OpenAlex IDs from search hits over synthesis prose.
