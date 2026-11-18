"""Replay a captured failing request (BFCL_ERRDUMP line) against Ollama /v1 with different
sampling overrides. Diagnostics only. usage: replay_errdump.py <errdump.jsonl> <line_idx> <pp|none> [n]"""
import json, sys, openai
d = [json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2])]
pp = sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 1
cli = openai.OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama", timeout=600)
tools = d["tools_full"] if d["n_tools"] else []
for i in range(n):
    kw = dict(model="qwen3.5-9b-bfcl32k", messages=d["messages"], temperature=0.001,
              extra_body={"reasoning_effort": "none"})
    if tools: kw["tools"] = tools
    if pp != "none": kw["presence_penalty"] = float(pp)
    try:
        r = cli.chat.completions.create(**kw)
        m = r.choices[0].message
        calls = [(c.function.name, c.function.arguments) for c in (m.tool_calls or [])]
        print(json.dumps({"id": d["id"], "pp": pp, "ok": True, "calls": calls, "text": (m.content or "")[:200]}))
    except Exception as e:
        print(json.dumps({"id": d["id"], "pp": pp, "ok": False, "error": str(e)[:200]}))
