#!/usr/bin/env python3
"""Thin wrapper around the official `bfcl` CLI (bfcl-eval==2026.3.23).

Only adds two registry entries (no change to data, prompts, or checker):
  local/qwen3.5-9b-FC       Ollama qwen3.5:9b (num_ctx 32768 tag qwen3.5-9b-bfcl32k), native FC,
                            full official tool list (baseline_no_gnca)
  local/qwen3.5-9b-FC-gnca  same actuator; GNCA dual-rule respond gate before each query (gnca_gate)
Both use the stock OpenAICompletionsHandler request path pointed at Ollama's /v1 endpoint.
"""
import os
import sys
from pathlib import Path

REPO = os.environ.get("NCA_REPO", "/home/s4il0r/nca_bfcl_official_1111")
sys.path.insert(0, str(Path(REPO) / "stage0"))
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")  # dummy, Ollama ignores it

from bfcl_eval.constants import model_config as mc  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: E402

import bfcl_gnca_router_hook as hook  # noqa: E402

OLLAMA_TAG = os.environ.get("BFCL_ACTUATOR_OLLAMA", "qwen3.5-9b-bfcl32k")
REASONING = os.environ.get("BFCL_REASONING_EFFORT", "none")  # non-thinking mode
# Sampling override sent with every request. Without it Ollama inherits the qwen3.5:9b Modelfile
# default presence_penalty 1.5 (plus top_k 20 / top_p 0.95). BFCL's OpenAI/vLLM protocol only sets
# temperature, i.e. presence_penalty 0. Unset env = old behaviour (no override sent).
PRESENCE_PENALTY = os.environ.get("BFCL_PRESENCE_PENALTY")


# Ollama's built-in qwen3.5 tool-call parser returns HTTP 500 ("XML syntax error ...") when the
# model emits malformed <function=...><parameter=...> markup -- even for requests that carry no
# tools (GNCA gate withheld them). The stock BFCL client then marks the WHOLE multi-turn entry as
# inference_error. A vLLM/OpenAI server would instead return the unparseable text as plain content
# (no tool_calls), which BFCL treats as "no call, end of turn". With BFCL_OLLAMA_PARSE_FALLBACK=1 we
# reproduce that: the step is recorded as an empty assistant reply with no tool calls (identical rule
# for both conditions), and every occurrence is logged to BFCL_PARSE_FALLBACK_LOG for reporting.
PARSE_FALLBACK = os.environ.get("BFCL_OLLAMA_PARSE_FALLBACK", "0").strip() in {"1", "true", "yes"}


def _is_ollama_toolcall_parse_error(exc) -> bool:
    import openai as _oa
    msg = str(exc)
    return isinstance(exc, _oa.InternalServerError) and (
        "XML syntax error" in msg or "expected element type" in msg)


def _empty_turn_response(handler, inference_data, kwargs, exc):
    import json as _json
    import time as _time
    from openai.types.chat import ChatCompletion
    log = os.environ.get("BFCL_PARSE_FALLBACK_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(_json.dumps({"ts": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  "id": inference_data.get("_gnca_id") or inference_data.get("_entry_id"),
                                  "model": handler.model_name, "n_tools": len(kwargs.get("tools", [])),
                                  "n_messages": len(kwargs["messages"]), "error": str(exc)[:300]}) + "\n")
    return ChatCompletion.model_validate({
        "id": "ollama-parse-fallback", "object": "chat.completion", "created": int(_time.time()),
        "model": handler.model_name,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": ""}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


class OllamaFCHandler(OpenAICompletionsHandler):
    def _tools_for_query(self, inference_data):
        return inference_data["tools"]

    def _query_FC(self, inference_data: dict):
        message = inference_data["message"]
        tools = self._tools_for_query(inference_data)
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        kwargs = {"messages": message, "model": self.model_name, "temperature": self.temperature,
                  "extra_body": {"reasoning_effort": REASONING}}
        if PRESENCE_PENALTY is not None:
            kwargs["presence_penalty"] = float(PRESENCE_PENALTY)
        if len(tools) > 0:
            kwargs["tools"] = tools
        try:
            return self.generate_with_backoff(**kwargs)
        except Exception as exc:  # diagnostics only; the error is re-raised unchanged
            dump = os.environ.get("BFCL_ERRDUMP")
            if dump:
                import json as _json
                with open(dump, "a") as fh:
                    fh.write(_json.dumps({"id": inference_data.get("_gnca_id") or inference_data.get("_entry_id"),
                                          "error": str(exc)[:500], "n_tools": len(tools), "tools_full": inference_data["tools"],
                                          "presence_penalty": kwargs.get("presence_penalty"),
                                          "messages": message}, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else repr(o)) + "\n")
            if PARSE_FALLBACK and _is_ollama_toolcall_parse_error(exc):
                return _empty_turn_response(self, inference_data, kwargs, exc), 0.0
            raise


    def _compile_tools(self, inference_data, test_entry):
        inference_data = super()._compile_tools(inference_data, test_entry)
        inference_data["_entry_id"] = test_entry.get("id")
        return inference_data


class GNCAGatedHandler(OllamaFCHandler):
    def _tools_for_query(self, inference_data):
        if hook.gate(inference_data["message"], inference_data.get("_gnca_id")):
            return []
        return inference_data["tools"]

    def _compile_tools(self, inference_data, test_entry):
        inference_data = super()._compile_tools(inference_data, test_entry)
        inference_data["_gnca_id"] = test_entry.get("id")
        return inference_data


def _cfg(name, handler, display):
    return mc.ModelConfig(model_name=OLLAMA_TAG, display_name=display, url="https://ollama.com/library/qwen3.5",
                          org="Qwen (local Ollama)", license="apache-2.0", model_handler=handler,
                          input_price=None, output_price=None, is_fc_model=True, underscore_to_dot=True)


for d in (mc.MODEL_CONFIG_MAPPING, mc.api_inference_model_map):
    d["local/qwen3.5-9b-FC"] = _cfg("local/qwen3.5-9b-FC", OllamaFCHandler, "Qwen3.5-9B Q4_K_M Ollama (FC, no-think)")
    d["local/qwen3.5-9b-FC-gnca"] = _cfg("local/qwen3.5-9b-FC-gnca", GNCAGatedHandler,
                                         "Qwen3.5-9B Q4_K_M Ollama (FC, no-think) + GNCA respond gate")

from bfcl_eval.__main__ import cli  # noqa: E402

if os.environ.get("BFCL_SEARCH_BACKEND", "").lower() in {"searxng", "searx"}:
    import searxng_patch  # noqa: E402

    searxng_patch.apply()

if __name__ == "__main__":
    cli()
