"""Optional hook for official bfcl-eval: GNCA tool-list pruning before Ollama generation.

Engine integration (not required for manifest-only runs):
  export GNCA_ROUTER_ENABLED=1
  export GNCA_CHECKPOINT=/Volumes/Extreme SSD/nca_research/results/checkpoints/...

When wired into the Dual-System harness, this module loads the headline Tree GNCA
and returns a reduced candidate tool subset for the actuator prompt. Until that
wiring lands in the bfcl model handler, official runs on engine use the baseline
actuator-only condition documented in scripts/run_bfcl_official_engine.sh.
"""

from __future__ import annotations

import os


def router_enabled() -> bool:
    return os.environ.get("GNCA_ROUTER_ENABLED", "0").strip() in {"1", "true", "yes"}


def checkpoint_path() -> str | None:
    return os.environ.get("GNCA_CHECKPOINT")
