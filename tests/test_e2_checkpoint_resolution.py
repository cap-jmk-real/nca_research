"""Fast checks for E2 E1-checkpoint resolution (no training)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage0"))

import run_e2_router_ablation as e2  # noqa: E402


class TestResolveE1Checkpoint(unittest.TestCase):
    def test_missing_star_exits_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            with patch.object(e2, "CKPT_DIR", ckpt_dir):
                with self.assertRaises(SystemExit) as ctx:
                    e2._resolve_e1_checkpoint(0, "star", dry_run=True)
                msg = str(ctx.exception)
                self.assertIn("star", msg)
                self.assertIn("seed0_star.pt", msg)
                self.assertIn("run_e1.sh --dry-run", msg)

    def test_primary_seed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            p = ckpt_dir / "seed1_star.pt"
            p.write_bytes(b"x")
            with patch.object(e2, "CKPT_DIR", ckpt_dir):
                self.assertEqual(e2._resolve_e1_checkpoint(1, "star", dry_run=False), p)

    def test_seed_fallback_to_seed0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            fb = ckpt_dir / "seed0_meta.pt"
            fb.write_bytes(b"x")
            with patch.object(e2, "CKPT_DIR", ckpt_dir):
                self.assertEqual(e2._resolve_e1_checkpoint(2, "meta", dry_run=False), fb)


if __name__ == "__main__":
    unittest.main()
