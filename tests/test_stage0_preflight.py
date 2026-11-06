"""E1/E2 preflight: NCA_DATA_ROOT and mandatory research-backup hook."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage0"))

import common  # noqa: E402


class TestRequireNcaDataRoot(unittest.TestCase):
    def test_unset_exits(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NCA_DATA_ROOT"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                common.require_nca_data_root()
            self.assertIn("NCA_DATA_ROOT is not set", str(ctx.exception))

    def test_missing_dir_exits(self) -> None:
        missing = Path(tempfile.gettempdir()) / "nca_nonexistent_test_dir_xyz"
        with patch.dict(os.environ, {"NCA_DATA_ROOT": str(missing)}, clear=False):
            with self.assertRaises(SystemExit) as ctx:
                common.require_nca_data_root()
            self.assertIn("does not exist", str(ctx.exception))

    def test_writable_dir_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"NCA_DATA_ROOT": tmp}, clear=False):
                root = common.require_nca_data_root()
                self.assertEqual(root, Path(tmp).resolve())


class TestMaybeResearchBackup(unittest.TestCase):
    def _home_without_hook(self) -> Path:
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        home = Path(tmp)
        (home / "bin").mkdir()
        return home

    def test_missing_hook_aborts_without_optional(self) -> None:
        home = self._home_without_hook()
        env = {k: v for k, v in os.environ.items() if k != "RESEARCH_BACKUP_OPTIONAL"}
        with patch.object(Path, "home", return_value=home):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit) as ctx:
                    common.maybe_research_backup("e1-seed0")
                self.assertIn("research-backup-now", str(ctx.exception))

    def test_missing_hook_skipped_when_optional(self) -> None:
        home = self._home_without_hook()
        with patch.object(Path, "home", return_value=home):
            with patch.dict(os.environ, {"RESEARCH_BACKUP_OPTIONAL": "1"}, clear=False):
                common.maybe_research_backup("e1")

    def test_nonzero_hook_aborts_without_optional(self) -> None:
        home = self._home_without_hook()
        hook = home / "bin" / "research-backup-now"
        hook.write_text("#!/bin/sh\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != "RESEARCH_BACKUP_OPTIONAL"}
        bad = subprocess.CompletedProcess(args=[str(hook), "e2"], returncode=2)
        with patch.object(Path, "home", return_value=home):
            with patch.dict(os.environ, env, clear=True):
                with patch("subprocess.run", return_value=bad):
                    with self.assertRaises(SystemExit) as ctx:
                        common.maybe_research_backup("e2")
                    self.assertIn("exit code 2", str(ctx.exception))

    def test_nonzero_hook_skipped_when_optional(self) -> None:
        home = self._home_without_hook()
        hook = home / "bin" / "research-backup-now"
        hook.write_text("#!/bin/sh\n", encoding="utf-8")
        bad = subprocess.CompletedProcess(args=[str(hook), "e2"], returncode=1)
        with patch.object(Path, "home", return_value=home):
            with patch.dict(os.environ, {"RESEARCH_BACKUP_OPTIONAL": "1"}, clear=False):
                with patch("subprocess.run", return_value=bad):
                    common.maybe_research_backup("e2")


if __name__ == "__main__":
    unittest.main()
