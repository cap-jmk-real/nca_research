"""E1v2/E2v2 evaluation helpers: classes_ mapping, majority, paired bootstrap, split, reload reproducibility."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stage0"))

import eval_utils as eu  # noqa: E402


class TestRanking(unittest.TestCase):
    def test_classes_mapping(self):
        # classifier only saw labels 2 and 7: column 1 is label 7, not label 1
        proba = np.array([[0.2, 0.8], [0.9, 0.1]])
        r = eu.rank_from_proba(proba, np.array([2, 7]))
        self.assertEqual(r[:, 0].tolist(), [7, 2])

    def test_sklearn_rank_matches_predict(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4)); y = np.where(X[:, 0] > 0, 5, 11)
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        r = eu.rank_from_proba(clf.predict_proba(X), clf.classes_)
        self.assertTrue(np.array_equal(r[:, 0], clf.predict(X)))

    def test_majority(self):
        r = eu.majority_ranked([3, 3, 1, 3, 2], 4, 5)
        self.assertEqual(r[:, 0].tolist(), [3] * 4)
        self.assertEqual(r[0, 1], 1)  # tie 1 vs 2 -> lower id


class TestStats(unittest.TestCase):
    def test_paired_identical(self):
        a = np.array([1, 0, 1, 1, 0] * 20)
        out = eu.paired_bootstrap(a, a, 500, 0)
        self.assertEqual(out["diff_pp"], 0.0); self.assertEqual(out["ci95"], [0.0, 0.0])

    def test_paired_clear_difference(self):
        a = np.ones(200); b = np.r_[np.ones(100), np.zeros(100)]
        out = eu.paired_bootstrap(a, b, 2000, 0)
        self.assertAlmostEqual(out["diff_pp"], 50.0); self.assertLess(out["p_two_sided"], 0.01)
        self.assertGreater(out["ci95"][0], 0)

    def test_bootstrap_varies_with_preds(self):
        a = eu.bootstrap_top1(np.r_[np.ones(80), np.zeros(20)], 500, 0)
        b = eu.bootstrap_top1(np.r_[np.ones(60), np.zeros(40)], 500, 0)
        self.assertNotEqual(a["boot_mean"], b["boot_mean"])

    def test_macro_f1_variants(self):
        m = eu.macro_f1s([0, 0, 1], [0, 0, 2], labels_all=[0, 1, 2])
        self.assertEqual(m["macro_f1_gold_labels"], [0, 1]); self.assertEqual(m["macro_f1_gold"], 50.0)
        self.assertAlmostEqual(m["macro_f1_all"], 33.33, places=2)


class TestSplit(unittest.TestCase):
    def test_stratified_grouped(self):
        from e1v2_prepare_splits import assign
        ex = [{"session_id": f"a{i}", "goal": f"g{i}", "target_tool": "x"} for i in range(50)]
        ex += [{"session_id": f"b{i}", "goal": f"h{i}", "target_tool": "y"} for i in range(10)]
        ex += [{"session_id": "dup1", "goal": "Same  text", "target_tool": "x"}, {"session_id": "dup2", "goal": "same text", "target_tool": "x"}]
        p = assign(ex, seed=1)
        cnt = {s: sum(r["target_tool"] == "y" for r in rows) for s, rows in p.items()}
        self.assertEqual(cnt, {"train": 6, "val": 2, "test": 2})
        where = {r["session_id"]: s for s, rows in p.items() for r in rows}
        self.assertEqual(where["dup1"], where["dup2"])
        self.assertEqual(assign(ex, seed=1), p)


class TestReloadReproducible(unittest.TestCase):
    def test_eval_from_disk_exact(self):
        import torch
        import run_e1v2_benchmark as rb
        from nca_text.data import Example
        labs = rb.LABELS[:3]
        rng = np.random.default_rng(0)
        words = [f"w{chr(97 + j % 26)}{chr(97 + j // 26)}x" for j in range(400)]
        exs = [Example(f"s{i}", labs[i % 3], " ".join([labs[i % 3]] + list(rng.choice(words, 8))), [], "t") for i in range(300)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gt = rb.fit_features(exs, 0)
            fsha = rb.save_features(gt, d / "f.pkl")
            from obs_pack import pack_arch
            X = torch.from_numpy(pack_arch("nca1d", exs, rb.L2I, "tfidf_svd128", gt)[0]).float()
            torch.manual_seed(0)
            m = rb.build_model(X.shape[1], "star")
            torch.save({"state": m.state_dict(), "feature_sha256": fsha}, d / "b.pt")
            expected = rb.ranked_logits(m, X)
            r1, y1 = rb.eval_nca_from_disk(d / "b.pt", d / "f.pkl", exs, "star")
            r2, _ = rb.eval_nca_from_disk(d / "b.pt", d / "f.pkl", exs, "star")
            self.assertTrue(np.array_equal(r1, r2)); self.assertTrue(np.array_equal(r1, expected))
            m1 = eu.full_metrics(y1, r1, sorted(set(y1.tolist())))
            m2 = eu.full_metrics(y1, r2, sorted(set(y1.tolist())))
            self.assertEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()
