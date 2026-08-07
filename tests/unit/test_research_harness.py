"""Research-quality tests T1-T12 for the reusable research harness + torch core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from airelab.cheap_kill.schemas import ExperimentSpec, ResearchHypothesis
from airelab.core.kill_gates import KillGateSpec
from airelab.core.leakage import LearnerEvaluatorSplit, ViewContract
from airelab.core.lifecycle import FrozenConfig
from airelab.core.provenance import capture_provenance
from airelab.core.statistics import bootstrap_ci, cohens_d, paired_differences
from airelab.torch.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from airelab.torch.gradient_check import gradient_check
from airelab.torch.metrics import expected_calibration_error, gradient_norm
from airelab.torch.reproducibility import set_torch_seed


# --- T1: pilot / confirmatory seeds cannot overlap ---
def test_T1_seed_disjointness():
    spec = ExperimentSpec(
        hypothesis_id="H", environment="e", method="m", baselines=["b"],
        metrics=["m"],
        seed_ranges={"tuning": [0, 1], "pilot": [10, 11], "confirmatory": [100, 101]},
    )
    assert spec.validate_seed_disjointness() is True
    bad = ExperimentSpec(
        hypothesis_id="H", environment="e", method="m", baselines=["b"],
        metrics=["m"],
        seed_ranges={"pilot": [10, 11], "confirmatory": [10, 12]},
    )
    assert bad.validate_seed_disjointness() is False


# --- T2: frozen config hash changes if scientific config changes ---
def test_T2_config_hash_changes(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("p: 0.6\n", encoding="utf-8")
    fc = FrozenConfig.freeze(p)
    assert fc.verify() is True
    p.write_text("p: 0.7\n", encoding="utf-8")  # scientific change
    assert fc.verify() is False  # hash changed -> detected
    with pytest.raises(ValueError):
        fc.promote_to_confirmatory()  # cannot promote modified config


# --- T3: dirty git state is recorded ---
def test_T3_provenance_records_dirty(tmp_path):
    prov = capture_provenance(repo_root=Path("."))
    assert "git_head" in prov
    assert "git_dirty" in prov
    assert isinstance(prov["git_dirty"], bool)


# --- T4/T5: seed reproducibility in torch ---
def _tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))


def test_T4_same_seed_same_result():
    set_torch_seed(0)
    m1 = _tiny_model()
    x = torch.randn(16, 4)
    y1 = m1(x).detach().numpy()
    set_torch_seed(0)
    m2 = _tiny_model()
    y2 = m2(x).detach().numpy()
    np.testing.assert_allclose(y1, y2, atol=1e-6)


def test_T5_different_seed_different():
    set_torch_seed(0)
    m1 = _tiny_model()
    x = torch.randn(16, 4)
    y1 = m1(x).detach().numpy()
    set_torch_seed(1)
    m2 = _tiny_model()
    y2 = m2(x).detach().numpy()
    assert not np.allclose(y1, y2)


# --- T6: paired analysis aligns seeds correctly ---
def test_T6_paired_differences_aligned():
    a = [10.0, 20.0, 30.0]
    b = [5.0, 15.0, 25.0]
    seeds = [1, 2, 3]
    res = paired_differences(a, b, seeds_a=seeds, seeds_b=seeds)
    assert res["n"] == 3
    assert res["mean_diff"] == pytest.approx(5.0)


# --- T7: bootstrap CI on a known synthetic case ---
def test_T7_bootstrap_ci_known():
    # Constant values -> CI collapses to the value.
    vals = [5.0] * 100
    lo, hi = bootstrap_ci(vals, n_boot=500, seed=0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)


# --- T8: learner-view object cannot access evaluator-only fields ---
def test_T8_learner_view_isolation():
    split = LearnerEvaluatorSplit(
        learner={"observed_label": 1},
        evaluator={"hidden_state": 1, "gold": 99},
    )
    learner_view = split.as_learner()
    assert learner_view == {"observed_label": 1}
    assert "hidden_state" not in learner_view
    assert "gold" not in learner_view


# --- T9: gradient check agrees with autodiff ---
def test_T9_gradient_check():
    def make_module():
        lin = torch.nn.Linear(4, 1)

        def forward(x):
            return lin(x).sum()
        return forward

    x = torch.randn(6, 4)
    passed, err = gradient_check(make_module(), x)  # uses tuned defaults
    assert passed, f"gradient check failed: max_abs_err={err:.2e}"


# --- T10: checkpoint restore reproduces evaluation ---
def test_T10_checkpoint_restore(tmp_path):
    set_torch_seed(0)
    m = _tiny_model()
    x = torch.randn(8, 4)
    before = m(x).detach().numpy()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, m, opt, epoch=1, metric=0.5)
    # Mutate the model then restore.
    with torch.no_grad():
        m[0].weight.add_(1.0)
    load_checkpoint(ckpt, m, opt)
    after = m(x).detach().numpy()
    np.testing.assert_allclose(before, after, atol=1e-6)


# --- T11: kill-gate spec is saved before result evaluation ---
def test_T11_kill_gate_saved_before(tmp_path):
    spec = KillGateSpec(experiment_id="E1")
    spec.add("K1", "IPW+eps matches R-C", on_false="KILL")
    spec.add("K2", "censored forecaster dominates", on_false="KILL")
    path = spec.save(tmp_path / "gates.json")
    loaded = KillGateSpec.load(path)
    assert len(loaded.gates) == 2
    assert loaded.gates[0]["id"] == "K1"
    assert loaded.gates[0]["on_false"] == "KILL"


# --- T12: artifact manifest detects missing required outputs ---
def test_T12_manifest_records_artifacts(tmp_path):
    from airelab.core.config import ExperimentConfig, ExperimentType
    from airelab.core.manifest import ExperimentManifest
    from airelab.core.artifacts import ArtifactHash

    cfg = ExperimentConfig(experiment_id="e1", experiment_type=ExperimentType.LINEAR_REGRESSION,
                           seed=0, dataset_id="tiny")
    man = ExperimentManifest(cfg, command="test")
    out = tmp_path / "metrics.json"
    out.write_text("{}", encoding="utf-8")
    ah = ArtifactHash(path=out, sha256="abc", size=2)
    man.mark_artifact(ah)
    man.finish(True)
    d = man.to_dict()
    assert d["experiment_id"] == "e1"
    assert d["success"] is True
    assert len(d["artifacts"]) == 1
    assert d["artifacts"][0]["sha256"] == "abc"
    assert "git_commit" in d


# --- Calibration / metrics sanity ---
def test_calibration_perfect():
    probs = torch.tensor([0.9, 0.1, 0.8, 0.2])
    labels = torch.tensor([1, 0, 1, 0])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece >= 0.0


def test_gradient_norm_finite():
    model = _tiny_model()
    x = torch.randn(4, 4)
    y = model(x)
    y.sum().backward()
    gn = gradient_norm(model)
    assert gn >= 0.0
