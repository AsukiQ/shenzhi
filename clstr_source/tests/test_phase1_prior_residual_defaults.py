"""Automatic AppWorld ranking keeps routing-only checkpoints recall-safe.

- `_resolve_auto_ranking_mode` in scripts/run_appworld_multistep_executor_eval.py
  selects `policy_blend` only for checkpoints explicitly marked multi-step and
  `skill_table` otherwise.
  It must NEVER default to `policy_head` (which dropped recall@1 from 0.96 to 0.23
  in the 2026-05-23 audit).
- The SkillRouter-init model config must carry `policy_skill_mode: prior_residual`
  so any optional policy-head ablation anchors skill logits to the routing prior.
- The multistep eval sbatch must default POLICY_BLEND_ALPHA to a conservative 0.25,
  not 0.5, to keep the base routing prior dominant when policy signal is weak."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml


REPO_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr")


def _load_resolver():
    spec = importlib.util.spec_from_file_location(
        "_phase1_eval_cli",
        REPO_ROOT / "scripts/run_appworld_multistep_executor_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._resolve_auto_ranking_mode


def test_auto_resolver_returns_policy_blend_for_explicit_multistep_checkpoint():
    resolver = _load_resolver()
    report = {"stage": "legacy_multistep_policy", "train_config": {"multi_step": True}}
    assert resolver("clstr_multistep", "auto", report) == "policy_blend"


def test_auto_resolver_falls_back_to_skill_table_for_plain_routing_checkpoint():
    resolver = _load_resolver()
    report = {"stage": "stage2_base_routing_only", "train_config": {"loss_mode": "routing_only"}}
    assert resolver("clstr_base", "auto", report) == "skill_table"


def test_auto_resolver_never_returns_policy_head_for_plain_method():
    resolver = _load_resolver()
    # Plain routing-only checkpoint must NOT fall through to policy_head
    report = {"stage": "stage2_base", "train_config": {"loss_mode": "routing_only"}}
    assert resolver("clstr_base", "auto", report) != "policy_head"


def test_explicit_request_is_respected():
    resolver = _load_resolver()
    # If the user explicitly asks for policy_head, the resolver should honor it
    # (the safety net is the default, not a hard ban)
    assert resolver("clstr_multistep", "policy_head", {}) == "policy_head"


def test_skillrouter_init_config_defaults_to_prior_residual_policy_mode():
    cfg = yaml.safe_load((REPO_ROOT / "configs/model/appworld_skillrouter_init.yaml").read_text())
    assert cfg.get("policy_skill_mode") == "prior_residual", (
        "SkillRouter-init config must default to prior_residual so policy-head ablations built "
        "on this config anchors skill logits to the routing prior."
    )
    assert cfg.get("routing_prior_strength") is not None, "routing_prior_strength must be set"
    assert cfg.get("policy_residual_scale") is not None, "policy_residual_scale must be set"


def test_multistep_eval_sbatch_defaults_to_policy_blend_alpha_025():
    text = (REPO_ROOT / "scripts/sbatch/run_appworld_multistep_executor_eval.sh").read_text()
    # We want POLICY_BLEND_ALPHA default to be 0.25
    match = re.search(r"POLICY_BLEND_ALPHA=\$\{POLICY_BLEND_ALPHA:-([0-9.]+)\}", text)
    assert match is not None, "POLICY_BLEND_ALPHA default line not found"
    assert float(match.group(1)) == 0.25, (
        f"POLICY_BLEND_ALPHA sbatch default must be 0.25, found {match.group(1)}"
    )
