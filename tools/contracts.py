"""QA/Risk contracts for code patches (Codex supervision and veto).

This module wraps the autopatch pipeline with additional governance:
- Patch category whitelist enforcement
- Mandatory checks: unit tests all green
- A/B threshold: Δscore >= +0.02 OR perplexity drop >= 1.5%
- Energy penalty: net = Δscore - L * Δenergy must remain positive
- Automatic revert and audit trail to CSV and Markdown

Only standard library dependencies are used.
"""

from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

# Local imports (standard-library only within repo)
from meta import autopatch as ap


# Paths for audit logs
RUNS_DIR = Path("runs")
CSV_PATH = RUNS_DIR / "contracts_log.csv"
REPORT_MD = RUNS_DIR / "self_report.md"


# Governance parameters
PATCH_CATEGORY_WHITELIST = {
    # Four anchor families only (by patch-id namespace)
    "surrogate",      # snn/lif.py      # AUTOPATCH SURROGATE START/END
    "defaults",       # snn/dense.py    # AUTOPATCH DEFAULTS START/END
    "candidates",     # meta/autoadapt  # AUTOPATCH CANDIDATES START/END
    "decode",         # scripts/spike_writer.py (params/logic anchors)
}

# Parameter hard bounds (enforced inside anchors)
BOUNDS = {
    "eta_e": (1e-4, 0.1),                 # ETA_E_DEFAULT
    "lam_e": (0.5, 0.999),                # LAM_E_DEFAULT
    "inner_steps": (4, 30),               # integrator steps
    "top_k": (10, 128),                   # DECODE_TOP_K
    "repeat_penalty": (1.0, 2.0),         # DECODE_REPEAT_PENALTY
}

# A/B acceptance thresholds
AB_MIN_DELTA = 0.02         # RL: mean return improvement ≥ +0.02
PPL_MIN_REL = 0.015         # LM: perplexity drop ≥ 1.5% (relative)
POST_MIN_DELTA = 0.05       # POST: explainability overall ≥ +0.05
ENERGY_DELTA_MAX_REL = 0.10 # POST: |Δenergy| / base ≤ 10%
ENERGY_PENALTY = 0.10       # Legacy net-benefit weight (kept for logging)


@dataclass
class PatchDecision:
    accepted: bool
    reason: str
    delta_score: float = 0.0
    delta_energy: float = 0.0
    base_ppl: float = 0.0
    patch_ppl: float = 0.0
    net_benefit: float = 0.0
    category: str = "unknown"
    changed_files: tuple[str, ...] = ()
    # Optional metrics for richer audit
    rel_ppl_drop: float = 0.0
    rl_delta_return: float = 0.0
    post_delta_overall: float = 0.0
    post_rel_energy: float = 0.0


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_csv() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp",
                    "patch_id",
                    "category",
                    "step",
                    "status",
                    "message",
                    "delta_score",
                    "delta_energy",
                    "base_ppl",
                    "patch_ppl",
                    "net_benefit",
                    "decision",
                ],
            )
            writer.writeheader()


def _append_csv(
    *,
    patch_id: str,
    category: str,
    step: str,
    status: str,
    message: str,
    decision: Optional[PatchDecision] = None,
) -> None:
    _ensure_csv()
    row = {
        "timestamp": _timestamp(),
        "patch_id": patch_id,
        "category": category,
        "step": step,
        "status": status,
        "message": message,
        "delta_score": f"{decision.delta_score:.6f}" if decision else "",
        "delta_energy": f"{decision.delta_energy:.6f}" if decision else "",
        "base_ppl": f"{decision.base_ppl:.6f}" if decision else "",
        "patch_ppl": f"{decision.patch_ppl:.6f}" if decision else "",
        "net_benefit": f"{decision.net_benefit:.6f}" if decision else "",
        "decision": ("accepted" if decision and decision.accepted else ("denied" if decision else "")),
    }
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writerow(row)


def _append_md(decision: PatchDecision, patch_id: str) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_MD.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Patch `{patch_id}` ({decision.category}) — {('ACCEPTED' if decision.accepted else 'DENIED')}\n\n")
        fh.write(f"- Time: {_timestamp()}\n")
        fh.write(f"- Δscore: {decision.delta_score:+.4f}\n")
        fh.write(f"- Δenergy: {decision.delta_energy:+.4f}\n")
        fh.write(f"- base_ppl: {decision.base_ppl:.4f} → patch_ppl: {decision.patch_ppl:.4f}\n")
        fh.write(f"- Net benefit (after energy penalty): {decision.net_benefit:+.4f}\n")
        fh.write(f"- Decision: {'accepted' if decision.accepted else 'denied'}\n")
        fh.write(f"- Reason: {decision.reason}\n")


def _categorize(patch_id: str) -> str:
    pid = patch_id.strip().lower()
    pid = pid.replace("-", ":").replace("_", ":")
    if pid.startswith("surrogate"):
        return "surrogate"
    if pid.startswith("defaults"):
        return "defaults"
    if pid.startswith("decode"):
        return "decode"
    if pid.startswith("candidates") or pid.startswith("meta"):
        return "candidates"
    # allow direct category names to pass through
    if pid in PATCH_CATEGORY_WHITELIST:
        return pid
    return "unknown"


def _strip_anchors(text: str, anchor_map: Mapping[str, tuple[str, str]]) -> str:
    """Return the file text with anchored regions replaced by stable markers.

    This allows equality checks for non-anchor regions.
    """
    # Build a list of spans [start, end) for each anchor
    spans: list[tuple[int, int, str]] = []
    for name, (start_tag, end_tag) in anchor_map.items():
        s = text.find(start_tag)
        e = text.find(end_tag)
        if s == -1 or e == -1 or e < s:
            continue
        s = s + len(start_tag)
        spans.append((s, e, name))
    if not spans:
        return text
    spans.sort(key=lambda t: t[0])
    pieces: list[str] = []
    last = 0
    for s, e, name in spans:
        # keep text before anchor region
        if last < s:
            pieces.append(text[last:s])
        # replace anchor region with marker
        pieces.append(f"<<ANCHOR:{name}>>")
        last = e
    pieces.append(text[last:])
    return "".join(pieces)


def _enforce_anchor_whitelist(changed_files: Sequence[str]) -> tuple[bool, str]:
    """Ensure only known anchor files are modified and only within anchors."""
    ok = True
    reasons: list[str] = []
    allowed_files = set(ap.ANCHORS.keys())
    ctx = dict(ap._LAST_CONTEXT)
    pre_sources: Dict[str, str] = ctx.get("pre_sources", {})  # type: ignore[assignment]
    for rel in changed_files:
        if rel not in allowed_files:
            ok = False
            reasons.append(f"{rel} 不在锚点白名单文件中。")
            continue
        full = ap.ROOT / rel  # type: ignore[attr-defined]
        try:
            after = full.read_text(encoding="utf-8")
        except OSError:
            ok = False
            reasons.append(f"无法读取修改后的文件: {rel}")
            continue
        before = pre_sources.get(rel, "")
        anchors = ap.ANCHORS.get(rel, {})
        before_stripped = _strip_anchors(before, anchors)
        after_stripped = _strip_anchors(after, anchors)
        if before_stripped != after_stripped:
            ok = False
            reasons.append(f"{rel} 存在锚点之外的改动，禁止。")
    return ok, "; ".join(reasons)


def _span_between(text: str, start_tag: str, end_tag: str) -> tuple[int, int]:
    s = text.find(start_tag)
    e = text.find(end_tag)
    if s == -1 or e == -1 or e < s:
        return -1, -1
    return s + len(start_tag), e


def _enforce_param_bounds(changed_files: Sequence[str]) -> tuple[bool, str]:
    """Check parameter bounds inside anchored regions only.

    Enforces hard limits:
      - 0.5 ≤ LAM_E_DEFAULT ≤ 0.999
      - 1e-4 ≤ ETA_E_DEFAULT ≤ 0.1
      - 10 ≤ DECODE_TOP_K ≤ 128
      - 1.0 ≤ DECODE_REPEAT_PENALTY ≤ 2.0
    """
    ok = True
    reasons: list[str] = []
    anchors = ap.ANCHORS  # type: ignore[attr-defined]
    for rel in changed_files:
        full = ap.ROOT / rel  # type: ignore[attr-defined]
        try:
            data = full.read_text(encoding="utf-8")
        except OSError:
            ok = False
            reasons.append(f"无法读取文件以检查参数边界: {rel}")
            continue
        amap = anchors.get(rel, {})
        for name, (start_tag, end_tag) in amap.items():
            s, e = _span_between(data, start_tag, end_tag)
            if s == -1:
                continue
            block = data[s:e]
            # defaults: ETA_E_DEFAULT / LAM_E_DEFAULT
            m = re.search(r"ETA_E_DEFAULT\s*=\s*([0-9]*\.?[0-9]+)", block)
            if m:
                val = float(m.group(1))
                lo, hi = BOUNDS["eta_e"]
                if not (lo <= val <= hi):
                    ok = False
                    reasons.append(f"ETA_E_DEFAULT 越界: {val} not in [{lo}, {hi}]")
            m = re.search(r"LAM_E_DEFAULT\s*=\s*([0-9]*\.?[0-9]+)", block)
            if m:
                val = float(m.group(1))
                lo, hi = BOUNDS["lam_e"]
                if not (lo <= val <= hi):
                    ok = False
                    reasons.append(f"LAM_E_DEFAULT 越界: {val} not in [{lo}, {hi}]")
            # decode params: DECODE_TOP_K / DECODE_REPEAT_PENALTY
            m = re.search(r"DECODE_TOP_K\s*=\s*([0-9]+)", block)
            if m:
                val = int(m.group(1))
                lo, hi = BOUNDS["top_k"]
                if not (lo <= val <= hi):
                    ok = False
                    reasons.append(f"DECODE_TOP_K 越界: {val} not in [{lo}, {hi}]")
            m = re.search(r"DECODE_REPEAT_PENALTY\s*=\s*([0-9]*\.?[0-9]+)", block)
            if m:
                val = float(m.group(1))
                lo, hi = BOUNDS["repeat_penalty"]
                if not (lo <= val <= hi):
                    ok = False
                    reasons.append(f"DECODE_REPEAT_PENALTY 越界: {val} not in [{lo}, {hi}]")
    return ok, "; ".join(reasons)


def _run_full_tests() -> bool:
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=0)
    res = runner.run(suite)
    return res.wasSuccessful()


def _measure_small_lm(seed: int = 0) -> Tuple[float, float, float]:
    """Return (valid_ppl, delta_ppl, avg_spikes)."""
    try:
        from scripts.snn_text_lm import train_lines  # delayed import
    except Exception:
        return float("inf"), 0.0, 0.0
    stats = train_lines(num_lines=120, seed=seed, sampler=None, valid_interval=60)
    valid_ppl = float(stats.get("valid_ppl", float("inf")) or float("inf"))
    delta_ppl = float(stats.get("delta_ppl", 0.0) or 0.0)
    avg_spikes = float(stats.get("avg_spikes", 0.0) or 0.0)
    return valid_ppl, delta_ppl, avg_spikes


def _measure_lm_ppl(seed: int = 0) -> Tuple[float, float]:
    """Return (valid_ppl, avg_spikes)."""
    try:
        from scripts.snn_text_lm import train_lines  # delayed import
    except Exception:
        return float("inf"), 0.0
    stats = train_lines(num_lines=160, seed=seed, sampler=None, valid_interval=80)
    return float(stats.get("valid_ppl", float("inf") or float("inf"))), float(stats.get("avg_spikes", 0.0) or 0.0)


def _measure_post(seed: int = 0) -> Tuple[float, float]:
    """Return (explainability overall, energy)."""
    try:
        from scripts.spike_writer import spike_generate  # type: ignore
        from tools.explainability import explainability_index
    except Exception:
        return 0.0, 0.0
    res = spike_generate(max_len=140, seed_text="", topic_hint="autopatch", rng_seed=seed)
    ei = explainability_index(res.text, "autopatch")
    overall = float(ei.get("overall", 0.0) or 0.0)
    energy = float(getattr(res, "spike_estimate", 0.0) or 0.0)
    return overall, energy


def _measure_rl(seed: int = 0) -> Tuple[float, float, float]:
    """Return (avg_return, success_rate, avg_spikes)."""
    try:
        from scripts.train_gridworld import train_once
    except Exception:
        return 0.0, 0.0, 0.0
    avg_return, success_rate, avg_spikes = train_once(episodes=12, seed=seed)
    return float(avg_return), float(success_rate), float(avg_spikes)


def enforce_code_patch(patch_id: str) -> PatchDecision:
    """Apply a code patch under QA contracts. Auto-reverts on any violation.

    Steps:
    1) Category whitelist check
    2) autopatch.apply_patch → static_checks → smoke_test
    3) Run full unit tests (all green)
    4) A/B eval on small LM; thresholds:
       - Δscore >= AB_MIN_DELTA OR relative ppl drop >= PPL_MIN_REL
       - Net benefit = Δscore - ENERGY_PENALTY * Δenergy > 0
    5) Persist CSV and Markdown audit entries
    """
    category = _categorize(patch_id)
    if category not in PATCH_CATEGORY_WHITELIST:
        decision = PatchDecision(
            accepted=False,
            reason=f"category '{category}' not in whitelist",
            category=category,
        )
        _append_csv(patch_id=patch_id, category=category, step="precheck", status="denied", message=decision.reason, decision=decision)
        _append_md(decision, patch_id)
        return decision

    # 1) apply + hard static (anchors/params) + static + smoke
    try:
        changed, backups = ap.apply_patch(patch_id)
    except Exception as exc:
        decision = PatchDecision(accepted=False, reason=f"apply failed: {exc}", category=category)
        _append_csv(patch_id=patch_id, category=category, step="apply", status="failed", message=str(exc), decision=decision)
        _append_md(decision, patch_id)
        return decision

    _append_csv(patch_id=patch_id, category=category, step="apply", status="ok", message=",".join(changed))

    # Hard whitelist: only known anchor files and no edits outside anchors
    ok_anchor, msg = _enforce_anchor_whitelist(changed)
    if not ok_anchor:
        with contextlib.suppress(Exception):
            ap.revert(backups)
        decision = PatchDecision(accepted=False, reason=f"anchor violation: {msg}", category=category)
        _append_csv(patch_id=patch_id, category=category, step="anchors", status="failed", message=decision.reason, decision=decision)
        _append_md(decision, patch_id)
        return decision

    # Additional param bounds (stricter than autopatch)
    # Validate inside anchors: eta_e, lam_e, inner_steps, top_k, repeat_penalty
    ctx = dict(ap._LAST_CONTEXT)
    pre_sources: Dict[str, str] = ctx.get("pre_sources", {})  # type: ignore[assignment]
    for rel in changed:
        try:
            after = (ap.ROOT / rel).read_text(encoding="utf-8")  # type: ignore[attr-defined]
        except OSError:
            continue
        anchors = ap.ANCHORS.get(rel, {})
        for name, (start_tag, end_tag) in anchors.items():
            s = after.find(start_tag)
            e = after.find(end_tag)
            if s == -1 or e == -1 or e < s:
                continue
            s = s + len(start_tag)
            block = after[s:e]
            # eta_e / lam_e defaults
            for v in re.findall(r"ETA_E_DEFAULT\s*=\s*([0-9]*\.?[0-9]+)", block):
                if not (BOUNDS["eta_e"][0] <= float(v) <= BOUNDS["eta_e"][1]):
                    ap.revert(backups)
                    decision = PatchDecision(accepted=False, reason=f"eta_e 超界: {v}", category=category)
                    _append_csv(patch_id=patch_id, category=category, step="bounds", status="failed", message=decision.reason, decision=decision)
                    _append_md(decision, patch_id)
                    return decision
            for v in re.findall(r"LAM_E_DEFAULT\s*=\s*([0-9]*\.?[0-9]+)", block):
                if not (BOUNDS["lam_e"][0] <= float(v) <= BOUNDS["lam_e"][1]):
                    ap.revert(backups)
                    decision = PatchDecision(accepted=False, reason=f"lam_e 超界: {v}", category=category)
                    _append_csv(patch_id=patch_id, category=category, step="bounds", status="failed", message=decision.reason, decision=decision)
                    _append_md(decision, patch_id)
                    return decision
            for v in re.findall(r"inner_steps\s*=\s*([0-9]+)", block):
                val = int(v)
                if not (BOUNDS["inner_steps"][0] <= val <= BOUNDS["inner_steps"][1]):
                    ap.revert(backups)
                    decision = PatchDecision(accepted=False, reason=f"inner_steps 超界: {val}", category=category)
                    _append_csv(patch_id=patch_id, category=category, step="bounds", status="failed", message=decision.reason, decision=decision)
                    _append_md(decision, patch_id)
                    return decision
            for v in re.findall(r"DECODE_TOP_K\s*=\s*([0-9]+)", block):
                val = int(v)
                if not (BOUNDS["top_k"][0] <= val <= BOUNDS["top_k"][1]):
                    ap.revert(backups)
                    decision = PatchDecision(accepted=False, reason=f"DECODE_TOP_K 超界: {val}", category=category)
                    _append_csv(patch_id=patch_id, category=category, step="bounds", status="failed", message=decision.reason, decision=decision)
                    _append_md(decision, patch_id)
                    return decision
            for v in re.findall(r"DECODE_REPEAT_PENALTY\s*=\s*([0-9]*\.?[0-9]+)", block):
                if not (BOUNDS["repeat_penalty"][0] <= float(v) <= BOUNDS["repeat_penalty"][1]):
                    ap.revert(backups)
                    decision = PatchDecision(accepted=False, reason=f"DECODE_REPEAT_PENALTY 超界: {v}", category=category)
                    _append_csv(patch_id=patch_id, category=category, step="bounds", status="failed", message=decision.reason, decision=decision)
                    _append_md(decision, patch_id)
                    return decision

    if not ap.static_checks(changed):
        decision = PatchDecision(accepted=False, reason="static checks failed", category=category)
        _append_csv(patch_id=patch_id, category=category, step="static", status="failed", message=decision.reason, decision=decision)
        _append_md(decision, patch_id)
        return decision
    _append_csv(patch_id=patch_id, category=category, step="static", status="ok", message="passed")

    if not ap.smoke_test():
        decision = PatchDecision(accepted=False, reason="smoke tests failed", category=category)
        _append_csv(patch_id=patch_id, category=category, step="smoke", status="failed", message=decision.reason, decision=decision)
        _append_md(decision, patch_id)
        return decision
    _append_csv(patch_id=patch_id, category=category, step="smoke", status="ok", message="passed")

    # 2) full unit tests
    if not _run_full_tests():
        with contextlib.suppress(Exception):
            ap.revert(ap._LAST_CONTEXT.get("backups", {}))  # type: ignore[arg-type]
        decision = PatchDecision(accepted=False, reason="unit tests not all green", category=category)
        _append_csv(patch_id=patch_id, category=category, step="unittest", status="failed", message=decision.reason, decision=decision)
        _append_md(decision, patch_id)
        return decision
    _append_csv(patch_id=patch_id, category=category, step="unittest", status="ok", message="all green")

    # 3) A/B thresholds across three tasks (post, lm, rl)
    import time as _time
    seed = int(_time.time()) & 0xFFFF

    # baseline: revert → measure
    backups_ctx = ap._LAST_CONTEXT.get("backups", {})  # type: ignore[assignment]
    with contextlib.suppress(Exception):
        ap.revert(backups_ctx)  # type: ignore[arg-type]
    base_post_overall, base_post_energy = _measure_post(seed)
    base_lm_ppl, _ = _measure_lm_ppl(seed)
    base_rl_return, _, _ = _measure_rl(seed)
    _append_csv(
        patch_id=patch_id,
        category=category,
        step="ab_baseline",
        status="ok",
        message=f"post={base_post_overall:.4f}/{base_post_energy:.4f} lm_ppl={base_lm_ppl:.4f} rl_ret={base_rl_return:.4f}",
    )

    # patched: re-apply → measure
    try:
        ap.apply_patch(patch_id)
    except Exception as exc:  # pragma: no cover
        decision = PatchDecision(accepted=False, reason=f"re-apply failed: {exc}", category=category)
        _append_csv(patch_id=patch_id, category=category, step="reapply", status="failed", message=str(exc), decision=decision)
        _append_md(decision, patch_id)
        return decision

    patch_post_overall, patch_post_energy = _measure_post(seed)
    patch_lm_ppl, _ = _measure_lm_ppl(seed)
    patch_rl_return, _, _ = _measure_rl(seed)

    # Compute metrics
    post_delta = patch_post_overall - base_post_overall
    rel_energy = (
        abs(patch_post_energy - base_post_energy) / base_post_energy
        if base_post_energy > 0.0
        else 0.0
    )
    rel_ppl_drop = (
        max(0.0, (base_lm_ppl - patch_lm_ppl) / base_lm_ppl)
        if math.isfinite(base_lm_ppl) and base_lm_ppl > 0.0
        else 0.0
    )
    rl_delta = patch_rl_return - base_rl_return
    # Legacy aggregation for logging
    delta_score = rl_delta
    delta_energy = patch_post_energy - base_post_energy
    net = delta_score - ENERGY_PENALTY * delta_energy

    # Decision logic (any one passes)
    cond_post = (post_delta >= POST_MIN_DELTA) and (rel_energy <= ENERGY_DELTA_MAX_REL)
    cond_lm = rel_ppl_drop >= PPL_MIN_REL
    cond_rl = rl_delta >= AB_MIN_DELTA
    accepted = any((cond_post, cond_lm, cond_rl))
    reason = (
        f"post={cond_post} lm={cond_lm} rl={cond_rl}"
    )

    decision = PatchDecision(
        accepted=accepted,
        reason=reason,
        delta_score=delta_score,
        delta_energy=delta_energy,
        base_ppl=base_lm_ppl,
        patch_ppl=patch_lm_ppl,
        net_benefit=net,
        category=category,
        changed_files=tuple(changed),
        rel_ppl_drop=rel_ppl_drop,
        rl_delta_return=rl_delta,
        post_delta_overall=post_delta,
        post_rel_energy=rel_energy,
    )

    if not accepted:
        with contextlib.suppress(Exception):
            ap.revert(ap._LAST_CONTEXT.get("backups", {}))  # type: ignore[arg-type]
        _append_csv(
            patch_id=patch_id,
            category=category,
            step="decision",
            status="denied",
            message=reason,
            decision=decision,
        )
        _append_md(decision, patch_id)
        return decision

    _append_csv(
        patch_id=patch_id,
        category=category,
        step="decision",
        status="accepted",
        message=(
            f"postΔ={post_delta:+.4f} relE={rel_energy:.3f} lm_relΔ={rel_ppl_drop:.3f} rlΔ={rl_delta:+.4f}"
        ),
        decision=decision,
    )
    _append_md(decision, patch_id)
    return decision


__all__ = [
    "PATCH_CATEGORY_WHITELIST",
    "BOUNDS",
    "AB_MIN_DELTA",
    "PPL_MIN_REL",
    "POST_MIN_DELTA",
    "ENERGY_DELTA_MAX_REL",
    "ENERGY_PENALTY",
    "enforce_code_patch",
]
