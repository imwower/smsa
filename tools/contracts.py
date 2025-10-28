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
    "surrogate_expr",
    "defaults_eta",
    "defaults_lambda",
    "meta_candidates",
}
AB_MIN_DELTA = 0.02  # absolute improvement on the proxy score
PPL_MIN_REL = 0.015  # at least 1.5% relative perplexity drop
ENERGY_PENALTY = 0.10  # weight for Δenergy in net benefit


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
        return "surrogate_expr"
    if pid.startswith("defaults"):
        # conservative label; both eta/lambda are governed here
        return "defaults_eta"
    if pid.startswith("candidates") or pid.startswith("meta"):
        return "meta_candidates"
    # allow direct category names to pass through
    if pid in PATCH_CATEGORY_WHITELIST:
        return pid
    return "unknown"


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

    # 1) apply + static + smoke
    try:
        changed, backups = ap.apply_patch(patch_id)
    except Exception as exc:
        decision = PatchDecision(accepted=False, reason=f"apply failed: {exc}", category=category)
        _append_csv(patch_id=patch_id, category=category, step="apply", status="failed", message=str(exc), decision=decision)
        _append_md(decision, patch_id)
        return decision

    _append_csv(patch_id=patch_id, category=category, step="apply", status="ok", message=",".join(changed))

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

    # 3) A/B thresholds (re-evaluate base vs patch with the same seed)
    import time as _time
    seed = int(_time.time()) & 0xFFFF

    # baseline: revert → measure
    with contextlib.suppress(Exception):
        ap.revert(ap._LAST_CONTEXT.get("backups", {}))  # type: ignore[arg-type]
    base_ppl, base_delta_ppl, base_energy = _measure_small_lm(seed)

    # patched: re-apply → measure
    try:
        ap.apply_patch(patch_id)
    except Exception as exc:  # pragma: no cover
        decision = PatchDecision(accepted=False, reason=f"re-apply failed: {exc}", category=category)
        _append_csv(patch_id=patch_id, category=category, step="reapply", status="failed", message=str(exc), decision=decision)
        _append_md(decision, patch_id)
        return decision
    patch_ppl, patch_delta_ppl, patch_energy = _measure_small_lm(seed)

    delta_score = (-patch_delta_ppl) - (-base_delta_ppl)
    delta_energy = patch_energy - base_energy
    rel_ppl_drop = 0.0
    if base_ppl and base_ppl != float("inf"):
        rel_ppl_drop = max(0.0, (base_ppl - patch_ppl) / base_ppl)
    net = delta_score - ENERGY_PENALTY * delta_energy

    # Decision logic
    cond_ab = delta_score >= AB_MIN_DELTA
    cond_ppl = rel_ppl_drop >= PPL_MIN_REL
    cond_net = net > 0.0
    accepted = cond_net and (cond_ab or cond_ppl)
    reason = (
        "accepted"
        if accepted
        else f"veto: cond_net={cond_net} cond_ab={cond_ab} cond_ppl={cond_ppl}"
    )

    decision = PatchDecision(
        accepted=accepted,
        reason=reason,
        delta_score=delta_score,
        delta_energy=delta_energy,
        base_ppl=base_ppl,
        patch_ppl=patch_ppl,
        net_benefit=net,
        category=category,
        changed_files=tuple(changed),
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
        message="meets thresholds",
        decision=decision,
    )
    _append_md(decision, patch_id)
    return decision


__all__ = [
    "PATCH_CATEGORY_WHITELIST",
    "AB_MIN_DELTA",
    "PPL_MIN_REL",
    "ENERGY_PENALTY",
    "enforce_code_patch",
]
