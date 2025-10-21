# Repository Guidelines

## Project Structure & Module Organization
SMSA organizes agent logic by lifecycle stage. Core spikes and network primitives live in `snn/` (`lif.py`, `dense.py`, `policy.py`, `selfmodel.py`). Experiment harnesses and CLI entry points sit under `scripts/`, while lightweight GridWorld environments reside in `envs/`. Meta-control and auto-adaptation helpers are in `meta/autoadapt.py`. Shared logging utilities are in `tools/logger.py`. Tests mirror the module layout inside `tests/` (e.g., `test_lif.py`, `test_meta.py`). Standalone prototypes (`snn_py_evo.py`, `snn_py_explore_auto.py`, `snn_self_agent.py`) provide compact baselines; keep them in sync with the modular folders when changing behavior.

## Build, Test, and Development Commands
- `python snn_py_evo.py` — quick XOR sanity check; expected ≥0.9 accuracy within ~30 epochs.
- `python scripts/train_gridworld.py --episodes 80 --seed 0 ...` — staged GridWorld exploration with tunable intrinsic rewards.
- `python snn_self_agent.py` — full self-model pipeline with meta-adaptation; monitor `[meta]` logs.
- `python -m unittest discover tests` — run all regression tests; prefer deterministic seeds via `--seed`.
- `python scripts/eval_metrics.py --run runs/exp_001` — recompute CSV analytics for a training run.

## Coding Style & Naming Conventions
Stick to Python 3.10+ standard-library only. Use 4-space indentation, `snake_case` for functions, `CapWords` for classes, and prefix auxiliary experiments with `experiment_`. Document public functions with concise PEP 257 docstrings; add inline comments only for non-obvious math (e.g., surrogate gradients). Keep configuration constants grouped near the top of each module and expose them via dataclasses when adding new knobs.

## Testing Guidelines
Add targeted unit tests mirroring the module name (e.g., `test_policy.py`). Favor pure functions over side effects so eligibility traces remain testable. When introducing stochasticity, gate it behind seeded RNG objects and assert on statistical envelopes (mean spikes, reward deltas). Update `tests/` whenever meta-parameters change defaults and include failure-mode coverage for self-modification rollbacks.

## Commit & Pull Request Guidelines
Write imperative, component-scoped commits (`snn: tune pseudo_gradient`). Include motivation and expected metrics deltas in the extended description. Pull requests should summarize scenario, list modified scripts/configs, attach relevant ASCII log excerpts, and note rollback safeguards. Link tracking issues when touching adaptive policies, and add checklist items for rerun commands and generated artifacts paths.
