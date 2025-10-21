# Repository Guidelines

## Project Structure & Module Organization
Core spiking-neuron modules live in `snn/`: `lif.py` for neuron dynamics, `dense.py` for synaptic layers, `policy.py` for decision heads, and `selfmodel.py` for self-prediction. Experiments and training entry points sit in `scripts/` (e.g., `scripts/train_gridworld.py`), while light-weight GridWorld environments are in `envs/`. Meta-control helpers reside in `meta/autoadapt.py`, and shared logging utilities are under `tools/logger.py`. Tests mirror this layout inside `tests/` (`tests/test_lif.py`, `tests/test_meta.py`). Standalone baselines (`snn_py_evo.py`, `snn_py_explore_auto.py`, `snn_self_agent.py`) should stay behaviorally aligned with the modular code.

## Build, Test, and Development Commands
- `python snn_py_evo.py` — run the XOR smoke test; expect ≥0.9 accuracy within ~30 epochs.
- `python scripts/train_gridworld.py --episodes 80 --seed 0` — launch the staged GridWorld exploration loop; adjust intrinsic reward flags as needed.
- `python snn_self_agent.py` — execute the full self-model pipeline and watch `[meta]` logs for adaptation checkpoints.
- `python -m unittest discover tests` — execute the regression suite; add `--seed <int>` for deterministic runs.
- `python scripts/eval_metrics.py --run runs/exp_001` — regenerate CSV analytics for a completed training run.

## Coding Style & Naming Conventions
Target Python 3.10+, 4-space indentation, and standard-library dependencies only. Use `snake_case` for functions and module-level helpers, `CapWords` for classes, and prefix exploratory scripts with `experiment_`. Group configuration constants near the top of each module and expose new tunables via `@dataclass` containers. Inline comments should clarify only non-obvious math (e.g., surrogate gradients).

## Testing Guidelines
Write unit tests alongside modules in `tests/`, mirroring filenames (`tests/test_policy.py` for `snn/policy.py`). Favor pure functions so eligibility traces can be asserted. Gate stochastic logic behind seeded RNG objects and assert on statistical envelopes (mean spike counts, reward deltas). Whenever defaults change, update the relevant tests and cover rollback paths for adaptive policies.

## Commit & Pull Request Guidelines
Craft imperative, component-scoped commits such as `snn: tune pseudo_gradient`, and include motivation plus expected metric deltas in the extended description. Pull requests should summarize the scenario, list adjusted scripts/configs, attach ASCII log excerpts, and note rollback safeguards. Link tracking issues when touching adaptive policies, and provide a checklist for rerun commands and generated artifact paths before requesting review.
