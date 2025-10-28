"""Replay utilities for GridWorld and LM.

- Transition replay for GridWorld: stores (obs_index, action, counts, reward, next_obs_index)
- Lightweight LM snippet replay: stores token sequences for augmentation
- Dream rollouts: generate short sequences using a Self-Model and update policy

Only standard library dependencies are used; types are intentionally loose to
avoid tight coupling with agent classes.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple


class ReplayBuffer:
    """Fixed-size replay buffer for dream and recovery phases.

    Supports two data stores:
    - GridWorld transitions: (obs_index, action, counts, reward, next_obs_index)
      where ``counts`` may be a sequence of float spike rates.
    - LM snippets: small token sequences (stored as tuples of str).
    """

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        # GridWorld transitions
        self.data: Deque[Tuple[int, int, List[float], float, int]] = deque(
            maxlen=capacity
        )
        # Optional LM token snippets
        self.lm_data: Deque[Tuple[str, ...]] = deque(maxlen=capacity)

    def add(
        self,
        obs_index: int,
        action: int,
        counts: Sequence[float],
        reward: float,
        next_obs_index: int,
    ) -> None:
        self.data.append((obs_index, action, [float(c) for c in counts], reward, next_obs_index))

    def sample(self) -> Tuple[int, int, List[float], float, int]:
        if not self.data:
            raise ValueError("回放缓存为空，无法采样。")
        return random.choice(self.data)

    def __len__(self) -> int:
        return len(self.data)

    # --- LM snippet helpers -------------------------------------------------
    def add_lm(self, sequence: Sequence[str]) -> None:
        """Append an LM token sequence to the snippet store.

        Empty sequences are ignored.
        """
        if not sequence:
            return
        self.lm_data.append(tuple(sequence))

    def sample_lm(self, count: int) -> List[List[str]]:
        """Sample up to ``count`` LM sequences uniformly at random."""
        if count <= 0 or not self.lm_data:
            return []
        count = min(count, len(self.lm_data))
        data_list = list(self.lm_data)
        indices = random.sample(range(len(data_list)), count)
        return [list(data_list[idx]) for idx in indices]

    # --- Dream rollouts for GridWorld --------------------------------------
    @staticmethod
    def _one_hot(index: int, size: int) -> List[float]:
        vec = [0.0 for _ in range(size)]
        if 0 <= index < size:
            vec[index] = 1.0
        return vec

    @staticmethod
    def _sample_from_probs(probs: Sequence[float]) -> int:
        total = max(1e-12, float(sum(max(0.0, p) for p in probs)))
        r = random.random() * total
        acc = 0.0
        for i, p in enumerate(probs):
            acc += max(0.0, p)
            if r <= acc:
                return i
        return max(0, len(probs) - 1)

    def dream(
        self,
        agent: object,
        self_model: object,
        *,
        state_size: int,
        sequences: int = 3,
        k_min: int = 3,
        k_max: int = 5,
        noise_std: float = 0.05,
    ) -> int:
        """Generate short imagined rollouts via ``self_model`` and update ``agent``.

        - Picks random starting states from stored transitions
        - Rolls out ``k ~ [k_min, k_max]`` steps per sequence
        - Uses ``agent``'s current policy to pick actions
        - Uses ``self_model`` to sample next state and predict reward
        - Mixes imagined samples back into this buffer and returns the count

        Returns the number of dream transitions added to the buffer.

        Requirements (duck-typed):
        - ``agent`` must provide methods/attributes:
          - ``begin_episode()``
          - ``_integrate_counts(obs: Sequence[float]) -> Sequence[float]``
          - ``policy`` with ``logits(counts)``, ``softmax(logits)`` and ``sample_action(probs)``
          - ``learn(counts, probs, action, reward)`` and ``baseline``/``baseline_beta``
          - ``hidden.n_out``, ``inner_steps``, ``hidden.params.v_th``
        - ``self_model`` must provide ``forward(features)`` -> state and ``update(...)``
          where ``state.probs_next`` and ``pred_reward`` exist.
        """
        if len(self.data) == 0:
            return 0

        dreamed = 0
        state_dim = int(state_size)
        for _ in range(max(1, sequences)):
            try:
                start_obs, _a, _c, _r, _n = self.sample()
            except ValueError:
                break
            current_idx = int(start_obs)
            if not (0 <= current_idx < state_dim):
                continue
            # reset agent temporal state if any
            begin = getattr(agent, "begin_episode", None)
            if callable(begin):
                begin()

            steps = random.randint(max(1, k_min), max(k_min, k_max))
            for _step in range(steps):
                obs_vec = self._one_hot(current_idx, state_dim)
                # policy interaction
                counts = list(getattr(agent, "_integrate_counts")(obs_vec))  # type: ignore[attr-defined]
                logits = agent.policy.logits(counts)  # type: ignore[attr-defined]
                probs = agent.policy.softmax(logits)  # type: ignore[attr-defined]
                action = agent.policy.sample_action(probs)  # type: ignore[attr-defined]

                # build self-model features [obs 1-hot | action 1-hot | mean_rate, sum_rate, eta_e, v_th]
                mean_rate = sum(counts) / float(max(1, len(counts)))
                sum_rate = min(1.0, sum(counts))
                action_vec = [0.0, 0.0, 0.0, 0.0]
                if 0 <= int(action) < 4:
                    action_vec[int(action)] = 1.0
                eta_e = float(getattr(agent, "eta_e", getattr(agent, "hidden_lr", 0.1)))
                v_th = float(getattr(getattr(agent, "hidden"), "params").v_th)  # type: ignore[attr-defined]
                features = obs_vec + action_vec + [
                    max(0.0, min(mean_rate, 1.0)),
                    max(0.0, min(sum_rate, 1.0)),
                    max(0.0, min(eta_e, 1.0)),
                    max(0.0, min(v_th, 1.0)),
                ]

                self_state = self_model.forward(features)  # type: ignore[call-arg]
                next_idx = int(self._sample_from_probs(getattr(self_state, "probs_next", [])))
                reward_pred = float(getattr(self_state, "pred_reward", 0.0)) + random.gauss(0.0, noise_std)
                advantage = reward_pred - float(getattr(agent, "baseline", 0.0))

                # policy update on dream
                getattr(agent, "learn")(counts, probs, int(action), reward_pred)  # type: ignore[attr-defined]
                # baseline update (ema)
                if hasattr(agent, "baseline") and hasattr(agent, "baseline_beta"):
                    agent.baseline += float(agent.baseline_beta) * advantage  # type: ignore[assignment]

                # self-model update: set energy target ~ mean_rate, cause label as self-caused (1)
                energy_target = max(0.0, min(mean_rate, 1.0))
                try:
                    self_model.update(  # type: ignore[call-arg]
                        state=self_state,
                        next_obs_index=next_idx,
                        reward_target=reward_pred,
                        energy_target=energy_target,
                        cause_label=1,
                    )
                except Exception:
                    # tolerate shape mismatches silently in dream mode
                    pass

                # persist imagined transition into buffer
                self.add(current_idx, int(action), counts, reward_pred, next_idx)
                dreamed += 1
                current_idx = next_idx

        return dreamed


__all__ = ["ReplayBuffer"]
