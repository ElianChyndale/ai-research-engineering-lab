from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, write_csv


SEED = 42
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
GOAL = (3, 3)


def step(state: tuple[int, int], action_idx: int) -> tuple[tuple[int, int], float, bool]:
    move = ACTIONS[action_idx]
    next_state = (min(3, max(0, state[0] + move[0])), min(3, max(0, state[1] + move[1])))
    done = next_state == GOAL
    reward = 1.0 if done else -0.03
    return next_state, reward, done


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = ensure_results_dir()
    q = np.zeros((4, 4, 4))
    curve_rows = []
    for episode in range(60):
        state = (0, 0)
        total_reward = 0.0
        epsilon = max(0.05, 0.4 - episode * 0.005)
        for _ in range(30):
            if rng.random() < epsilon:
                action = int(rng.integers(0, 4))
            else:
                action = int(np.argmax(q[state[0], state[1]]))
            next_state, reward, done = step(state, action)
            q[state[0], state[1], action] += 0.4 * (
                reward + 0.9 * np.max(q[next_state[0], next_state[1]]) - q[state[0], state[1], action]
            )
            state = next_state
            total_reward += reward
            if done:
                break
        curve_rows.append({"episode": episode + 1, "reward": round(total_reward, 6)})
    eval_rows = []
    state = (0, 0)
    for task_id in range(1, 6):
        action = int(np.argmax(q[state[0], state[1]]))
        state, reward, done = step(state, action)
        eval_rows.append(
            {
                "task_id": f"T{task_id}",
                "state": str(state),
                "task_success_score": 1.0 if done else 0.0,
                "reward": round(reward, 6),
            }
        )
        if done:
            break
    write_csv(results / "rl_learning_curve.csv", curve_rows)
    write_csv(results / "agent_task_eval.csv", eval_rows)


if __name__ == "__main__":
    main()
