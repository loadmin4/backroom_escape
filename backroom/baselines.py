"""비교용 기준 에이전트들.

모든 에이전트는 act(obs, env) -> (n,) 행동 배열 인터페이스를 가진다.
기준 에이전트는 env 의 상태 중 에이전트가 원래 알 수 있는 것(걸음 수)만 읽는다.
flag 위치(env.flag)는 절대 보지 않는다.
"""

from __future__ import annotations

import numpy as np

from .env import DOWN, LEFT, RIGHT, UP, BackroomVecEnv


class RandomWalkAgent:
    """아무 문이나 연다."""

    name = "random-walk"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env: BackroomVecEnv):
        return self.rng.integers(0, 4, size=env.n)


def spiral_actions(n_steps: int) -> np.ndarray:
    """시작점에서 바깥으로 도는 사각 나선: 오1 위1 왼2 아2 오3 위3 ...
    같은 방을 두 번 가지 않으면서 반경 k 의 정사각형을 (2k+1)^2 - 1 걸음에 모두 훑는다."""
    out = []
    dirs = (RIGHT, UP, LEFT, DOWN)
    length, d = 1, 0
    while len(out) < n_steps:
        for _ in range(2):
            out.extend([dirs[d % 4]] * length)
            d += 1
        length += 1
    return np.array(out[:n_steps], dtype=np.int64)


class SpiralAgent:
    """단서가 없을 때 할 수 있는 최선. 걸음 수만 세면 되므로 기억도 필요 없다."""

    name = "spiral"

    def __init__(self, max_steps: int):
        self.plan = spiral_actions(max_steps + 1)

    def act(self, obs, env: BackroomVecEnv):
        return self.plan[np.minimum(env.t, len(self.plan) - 1)]
