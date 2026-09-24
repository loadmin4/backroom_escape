"""비교용 기준 에이전트들.

모든 에이전트는 act(obs, env) -> (n,) 행동 배열 인터페이스를 가진다.
기준 에이전트는 env 의 상태 중 에이전트가 원래 알 수 있는 것(자기 상대 좌표, 걸음 수,
방문 기록, 방금 들은 소리)만 읽는다. flag 위치(env.flag)는 절대 보지 않는다.
"""

from __future__ import annotations

import numpy as np

from .env import DOWN, LEFT, MOVES, RIGHT, UP, BackroomVecEnv, sound_level


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
    """힌트 없이 할 수 있는 최선. 걸음 수만 세면 되므로 기억도 필요 없다."""

    name = "spiral"

    def __init__(self, max_steps: int):
        self.plan = spiral_actions(max_steps + 1)

    def act(self, obs, env: BackroomVecEnv):
        return self.plan[np.minimum(env.t, len(self.plan) - 1)]


class BayesSearchAgent:
    """flag 의 사전분포(시작점 주변 정사각형 균등)와 소리 모델을 알고 있는 모델 기반 탐색.

    flag 가 각 방에 있을 확률(사후분포)을 유지하면서
      - 들어가 본 방은 확률 0,
      - 소리를 들을 때마다 베이즈 규칙으로 갱신,
    한 뒤 '확률 / 거리' 가 가장 큰 방을 향해 한 칸 움직인다.
    학습된 정책이 얼마나 잘하는지 가늠하기 위한 강한 기준선이다.
    """

    name = "bayes-search"

    def __init__(self, env: BackroomVecEnv):
        cfg = env.cfg
        rng = np.arange(-cfg.flag_range, cfg.flag_range + 1)
        cx, cy = np.meshgrid(rng, rng)
        self.cells = np.stack([cx.ravel(), cy.ravel()], axis=1)  # (K, 2)
        self.start_idx = int(np.flatnonzero((self.cells == 0).all(axis=1))[0])
        self.use_sound = cfg.hint == "sound"
        self.inv_two_var = 1.0 / (2.0 * cfg.sound_noise**2)
        self.logp = np.zeros((env.n, len(self.cells)))

    def act(self, obs, env: BackroomVecEnv):
        fresh = env.t == 0
        self.logp[fresh] = 0.0
        self.logp[fresh, self.start_idx] = -np.inf

        # 에이전트 -> 모든 후보 방 거리 (n, K)
        dist = np.abs(env.pos[:, None, :] - self.cells[None, :, :]).sum(axis=2)
        if self.use_sound:
            expected = sound_level(dist)
            self.logp -= (env.last_sound[:, None] - expected) ** 2 * self.inv_two_var
        self.logp[dist == 0] = -np.inf  # 지금 방에는 없었다

        score = self.logp - np.log(np.maximum(dist, 1))
        target = self.cells[np.argmax(score, axis=1)]  # (n, 2)

        # 목표에 가까워지는 문을 고르되, 같은 조건이면 안 가본 방을 우선
        cand = env.pos[:, None, :] + MOVES[None, :, :]  # (n, 4, 2)
        closer = np.abs(cand - target[:, None, :]).sum(axis=2) < np.abs(env.pos - target).sum(axis=1)[:, None]
        fresh_room = np.stack([~env.visited_at(cand[:, a]) for a in range(4)], axis=1)
        return np.argmax(2 * closer + fresh_room, axis=1)
