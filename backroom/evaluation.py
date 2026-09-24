"""에이전트 평가: 같은 seed 로 같은 배치(시작점/flag)의 회차들을 돌려서 비교한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import BackroomConfig, BackroomVecEnv


@dataclass
class EvalResult:
    name: str
    steps: np.ndarray  # 회차별 걸음 수 (실패는 max_steps)
    success: np.ndarray

    @property
    def success_rate(self) -> float:
        return float(self.success.mean())

    @property
    def mean_steps(self) -> float:
        return float(self.steps.mean())

    def row(self) -> str:
        return (
            f"{self.name:<14} {100 * self.success_rate:7.1f}% {self.mean_steps:9.1f} "
            f"{np.median(self.steps):8.0f} {np.percentile(self.steps, 90):8.0f}"
        )


HEADER = f"{'agent':<14} {'success':>8} {'mean':>9} {'median':>8} {'p90':>8}"


def evaluate(make_agent, config: BackroomConfig, episodes: int, seed: int = 12345, batch: int = 250) -> EvalResult:
    """episodes 개의 회차를 한 번씩 돌린다. make_agent(env) -> agent.

    메모리를 아끼려고 batch 개씩 나눠 돌리며, 묶음마다 seed 가 정해져 있으므로
    어떤 에이전트든 똑같은 시작점/flag 배치에서 평가된다."""
    steps, success, name = [], [], None
    for k, start in enumerate(range(0, episodes, batch)):
        env = BackroomVecEnv(config, n_envs=min(batch, episodes - start), seed=seed + k)
        agent = make_agent(env)
        name = getattr(agent, "name", type(agent).__name__)
        s, ok = _run_once(agent, env)
        steps.append(s)
        success.append(ok)
    return EvalResult(name, np.concatenate(steps), np.concatenate(success))


def _run_once(agent, env: BackroomVecEnv):
    """env 의 각 슬롯에서 첫 회차만 끝까지 돌린다."""
    n = env.n
    obs = env.reset()
    steps = np.full(n, env.cfg.max_steps, dtype=np.int64)
    success = np.zeros(n, dtype=bool)
    finished = np.zeros(n, dtype=bool)
    while not finished.all():
        obs, _, terminated, truncated, info = env.step(agent.act(obs, env))
        done = terminated | truncated
        newly = done & ~finished
        # 끝난 슬롯은 이미 자동 리셋됐으므로 걸음 수는 info 에서 읽는다 (info 는 done 슬롯 순서)
        steps[newly] = info["done_len"][newly[done]]
        success[newly] = terminated[newly]
        finished |= newly
    return steps, success
