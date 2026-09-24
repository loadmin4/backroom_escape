import numpy as np
import pytest
import torch

from backroom.baselines import BayesSearchAgent, SpiralAgent, spiral_actions
from backroom.env import MOVES, RIGHT, UP, VEC_DIM, BackroomConfig, BackroomVecEnv
from backroom.evaluation import evaluate
from backroom.model import ActorCritic, PolicyAgent


def test_flag_is_in_range_and_never_at_start():
    cfg = BackroomConfig(flag_range=3)
    env = BackroomVecEnv(cfg, n_envs=4000, seed=0)
    env.reset()
    assert np.abs(env.flag).max() <= 3
    assert not np.all(env.flag == 0, axis=1).any()
    # 후보 방 48개가 모두 나온다
    assert len({tuple(f) for f in env.flag}) == cfg.n_candidates


def test_rooms_look_identical_without_hint():
    """시작점과 flag 가 달라도, 힌트가 없으면 처음 보는 관측은 모든 회차에서 똑같다."""
    env = BackroomVecEnv(BackroomConfig(hint="none"), n_envs=50, seed=1)
    obs = env.reset()
    assert len({tuple(w) for w in env.world_start}) == 50
    assert np.all(obs["map"] == obs["map"][0])
    assert np.all(obs["vec"] == obs["vec"][0])


def test_entering_flag_room_ends_episode_and_resets():
    cfg = BackroomConfig()
    env = BackroomVecEnv(cfg, n_envs=1, seed=0)
    env.reset()
    env.flag[0] = (1, 0)
    _, reward, terminated, truncated, info = env.step([RIGHT])
    assert terminated[0] and not truncated[0]
    assert reward[0] == pytest.approx(cfg.flag_reward - cfg.step_penalty)
    assert info["done_len"].tolist() == [1] and info["done_success"].tolist() == [True]
    assert env.t[0] == 0 and tuple(env.pos[0]) == (0, 0)


def test_time_limit():
    cfg = BackroomConfig(max_steps=5)
    env = BackroomVecEnv(cfg, n_envs=1, seed=0)
    env.reset()
    env.flag[0] = (6, 6)
    for i in range(5):
        _, _, terminated, truncated, info = env.step([UP if i % 2 else RIGHT])
    assert truncated[0] and not terminated[0]
    assert info["done_success"].tolist() == [False]


def test_spiral_covers_square_without_revisits():
    k = 5
    pos, seen = np.zeros(2, dtype=np.int64), {(0, 0)}
    for a in spiral_actions((2 * k + 1) ** 2 - 1):
        pos = pos + MOVES[a]
        assert tuple(pos) not in seen
        seen.add(tuple(pos))
    assert seen == {(x, y) for x in range(-k, k + 1) for y in range(-k, k + 1)}


@pytest.mark.parametrize("drift", [0.0, 0.8])
def test_memory_matches_true_infinite_grid(drift):
    """원환 배열로 접은 기억이 진짜 무한 격자(집합)와 똑같이 보이는지 확인한다.
    drift > 0 이면 한쪽으로 멀리 걸어가서 배열 경계를 여러 번 넘는다."""
    cfg = BackroomConfig(hint="sound", max_steps=40, view_radius=6, flag_range=3, sound_noise=0.0)
    n = 16
    env = BackroomVecEnv(cfg, n_envs=n, seed=2)
    obs = env.reset()
    env.flag[:] = 10**6  # 회차가 끝나지 않게
    rng = np.random.default_rng(0)
    pos = np.zeros((n, 2), dtype=np.int64)
    visited = [{(0, 0)} for _ in range(n)]
    r = cfg.view_radius
    for _ in range(cfg.max_steps - 1):
        actions = np.where(rng.random(n) < drift, RIGHT, rng.integers(0, 4, n))
        pos += MOVES[actions]
        for i in range(n):
            visited[i].add(tuple(pos[i]))
        obs, *_ = env.step(actions)
        for i in range(n):
            expect = np.zeros((2 * r + 1, 2 * r + 1))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    expect[dy + r, dx + r] = (pos[i, 0] + dx, pos[i, 1] + dy) in visited[i]
            np.testing.assert_array_equal(obs["map"][i, 0], expect)


def test_sound_level_without_noise():
    cfg = BackroomConfig(hint="sound", sound_noise=0.0)
    env = BackroomVecEnv(cfg, n_envs=1, seed=0)
    env.reset()
    env.flag[0] = (3, 0)
    env.step([RIGHT])
    env.step([RIGHT])  # 거리 1
    assert env.last_sound[0] == pytest.approx(0.5)


def test_evaluation_uses_same_layouts_for_every_agent():
    cfg = BackroomConfig(hint="sound")
    a = BackroomVecEnv(cfg, n_envs=20, seed=7)
    b = BackroomVecEnv(cfg, n_envs=20, seed=7)
    a.reset(), b.reset()
    np.testing.assert_array_equal(a.flag, b.flag)


def test_baselines_always_escape():
    cfg = BackroomConfig(hint="sound")
    assert evaluate(lambda env: SpiralAgent(cfg.max_steps), cfg, 200).success_rate == 1.0
    assert evaluate(lambda env: BayesSearchAgent(env), cfg, 200).success_rate == 1.0


def test_policy_agent_runs():
    cfg = BackroomConfig()
    env = BackroomVecEnv(cfg, n_envs=3, seed=0)
    obs = env.reset()
    assert obs["map"].shape == (3, 3, cfg.view_size, cfg.view_size)
    assert obs["vec"].shape == (3, VEC_DIM)
    logits, value = ActorCritic(cfg.view_size)(torch.from_numpy(obs["map"]), torch.from_numpy(obs["vec"]))
    assert logits.shape == (3, 4) and value.shape == (3,)
    assert PolicyAgent(ActorCritic(cfg.view_size)).act(obs, env).shape == (3,)
