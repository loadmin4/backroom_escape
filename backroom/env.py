"""백룸 환경 (여러 회차를 동시에 돌리는 벡터 환경).

세계의 규칙
-----------
- 무한히 펼쳐진 격자. 모든 방은 똑같이 생겼고 상/하/좌/우에 문이 하나씩 있다.
- 회차마다 시작점(월드 좌표)과 flag 위치가 무작위로 바뀐다.
- flag 가 있는 방에 들어가면 탈출(성공).

에이전트가 알 수 있는 것
-----------------------
방이 전부 똑같으므로 절대 좌표는 아무 의미가 없다. 에이전트가 쓸 수 있는 정보는
  1) 추측 항법: 시작점에서 내가 몇 칸 움직였는가 (상대 좌표)
  2) 분필 표시: 이미 들어가 본 방이 어디인가 (방문 기록)
  3) (hint="sound" 일 때만) 방마다 들리는 flag 의 소리 크기. 거리가 멀수록 작아지고 잡음이 섞인다.
뿐이다. 월드 좌표(`world_start`)와 flag 위치(`flag`)는 채점/시각화용이며 정책에 넘기지 않는다.

좌표계: 시작점 = (0, 0), x 는 오른쪽, y 는 위쪽이 +.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ("up", "right", "down", "left")
# 행동별 (dx, dy)
MOVES = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int64)

HINTS = ("none", "sound")

# 관측 지도의 채널: 방문 여부, 시작점 표시, 그 방에서 들은 소리의 평균
MAP_CHANNELS = 3
LOCAL_RADIUS = 2  # 벡터 관측에 그대로 펴서 넣는 주변 영역 (5x5)
VEC_DIM = 4 + MAP_CHANNELS * (2 * LOCAL_RADIUS + 1) ** 2


@dataclass
class BackroomConfig:
    # flag 는 시작점에서 가로/세로 각각 flag_range 칸 이내의 방 중 하나에 균등하게 놓인다.
    flag_range: int = 6
    max_steps: int = 250
    hint: str = "none"
    # 들리는 소리 = 1 / (1 + 맨해튼 거리) + N(0, sound_noise^2)
    sound_noise: float = 0.1
    # 에이전트가 보는 자기 중심 기억 지도의 반경 ((2r+1) x (2r+1))
    view_radius: int = 12
    step_penalty: float = 0.01
    flag_reward: float = 1.0

    def __post_init__(self):
        if self.hint not in HINTS:
            raise ValueError(f"hint must be one of {HINTS}, got {self.hint!r}")
        if self.flag_range < 1:
            raise ValueError("flag_range must be >= 1")
        if self.view_radius < LOCAL_RADIUS:
            raise ValueError(f"view_radius must be >= {LOCAL_RADIUS}")

    @property
    def view_size(self) -> int:
        return 2 * self.view_radius + 1

    @property
    def n_candidates(self) -> int:
        """flag 가 있을 수 있는 방의 수 (시작 방 제외)."""
        return (2 * self.flag_range + 1) ** 2 - 1

    def to_dict(self) -> dict:
        return asdict(self)


def sound_level(dist):
    """잡음이 없을 때 거리 dist 에서 들리는 소리 크기."""
    return 1.0 / (1.0 + dist)


class BackroomVecEnv:
    """n_envs 개의 백룸 회차를 한꺼번에 진행한다. 끝난 회차는 자동으로 새 회차로 리셋된다.

    step() 은 (obs, reward, terminated, truncated, info) 를 돌려준다.
    obs = {"map": (n, C, W, W) float32, "vec": (n, VEC_DIM) float32}
    info = {"done_len": 끝난 회차들의 걸음 수, "done_success": 성공 여부}
    """

    def __init__(self, config: BackroomConfig, n_envs: int = 1, seed: int | None = None):
        self.cfg = config
        self.n = n_envs
        layout_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        # 시작점/flag 배치와 소리 잡음의 난수를 분리해 둔다. 같은 seed 면 어떤 에이전트로
        # 평가하든 각 슬롯의 첫 회차 배치가 똑같아서 공정하게 비교할 수 있다.
        self.layout_rng = np.random.default_rng(layout_seed)
        self.noise_rng = np.random.default_rng(noise_seed)

        # 무한한 세계를 한 변이 P 인 원환(torus) 배열에 접어서 기억한다. 한 회차의 경로는
        # 가로/세로로 max_steps 칸보다 넓게 퍼질 수 없으므로, P > max_steps + view_radius 이면
        # 관측 창 안에서 서로 다른 두 방이 같은 칸에 겹치는 일이 없다. 즉 무한 격자와 똑같이 동작한다.
        self._P = config.max_steps + config.view_radius + 1
        self.visits = np.zeros((n_envs, self._P, self._P), dtype=np.uint16)
        self.sound_sum = np.zeros((n_envs, self._P, self._P), dtype=np.float32) if config.hint == "sound" else None

        self.pos = np.zeros((n_envs, 2), dtype=np.int64)
        self.flag = np.zeros((n_envs, 2), dtype=np.int64)
        self.world_start = np.zeros((n_envs, 2), dtype=np.int64)
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.last_sound = np.zeros(n_envs, dtype=np.float32)

        r = config.view_radius
        self._offsets = np.arange(-r, r + 1)
        self._idx = np.arange(n_envs)

    # ------------------------------------------------------------------ 리셋
    def reset(self):
        for i in range(self.n):
            self._reset_slot(i)
        return self.observe()

    def _reset_slot(self, i: int):
        cfg = self.cfg
        self.visits[i] = 0
        if self.sound_sum is not None:
            self.sound_sum[i] = 0.0
        self.pos[i] = 0
        self.t[i] = 0
        # 월드 좌표상의 시작점. 방이 모두 똑같으므로 에이전트에게는 보이지 않는다.
        self.world_start[i] = self.layout_rng.integers(-10**9, 10**9, size=2)
        k = self.layout_rng.integers(cfg.n_candidates)
        side = 2 * cfg.flag_range + 1
        k = k + (k >= cfg.n_candidates // 2)  # 정중앙(시작 방)은 건너뛴다
        self.flag[i] = (k % side - cfg.flag_range, k // side - cfg.flag_range)
        self._enter(np.array([i]))

    # ------------------------------------------------------------------ 진행
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        cfg = self.cfg
        self.pos += MOVES[actions]
        self.t += 1

        found = np.all(self.pos == self.flag, axis=1)
        self._enter(self._idx[~found])

        reward = np.where(found, cfg.flag_reward, 0.0) - cfg.step_penalty
        terminated = found
        truncated = ~found & (self.t >= cfg.max_steps)
        done = terminated | truncated
        info = {"done_len": self.t[done].copy(), "done_success": found[done].copy()}
        for i in np.flatnonzero(done):
            self._reset_slot(i)
        return self.observe(), reward.astype(np.float32), terminated, truncated, info

    def _enter(self, ids):
        """ids 회차의 에이전트가 현재 방에 들어섰다: 분필 표시를 하고 소리를 듣는다."""
        if len(ids) == 0:
            return
        rows = self.pos[ids, 1] % self._P
        cols = self.pos[ids, 0] % self._P
        self.visits[ids, rows, cols] += 1
        if self.cfg.hint == "sound":
            dist = np.abs(self.pos[ids] - self.flag[ids]).sum(axis=1)
            noise = self.noise_rng.normal(0.0, self.cfg.sound_noise, size=len(ids))
            heard = (sound_level(dist) + noise).astype(np.float32)
            self.last_sound[ids] = heard
            self.sound_sum[ids, rows, cols] += heard

    # ------------------------------------------------------------------ 관측
    def observe(self):
        cfg = self.cfg
        r = cfg.view_radius
        rows = (self.pos[:, 1, None, None] + self._offsets[None, :, None]) % self._P
        cols = (self.pos[:, 0, None, None] + self._offsets[None, None, :]) % self._P
        idx = self._idx[:, None, None]
        visits = self.visits[idx, rows, cols].astype(np.float32)
        visited = np.minimum(visits, 1.0)
        if self.sound_sum is not None:
            sound = self.sound_sum[idx, rows, cols] / np.maximum(visits, 1.0)
        else:
            sound = np.zeros_like(visited)

        start = np.zeros_like(visited)
        sy, sx = r - self.pos[:, 1], r - self.pos[:, 0]
        inside = (np.abs(sx - r) <= r) & (np.abs(sy - r) <= r)
        start[self._idx[inside], sy[inside], sx[inside]] = 1.0

        grid = np.stack([visited, start, sound], axis=1)  # (n, C, W, W)
        lo, hi = r - LOCAL_RADIUS, r + LOCAL_RADIUS + 1
        vec = np.concatenate(
            [
                self.pos / cfg.flag_range,
                (self.t / cfg.max_steps)[:, None],
                self.last_sound[:, None],
                grid[:, :, lo:hi, lo:hi].reshape(self.n, -1),
            ],
            axis=1,
        ).astype(np.float32)
        return {"map": grid, "vec": vec}

    def visited_at(self, xy):
        """(n, 2) 상대 좌표들이 이미 방문한 방인지. 에이전트 자신의 기억이므로 정책이 써도 된다."""
        return self.visits[self._idx, xy[:, 1] % self._P, xy[:, 0] % self._P] > 0
