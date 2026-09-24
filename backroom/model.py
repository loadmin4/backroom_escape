"""정책/가치 신경망."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .env import MAP_CHANNELS, VEC_DIM


def _init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """자기 중심 기억 지도(CNN) + 좌표/주변 정보(벡터) -> 4개 문에 대한 정책과 가치."""

    def __init__(self, view_size: int, hidden: int = 256):
        super().__init__()
        self.view_size = view_size
        # CPU 에서도 빨리 학습되도록 처음부터 해상도를 줄인다. 바로 옆 방들의 정확한 정보는
        # 벡터 관측(주변 5x5)으로 따로 들어간다.
        self.conv = nn.Sequential(
            _init(nn.Conv2d(MAP_CHANNELS, 16, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, MAP_CHANNELS, view_size, view_size)).shape[1]
        self.body = nn.Sequential(
            _init(nn.Linear(flat + VEC_DIM, hidden)), nn.ReLU(),
            _init(nn.Linear(hidden, hidden)), nn.ReLU(),
        )
        self.pi = _init(nn.Linear(hidden, 4), gain=0.01)
        self.v = _init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, grid, vec):
        h = self.body(torch.cat([self.conv(grid), vec], dim=1))
        return self.pi(h), self.v(h).squeeze(-1)


class PolicyAgent:
    """학습된 모델을 기준 에이전트와 같은 인터페이스로 감싼다."""

    name = "ppo"

    def __init__(self, model: ActorCritic, greedy: bool = False, seed: int | None = None):
        self.model = model.eval()
        self.greedy = greedy
        self.gen = torch.Generator().manual_seed(0 if seed is None else seed)

    @torch.no_grad()
    def act(self, obs, env):
        logits, _ = self.model(torch.from_numpy(obs["map"]), torch.from_numpy(obs["vec"]))
        if self.greedy:
            return logits.argmax(dim=1).numpy()
        return torch.multinomial(torch.softmax(logits, dim=1), 1, generator=self.gen).squeeze(1).numpy()


def save_checkpoint(path, model: ActorCritic, env_config: dict, extra: dict | None = None):
    torch.save(
        {"model": model.state_dict(), "view_size": model.view_size, "env_config": env_config, **(extra or {})},
        path,
    )


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(ckpt["view_size"])
    model.load_state_dict(ckpt["model"])
    return model, ckpt
