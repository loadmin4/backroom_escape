"""PPO 학습 루프."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .env import BackroomConfig, BackroomVecEnv
from .model import ActorCritic, save_checkpoint


@dataclass
class PPOConfig:
    total_steps: int = 3_000_000
    n_envs: int = 64
    n_steps: int = 128
    gamma: float = 0.995
    gae_lambda: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0


def _to_torch(obs):
    return torch.from_numpy(obs["map"]), torch.from_numpy(obs["vec"])


def train(env_cfg: BackroomConfig, cfg: PPOConfig, out_path: str, log=print) -> ActorCritic:
    torch.manual_seed(cfg.seed)
    env = BackroomVecEnv(env_cfg, n_envs=cfg.n_envs, seed=cfg.seed)
    model = ActorCritic(env_cfg.view_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.n_steps, cfg.n_envs
    obs = env.reset()
    buf_map = torch.zeros((T, N) + obs["map"].shape[1:])
    buf_vec = torch.zeros((T, N, obs["vec"].shape[1]))
    buf_act = torch.zeros((T, N), dtype=torch.long)
    buf_logp = torch.zeros((T, N))
    buf_val = torch.zeros((T, N))
    buf_rew = torch.zeros((T, N))
    buf_done = torch.zeros((T, N))

    n_updates = max(1, cfg.total_steps // (T * N))
    batch = T * N
    mb_size = batch // cfg.minibatches
    ep_len, ep_ok = [], []
    t0 = time.time()

    for update in range(1, n_updates + 1):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)

        # ---------------------------------------------------------- 경험 수집
        model.eval()
        for t in range(T):
            grid, vec = _to_torch(obs)
            with torch.no_grad():
                logits, value = model(grid, vec)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            buf_map[t], buf_vec[t] = grid, vec
            buf_act[t], buf_logp[t], buf_val[t] = action, dist.log_prob(action), value

            obs, reward, terminated, truncated, info = env.step(action.numpy())
            # 시간 초과도 종료로 취급한다. 남은 시간(t/max_steps)이 관측에 있으므로 마르코프성이 유지된다.
            buf_rew[t] = torch.from_numpy(reward)
            buf_done[t] = torch.from_numpy((terminated | truncated).astype(np.float32))
            ep_len.extend(info["done_len"].tolist())
            ep_ok.extend(info["done_success"].tolist())

        # ---------------------------------------------------------- GAE
        with torch.no_grad():
            _, next_value = model(*_to_torch(obs))
        adv = torch.zeros((T, N))
        last = torch.zeros(N)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + cfg.gamma * nv * nonterm - buf_val[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + buf_val

        # ---------------------------------------------------------- 정책 갱신
        model.train()
        f_map = buf_map.reshape((batch,) + buf_map.shape[2:])
        f_vec = buf_vec.reshape(batch, -1)
        f_act, f_logp = buf_act.reshape(-1), buf_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), ret.reshape(-1)
        stats = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch)
            for s in range(0, batch, mb_size):
                mb = perm[s : s + mb_size]
                logits, value = model(f_map[mb], f_vec[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(f_act[mb])
                ratio = (logp - f_logp[mb]).exp()
                a = f_adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_loss = 0.5 * (value - f_ret[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                stats.append((pg_loss.item(), v_loss.item(), entropy.item()))

        if update % 10 == 0 or update == n_updates:
            steps = update * batch
            pg, vl, ent = np.mean(stats, axis=0)
            if ep_len:
                ok = 100 * np.mean(ep_ok)
                log(
                    f"[{steps:>9,d} steps | {steps / (time.time() - t0):6.0f}/s] "
                    f"episodes={len(ep_len):5d} success={ok:5.1f}% mean_steps={np.mean(ep_len):6.1f} "
                    f"entropy={ent:.3f} v_loss={vl:.4f}"
                )
            ep_len, ep_ok = [], []
            save_checkpoint(out_path, model, env_cfg.to_dict(), {"ppo_config": asdict(cfg), "steps": steps})

    return model
