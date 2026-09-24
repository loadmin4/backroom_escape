"""PPO 학습 루프."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .env import BackroomConfig, BackroomVecEnv
from .model import ActorCritic, resolve_device, save_checkpoint


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


def train(
    env_cfg: BackroomConfig,
    cfg: PPOConfig,
    out_path: str,
    device: str = "auto",
    writer=None,
    log=print,
    resume: dict | None = None,
) -> ActorCritic:
    """PPO 로 학습하고 out_path 에 체크포인트를 저장한다.

    환경은 numpy 로 CPU 에서 돌고, 신경망 계산(행동 선택, 역전파)은 device(cpu/cuda)에서 한다.
    writer 에 TensorBoard SummaryWriter 를 넘기면 학습 곡선을 기록한다.
    resume 에 이전 체크포인트(dict)를 넘기면 그 가중치(와 옵티마이저 상태)에서 이어서 학습한다."""
    device = resolve_device(device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 입력 크기가 고정이라 가장 빠른 합성곱 알고리즘을 골라 둔다
    env = BackroomVecEnv(env_cfg, n_envs=cfg.n_envs, seed=cfg.seed)
    model = ActorCritic(env_cfg.view_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    step_offset = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        if "optimizer" in resume:
            opt.load_state_dict(resume["optimizer"])
        step_offset = int(resume.get("steps", 0))

    def to_device(obs):
        return torch.from_numpy(obs["map"]).to(device), torch.from_numpy(obs["vec"]).to(device)

    T, N = cfg.n_steps, cfg.n_envs
    obs = env.reset()
    buf_map = torch.zeros((T, N) + obs["map"].shape[1:], device=device)
    buf_vec = torch.zeros((T, N, obs["vec"].shape[1]), device=device)
    buf_act = torch.zeros((T, N), dtype=torch.long, device=device)
    buf_logp = torch.zeros((T, N), device=device)
    buf_val = torch.zeros((T, N), device=device)
    buf_rew = torch.zeros((T, N), device=device)
    buf_done = torch.zeros((T, N), device=device)

    n_updates = max(1, cfg.total_steps // (T * N))
    batch = T * N
    mb_size = batch // cfg.minibatches
    ep_len, ep_ok = [], []
    n_logged = 0  # ep_len 중 TensorBoard 에 이미 기록한 개수
    t0 = time.time()
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    for update in range(1, n_updates + 1):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)

        # ---------------------------------------------------------- 경험 수집
        model.eval()
        for t in range(T):
            grid, vec = to_device(obs)
            with torch.no_grad():
                logits, value = model(grid, vec)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            buf_map[t], buf_vec[t] = grid, vec
            buf_act[t], buf_logp[t], buf_val[t] = action, dist.log_prob(action), value

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            # 시간 초과도 종료로 취급한다. 남은 시간(t/max_steps)이 관측에 있으므로 마르코프성이 유지된다.
            buf_rew[t] = torch.from_numpy(reward).to(device)
            buf_done[t] = torch.from_numpy((terminated | truncated).astype(np.float32)).to(device)
            ep_len.extend(info["done_len"].tolist())
            ep_ok.extend(info["done_success"].tolist())

        # ---------------------------------------------------------- GAE
        with torch.no_grad():
            _, next_value = model(*to_device(obs))
        adv = torch.zeros((T, N), device=device)
        last = torch.zeros(N, device=device)
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
            perm = torch.randperm(batch, device=device)
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
                # .item() 은 GPU 를 기다리게 하므로 텐서로 모아 두었다가 한 번에 꺼낸다
                stats.append(torch.stack([pg_loss, v_loss, entropy]).detach())

        # TensorBoard 에는 매 업데이트(약 8천 걸음)마다 기록해서 거의 실시간으로 볼 수 있게 한다
        steps = step_offset + update * batch
        pg, vl, ent = torch.stack(stats).mean(dim=0).tolist()
        elapsed = time.time() - t0
        sps = update * batch / elapsed
        lr_now = opt.param_groups[0]["lr"]
        if writer is not None:
            new_len, new_ok = ep_len[n_logged:], ep_ok[n_logged:]
            if new_len:
                writer.add_scalar("episode/success_rate", 100 * np.mean(new_ok), steps)
                writer.add_scalar("episode/mean_steps", float(np.mean(new_len)), steps)
            writer.add_scalar("train/learning_rate", lr_now, steps)
            writer.add_scalar("train/progress_percent", 100 * update / n_updates, steps)
            writer.add_scalar("loss/policy", pg, steps)
            writer.add_scalar("loss/value", vl, steps)
            writer.add_scalar("loss/entropy", ent, steps)
            writer.add_scalar("speed/steps_per_sec", sps, steps)
            writer.flush()
        n_logged = len(ep_len)

        if update % 10 == 0 or update == n_updates:
            eta = (n_updates - update) * elapsed / update
            done = update / n_updates
            bar = "#" * int(20 * done) + "-" * (20 - int(20 * done))
            if ep_len:
                log(
                    f"[{bar}] {100 * done:5.1f}% 남은 시간 {eta / 60:5.1f}분 | {steps:>10,d} steps {sps:6.0f}/s | "
                    f"success={100 * np.mean(ep_ok):5.1f}% mean_steps={np.mean(ep_len):6.1f} "
                    f"lr={lr_now:.2e} entropy={ent:.3f} v_loss={vl:.4f}"
                )
            ep_len, ep_ok = [], []
            n_logged = 0
            save_checkpoint(
                out_path,
                model,
                env_cfg.to_dict(),
                {"ppo_config": asdict(cfg), "steps": steps, "optimizer": opt.state_dict()},
            )

    return model
