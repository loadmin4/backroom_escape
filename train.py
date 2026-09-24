"""백룸 탈출 에이전트 학습.

    python train.py --hint none  --steps 3000000
    python train.py --hint sound --steps 3000000
"""

import argparse
import os

import torch

from backroom.env import HINTS, BackroomConfig
from backroom.ppo import PPOConfig, train


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hint", choices=HINTS, default="sound", help="none: 방이 전부 똑같음 / sound: flag 의 소리가 들림")
    p.add_argument("--steps", type=int, default=3_000_000, help="총 학습 걸음 수")
    p.add_argument("--flag-range", type=int, default=6, help="flag 는 시작점에서 가로/세로 이 칸 수 이내")
    p.add_argument("--max-steps", type=int, default=250, help="한 회차의 최대 걸음 수")
    p.add_argument("--sound-noise", type=float, default=0.1)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--out", default=None, help="체크포인트 경로 (기본: checkpoints/ppo_<hint>.pt)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    env_cfg = BackroomConfig(
        flag_range=args.flag_range,
        max_steps=args.max_steps,
        hint=args.hint,
        sound_noise=args.sound_noise,
        view_radius=2 * args.flag_range,
    )
    out = args.out or f"checkpoints/ppo_{args.hint}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"env: {env_cfg}")
    train(env_cfg, PPOConfig(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed), out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
