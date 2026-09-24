"""백룸 탈출 에이전트 학습.

    python train.py --hint sound --steps 4000000
    python train.py --hint none  --steps 4000000
    python train.py --hint sound --device cpu      # GPU 가 있어도 CPU 로
    python train.py --resume checkpoints/ppo_sound.pt --steps 4000000   # 저장된 모델에서 400만 걸음 더

학습 곡선은 runs/ 에 TensorBoard 형식으로 기록된다:  tensorboard --logdir runs
"""

import argparse
import os
import time

try:
    import torch
except ModuleNotFoundError as e:  # .venv 가 아닌 파이썬으로 실행했거나 설치가 안 된 경우
    from backroom import exit_missing_package

    exit_missing_package(e)

from backroom.env import HINTS, BackroomConfig
from backroom.ppo import PPOConfig, train


def make_writer(logdir: str, name: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("(tensorboard 가 설치돼 있지 않아 학습 곡선은 기록하지 않습니다: pip install tensorboard)")
        return None
    path = os.path.join(logdir, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"TensorBoard 로그: {path}")
    return SummaryWriter(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hint", choices=HINTS, default="sound", help="none: 방이 전부 똑같음 / sound: flag 의 소리가 들림")
    p.add_argument("--steps", type=int, default=4_000_000, help="총 학습 걸음 수")
    p.add_argument("--device", default="auto", help="auto(GPU 있으면 GPU) / cuda / cpu")
    p.add_argument("--flag-range", type=int, default=6, help="flag 는 시작점에서 가로/세로 이 칸 수 이내")
    p.add_argument("--max-steps", type=int, default=250, help="한 회차의 최대 걸음 수")
    p.add_argument("--sound-noise", type=float, default=0.1)
    p.add_argument("--n-envs", type=int, default=64, help="동시에 돌리는 회차 수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="CPU 계산에 쓸 스레드 수")
    p.add_argument("--out", default=None, help="체크포인트 경로 (기본: checkpoints/ppo_<hint>.pt)")
    p.add_argument("--logdir", default="runs", help="TensorBoard 로그 폴더 ('' 이면 기록 안 함)")
    p.add_argument(
        "--resume",
        default=None,
        help="이 체크포인트에서 이어서 --steps 만큼 더 학습 (환경 설정은 체크포인트 것을 쓰고, 기본 저장 위치도 같은 파일)",
    )
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        env_cfg = BackroomConfig(**resume["env_config"])
        print(f"이어서 학습: {args.resume} ({resume.get('steps', 0):,} 걸음 학습된 모델)")
    else:
        env_cfg = BackroomConfig(
            flag_range=args.flag_range,
            max_steps=args.max_steps,
            hint=args.hint,
            sound_noise=args.sound_noise,
            view_radius=2 * args.flag_range,
        )
    out = args.out or args.resume or f"checkpoints/ppo_{env_cfg.hint}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"env: {env_cfg}")
    writer = make_writer(args.logdir, f"ppo_{env_cfg.hint}") if args.logdir else None
    ppo_cfg = PPOConfig(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed)
    try:
        train(env_cfg, ppo_cfg, out, device=args.device, writer=writer, resume=resume)
        print(f"saved -> {out}")
    except KeyboardInterrupt:
        print(f"\n학습을 중단했습니다. 체크포인트는 약 8만 걸음마다 {out} 에 저장됩니다 (그 전에 멈췄다면 없음).")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
