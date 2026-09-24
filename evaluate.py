"""학습된 모델과 기준 에이전트를 같은 회차들에서 비교한다.

    python evaluate.py --checkpoint checkpoints/ppo_sound.pt
    python evaluate.py --hint none            # 모델 없이 기준 에이전트만
    python evaluate.py --checkpoint checkpoints/ppo_sound.pt --device cuda
"""

import argparse

try:
    import torch
except ModuleNotFoundError as e:  # .venv 가 아닌 파이썬으로 실행했거나 설치가 안 된 경우
    from backroom import exit_missing_package

    exit_missing_package(e)

from backroom.baselines import BayesSearchAgent, RandomWalkAgent, SpiralAgent
from backroom.env import HINTS, BackroomConfig
from backroom.evaluation import HEADER, evaluate
from backroom.model import PolicyAgent, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--hint", choices=HINTS, default=None, help="체크포인트가 없을 때 쓸 모드")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--greedy", action="store_true", help="모델이 확률 대신 가장 높은 행동만 고르게 한다")
    p.add_argument("--device", default="auto", help="모델을 돌릴 장치: auto(GPU 있으면 GPU) / cuda / cpu")
    args = p.parse_args()

    torch.set_num_threads(1)
    model = None
    if args.checkpoint:
        model, ckpt = load_checkpoint(args.checkpoint, device=args.device)
        cfg = BackroomConfig(**ckpt["env_config"])
        print(f"checkpoint: {args.checkpoint} ({ckpt.get('steps', '?'):,} steps, device={next(model.parameters()).device})")
    else:
        cfg = BackroomConfig(hint=args.hint or "sound")

    print(f"env: {cfg}")
    print(f"episodes: {args.episodes}, seed: {args.seed}")
    print(f"힌트 없이 가능한 최선의 평균 걸음 수 = (후보 방 수 + 1) / 2 = {(cfg.n_candidates + 1) / 2:.1f}\n")

    agents = [
        lambda env: RandomWalkAgent(seed=0),
        lambda env: SpiralAgent(cfg.max_steps),
        lambda env: BayesSearchAgent(env),
    ]
    if model is not None:
        agents.append(lambda env: PolicyAgent(model, greedy=args.greedy, seed=0))

    print(HEADER)
    for make in agents:
        print(evaluate(make, cfg, args.episodes, seed=args.seed).row())
    print("\n(mean/median/p90 = flag 를 찾기까지 걸음 수, 실패한 회차는 max_steps 로 계산)")


if __name__ == "__main__":
    main()
