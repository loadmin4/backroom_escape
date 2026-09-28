"""한 회차를 터미널에 그려서 보여준다.

    python play.py --checkpoint checkpoints/ppo.pt
    python play.py --agent spiral
    python play.py --checkpoint checkpoints/ppo.pt --animate

S = 시작 방, F = flag, @ = 에이전트, o = 지나온 방, . = flag 가 있을 수 있는 방
"""

import argparse
import os
import time

try:
    import torch
except ModuleNotFoundError as e:  # .venv 가 아닌 파이썬으로 실행했거나 설치가 안 된 경우
    from backroom import exit_missing_package

    exit_missing_package(e)

from backroom.baselines import RandomWalkAgent, SpiralAgent
from backroom.env import MOVES, BackroomConfig, BackroomVecEnv
from backroom.model import PolicyAgent, load_checkpoint


def render(cfg: BackroomConfig, pos, flag, path) -> str:
    """path = 지나온 방들의 (x, y)"""
    radius = cfg.flag_range
    seen = set(path)
    lo_x = min(-radius, min(x for x, _ in path))
    hi_x = max(radius, max(x for x, _ in path))
    lo_y = min(-radius, min(y for _, y in path))
    hi_y = max(radius, max(y for _, y in path))
    lines = []
    for y in range(hi_y, lo_y - 1, -1):
        row = []
        for x in range(lo_x, hi_x + 1):
            if (x, y) == pos:
                c = "@"
            elif (x, y) == flag:
                c = "F"
            elif (x, y) == (0, 0):
                c = "S"
            elif (x, y) in seen:
                c = "o"
            elif max(abs(x), abs(y)) <= cfg.flag_range:
                c = "."
            else:
                c = " "
            row.append(c)
        lines.append(" ".join(row))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--agent", choices=["ppo", "spiral", "random"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--delay", type=float, default=0.15)
    args = p.parse_args()

    torch.set_num_threads(1)
    if os.name == "nt":
        os.system("")  # 윈도우 콘솔(cmd)에서 화면 지우기용 ANSI 코드가 동작하도록 켠다
    agent_name = args.agent or ("ppo" if args.checkpoint else "spiral")
    if args.checkpoint:
        model, ckpt = load_checkpoint(args.checkpoint)
        cfg = BackroomConfig(**ckpt["env_config"])
    else:
        if agent_name == "ppo":
            p.error("--agent ppo 에는 --checkpoint 가 필요합니다")
        cfg = BackroomConfig()

    env = BackroomVecEnv(cfg, n_envs=1, seed=args.seed)
    obs = env.reset()
    agent = {
        "ppo": lambda: PolicyAgent(model, seed=args.seed),
        "spiral": lambda: SpiralAgent(cfg.max_steps),
        "random": lambda: RandomWalkAgent(args.seed),
    }[agent_name]()

    print(f"agent={agent_name}")
    flag = tuple(int(v) for v in env.flag[0])  # 회차가 끝나면 env 가 자동 리셋되므로 미리 적어 둔다
    world = tuple(int(v) for v in env.world_start[0])
    print(f"월드 좌표상 시작점 {world} (에이전트는 모름), 시작점 기준 flag 위치 {flag}")
    pos = (0, 0)
    path = [pos]
    for step in range(1, cfg.max_steps + 1):
        action = int(agent.act(obs, env)[0])
        pos = (pos[0] + int(MOVES[action][0]), pos[1] + int(MOVES[action][1]))
        obs, _, terminated, truncated, _ = env.step([action])
        if terminated[0] or truncated[0]:
            print(render(cfg, pos, flag, path))
            print(f"\n{'탈출 성공!' if terminated[0] else '실패 (시간 초과)'}  걸음 수 = {step}")
            return
        path.append(pos)
        if args.animate:
            print("\033[H\033[J" + render(cfg, pos, flag, path) + f"\n\nstep {step}")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
