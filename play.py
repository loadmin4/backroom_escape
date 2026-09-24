"""한 회차를 터미널에 그려서 보여준다.

    python play.py --checkpoint checkpoints/ppo_sound.pt
    python play.py --agent spiral --hint none
    python play.py --checkpoint checkpoints/ppo_sound.pt --animate

S = 시작 방, F = flag, @ = 에이전트, 숫자/점 = 지나온 방 (sound 모드에서는 들은 소리 크기 0~9)
"""

import argparse
import time

import numpy as np
import torch

from backroom.baselines import BayesSearchAgent, RandomWalkAgent, SpiralAgent
from backroom.env import HINTS, MOVES, BackroomConfig, BackroomVecEnv
from backroom.model import PolicyAgent, load_checkpoint


def render(cfg: BackroomConfig, pos, flag, path) -> str:
    """path = [((x, y), 들은 소리), ...]"""
    radius = cfg.flag_range
    heard = {}
    for xy, s in path:
        heard.setdefault(xy, []).append(s)
    lo_x = min(-radius, min(x for (x, _), _ in path))
    hi_x = max(radius, max(x for (x, _), _ in path))
    lo_y = min(-radius, min(y for (_, y), _ in path))
    hi_y = max(radius, max(y for (_, y), _ in path))
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
            elif (x, y) in heard:
                c = str(min(9, int(10 * np.mean(heard[(x, y)])))) if cfg.hint == "sound" else "o"
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
    p.add_argument("--agent", choices=["ppo", "spiral", "bayes", "random"], default=None)
    p.add_argument("--hint", choices=HINTS, default="sound")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--delay", type=float, default=0.15)
    args = p.parse_args()

    torch.set_num_threads(1)
    agent_name = args.agent or ("ppo" if args.checkpoint else "spiral")
    if args.checkpoint:
        model, ckpt = load_checkpoint(args.checkpoint)
        cfg = BackroomConfig(**ckpt["env_config"])
    else:
        if agent_name == "ppo":
            p.error("--agent ppo 에는 --checkpoint 가 필요합니다")
        cfg = BackroomConfig(hint=args.hint)

    env = BackroomVecEnv(cfg, n_envs=1, seed=args.seed)
    obs = env.reset()
    agent = {
        "ppo": lambda: PolicyAgent(model, seed=args.seed),
        "spiral": lambda: SpiralAgent(cfg.max_steps),
        "bayes": lambda: BayesSearchAgent(env),
        "random": lambda: RandomWalkAgent(args.seed),
    }[agent_name]()

    print(f"agent={agent_name} hint={cfg.hint}")
    flag = tuple(int(v) for v in env.flag[0])  # 회차가 끝나면 env 가 자동 리셋되므로 미리 적어 둔다
    world = tuple(int(v) for v in env.world_start[0])
    print(f"월드 좌표상 시작점 {world} (에이전트는 모름), 시작점 기준 flag 위치 {flag}")
    pos = (0, 0)
    path = [(pos, float(env.last_sound[0]))]
    for step in range(1, cfg.max_steps + 1):
        action = int(agent.act(obs, env)[0])
        pos = (pos[0] + int(MOVES[action][0]), pos[1] + int(MOVES[action][1]))
        obs, _, terminated, truncated, _ = env.step([action])
        if terminated[0] or truncated[0]:
            print(render(cfg, pos, flag, path))
            print(f"\n{'탈출 성공!' if terminated[0] else '실패 (시간 초과)'}  걸음 수 = {step}")
            return
        path.append((pos, float(env.last_sound[0])))
        if args.animate:
            print("\033[H\033[J" + render(cfg, pos, flag, path) + f"\n\nstep {step}")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
