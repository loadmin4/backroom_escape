"""설치가 제대로 됐는지, GPU 로 학습할 수 있는지 확인하고 학습 속도를 잰다.

    python check_gpu.py            # 확인 + 속도 측정
    python check_gpu.py --no-bench # 확인만
"""

import os
import shutil
import sys
import tempfile
import time


def check_cuda(torch) -> bool:
    has_smi = shutil.which("nvidia-smi") is not None
    if not torch.cuda.is_available():
        print("GPU        : 사용할 수 없음 -> CPU 로 학습합니다")
        if torch.version.cuda is None and has_smi:
            print("  ! NVIDIA GPU 는 있는데 CPU 전용 PyTorch 가 설치돼 있습니다.")
            print("    설치 스크립트를 GPU 모드로 다시 실행하세요: scripts\\setup_windows.ps1")
        elif not has_smi:
            print("  ! NVIDIA 드라이버(nvidia-smi)를 찾지 못했습니다. NVIDIA GPU 가 있다면 드라이버를 설치하세요.")
        else:
            print("  ! GPU 용 PyTorch 는 설치됐지만 GPU 를 쓸 수 없습니다. NVIDIA 드라이버를 최신으로 업데이트해 보세요.")
        return False

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}      : {props.name} (compute {props.major}.{props.minor}, {props.total_memory / 2**30:.1f} GB)")
    try:
        x = torch.randn(512, 512, device="cuda")
        (x @ x).sum().item()
    except RuntimeError as e:
        print(f"  ! GPU 에서 계산하다가 오류가 났습니다: {e}")
        print("    GPU 가 이 PyTorch 빌드를 지원하지 않는 경우가 많습니다 (예: RTX 50 시리즈 + cu126).")
        print("    드라이버를 최신으로 올린 뒤 scripts\\setup_windows.ps1 -Cuda cu130 으로 다시 설치하세요.")
        return False
    print("  -> GPU 로 학습할 수 있습니다.")
    return True


def benchmark(devices):
    from backroom.env import BackroomConfig
    from backroom.ppo import PPOConfig, train

    cfg = PPOConfig(total_steps=64 * 128 * 3)  # PPO 업데이트 3번 분량
    print("\n학습 속도 측정 (대략적인 값):")
    for device in devices:
        with tempfile.TemporaryDirectory() as d:
            t0 = time.time()
            train(BackroomConfig(), cfg, os.path.join(d, "bench.pt"), device=device, log=lambda *_: None)
            sps = cfg.total_steps / (time.time() - t0)
        print(f"  {device:<5}: 초당 {sps:7,.0f} 걸음 -> 400만 걸음 학습에 약 {4e6 / sps / 60:.0f}분")


def main():
    print(f"Python     : {sys.version.split()[0]} ({sys.executable})")
    if sys.prefix == sys.base_prefix:
        print("  ! 가상환경(.venv) 밖의 파이썬입니다. VS Code 에서는 Ctrl+Shift+P -> 'Python: Select Interpreter' 로 .venv 를 고르세요.")
    try:
        import numpy
        import torch
    except ModuleNotFoundError as e:
        from backroom import exit_missing_package

        exit_missing_package(e)
    print(f"numpy      : {numpy.__version__}")
    print(f"PyTorch    : {torch.__version__} (CUDA 빌드: {torch.version.cuda or '없음, CPU 전용'})")
    try:
        import tensorboard

        print(f"TensorBoard: {tensorboard.__version__}")
    except ImportError:
        print("TensorBoard: 없음 (학습 곡선이 기록되지 않습니다. pip install tensorboard)")

    cuda_ok = check_cuda(torch)
    if "--no-bench" not in sys.argv:
        benchmark(["cpu", "cuda"] if cuda_ok else ["cpu"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
