"""백룸 탈출: 무한 백룸에서 flag 를 빨리 찾도록 강화학습으로 에이전트를 학습시킨다."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def venv_python() -> Path:
    """설치 스크립트가 만드는 가상환경의 파이썬 경로."""
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def exit_missing_package(error: ModuleNotFoundError):
    """numpy/torch 가 없을 때 원인(설치 안 함 / 설치 실패 / 다른 파이썬 사용 / 다른 폴더를 엶)을 짚어 주고 끝낸다."""
    venv = PROJECT_ROOT / ".venv"
    running_venv = sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == venv.resolve()
    lines = [
        f"[오류] '{error.name}' 패키지를 찾을 수 없습니다.",
        f"  지금 실행 중인 파이썬: {sys.executable}",
    ]
    if running_venv:
        lines += [
            "  -> .venv 가상환경은 맞는데 패키지가 없습니다. 설치가 중간에 실패한 것 같습니다.",
            "     VS Code: 터미널 > 작업 실행... > '1. 환경 설치 (GPU 자동 감지)' 를 다시 실행하고",
            "     마지막에 '설치 완료!' 가 나오는지, 아니면 [오류] 가 나오는지 확인하세요.",
        ]
    elif venv_python().exists():
        lines += [
            f"  -> 설치된 가상환경({venv})이 아닌 다른 파이썬으로 실행했습니다.",
            "     VS Code: Ctrl+Shift+P > 'Python: Select Interpreter' > '.venv' 를 고르거나,",
            "     실행 및 디버그(Ctrl+Shift+D)의 구성으로 실행하세요 (항상 .venv 를 씁니다).",
            f"     터미널이라면: {venv_python()} {Path(sys.argv[0]).name} ...",
        ]
    else:
        lines += [
            f"  -> 아직 환경 설치를 하지 않았습니다 ({venv} 가 없음).",
            "     VS Code: 터미널 > 작업 실행... > '1. 환경 설치 (GPU 자동 감지)' 를 먼저 실행하세요.",
        ]
    if Path.cwd().resolve() != PROJECT_ROOT:
        lines += [
            f"  * VS Code 에서 연 폴더({Path.cwd()})가 프로젝트 폴더가 아닌 것 같습니다.",
            f"    파일 > 폴더 열기로 train.py 가 들어 있는 {PROJECT_ROOT} 를 여세요",
            "    (압축을 풀면 폴더가 두 겹일 수 있습니다). 그래야 작업/실행 구성이 보입니다.",
        ]
    sys.exit("\n".join(lines))
