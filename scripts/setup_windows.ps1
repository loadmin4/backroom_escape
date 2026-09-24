<#
  백룸 탈출 - 윈도우 개발 환경 설치
  - .venv 가상환경을 만들고
  - NVIDIA 드라이버 버전을 보고 알맞은 GPU(CUDA) 버전 PyTorch 를 설치한 뒤
  - 나머지 패키지를 설치하고 GPU 를 쓸 수 있는지 확인한다.

  사용법 (프로젝트 폴더에서, 또는 VS Code: 터미널 > 작업 실행 > "1. 환경 설치"):
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1              # GPU 자동 감지
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cpu         # CPU 전용
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cuda cu126  # CUDA 빌드 직접 지정
#>
param(
    [switch]$Cpu,
    [string]$Cuda = "auto"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`n[오류] $msg" -ForegroundColor Red; exit 1 }
function Check($msg) { if ($LASTEXITCODE -ne 0) { Fail $msg } }

# ---------------------------------------------------------------- 1. 파이썬 찾기
Step "파이썬 찾기"
$usePy = $null -ne (Get-Command py -ErrorAction SilentlyContinue)
if (-not $usePy) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    # WindowsApps 의 python.exe 는 Microsoft Store 를 여는 가짜 실행 파일이다
    if ($null -eq $python -or $python.Source -like "*WindowsApps*") {
        Fail "파이썬이 없습니다. https://www.python.org/downloads/windows/ 에서 3.11 또는 3.12 를 설치하세요 (설치 화면에서 'Add python.exe to PATH' 체크)."
    }
}
function SysPython { if ($usePy) { & py -3 @args } else { & python @args } }

$version = SysPython -c "import sys; print('%d.%d' % sys.version_info[:2])"
Check "파이썬을 실행하지 못했습니다."
Write-Host "Python $version"
$major, $minor = $version.Split(".") | ForEach-Object { [int]$_ }
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) { Fail "Python 3.10 이상이 필요합니다 (지금: $version)." }

# ---------------------------------------------------------------- 2. 가상환경
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Step "기존 가상환경 사용 (.venv)"
} else {
    Step "가상환경 만들기 (.venv)"
    SysPython -m venv .venv
    Check "가상환경을 만들지 못했습니다."
}
& $venvPython -m pip install --upgrade pip --quiet
Check "pip 업그레이드에 실패했습니다."

# ---------------------------------------------------------------- 3. PyTorch 빌드 고르기
Step "PyTorch 버전 고르기"
if ($Cpu) {
    $Cuda = "cpu"
} elseif ($Cuda -eq "auto") {
    if ($null -eq (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Host "NVIDIA 드라이버(nvidia-smi)를 찾지 못했습니다. CPU 버전을 설치합니다." -ForegroundColor Yellow
        Write-Host "NVIDIA GPU 가 있다면 https://www.nvidia.com/drivers 에서 드라이버를 설치하고 다시 실행하세요." -ForegroundColor Yellow
        $Cuda = "cpu"
    } else {
        $info = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1).Split(",")
        $gpuName = $info[0].Trim()
        $driver = $info[1].Trim()
        $driverMajor = [int]$driver.Split(".")[0]
        Write-Host "GPU: $gpuName / 드라이버: $driver"
        # CUDA 13 빌드는 드라이버 580 이상, 최신 GPU(RTX 50 시리즈 등)까지 지원한다.
        # 드라이버가 그보다 오래됐으면 CUDA 12.6 빌드를 쓴다.
        if ($driverMajor -ge 580) {
            $Cuda = "cu130"
        } else {
            $Cuda = "cu126"
            if ($gpuName -match "RTX 50") {
                Write-Host "RTX 50 시리즈는 드라이버 580 이상이 필요합니다. 드라이버를 업데이트한 뒤 다시 실행하세요." -ForegroundColor Yellow
            }
        }
    }
}
Write-Host "설치할 PyTorch 빌드: $Cuda"

# ---------------------------------------------------------------- 4. PyTorch 설치
# torch 를 import 하지 않고 설치된 버전만 읽는다 (없으면 none). stderr 를 쓰지 않으므로
# Windows PowerShell 5.1 에서 ErrorActionPreference=Stop 과 부딪히지 않는다.
$installed = & $venvPython -c "import importlib.metadata as m, importlib.util as u; print(m.version('torch') if u.find_spec('torch') else 'none')"
Check "설치된 PyTorch 버전을 확인하지 못했습니다."
if ($installed -like "*+$Cuda") {
    Step "PyTorch $installed 이 이미 설치돼 있습니다"
} else {
    if ($installed -ne "none") {
        Step "기존 PyTorch ($installed) 제거"
        & $venvPython -m pip uninstall -y torch
        Check "기존 PyTorch 를 제거하지 못했습니다."
    }
    Step "PyTorch ($Cuda) 설치 - GPU 버전은 2~3GB 라 몇 분 걸립니다"
    & $venvPython -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
    Check "PyTorch 설치에 실패했습니다. 인터넷 연결과 디스크 공간을 확인하세요."
}

# ---------------------------------------------------------------- 5. 나머지 패키지
Step "나머지 패키지 설치 (numpy, tensorboard, pytest)"
& $venvPython -m pip install -r requirements.txt
Check "패키지 설치에 실패했습니다."

# ---------------------------------------------------------------- 6. 확인
Step "설치 확인"
& $venvPython check_gpu.py
Check "설치 확인 중 오류가 났습니다."

Write-Host "`n설치 완료! VS Code 에서 Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv 를 고른 뒤" -ForegroundColor Green
Write-Host "실행 및 디버그(Ctrl+Shift+D)에서 '학습: sound' 를 골라 Ctrl+F5 로 실행하세요." -ForegroundColor Green
