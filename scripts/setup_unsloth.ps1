[CmdletBinding()]
param(
    [switch]$SkipDevDependencies
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv-unsloth"

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvExe = $uvCommand.Source
} else {
    # O winget instala uv antes de atualizar a variável PATH da sessão atual.
    $wingetUv = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
    if (Test-Path -LiteralPath $wingetUv) {
        $uvExe = $wingetUv
    } else {
        throw "uv não encontrado. Instale com: winget install --id astral-sh.uv -e"
    }
}

Write-Host "Criando ambiente Python 3.13 em $venvPath"
& $uvExe venv $venvPath --python 3.13
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # A resolução automática do uv pode escolher o wheel CPU no Windows. A RTX 50
    # requer um wheel CUDA moderno; o índice oficial de testes 2.11 fornece cu130.
    Write-Host "Instalando PyTorch CUDA 13.0 para NVIDIA (Windows/RTX 50)"
    & $uvExe pip install --python $pythonExe --reinstall `
        "torch==2.11.0" torchvision torchaudio `
        --index-url https://download.pytorch.org/whl/test/cu130
} else {
    Write-Warning "nvidia-smi não encontrado; instalando backend automático. Treino acelerado por CUDA ficará indisponível."
    & $uvExe pip install --python $pythonExe --upgrade torch --torch-backend=auto
}

Write-Host "Instalando Unsloth"
& $uvExe pip install --python $pythonExe --upgrade unsloth

$extras = ".[train,rag,vision]"
if (-not $SkipDevDependencies) {
    $extras = ".[train,rag,vision,dev]"
}
Push-Location $projectRoot
try {
    & $uvExe pip install --python $pythonExe -e $extras
    & $pythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA não foi detectada; não inicie treino.'; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0), 'bf16': torch.cuda.is_bf16_supported()})"
}
finally {
    Pop-Location
}

Write-Host "Ambiente pronto. Ative com: $venvPath\Scripts\Activate.ps1"
