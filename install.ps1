<#
.SYNOPSIS
    Set up local-upscale: restore and upscale photos locally (Qwen-Image clean-up through the
    local-image tool, then SeedVR2-7B upscaling), on this machine's NVIDIA GPU.

.DESCRIPTION
    Clones the SeedVR2 standalone CLI at a pinned commit into engine\SeedVR2, builds .venv\ with a
    CUDA build of PyTorch, and fetches the SeedVR2-7B weights and the face detector with checksum
    verification into models\. Nothing here uploads anything; your photos never leave the machine.

    local-image (the Qwen-Image edit tool) is a separate install: see README.md. This script only
    checks that it can be found.

    Re-running is safe: every finished step is skipped and downloads resume.
    Works on Windows PowerShell 5.1 and PowerShell 7+.

.PARAMETER PythonVersion
    Python version for the venv, passed to the 'py' launcher (3.12 or 3.13).

.PARAMETER CudaIndex
    PyTorch wheel index. cu130 is what this repo was tested with; cu128 also supports RTX 50-series.

.PARAMETER ReuseModels
    One or more folders that already hold some of the weights (another SeedVR2 install, for example).
    Matching files are hard-linked (or copied, across drives) instead of downloaded, then verified.

.PARAMETER SkipModels
    Build the environment only; fetch the weights later with scripts\download_models.py.

.PARAMETER NoSkillLink
    Don't link this folder into ~\.claude\skills\local-upscale (the Claude Code skill).

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -ReuseModels D:\old\seedvr2\models
#>
[CmdletBinding()]
param(
    [string]   $PythonVersion = '3.13',
    [string]   $CudaIndex     = 'cu130',
    [string[]] $ReuseModels   = @(),
    [switch]   $SkipModels,
    [switch]   $NoSkillLink
)

$ErrorActionPreference = 'Stop'

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }

$TORCH_VERSION       = '2.14.1'
$TORCHVISION_VERSION = '0.29.1'

function Step ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "    $m"   -ForegroundColor Green }
function Note ($m) { Write-Host "    $m"   -ForegroundColor DarkGray }
function Warn ($m) { Write-Host "    $m"   -ForegroundColor Yellow }
function Fail ($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

function Probe {
    param([Parameter(Mandatory)][string] $Exe, [string[]] $Arguments = @())
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = & $Exe @Arguments 2>&1 | Out-String; $code = $LASTEXITCODE }
    catch { $out = $_.ToString(); $code = 1 }
    finally { $ErrorActionPreference = $old }
    [pscustomobject]@{ Code = $code; Out = $out.Trim() }
}

$manifest = Get-Content (Join-Path $RepoRoot 'models.json') -Raw | ConvertFrom-Json

# --- prerequisites --------------------------------------------------------
Step 'Checking prerequisites'
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'git is not on PATH. Install Git for Windows.' }
Ok "git $((git --version) -replace 'git version ')"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail "The 'py' launcher is not on PATH. Install Python from python.org with the launcher option ticked."
}
$probe = Probe 'py' @("-$PythonVersion", '-c', 'import sys; sys.stdout.write(sys.executable)')
if ($probe.Code -ne 0) { Fail "Python $PythonVersion not found. Installed:`n$((Probe 'py' @('--list')).Out)" }
Ok "Python $PythonVersion"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Ok "GPU: $((Probe 'nvidia-smi' @('--query-gpu=name,memory.total', '--format=csv,noheader')).Out)"
} else {
    Warn 'nvidia-smi not found. SeedVR2 and local-image need an NVIDIA GPU (16 GB recommended).'
}
$drive = Get-PSDrive -Name ($RepoRoot.Substring(0, 1)) -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Ok "$freeGB GB free"
    if ($freeGB -lt 20) { Warn 'Under 20 GB free. Weights are ~9 GB, the venv ~6 GB.' }
}
$li = if ($env:LOCAL_IMAGE_DIR) { $env:LOCAL_IMAGE_DIR } else { Join-Path $HOME '.claude\skills\local-image' }
if (Test-Path (Join-Path $li 'scripts\generate.py')) { Ok "local-image found at $li" }
else { Warn "local-image not found at $li. Install it (see README.md) or set LOCAL_IMAGE_DIR." }

# --- engine code ----------------------------------------------------------
Step "SeedVR2 standalone CLI ($($manifest.engine.dest))"
$dest = Join-Path $RepoRoot $manifest.engine.dest
$commit = $manifest.engine.commit
if (Test-Path (Join-Path $dest '.git')) {
    $head = (Probe 'git' @('-C', $dest, 'rev-parse', 'HEAD')).Out
    if ($head -eq $commit) { Ok "Already at $($commit.Substring(0, 10))" }
    else {
        git -C $dest fetch --quiet origin
        git -C $dest checkout --quiet $commit
        if ($LASTEXITCODE -ne 0) { Fail "could not check out $commit in $dest" }
        Ok "Checked out $($commit.Substring(0, 10))"
    }
} else {
    if (Test-Path $dest) { Fail "$dest exists but is not a git checkout. Move it aside." }
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    git clone --quiet $manifest.engine.repo $dest
    if ($LASTEXITCODE -ne 0) { Fail 'git clone failed' }
    git -C $dest checkout --quiet $commit
    if ($LASTEXITCODE -ne 0) { Fail "could not check out $commit" }
    Ok 'Cloned'
}

# --- venv -------------------------------------------------------------------
Step 'Python environment (.venv)'
$venv = Join-Path $RepoRoot '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    & py "-$PythonVersion" -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail 'venv creation failed' }
    Ok 'Created'
}
$env:PIP_CACHE_DIR = Join-Path $RepoRoot '.cache\pip'
& $py -m pip install --quiet --upgrade pip
$have = Probe $py @('-c', 'import torch;print(torch.__version__)')
if ($have.Code -eq 0 -and $have.Out -like "$TORCH_VERSION*") { Ok "torch $($have.Out) already installed" }
else {
    Note "Installing torch $TORCH_VERSION from the $CudaIndex index (~3 GB download unless cached)..."
    & $py -m pip install --quiet --index-url "https://download.pytorch.org/whl/$CudaIndex" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    if ($LASTEXITCODE -ne 0) { Fail 'torch install failed' }
}
& $py -m pip install --quiet -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Fail 'requirements install failed' }
$cuda = Probe $py @('-c', 'import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 0)')
if ($cuda.Out -like 'True*') { Ok "CUDA OK: $($cuda.Out -replace '^True ')" } else { Warn "torch cannot see a CUDA GPU ($($cuda.Out))" }

# --- weights ----------------------------------------------------------------
if ($SkipModels) { Step 'Weights: skipped (-SkipModels)' }
else {
    Step 'Weights (models\): SeedVR2-7B mixed FP8 + VAE (~9 GB), face detector'
    $dl = @((Join-Path $RepoRoot 'scripts\download_models.py'))
    if ($ReuseModels.Count) { $dl += '--reuse'; $dl += $ReuseModels }
    & $py @dl
    if ($LASTEXITCODE -ne 0) { Fail 'weight download failed (re-run to resume)' }
}

# --- skill link -------------------------------------------------------------
if (-not $NoSkillLink) {
    Step 'Claude Code skill (~\.claude\skills\local-upscale)'
    $skills = Join-Path $HOME '.claude\skills'
    $link = Join-Path $skills 'local-upscale'
    New-Item -ItemType Directory -Path $skills -Force | Out-Null
    if (Test-Path $link) {
        $item = Get-Item $link -Force
        if ($item.LinkType -and ($item.Target -contains $RepoRoot -or "$($item.Target)" -eq $RepoRoot)) { Ok 'Already linked' }
        else { Warn "$link already exists and points elsewhere; left alone." }
    } else {
        New-Item -ItemType Junction -Path $link -Target $RepoRoot | Out-Null
        Ok "Linked $link -> $RepoRoot"
    }
}

New-Item -ItemType Directory -Path (Join-Path $RepoRoot 'input'), (Join-Path $RepoRoot 'output') -Force | Out-Null
Step 'Done'
& $py (Join-Path $RepoRoot 'scripts\upscale.py') status
Write-Host "`n  Put photos in input\ and run:  .venv\Scripts\python.exe scripts\upscale.py variants input\<photo>`n"
