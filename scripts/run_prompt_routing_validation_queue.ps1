param(
    [Parameter(Mandatory = $true)]
    [string]$PrerequisiteManifest
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$env:PYTHONPATH = $projectRoot
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$python = 'python'

function Resolve-ProjectPath([string]$Path) {
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Wait-For-Prerequisite([string]$ManifestPath) {
    $resolved = Resolve-ProjectPath $ManifestPath
    while ($true) {
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "Missing prerequisite manifest: $resolved"
        }
        $payload = Get-Content -LiteralPath $resolved -Raw | ConvertFrom-Json
        if ($payload.status -eq 'completed' -and $payload.returncode -eq 0) {
            return
        }
        if ($payload.status -eq 'failed') {
            throw "Prerequisite failed: $resolved (returncode=$($payload.returncode))"
        }
        Start-Sleep -Seconds 15
    }
}

function Invoke-RegisteredRun(
    [string]$Name,
    [string]$Spec,
    [string]$WorkDir,
    [string]$Resume = ''
) {
    $arguments = @(
        'scripts/run_experiment.py',
        $Spec,
        '--execute',
        '--work-dir',
        $WorkDir
    )
    if ($Resume) {
        $resumePath = Resolve-ProjectPath $Resume
        if (-not (Test-Path -LiteralPath $resumePath -PathType Leaf)) {
            throw "Missing resume checkpoint for ${Name}: $resumePath"
        }
        $arguments += @('--resume', $resumePath)
    }
    Write-Output "queue_start=$Name local=$((Get-Date).ToString('o'))"
    & $python @arguments
    $code = $LASTEXITCODE
    Write-Output "queue_finish=$Name returncode=$code local=$((Get-Date).ToString('o'))"
    if ($code -ne 0) {
        throw "Registered run failed: $Name (returncode=$code)"
    }
}

Wait-For-Prerequisite $PrerequisiteManifest

$runs = @(
    @{
        Name = 'uniform_seed0_epoch13'
        Spec = 'configs/ablations/generated_specs/pothole_uniform_prompt_lora_r16_seed0_extend13.yaml'
        WorkDir = 'experiments/odinw_pothole_uniform_prompt_lora_r16_seed0_extend13'
        Resume = 'experiments/odinw_pothole_uniform_prompt_lora_r16_seed0/epoch_12.pth'
    },
    @{
        Name = 'lora_seed0_epoch13'
        Spec = 'configs/ablations/generated_specs/pothole_lora_r16_seed0_extend13.yaml'
        WorkDir = 'experiments/odinw_pothole_lora_r16_seed0_extend13'
        Resume = 'experiments/odinw_pothole_fusion_lora_r16_seed0/epoch_12.pth'
    },
    @{
        Name = 'lora_seed21_epoch13'
        Spec = 'configs/ablations/generated_specs/pothole_lora_r16_seed21_extend13.yaml'
        WorkDir = 'experiments/odinw_pothole_lora_r16_seed21_extend13'
        Resume = 'experiments/odinw_pothole_fusion_lora_r16_seed21/epoch_12.pth'
    },
    @{
        Name = 'learned_seed0_epoch13'
        Spec = 'configs/ablations/generated_specs/pothole_learned_prompt_lora_r16_seed0_extend13.yaml'
        WorkDir = 'experiments/odinw_pothole_learned_prompt_lora_r16_seed0_extend13'
        Resume = 'experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed0/epoch_12.pth'
    },
    @{
        Name = 'learned_seed21_epoch13'
        Spec = 'configs/ablations/generated_specs/pothole_learned_prompt_lora_r16_seed21_extend13.yaml'
        WorkDir = 'experiments/odinw_pothole_learned_prompt_lora_r16_seed21_extend13'
        Resume = 'experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed21/epoch_12.pth'
    },
    @{
        Name = 'uniform_seed21_base12'
        Spec = 'configs/ablations/generated_specs/pothole_uniform_prompt_lora_r16_seed21.yaml'
        WorkDir = 'experiments/odinw_pothole_uniform_prompt_lora_r16_seed21'
        Resume = ''
    },
    @{
        Name = 'uniform_seed21_epoch13'
        Spec = 'configs/ablations/generated_specs/pothole_uniform_prompt_lora_r16_seed21_extend13.yaml'
        WorkDir = 'experiments/odinw_pothole_uniform_prompt_lora_r16_seed21_extend13'
        Resume = 'experiments/odinw_pothole_uniform_prompt_lora_r16_seed21/epoch_12.pth'
    }
)

foreach ($run in $runs) {
    Invoke-RegisteredRun `
        -Name $run.Name `
        -Spec $run.Spec `
        -WorkDir $run.WorkDir `
        -Resume $run.Resume
}

Write-Output "queue_status=completed local=$((Get-Date).ToString('o'))"
