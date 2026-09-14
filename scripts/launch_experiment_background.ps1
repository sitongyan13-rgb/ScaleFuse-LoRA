param(
    [Parameter(Mandatory = $true)]
    [string]$Spec,
    [Parameter(Mandatory = $true)]
    [string]$WorkDir,
    [Parameter(Mandatory = $true)]
    [string]$LauncherLog,
    [string]$Resume = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$env:PYTHONPATH = $projectRoot
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

$logPath = Join-Path $projectRoot $LauncherLog
$logParent = Split-Path -Parent $logPath
New-Item -ItemType Directory -Path $logParent -Force | Out-Null

$launcherArgs = @(
    'scripts/run_experiment.py',
    $Spec,
    '--execute',
    '--work-dir',
    $WorkDir
)
if ($Resume) {
    $launcherArgs += @('--resume', $Resume)
}
python @launcherArgs *>> $logPath
$exitCode = $LASTEXITCODE
"background_launcher_exit=$exitCode finished_local=$((Get-Date).ToString('o'))" | `
    Add-Content -LiteralPath $logPath -Encoding UTF8
exit $exitCode
