param([string]$PythonExecutable = "", [switch]$NoPlots)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $PythonExecutable) {
    $venvPython = Join-Path $projectRoot "work\.venv\Scripts\python.exe"
    $PythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}
# check.ps1 prepares project-local scratch/cache paths, runs all tests and smoke.
& (Join-Path $PSScriptRoot "check.ps1") -PythonExecutable $PythonExecutable
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

$part1Arguments = @("-m", "coordinate_geometry.part1_experiments", "--config", "configs/part1.json", "--output", "outputs/part1")
if ($NoPlots) { $part1Arguments += "--no-plots" }
& $PythonExecutable @part1Arguments
if ($LASTEXITCODE -ne 0) { throw "Module 1 experiment failed." }
& $PythonExecutable -m coordinate_geometry.verify_part1 --output outputs/part1
if ($LASTEXITCODE -ne 0) { throw "Module 1 independent artifact verification failed." }

$part2Arguments = @("-m", "coordinate_geometry.adam_diagnostics", "--config", "configs/adam_diagnostics.json", "--output", "outputs/adam_diagnostics")
if (-not $NoPlots) { $part2Arguments += "--plot" }
& $PythonExecutable @part2Arguments
if ($LASTEXITCODE -ne 0) { throw "Module 2 diagnostic failed." }

$part5Arguments = @("-m", "coordinate_geometry.part5_experiments", "--config", "configs/part5.json", "--output", "outputs/part5")
if ($NoPlots) { $part5Arguments += "--no-plots" }
& $PythonExecutable @part5Arguments
if ($LASTEXITCODE -ne 0) { throw "Module 5 experiment failed." }
& $PythonExecutable -m coordinate_geometry.verify_part5 --output outputs/part5
if ($LASTEXITCODE -ne 0) { throw "Module 5 independent artifact verification failed." }
# Module 3: seconds-long pipeline smoke only. The main study is run separately with scripts/part3.ps1.
& $PythonExecutable -m coordinate_geometry.part3_experiments --config configs/part3_smoke.json --output outputs/part3_smoke --stage all
if ($LASTEXITCODE -ne 0) { throw "Module 3 pipeline smoke failed." }
& $PythonExecutable -m coordinate_geometry.verify_part3 --output outputs/part3_smoke
if ($LASTEXITCODE -ne 0) { throw "Module 3 independent artifact verification failed." }
Write-Host "All integrated module checks passed. Artifacts are in outputs/."
