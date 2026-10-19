param(
    [string]$PythonExecutable = "",
    [ValidateSet("smoke", "pilot", "tune", "evaluate", "report", "all")][string]$Stage = "smoke",
    [int]$Workers = 0,
    [string]$Config = "configs/part3.json",
    [string]$Output = "outputs/part3",
    [switch]$NoPlots,
    [switch]$SkipTests
)
# smoke: tests + seconds-long end-to-end check (configs/part3_smoke.json -> outputs/part3_smoke)
# pilot/tune/evaluate/report/all: main protocol stages; every stage resumes from saved raw runs.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$tempPath = Join-Path $projectRoot "work\temp"
$cachePath = Join-Path $projectRoot "work\cache"
New-Item -ItemType Directory -Force -Path $tempPath, $cachePath | Out-Null
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:TMPDIR = $tempPath
$env:PIP_CACHE_DIR = Join-Path $cachePath "pip"
$env:PYTHONPYCACHEPREFIX = Join-Path $cachePath "pycache"
$env:MPLCONFIGDIR = Join-Path $cachePath "matplotlib"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONPATH = Join-Path $projectRoot "src"
if (-not $PythonExecutable) {
    $venvPython = Join-Path $projectRoot "work\.venv\Scripts\python.exe"
    $PythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}
if ($Workers -le 0) { $Workers = [Math]::Max(1, [Environment]::ProcessorCount - 1) }
if (-not $SkipTests) {
    & $PythonExecutable -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Verification tests failed." }
}
if ($Stage -eq "smoke") {
    $Config = "configs/part3_smoke.json"
    $Output = "outputs/part3_smoke"
    $Stage = "all"
}
$arguments = @("-m", "coordinate_geometry.part3_experiments", "--config", $Config, "--output", $Output,
               "--stage", $Stage, "--workers", "$Workers")
if (-not $NoPlots) { $arguments += "--plot" }
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "Module 3 stage '$Stage' failed." }
if ($Stage -in @("all", "evaluate", "report")) {
    & $PythonExecutable -m coordinate_geometry.verify_part3 --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Independent module 3 verification failed." }
}
