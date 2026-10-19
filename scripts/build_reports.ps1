$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$tempPath = Join-Path $projectRoot "work\temp"
$logPath = Join-Path $projectRoot "work\report-build"
New-Item -ItemType Directory -Force -Path $tempPath, $logPath, "outputs/reports" | Out-Null
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:TMPDIR = $tempPath
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    throw "Install a TeX distribution providing pdflatex, such as MiKTeX or TeX Live."
}
foreach ($partNumber in @(1, 2, 3, 4, 5)) {
    $sourcePath = "part${partNumber}_report.tex"
    $outputPath = Join-Path $projectRoot "outputs/reports"
    Push-Location -LiteralPath (Join-Path $projectRoot "docs")
    try {
    foreach ($passNumber in @(1, 2)) {
        $passLogPath = Join-Path $logPath "part${partNumber}_pass${passNumber}.txt"
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "-output-directory=$outputPath" $sourcePath *> $passLogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Module $partNumber report compilation failed; see $passLogPath"
        }
    }
    } finally { Pop-Location }
    $buildLogPath = "outputs/reports/part${partNumber}_report.log"
    $unresolved = Select-String -Path $buildLogPath -Pattern 'undefined|Rerun to get|Overfull \\[hv]box|LaTeX Error'
    if ($unresolved) {
        $unresolved | ForEach-Object { Write-Host $_.Line }
        throw "Module $partNumber report has unresolved references or overflowing content."
    }
    Write-Host "Verified outputs/reports/part${partNumber}_report.pdf"
}
