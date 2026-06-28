[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$OUT = "$LOG\results"
$MM_CACHE = "C:\Temp\geniex-cache"
$TC = "C:\Temp\TestContent"

New-Item -ItemType Directory -Force -Path $LOG, $OUT, $MM_CACHE | Out-Null
Start-Transcript -Path "$LOG\script.log" -Force | Out-Null

# geniex-bench.exe writes informational lines to stderr even on success.
# Called via the bare `&` operator, every such line becomes a NativeCommandError
# ErrorRecord that QDC's parser flags as Unsuccessful — same trap that bit
# genie's run_windows.ps1. Start-Process redirects at the OS-process level so
# stderr bypasses PowerShell's error stream entirely. One retry mirrors
# Invoke-GenieRetry; throw on double-failure so the script aborts.
function Invoke-GenieXBenchRetry {
    param([Parameter(Mandatory = $true)][string[]]$BenchArgs)
    foreach ($attempt in 1, 2) {
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process -FilePath "$BUNDLE\bin\geniex-bench.exe" `
                -ArgumentList $BenchArgs `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
            $exitCode = $proc.ExitCode
            $stdout = Get-Content $stdoutFile -Raw -Encoding UTF8
            $stderr = Get-Content $stderrFile -Raw -Encoding UTF8
        } finally {
            Remove-Item $stdoutFile -ErrorAction SilentlyContinue
            Remove-Item $stderrFile -ErrorAction SilentlyContinue
        }
        $captured = @($stdout) + @($stderr)
        $captured | Out-String | Write-Host
        if ($exitCode -eq 0) { return }
        Write-Host "Invoke-GenieXBenchRetry: geniex-bench.exe failed (exit $exitCode)"
    }
    throw "geniex-bench.exe failed twice: $($BenchArgs -join ' ')"
}

$ZIP = "$TC\geniex-bench.zip"
$URL = "{WINDOWS_BENCH_URL}"
& curl.exe -fSL --retry 3 --retry-delay 5 -o $ZIP $URL
if ($LASTEXITCODE -ne 0) { throw "geniex-bench download failed: $LASTEXITCODE" }
Expand-Archive -Path $ZIP -DestinationPath $TC -Force
Remove-Item $ZIP

$BUNDLE = (Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' | Select-Object -First 1).FullName
if (-not $BUNDLE) { throw "extracted bundle dir missing under $TC" }
if (-not (Test-Path "$BUNDLE\bin\geniex-bench.exe")) {
    throw "geniex-bench.exe not found at $BUNDLE\bin"
}

Set-Location $BUNDLE
$env:GENIEX_PLUGIN_PATH = "$BUNDLE\lib"
$env:PATH = "$BUNDLE\lib;$BUNDLE\lib\llama_cpp;$BUNDLE\lib\qairt;$BUNDLE\lib\qairt\htp-files;$env:PATH"

$rows = @'
{MODELS}
'@ -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }

$IMG = "$TC/test.png" -replace '\\', '/'

$ctxList = @({CTX_LIST})
$tsvByCtx = @{}
foreach ($ctx in $ctxList) {
    $tsvByCtx[$ctx] = "C:\Temp\matrix-$ctx.tsv"
    Remove-Item $tsvByCtx[$ctx] -ErrorAction SilentlyContinue
}

foreach ($row in $rows) {
    $name, $plugin, $devs, $model_id, $vlm, $image = $row -split '\|'
    Write-Output "=== plan $name id=$model_id ==="
    $imgpath = if ($image -eq "1") { $IMG } else { "" }
    foreach ($d in $devs -split ',') {
        foreach ($ctx in $ctxList) {
            "{0}-{1}-{2}-c{3}`t{1}`t{2}`t{4}`t`t`t{5}`t{6}" -f `
                $name, $plugin, $d, $ctx, $model_id, $imgpath, $vlm `
                | Add-Content $tsvByCtx[$ctx]
        }
    }
}

foreach ($ctx in $ctxList) {
    $tsv = $tsvByCtx[$ctx]
    Write-Output "=== matrix ctx=$ctx ==="
    if (Test-Path $tsv) { Get-Content $tsv }
    Invoke-GenieXBenchRetry -BenchArgs @(
        "--matrix-file", $tsv, "--output-json-dir", $OUT, "-r", "3",
        {BENCH_SIZE_FLAGS_ARGS}
        "--mm-data-dir", $MM_CACHE, "--chipset", "{CHIPSET}"
    )
    Write-Output "$((Get-ChildItem $OUT).Count) cell json files so far"
}

Write-Output "=== done ==="
Stop-Transcript | Out-Null
exit 0
