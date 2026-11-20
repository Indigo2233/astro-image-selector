param(
    [string]$Executable = "dist\AstroImageSelector.exe",
    [string]$Frames = "bench\data\frames",
    [string]$Scan = "bench\data\scan",
    [int]$Repeats = 5
)

$ErrorActionPreference = "Stop"
$exe = (Resolve-Path -LiteralPath $Executable).Path
$frames = (Resolve-Path -LiteralPath $Frames).Path
$scan = (Resolve-Path -LiteralPath $Scan).Path

function Measure-Process {
    param(
        [string[]]$Arguments,
        [int]$Runs
    )
    $values = foreach ($run in 1..$Runs) {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $exe -ArgumentList $Arguments -Wait -PassThru -RedirectStandardOutput ([IO.Path]::GetTempFileName()) -RedirectStandardError ([IO.Path]::GetTempFileName())
        $timer.Stop()
        if ($process.ExitCode -ne 0) {
            throw "Process failed with exit code $($process.ExitCode)"
        }
        $timer.Elapsed.TotalMilliseconds
    }
    $ordered = @($values | Sort-Object)
    [pscustomobject]@{
        median_ms = [Math]::Round($ordered[[Math]::Floor($ordered.Count / 2)], 3)
        min_ms = [Math]::Round($ordered[0], 3)
        max_ms = [Math]::Round($ordered[-1], 3)
    }
}

$cacheFile = Join-Path $frames ".astro_selector_analysis_cache.json"
if (Test-Path -LiteralPath $cacheFile) {
    Remove-Item -LiteralPath $cacheFile -Force
}

$cold = Measure-Process -Arguments @("__replay-analysis", "`"$frames`"") -Runs 1
$warm = Measure-Process -Arguments @("__replay-analysis", "`"$frames`"") -Runs $Repeats
$gpu = Measure-Process -Arguments @("__replay-analysis", "`"$frames`"", "--gpu") -Runs $Repeats

[pscustomobject]@{
    cold_frames_ms = $cold.median_ms
    warm_frames_ms = $warm.median_ms
    gpu_frames_ms = $gpu.median_ms
    frame_count = (Get-ChildItem -LiteralPath $frames -File -Filter "*.fits").Count
} | ConvertTo-Json
