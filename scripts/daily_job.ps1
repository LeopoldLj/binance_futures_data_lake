param(
    [switch]$FullVerify = $false
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [scriptblock]$Command
    )
    Write-Host "`n$Title"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code $LASTEXITCODE : $Title"
    }
}

function Read-SymbolsFromYaml {
    param(
        [string]$YamlPath
    )
    if (!(Test-Path $YamlPath)) {
        throw "symbols.yml introuvable: $YamlPath"
    }

    $lines = Get-Content $YamlPath
    $symbols = @()

    foreach ($line in $lines) {
        $t = $line.Trim()

        # Ignore commentaires / lignes vides
        if ($t -eq "" -or $t.StartsWith("#")) { continue }

        # Format attendu:
        # symbols:
        #   - BTCUSDT
        #   - ETHUSDT
        if ($t.StartsWith("-")) {
            $sym = $t.TrimStart("-").Trim()
            if ($sym -ne "") { $symbols += $sym.ToUpper() }
        }
    }

    if ($symbols.Count -eq 0) {
        throw "Aucun symbole trouvé dans $YamlPath (attendu: liste YAML sous 'symbols:')."
    }

    return $symbols
}

$projectRoot = (Get-Location).Path
$env:PYTHONPATH = "$projectRoot\src"

# Logs
$logsDir = Join-Path $projectRoot "logs"
if (!(Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logsDir ("daily_job_{0}.txt" -f $stamp)

Start-Transcript -Path $logFile -Append | Out-Null

Write-Host "=== DAILY JOB START ==="
Write-Host "ProjectRoot: $projectRoot"
Write-Host "PYTHONPATH:  $env:PYTHONPATH"
Write-Host "LogFile:     $logFile"

try {
    $symbolsFile = Join-Path $projectRoot "config\symbols.yml"
    $symbols = Read-SymbolsFromYaml -YamlPath $symbolsFile

    Write-Host "`nSymbols loaded from config/symbols.yml:"
    foreach ($s in $symbols) { Write-Host " - $s" }

    # 1) Collect (multi-symbol déjà géré par bfdl.cli.collect)
    Run-Step "[1/4] Collect incremental (all symbols)..." { python -m bfdl.cli.collect }

    # 2) Compact par symbole
    Write-Host "`n[2/4] Compact staging (per symbol)..."
    foreach ($s in $symbols) {
        Run-Step "  - Compact $s" { python -m bfdl.transforms.compact_staging --symbol $s }
    }

    # 3) Quick verify + last minute par symbole
    Write-Host "`n[3/4] Quick verify (gaps_report) + Last minute (per symbol)..."
    foreach ($s in $symbols) {
        Write-Host "`n  - Gaps report $s"
        python -m bfdl.transforms.gaps_report --symbol $s
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[KO] gaps_report returned $LASTEXITCODE for $s"
            Stop-Transcript | Out-Null
            exit 1
        }

        Run-Step "  - Last minute $s" { python -m bfdl.transforms.last_minute --symbol $s }
    }

    # 4) Full verify (optionnel) par symbole
    if ($FullVerify) {
        Write-Host "`n[4/4] Full verify (verify_all) per symbol..."
        foreach ($s in $symbols) {
            Write-Host "`n  - Full verify $s"
            python -m bfdl.transforms.verify_all --symbol $s
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[KO] verify_all returned $LASTEXITCODE for $s"
                Stop-Transcript | Out-Null
                exit 1
            }
        }
    }
    else {
        Write-Host "`n[4/4] Full verify skipped (use -FullVerify to run it)"
    }

    Write-Host "`n=== DAILY JOB DONE ==="
    Stop-Transcript | Out-Null
    exit 0
}
catch {
    Write-Host "`n[KO] Exception: $($_.Exception.Message)"
    Write-Host "=== DAILY JOB FAILED ==="
    Stop-Transcript | Out-Null
    exit 2
}
