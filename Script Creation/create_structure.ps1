param(
  [string]$ProjectPath = "C:\Users\lolo_\PycharmProjects\binance_futures_data_lake"
)

function New-Dir($p) {
  if (-not (Test-Path $p)) {
    New-Item -ItemType Directory -Path $p | Out-Null
  }
}

function Touch-File($p) {
  if (-not (Test-Path $p)) {
    New-Item -ItemType File -Path $p | Out-Null
  }
}

$dirs = @(
  "config",
  "src\bfdl",
  "src\bfdl\core",
  "src\bfdl\binance",
  "src\bfdl\collectors",
  "src\bfdl\transforms",
  "src\bfdl\cli",
  "data\raw\binance_um\klines_m1",
  "data\raw\binance_um\premium_index_klines_m1",
  "data\raw\binance_um\funding_rate_events",
  "data\raw\binance_um\open_interest_snapshots",
  "data\curated\um_m5",
  "data\curated\um_h1",
  "data\curated\um_h4",
  "logs"
)

foreach ($d in $dirs) {
  New-Dir (Join-Path $ProjectPath $d)
}

$files = @(
  "src\bfdl\__init__.py",
  "src\bfdl\core\__init__.py",
  "src\bfdl\binance\__init__.py",
  "src\bfdl\collectors\__init__.py",
  "src\bfdl\transforms\__init__.py",
  "src\bfdl\cli\__init__.py",

  "src\bfdl\core\log.py",
  "src\bfdl\core\timeutil.py",
  "src\bfdl\core\http.py",
  "src\bfdl\core\io.py",
  "src\bfdl\core\checkpoint.py",
  "src\bfdl\core\schema.py",

  "src\bfdl\binance\client.py",
  "src\bfdl\binance\endpoints.py",
  "src\bfdl\binance\rate_limit.py",

  "src\bfdl\collectors\klines_m1.py",
  "src\bfdl\collectors\premium_index_m1.py",
  "src\bfdl\collectors\funding_events.py",
  "src\bfdl\collectors\open_interest_snapshots.py",

  "src\bfdl\transforms\aggregate_tf.py",
  "src\bfdl\transforms\integrity_checks.py",

  "src\bfdl\cli\collect.py",
  "src\bfdl\cli\validate.py",
  "src\bfdl\cli\aggregate.py",

  "README.md",
  ".env.example",
  ".gitignore",
  "pyproject.toml"
)

foreach ($f in $files) {
  Touch-File (Join-Path $ProjectPath $f)
}

@"
symbols:
  - BTCUSDT
  - ETHUSDT
"@ | Set-Content "$ProjectPath\config\symbols.yml"

@"
storage:
  base_dir: ./data/raw/binance_um
  write_parquet: true
  write_csv: true

collector:
  start_date_utc: "2019-10-01"
  interval: "1m"
  limit_per_request: 1500
  sleep_s: 0.25
"@ | Set-Content "$ProjectPath\config\collector.yml"

Write-Host "Structure créée avec succès."
