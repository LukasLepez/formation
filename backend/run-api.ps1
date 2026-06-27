$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$backendPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$rootPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $backendPython) { $backendPython } else { $rootPython }

if (-not (Test-Path $python)) {
    throw "Aucune venv Python utilisable n'a été trouvée. Lance d'abord : uv sync"
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
& $python -m uvicorn indusense.api:app --host 127.0.0.1 --port 8000
