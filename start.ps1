<#
.SYNOPSIS
    1-Click Launcher for The5ers Multi-Agent Trading Co-Pilot (FastAPI + Streamlit).
.DESCRIPTION
    Cleans up any lingering processes on ports 8000/8501, activates .venv in D:\Trading Agent,
    verifies .env configuration, launches FastAPI in the background, waits for /health,
    and runs Streamlit in the foreground. Pressing Ctrl+C cleanly terminates both servers.
#>

$ErrorActionPreference = "Stop"

# 1. Lock working directory to script root (D:\Trading Agent)
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "=== The5ers Multi-Agent Trading Co-Pilot Launcher ===" -ForegroundColor Cyan

# Helper: Free a TCP port if an orphaned process is holding it
function Clear-PortIfBusy {
    param([int]$Port)
    $Connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($Connections) {
        $Pids = $Connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($ProcId in $Pids) {
            if ($ProcId -and $ProcId -gt 0) {
                $Proc = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
                $ProcName = if ($Proc) { $Proc.ProcessName } else { "Unknown" }
                Write-Host "[CLEANUP] Stopping stale process '$ProcName' (PID $ProcId) holding port $Port..." -ForegroundColor Yellow
                Stop-Process -Id $ProcId -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Milliseconds 500
    }
}

# 2. Clear stale FastAPI (8000) and Streamlit (8501) processes if present
Clear-PortIfBusy -Port 8000
Clear-PortIfBusy -Port 8501

# 3. Verify virtual environment exists
$VenvActivate = Join-Path $RootDir ".venv\Scripts\Activate.ps1"
$PythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at $PythonExe" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 4. Verify .env exists and contains a non-placeholder GEMINI_API_KEY
$EnvFile = Join-Path $RootDir ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "[ERROR] Missing .env file in $RootDir" -ForegroundColor Red
    Write-Host "Copy .env.example to .env and add your GEMINI_API_KEY and MT5 credentials." -ForegroundColor Yellow
    exit 1
}

$EnvContent = Get-Content $EnvFile -Raw
if ($EnvContent -match "your_gemini_api_key_here" -or $EnvContent -notmatch "GEMINI_API_KEY=.+") {
    Write-Host "[WARNING] GEMINI_API_KEY in .env looks unset or still uses the placeholder." -ForegroundColor Yellow
}

# 5. Activate .venv for the current PowerShell process
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
& $VenvActivate

# 6. Ensure logs directory exists
$LogsDir = Join-Path $RootDir "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
}
$ApiOutLog = Join-Path $LogsDir "fastapi_out.log"
$ApiErrLog = Join-Path $LogsDir "fastapi_err.log"

# 7. Start FastAPI backend as a managed background process
Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
$FastApiProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $RootDir `
    -RedirectStandardOutput $ApiOutLog `
    -RedirectStandardError $ApiErrLog `
    -PassThru `
    -NoNewWindow

try {
    # Wait up to 15 seconds for FastAPI /health endpoint to come online
    $Ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if ($FastApiProc.HasExited) {
            Write-Host "[ERROR] FastAPI exited unexpectedly. Check $ApiErrLog" -ForegroundColor Red
            Get-Content $ApiErrLog -Tail 20
            exit 1
        }
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 1
            if ($Health.status -eq "ok") {
                $Ready = $true
                break
            }
        } catch {
            # Still booting
        }
    }

    if (-not $Ready) {
        Write-Host "[ERROR] Timed out waiting for FastAPI on port 8000. Check $ApiErrLog" -ForegroundColor Red
        exit 1
    }

    Write-Host "[OK] FastAPI Backend Online ($($Health.phase))" -ForegroundColor Green

    # 8. Launch Streamlit Dashboard in foreground (Ctrl+C stops both)
    Write-Host "[2/2] Launching Streamlit Dashboard on http://localhost:8501 ..." -ForegroundColor Green
    Write-Host "Press Ctrl+C in this terminal to stop both Streamlit and FastAPI.`n" -ForegroundColor DarkGray

    & $PythonExe -m streamlit run "ui\app.py" --server.port 8501 --server.headless false
}
finally {
    # 9. Clean shutdown of background FastAPI server on exit / Ctrl+C
    if ($FastApiProc -and -not $FastApiProc.HasExited) {
        Write-Host "`nShutting down FastAPI Backend (PID $($FastApiProc.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $FastApiProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Co-Pilot stopped cleanly." -ForegroundColor Cyan
}