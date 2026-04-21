param(
    [switch]$SmokeTest,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-PythonVersion($Command, $Arguments) {
    try {
        $output = & $Command @Arguments --version 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }
        $text = ($output | Out-String).Trim()
        if ($text -match "Python\s+(\d+)\.(\d+)\.(\d+)") {
            return [pscustomobject]@{
                Text = $text
                Major = [int]$Matches[1]
                Minor = [int]$Matches[2]
                Patch = [int]$Matches[3]
                Command = $Command
                Arguments = $Arguments
            }
        }
    } catch {
        return $null
    }
    return $null
}

function Find-CompatiblePython {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.11") },
        @{ Command = "py"; Args = @("-3.12") },
        @{ Command = "py"; Args = @("-3.13") },
        @{ Command = "py"; Args = @("-3.10") },
        @{ Command = "py"; Args = @("-3.9") },
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "python3"; Args = @() },
        @{ Command = "python"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $version = Get-PythonVersion $candidate.Command $candidate.Args
        if ($null -eq $version) { continue }
        if ($version.Major -eq 3 -and $version.Minor -ge 9 -and $version.Minor -le 13) {
            return $version
        }
    }

    throw "Python 3.9 to 3.13 was not found. Please install Python 3.11 from https://www.python.org/downloads/windows/ and run this file again."
}

function Run-ProcessChecked($FilePath, $Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

function Test-AppHealth($PortToCheck) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$PortToCheck/_stcore/health" -TimeoutSec 2
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Test-PortFree($PortToCheck) {
    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $listener = New-Object System.Net.Sockets.TcpListener($address, $PortToCheck)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Resolve-AppPort($PreferredPort) {
    if (Test-AppHealth $PreferredPort) {
        return [pscustomobject]@{ Port = $PreferredPort; AlreadyRunning = $true }
    }
    if (Test-PortFree $PreferredPort) {
        return [pscustomobject]@{ Port = $PreferredPort; AlreadyRunning = $false }
    }
    foreach ($candidate in 8502..8520) {
        if (Test-AppHealth $candidate) {
            return [pscustomobject]@{ Port = $candidate; AlreadyRunning = $true }
        }
        if (Test-PortFree $candidate) {
            return [pscustomobject]@{ Port = $candidate; AlreadyRunning = $false }
        }
    }
    throw "No free local port was found between 8501 and 8520. Close old Streamlit windows and try again."
}

Write-Step "Finding a compatible Python version"
$python = Find-CompatiblePython
Write-Host "Using $($python.Text)"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating local virtual environment (.venv)"
    Run-ProcessChecked $python.Command ($python.Arguments + @("-m", "venv", ".venv"))
}

Write-Step "Checking Python environment"
Run-ProcessChecked $VenvPython @("--version")

$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$HashPath = Join-Path $ProjectRoot ".venv\requirements.sha256"
$currentHash = (Get-FileHash $RequirementsPath -Algorithm SHA256).Hash
$storedHash = if (Test-Path $HashPath) { (Get-Content $HashPath -Raw).Trim() } else { "" }

if ($currentHash -ne $storedHash) {
    Write-Step "Installing required packages. First run may take several minutes"
    Run-ProcessChecked $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Run-ProcessChecked $VenvPython @("-m", "pip", "install", "-r", "requirements.txt")
    Set-Content -Path $HashPath -Value $currentHash -Encoding ASCII
} else {
    Write-Step "Packages are already installed"
}

Write-Step "Running a quick import check"
Run-ProcessChecked $VenvPython @("-c", "import streamlit, pandas, rdkit, sklearn, plotly; print('Import check OK')")

if ($SmokeTest) {
    Write-Step "Starting Streamlit smoke test on port $Port"
    $log = Join-Path $ProjectRoot "streamlit_local_test.log"
    $err = Join-Path $ProjectRoot "streamlit_local_test.err.log"
    Remove-Item $log, $err -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath $VenvPython -ArgumentList @("-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", "$Port") -WorkingDirectory $ProjectRoot -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    try {
        $ready = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/_stcore/health" -TimeoutSec 3
                if ($response.StatusCode -eq 200) { $ready = $true; break }
            } catch {}
        }
        if (-not $ready) { throw "Streamlit did not become ready." }
        $page = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/" -TimeoutSec 10
        Write-Host "LOCALHOST_OK status=$($page.StatusCode) length=$($page.Content.Length)"
        $stderr = ""
        if (Test-Path $err) {
            $rawErrorLog = Get-Content $err -Raw
            if ($null -ne $rawErrorLog) {
                $stderr = $rawErrorLog.Trim()
            }
        }
        if ($stderr) { Write-Host "STDERR:"; Write-Host $stderr } else { Write-Host "STDERR_EMPTY" }
    } finally {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
        Remove-Item $log, $err -ErrorAction SilentlyContinue
    }
    exit 0
}

Write-Step "Starting the app"
$portInfo = Resolve-AppPort $Port
$Port = [int]$portInfo.Port
$url = "http://localhost:$Port"

if ($portInfo.AlreadyRunning) {
    Write-Host "The app is already running at $url"
    Start-Process $url
    Write-Host "Opened the existing local app. You can close this launcher window."
    exit 0
}

Write-Host "Opening $url"
Start-Job -ScriptBlock { param($TargetUrl) Start-Sleep -Seconds 4; Start-Process $TargetUrl } -ArgumentList $url | Out-Null
& $VenvPython @("-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1", "--server.port", "$Port", "--browser.gatherUsageStats", "false")
$streamlitExit = $LASTEXITCODE
if ($streamlitExit -ne 0) {
    Write-Host ""
    Write-Host "Streamlit stopped with exit code $streamlitExit." -ForegroundColor Yellow
    Write-Host "If you closed the server manually, this is safe to ignore. If it failed immediately, copy the messages above."
}
exit 0
