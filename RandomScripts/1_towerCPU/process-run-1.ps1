# script-organized-TowerCPU-0-enhanced.ps1
# ==============================================
# Enhanced version with comprehensive validity checks and email reporting
# ==============================================

# ========== CONFIGURATION ==========
$VENV_ROOT = "C:\raihan\dokumen\project\global-env\faceRecog\.venv"
$PYTHON_SCRIPT = "C:\raihan\dokumen\project\global-env\faceRecog\run_py\autoEmbedPerson.py"
$POWERSHELL_SCRIPT_NAME = Split-Path -Leaf $MyInvocation.MyCommand.Path

# Python parameters
$PYTHON_PARAMS = @{
    "--multi-source" = $null
}

# Base path for all runs
$RUNS_BASE_PATH = "C:\raihan\dokumen\project\global-env\faceRecog\process-run"

# Email Configuration
$EMAIL_CONFIG = @{
    Enabled = $true  # Set to $false to disable email notifications
    SmtpServer = "smtp.gmail.com"  # Change to your SMTP server
    SmtpPort = 587
    UseSsl = $true
    SenderEmail = "your-email@gmail.com"  # Update with your email
    SenderPassword = "your-app-password"  # Use app-specific password for Gmail
    RecipientEmail = "recipient@example.com"  # Update with recipient email
    SubjectPrefix = "[FaceRecog Process]"
}

# Completion File Configuration
$COMPLETION_FILE_CONFIG = @{
    Enabled = $true
    FileName = "PROCESS_COMPLETE.txt"
    CheckDelaySeconds = 30  # Wait before checking for completion file
    MaxRetries = 6  # Check up to 6 times (total 3 minutes wait)
}
# ===================================

# ========== VALIDITY CHECK FUNCTIONS ==========
function Test-PathWithDetail {
    param(
        [string]$Path,
        [string]$PathType = "Any",  # "File", "Directory", or "Any"
        [string]$Description = "Path"
    )
    
    $result = @{
        Valid = $false
        Path = $Path
        Description = $Description
        Type = $PathType
        Error = ""
        Details = @{}
    }
    
    if ([string]::IsNullOrWhiteSpace($Path)) {
        $result.Error = "Path is null or empty"
        return $result
    }
    
    try {
        if (Test-Path $Path) {
            $item = Get-Item $Path -ErrorAction Stop
            
            switch ($PathType) {
                "File" {
                    if ($item.PSIsContainer) {
                        $result.Error = "Path is a directory, but a file was expected"
                        $result.Details.ItemType = "Directory"
                    } else {
                        $result.Valid = $true
                        $result.Details.ItemType = "File"
                        $result.Details.Size = if ($item.Length) { "$([math]::Round($item.Length/1MB, 2)) MB" } else { "0 MB" }
                        $result.Details.LastModified = $item.LastWriteTime
                    }
                }
                "Directory" {
                    if (-not $item.PSIsContainer) {
                        $result.Error = "Path is a file, but a directory was expected"
                        $result.Details.ItemType = "File"
                    } else {
                        $result.Valid = $true
                        $result.Details.ItemType = "Directory"
                        $result.Details.ItemCount = @(Get-ChildItem $Path -Force -ErrorAction SilentlyContinue).Count
                        $result.Details.LastModified = $item.LastWriteTime
                    }
                }
                "Any" {
                    $result.Valid = $true
                    $result.Details.ItemType = if ($item.PSIsContainer) { "Directory" } else { "File" }
                    $result.Details.LastModified = $item.LastWriteTime
                }
            }
            
            if ($result.Valid) {
                $result.Details.FullPath = $item.FullName
                $result.Details.CreationTime = $item.CreationTime
                $result.Details.Attributes = $item.Attributes
            }
        } else {
            $result.Error = "Path does not exist"
            
            # Check if parent directory exists
            $parentDir = Split-Path $Path -Parent
            if (Test-Path $parentDir) {
                $result.Details.ParentExists = $true
                $result.Details.ParentPath = $parentDir
                
                # Check permissions on parent directory
                try {
                    $testFile = Join-Path $parentDir "test_permission_$(Get-Random).tmp"
                    $null = New-Item $testFile -ItemType File -Force -ErrorAction Stop
                    Remove-Item $testFile -Force -ErrorAction Stop
                    $result.Details.WritePermission = $true
                } catch {
                    $result.Details.WritePermission = $false
                    $result.Details.PermissionError = $_.Exception.Message
                }
            } else {
                $result.Details.ParentExists = $false
            }
        }
    } catch {
        $result.Error = "Error testing path: $($_.Exception.Message)"
        $result.Details.Exception = $_.Exception
    }
    
    return $result
}

function Test-VenvConfiguration {
    param([string]$VenvPath)
    
    $results = @()
    
    # Check virtual environment root
    $venvRootCheck = Test-PathWithDetail -Path $VenvPath -PathType "Directory" -Description "Virtual Environment Root"
    $results += $venvRootCheck
    
    if ($venvRootCheck.Valid) {
        # Check Python executable
        $pythonExe = "$VenvPath\Scripts\python.exe"
        $pythonCheck = Test-PathWithDetail -Path $pythonExe -PathType "File" -Description "Python Executable"
        $results += $pythonCheck
        
        # Check pip
        $pipExe = "$VenvPath\Scripts\pip.exe"
        $pipCheck = Test-PathWithDetail -Path $pipExe -PathType "File" -Description "Pip Executable"
        $results += $pipCheck
        
        # Check activate script
        $activateScript = "$VenvPath\Scripts\Activate.ps1"
        $activateCheck = Test-PathWithDetail -Path $activateScript -PathType "File" -Description "Activation Script"
        $results += $activateCheck
    }
    
    return $results
}

function Test-PythonScriptConfiguration {
    param([string]$ScriptPath)
    
    $results = @()
    
    # Check Python script exists
    $scriptCheck = Test-PathWithDetail -Path $ScriptPath -PathType "File" -Description "Python Script"
    $results += $scriptCheck
    
    if ($scriptCheck.Valid) {
        # Check if it's a Python file
        if ($ScriptPath -notmatch '\.py$') {
            $results += @{
                Valid = $false
                Path = $ScriptPath
                Description = "Python Script Extension"
                Error = "File does not have .py extension"
            }
        }
        
        # Try to check Python syntax (basic check)
        try {
            $content = Get-Content $ScriptPath -First 5 -ErrorAction Stop
            $hasPythonShebang = $content -match '^#!.*python'
            $results += @{
                Valid = $hasPythonShebang
                Path = $ScriptPath
                Description = "Python Shebang Check"
                Error = if (-not $hasPythonShebang) { "No Python shebang found in first 5 lines" } else { "" }
                Details = @{FirstLines = $content}
            }
        } catch {
            # Ignore syntax check errors
        }
    }
    
    return $results
}

function Test-SystemRequirements {
    $results = @()
    
    # Check PowerShell version
    $psVersion = $PSVersionTable.PSVersion
    $results += @{
        Valid = $psVersion -ge [Version]"5.1"
        Description = "PowerShell Version"
        Details = @{Version = $psVersion.ToString()}
        Error = if ($psVersion -lt [Version]"5.1") { "PowerShell 5.1 or higher required" } else { "" }
    }
    
    # Check execution policy
    $executionPolicy = Get-ExecutionPolicy
    $results += @{
        Valid = $executionPolicy -in @("RemoteSigned", "Unrestricted", "Bypass")
        Description = "Execution Policy"
        Details = @{Policy = $executionPolicy}
        Error = if ($executionPolicy -notin @("RemoteSigned", "Unrestricted", "Bypass")) { 
            "Execution policy too restrictive. Current: $executionPolicy" 
        } else { "" }
    }
    
    # Check .NET Framework (optional)
    try {
        $dotNetVersion = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" -ErrorAction SilentlyContinue).Release
        $results += @{
            Valid = $true
            Description = ".NET Framework"
            Details = @{Release = $dotNetVersion}
            Error = ""
        }
    } catch {
        $results += @{
            Valid = $true  # Not critical
            Description = ".NET Framework"
            Details = @{}
            Error = "Could not determine .NET version: $_"
        }
    }
    
    return $results
}

function Run-ComprehensiveValidityChecks {
    Write-Host "`n=================================================="
    Write-Host "RUNNING COMPREHENSIVE VALIDITY CHECKS"
    Write-Host "=================================================="
    
    $allResults = @()
    $criticalFailures = @()
    
    # 1. System Requirements
    Write-Host "`n[1] Checking System Requirements..." -ForegroundColor Cyan
    $systemChecks = Test-SystemRequirements
    $allResults += $systemChecks
    $criticalFailures += $systemChecks | Where-Object { -not $_.Valid }
    Display-CheckResults $systemChecks
    
    # 2. Virtual Environment
    Write-Host "`n[2] Checking Virtual Environment..." -ForegroundColor Cyan
    $venvChecks = Test-VenvConfiguration -VenvPath $VENV_ROOT
    $allResults += $venvChecks
    $criticalFailures += $venvChecks | Where-Object { -not $_.Valid -and $_.Description -eq "Python Executable" }
    Display-CheckResults $venvChecks
    
    # 3. Python Script
    Write-Host "`n[3] Checking Python Script..." -ForegroundColor Cyan
    $scriptChecks = Test-PythonScriptConfiguration -ScriptPath $PYTHON_SCRIPT
    $allResults += $scriptChecks
    $criticalFailures += $scriptChecks | Where-Object { -not $_.Valid -and $_.Description -eq "Python Script" }
    Display-CheckResults $scriptChecks
    
    # 4. Runs Base Path
    Write-Host "`n[4] Checking Runs Base Path..." -ForegroundColor Cyan
    $runsPathCheck = Test-PathWithDetail -Path $RUNS_BASE_PATH -PathType "Directory" -Description "Runs Base Directory"
    $allResults += $runsPathCheck
    if (-not $runsPathCheck.Valid) {
        Write-Host "  Attempting to create directory..." -ForegroundColor Yellow
        try {
            New-Item -ItemType Directory -Path $RUNS_BASE_PATH -Force | Out-Null
            $runsPathCheck.Valid = $true
            $runsPathCheck.Error = ""
            $runsPathCheck.Details.Created = $true
            Write-Host "  ✓ Created directory successfully" -ForegroundColor Green
        } catch {
            $runsPathCheck.Error = "Failed to create directory: $_"
            $criticalFailures += $runsPathCheck
        }
    }
    Display-CheckResults @($runsPathCheck)
    
    # 5. Email Configuration (if enabled)
    if ($EMAIL_CONFIG.Enabled) {
        Write-Host "`n[5] Checking Email Configuration..." -ForegroundColor Cyan
        $emailChecks = @()
        
        # Check required fields
        $requiredFields = @("SmtpServer", "SenderEmail", "RecipientEmail")
        foreach ($field in $requiredFields) {
            $isValid = -not [string]::IsNullOrWhiteSpace($EMAIL_CONFIG[$field])
            $emailChecks += @{
                Valid = $isValid
                Description = "Email $field"
                Error = if (-not $isValid) { "Field is empty" } else { "" }
                Details = @{Value = $EMAIL_CONFIG[$field]}
            }
        }
        
        $allResults += $emailChecks
        Display-CheckResults $emailChecks
    }
    
    # Summary
    Write-Host "`n=================================================="
    Write-Host "VALIDITY CHECK SUMMARY"
    Write-Host "=================================================="
    
    $totalChecks = $allResults.Count
    $passedChecks = ($allResults | Where-Object { $_.Valid }).Count
    $failedChecks = $totalChecks - $passedChecks
    $criticalCount = $criticalFailures.Count
    
    Write-Host "Total Checks: $totalChecks" -ForegroundColor White
    Write-Host "Passed: $passedChecks" -ForegroundColor Green
    Write-Host "Failed: $failedChecks" -ForegroundColor $(if ($failedChecks -gt 0) { "Red" } else { "White" })
    Write-Host "Critical Failures: $criticalCount" -ForegroundColor $(if ($criticalCount -gt 0) { "Red" } else { "White" })
    
    if ($criticalCount -gt 0) {
        Write-Host "`nCRITICAL FAILURES:" -ForegroundColor Red
        foreach ($failure in $criticalFailures) {
            Write-Host "  • $($failure.Description): $($failure.Error)" -ForegroundColor Red
        }
        return $false
    }
    
    return $true
}

function Display-CheckResults {
    param([array]$Results)
    
    foreach ($result in $Results) {
        $status = if ($result.Valid) { "✓" } else { "✗" }
        $color = if ($result.Valid) { "Green" } else { "Red" }
        
        Write-Host "  $status $($result.Description)" -ForegroundColor $color
        if (-not $result.Valid -and -not [string]::IsNullOrEmpty($result.Error)) {
            Write-Host "    Error: $($result.Error)" -ForegroundColor Yellow
        }
        if ($result.Details.Count -gt 0 -and $result.Valid) {
            foreach ($detail in $result.Details.Keys) {
                Write-Host "    $detail: $($result.Details[$detail])" -ForegroundColor Gray
            }
        }
    }
}
# ===================================

# ========== EMAIL FUNCTIONS ==========
function Send-ProcessCompletionEmail {
    param(
        [hashtable]$Config,
        [string]$RunFolder,
        [string]$RunId,
        [int]$ExitCode,
        [string]$Duration,
        [string]$CompletionStatus,
        [string]$CompletionFilePath
    )
    
    if (-not $Config.Enabled) {
        Write-Host "Email notifications are disabled" -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "`nPreparing email notification..." -ForegroundColor Cyan
    
    try {
        # Create email credentials
        $securePassword = ConvertTo-SecureString $Config.SenderPassword -AsPlainText -Force
        $credential = New-Object System.Management.Automation.PSCredential ($Config.SenderEmail, $securePassword)
        
        # Prepare email body
        $emailBody = @"
Face Recognition Processing Report
==================================================
Run ID: $RunId
Completion Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Status: $CompletionStatus
Exit Code: $ExitCode
Duration: $Duration

LOCATION DETAILS
==================================================
Run Folder: $RunFolder

COMPLETION FILE
==================================================
Status: $(if (Test-Path $CompletionFilePath) { "Found at $CompletionFilePath" } else { "NOT FOUND" })

SYSTEM INFORMATION
==================================================
Host Name: $env:COMPUTERNAME
User: $env:USERNAME
PowerShell Version: $($PSVersionTable.PSVersion)
Execution Time: $(Get-Date)

LOG FILES
==================================================
• Main Log: $RunFolder\logs\run_$RunId.log
• Python Output: $RunFolder\logs\python_output.txt
• Completion Summary: $RunFolder\logs\completion_summary.txt
• Metadata: $RunFolder\metadata.json

NOTES
==================================================
This is an automated notification from the Face Recognition processing system.
"@
        
        $subject = "$($Config.SubjectPrefix) Process $CompletionStatus - $RunId"
        
        # Send email
        Send-MailMessage `
            -From $Config.SenderEmail `
            -To $Config.RecipientEmail `
            -Subject $subject `
            -Body $emailBody `
            -SmtpServer $Config.SmtpServer `
            -Port $Config.SmtpPort `
            -UseSsl:$Config.UseSsl `
            -Credential $credential `
            -ErrorAction Stop
        
        Write-Host "✓ Email notification sent successfully" -ForegroundColor Green
        return $true
        
    } catch {
        Write-Host "✗ Failed to send email: $_" -ForegroundColor Red
        Write-Host "  Check email configuration and network connectivity" -ForegroundColor Yellow
        return $false
    }
}

function Wait-ForCompletionFile {
    param(
        [string]$RunFolderPath,
        [hashtable]$Config
    )
    
    if (-not $Config.Enabled) {
        Write-Host "Completion file detection is disabled" -ForegroundColor Yellow
        return $null
    }
    
    $completionFile = Join-Path $RunFolderPath $Config.FileName
    $scriptOutputPath = Join-Path $RunFolderPath "script_output"
    
    # Also check in script_output subdirectory
    $completionFileAlt = Join-Path $scriptOutputPath $Config.FileName
    
    Write-Host "`nWaiting for completion file..." -ForegroundColor Cyan
    Write-Host "  Primary location: $completionFile" -ForegroundColor Gray
    Write-Host "  Alternate location: $completionFileAlt" -ForegroundColor Gray
    Write-Host "  Check delay: $($Config.CheckDelaySeconds) seconds" -ForegroundColor Gray
    Write-Host "  Max retries: $($Config.MaxRetries)" -ForegroundColor Gray
    
    for ($i = 1; $i -le $Config.MaxRetries; $i++) {
        Write-Host "  Check attempt $i of $($Config.MaxRetries)..." -ForegroundColor Gray
        
        # Check both locations
        if (Test-Path $completionFile) {
            Write-Host "  ✓ Completion file found at primary location" -ForegroundColor Green
            return $completionFile
        }
        
        if (Test-Path $completionFileAlt) {
            Write-Host "  ✓ Completion file found at alternate location" -ForegroundColor Green
            return $completionFileAlt
        }
        
        if ($i -lt $Config.MaxRetries) {
            Write-Host "  Not found. Waiting $($Config.CheckDelaySeconds) seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds $Config.CheckDelaySeconds
        }
    }
    
    Write-Host "  ✗ Completion file not found after $($Config.MaxRetries) attempts" -ForegroundColor Red
    return $null
}
# ===================================

# ========== TIME DETECTION FUNCTIONS ==========
function Get-SystemTimeInfo {
    $timeInfo = @{
        LocalTime = Get-Date
        UtcTime = (Get-Date).ToUniversalTime()
        TimeZone = [System.TimeZoneInfo]::Local
        Uptime = Get-Uptime
        SystemBootTime = (Get-Date).AddSeconds(-(Get-Uptime).TotalSeconds)
    }
    
    # Get high precision timestamp (if available)
    try {
        $timeInfo.HighPrecisionTime = [System.Diagnostics.Stopwatch]::StartNew()
    } catch {
        # Ignore if not available
    }
    
    return $timeInfo
}

function Get-Uptime {
    try {
        $os = Get-WmiObject Win32_OperatingSystem
        $uptime = (Get-Date) - ($os.ConvertToDateTime($os.LastBootUpTime))
        return $uptime
    } catch {
        # Fallback method
        $bootTime = (Get-CimInstance -ClassName Win32_OperatingSystem).LastBootUpTime
        return (Get-Date) - $bootTime
    }
}

function Format-TimeSpanForDisplay {
    param([TimeSpan]$TimeSpan)
    
    if ($TimeSpan.Days -gt 0) {
        return "$($TimeSpan.Days)d $($TimeSpan.Hours)h $($TimeSpan.Minutes)m $($TimeSpan.Seconds)s"
    } elseif ($TimeSpan.Hours -gt 0) {
        return "$($TimeSpan.Hours)h $($TimeSpan.Minutes)m $($TimeSpan.Seconds)s"
    } elseif ($TimeSpan.Minutes -gt 0) {
        return "$($TimeSpan.Minutes)m $($TimeSpan.Seconds)s"
    } else {
        return "$($TimeSpan.Seconds)s"
    }
}
# ===================================

# Helper function to convert hashtable to argument array
function Convert-HashtableToArgs {
    param([hashtable]$Params)
    
    $argsArray = @()
    foreach ($key in $Params.Keys) {
        $argsArray += $key
        if ($Params[$key] -ne $null) {
            $argsArray += $Params[$key]
        }
    }
    return $argsArray
}

# ========== MAIN EXECUTION ==========

# Get system time info at start
Write-Host "=================================================="
Write-Host "PROCESS INITIALIZATION"
Write-Host "=================================================="
$systemStartTime = Get-SystemTimeInfo
Write-Host "System Local Time: $($systemStartTime.LocalTime)" -ForegroundColor Cyan
Write-Host "System UTC Time: $($systemStartTime.UtcTime)" -ForegroundColor Cyan
Write-Host "Time Zone: $($systemStartTime.TimeZone.DisplayName)" -ForegroundColor Cyan
Write-Host "System Uptime: $(Format-TimeSpanForDisplay -TimeSpan $systemStartTime.Uptime)" -ForegroundColor Cyan
Write-Host "System Boot Time: $($systemStartTime.SystemBootTime)" -ForegroundColor Cyan

# Run comprehensive validity checks
$checksPassed = Run-ComprehensiveValidityChecks

if (-not $checksPassed) {
    Write-Host "`n✗ Validity checks failed. Exiting script." -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ All validity checks passed!" -ForegroundColor Green

# ========== SCRIPT EXECUTION ==========

# Convert parameters to arguments
$PYTHON_ARGS = Convert-HashtableToArgs -Params $PYTHON_PARAMS

# Create runs base directory if it doesn't exist
if (-not (Test-Path $RUNS_BASE_PATH)) {
    New-Item -ItemType Directory -Path $RUNS_BASE_PATH -Force | Out-Null
    Write-Host "Created runs base directory: $RUNS_BASE_PATH" -ForegroundColor Green
}

# Generate run folder with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$RUN_FOLDER_NAME = "Run_Process_TowerCPU_Embedding_$timestamp"
$RUN_FOLDER_PATH = Join-Path $RUNS_BASE_PATH $RUN_FOLDER_NAME

# Create run folder structure
$RUN_LOGS_PATH = Join-Path $RUN_FOLDER_PATH "logs"
$RUN_SCRIPT_OUTPUT_PATH = Join-Path $RUN_FOLDER_PATH "script_output"
$RUN_METADATA_PATH = Join-Path $RUN_FOLDER_PATH "metadata.json"

try {
    # Create all directories
    New-Item -ItemType Directory -Path $RUN_FOLDER_PATH -Force | Out-Null
    New-Item -ItemType Directory -Path $RUN_LOGS_PATH -Force | Out-Null
    New-Item -ItemType Directory -Path $RUN_SCRIPT_OUTPUT_PATH -Force | Out-Null
    
    Write-Host "`n✓ Created run folder structure at: $RUN_FOLDER_PATH" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to create run folder structure: $_" -ForegroundColor Red
    exit 1
}

# ========== LOG FILE SETUP ==========
$MAIN_LOG_FILE = Join-Path $RUN_LOGS_PATH "run_$timestamp.log"
$PYTHON_OUTPUT_FILE = Join-Path $RUN_LOGS_PATH "python_output.txt"
$COMPLETION_SUMMARY_FILE = Join-Path $RUN_LOGS_PATH "completion_summary.txt"

# Format arguments for logging
$argsString = $PYTHON_ARGS -join " "

# Write initial log header with system info
$logHeader = @"
==================================================
RUN STARTED: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
System Local Time: $($systemStartTime.LocalTime)
System UTC Time: $($systemStartTime.UtcTime)
Time Zone: $($systemStartTime.TimeZone.DisplayName)
System Uptime: $(Format-TimeSpanForDisplay -TimeSpan $systemStartTime.Uptime)
==================================================
Run Folder: $RUN_FOLDER_PATH
PowerShell Script: $POWERSHELL_SCRIPT_NAME
Virtual Environment: $VENV_ROOT
Python Script: $PYTHON_SCRIPT
Arguments: $argsString
Number of arguments: $($PYTHON_ARGS.Count)
==================================================
"@

$logHeader | Out-File $MAIN_LOG_FILE

# ========== CREATE METADATA FILE ==========
$metadata = @{
    run_id = $timestamp
    start_time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    system_info = @{
        local_time = $systemStartTime.LocalTime
        utc_time = $systemStartTime.UtcTime
        timezone = $systemStartTime.TimeZone.DisplayName
        uptime = $systemStartTime.Uptime.TotalSeconds
        boot_time = $systemStartTime.SystemBootTime
    }
    powershell_script = $POWERSHELL_SCRIPT_NAME
    virtual_env = $VENV_ROOT
    python_script = $PYTHON_SCRIPT
    arguments = $PYTHON_ARGS
    arguments_count = $PYTHON_ARGS.Count
    run_folder = $RUN_FOLDER_PATH
    email_config = @{
        enabled = $EMAIL_CONFIG.Enabled
        recipient = $EMAIL_CONFIG.RecipientEmail
    }
}

$metadata | ConvertTo-Json -Depth 5 | Out-File $RUN_METADATA_PATH
Write-Host "✓ Created metadata file: $RUN_METADATA_PATH" -ForegroundColor Green

# ========== EXECUTION ==========
Write-Host "`n=================================================="
Write-Host "STARTING PYTHON SCRIPT EXECUTION"
Write-Host "=================================================="
Write-Host "Run folder: $RUN_FOLDER_PATH" -ForegroundColor Cyan
Write-Host "Arguments: $argsString" -ForegroundColor Cyan
Write-Host ""

$pythonExe = "$VENV_ROOT\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    $errorMsg = "ERROR: Python executable not found at $pythonExe"
    $errorMsg | Out-File $MAIN_LOG_FILE -Append
    Write-Host "✗ $errorMsg" -ForegroundColor Red
    exit 1
}

# Capture precise start time
$processStartTime = Get-Date
"`n==================================================" | Out-File $MAIN_LOG_FILE -Append
"PYTHON PROCESS STARTED: $($processStartTime.ToString('yyyy-MM-dd HH:mm:ss.fff'))" | Out-File $MAIN_LOG_FILE -Append
"==================================================" | Out-File $MAIN_LOG_FILE -Append

try {
    # Prepare the command for logging
    $commandString = "`"$pythonExe`" `"$PYTHON_SCRIPT`" $argsString"
    Write-Host "Executing: $commandString" -ForegroundColor Yellow
    $commandString | Out-File $MAIN_LOG_FILE -Append
    "" | Out-File $MAIN_LOG_FILE -Append
    
    Write-Host "Python output will be saved to: $PYTHON_OUTPUT_FILE" -ForegroundColor Cyan
    
    # Set working directory to script_output
    $originalLocation = Get-Location
    Set-Location $RUN_SCRIPT_OUTPUT_PATH
    
    # Run Python with array of arguments
    & $pythonExe $PYTHON_SCRIPT $PYTHON_ARGS 2>&1 | Tee-Object -FilePath $PYTHON_OUTPUT_FILE
    
    $exitCode = $LASTEXITCODE
    
    # Return to original location
    Set-Location $originalLocation
    
} catch {
    $errorMessage = $_.Exception.Message
    "EXCEPTION: $errorMessage" | Out-File $MAIN_LOG_FILE -Append
    $errorMessage | Out-File $PYTHON_OUTPUT_FILE -Append
    Write-Host "✗ Exception during Python script execution: $errorMessage" -ForegroundColor Red
}

# Capture precise end time
$processEndTime = Get-Date
$processDuration = $processEndTime - $processStartTime
$processDurationFormatted = Format-TimeSpanForDisplay -TimeSpan $processDuration

# ========== COMPLETION FILE DETECTION ==========
$completionFile = $null
if ($COMPLETION_FILE_CONFIG.Enabled) {
    $completionFile = Wait-ForCompletionFile -RunFolderPath $RUN_FOLDER_PATH -Config $COMPLETION_FILE_CONFIG
    
    if ($completionFile) {
        # Read completion file content if it exists
        try {
            $completionContent = Get-Content $completionFile -Raw -ErrorAction Stop
            "Completion file content:`n$completionContent" | Out-File $MAIN_LOG_FILE -Append
            Write-Host "✓ Completion file content loaded" -ForegroundColor Green
        } catch {
            "Could not read completion file: $_" | Out-File $MAIN_LOG_FILE -Append
        }
    }
}

# Get system time info at completion
$systemEndTime = Get-SystemTimeInfo
$totalUptimeChange = $systemEndTime.Uptime - $systemStartTime.Uptime

# ========== POST-EXECUTION LOGGING ==========
$executionSummary = @"

==================================================
EXECUTION SUMMARY
==================================================
Python Process Start: $($processStartTime.ToString('yyyy-MM-dd HH:mm:ss.fff'))
Python Process End: $($processEndTime.ToString('yyyy-MM-dd HH:mm:ss.fff'))
Python Process Duration: $processDurationFormatted

System Time at Start: $($systemStartTime.LocalTime)
System Time at End: $($systemEndTime.LocalTime)
Total Script Duration: $(Format-TimeSpanForDisplay -TimeSpan ($systemEndTime.LocalTime - $systemStartTime.LocalTime))
System Uptime Change: $(Format-TimeSpanForDisplay -TimeSpan $totalUptimeChange)

PowerShell Script: $POWERSHELL_SCRIPT_NAME
Exit code: $exitCode

Completion File: $(if ($completionFile) { "Found at $completionFile" } else { "Not found" })

Arguments passed to Python script:
$($PYTHON_ARGS | ForEach-Object { "  [$(($PYTHON_ARGS.IndexOf($_)+1))]: $_" } | Out-String)

Output Analysis:
"@

$executionSummary | Out-File $MAIN_LOG_FILE -Append

# Analyze script output
try {
    $createdFolders = Get-ChildItem -Path $RUN_SCRIPT_OUTPUT_PATH -Directory -ErrorAction SilentlyContinue | 
                     Where-Object { $_.CreationTime -ge $processStartTime } | 
                     Select-Object -ExpandProperty FullName
    
    if ($createdFolders) {
        "Python script created folders:" | Out-File $MAIN_LOG_FILE -Append
        foreach ($folder in $createdFolders) {
            $folderSize = "{0:N2} MB" -f ((Get-ChildItem $folder -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
            $itemCount = @(Get-ChildItem $folder -Recurse -File).Count
            "  - $folder (Size: $folderSize, Files: $itemCount)" | Out-File $MAIN_LOG_FILE -Append
        }
    } else {
        "No new folders were created by Python script." | Out-File $MAIN_LOG_FILE -Append
    }
    
    # Check for any output files
    $outputFiles = Get-ChildItem -Path $RUN_SCRIPT_OUTPUT_PATH -File -Recurse -ErrorAction SilentlyContinue
    "Total files in output: $($outputFiles.Count)" | Out-File $MAIN_LOG_FILE -Append
} catch {
    "Could not analyze output: $_" | Out-File $MAIN_LOG_FILE -Append
}

# ========== CREATE COMPLETION SUMMARY ==========
$completionStatus = if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED" }
if ($completionFile) { $completionStatus += " (With Completion File)" }

$completionSummary = @"
==================================================
RUN COMPLETION SUMMARY
==================================================
Completion Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Run ID: $timestamp
PowerShell Script: $POWERSHELL_SCRIPT_NAME
Status: $completionStatus
Exit Code: $exitCode

TIMING INFORMATION
==================================================
System Start Time: $($systemStartTime.LocalTime)
System End Time: $($systemEndTime.LocalTime)
Total Script Duration: $(Format-TimeSpanForDisplay -TimeSpan ($systemEndTime.LocalTime - $systemStartTime.LocalTime))

Python Process Duration: $processDurationFormatted
Python Start: $($processStartTime.ToString('HH:mm:ss.fff'))
Python End: $($processEndTime.ToString('HH:mm:ss.fff'))

FOLDER STRUCTURE
==================================================
Run Folder: $RUN_FOLDER_PATH
├── logs\
│   ├── run_$timestamp.log          (This main log file)
│   ├── python_output.txt           (Full Python script output)
│   └── completion_summary.txt      (This summary)
├── script_output\                  (Python script's working directory)
│   └── [Output folders created by Python]
└── metadata.json                   (Run configuration metadata)

EXECUTION DETAILS
==================================================
Virtual Environment: $VENV_ROOT
Python Script: $PYTHON_SCRIPT

COMPLETION FILE STATUS
==================================================
$(if ($completionFile) { 
    "✓ Completion file found at:`n  $completionFile" 
} else { 
    if ($COMPLETION_FILE_CONFIG.Enabled) {
        "✗ Completion file not found after waiting"
    } else {
        "ℹ Completion file detection disabled"
    }
})

EMAIL NOTIFICATION
==================================================
$(if ($EMAIL_CONFIG.Enabled) {
    "✓ Email notifications enabled"
    "  Recipient: $($EMAIL_CONFIG.RecipientEmail)"
} else {
    "ℹ Email notifications disabled"
})

LOG FILES
==================================================
1. Main Execution Log: $MAIN_LOG_FILE
2. Python Output: $PYTHON_OUTPUT_FILE

SCRIPT OUTPUT LOCATION
==================================================
Python script output location:
$RUN_SCRIPT_OUTPUT_PATH

$(if ($exitCode -ne 0) {
    "DEBUGGING TIPS:`n- Check python_output.txt for detailed error messages`n- Verify Python script parameters`n- Check disk space and permissions"
})
==================================================
"@

$completionSummary | Out-File $COMPLETION_SUMMARY_FILE

# Update metadata with completion info
$completionMetadata = Get-Content $RUN_METADATA_PATH | ConvertFrom-Json
$completionMetadata | Add-Member -NotePropertyName "end_time" -NotePropertyValue (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Force
$completionMetadata | Add-Member -NotePropertyName "system_end_info" -NotePropertyValue @{
    local_time = $systemEndTime.LocalTime
    utc_time = $systemEndTime.UtcTime
    uptime = $systemEndTime.Uptime.TotalSeconds
} -Force
$completionMetadata | Add-Member -NotePropertyName "process_timing" -NotePropertyValue @{
    python_start = $processStartTime
    python_end = $processEndTime
    python_duration_seconds = $processDuration.TotalSeconds
    python_duration_formatted = $processDurationFormatted
} -Force
$completionMetadata | Add-Member -NotePropertyName "exit_code" -NotePropertyValue $exitCode -Force
$completionMetadata | Add-Member -NotePropertyName "completion_status" -NotePropertyValue $completionStatus -Force
$completionMetadata | Add-Member -NotePropertyName "completion_file" -NotePropertyValue $(if ($completionFile) { $completionFile } else { $null }) -Force
$completionMetadata | ConvertTo-Json -Depth 6 | Out-File $RUN_METADATA_PATH

# ========== EMAIL NOTIFICATION ==========
$emailSent = Send-ProcessCompletionEmail `
    -Config $EMAIL_CONFIG `
    -RunFolder $RUN_FOLDER_PATH `
    -RunId $timestamp `
    -ExitCode $exitCode `
    -Duration $processDurationFormatted `
    -CompletionStatus $completionStatus `
    -CompletionFilePath $completionFile

# Update metadata with email status
$completionMetadata = Get-Content $RUN_METADATA_PATH | ConvertFrom-Json
$completionMetadata | Add-Member -NotePropertyName "email_notification" -NotePropertyValue @{
    sent = $emailSent
    sent_time = if ($emailSent) { (Get-Date -Format "yyyy-MM-dd HH:mm:ss") } else { $null }
    recipient = if ($EMAIL_CONFIG.Enabled) { $EMAIL_CONFIG.RecipientEmail } else { $null }
} -Force
$completionMetadata | ConvertTo-Json -Depth 6 | Out-File $RUN_METADATA_PATH

# ========== FINAL OUTPUT ==========
Write-Host "`n=================================================="
Write-Host "RUN COMPLETED"
Write-Host "=================================================="
Write-Host "PowerShell Script: $POWERSHELL_SCRIPT_NAME" -ForegroundColor Cyan
Write-Host "Run Folder: $RUN_FOLDER_PATH" -ForegroundColor Cyan
Write-Host "Status: $completionStatus" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Exit Code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Duration: $processDurationFormatted" -ForegroundColor Cyan
Write-Host ""
Write-Host "LOG FILES:" -ForegroundColor White
Write-Host "  • Main Log: $MAIN_LOG_FILE" -ForegroundColor Gray
Write-Host "  • Python Output: $PYTHON_OUTPUT_FILE" -ForegroundColor Gray
Write-Host "  • Completion Summary: $COMPLETION_SUMMARY_FILE" -ForegroundColor Gray
Write-Host "  • Metadata: $RUN_METADATA_PATH" -ForegroundColor Gray
Write-Host ""
Write-Host "COMPLETION FILE:" -ForegroundColor White
Write-Host "  Status: $(if ($completionFile) { "Found ✓" } else { if ($COMPLETION_FILE_CONFIG.Enabled) { "Not Found ✗" } else { "Detection Disabled" } })" -ForegroundColor $(if ($completionFile) { "Green" } else { "Yellow" })
Write-Host ""
Write-Host "EMAIL NOTIFICATION:" -ForegroundColor White
Write-Host "  Status: $(if ($EMAIL_CONFIG.Enabled) { if ($emailSent) { "Sent ✓" } else { "Failed ✗" } } else { "Disabled" })" -ForegroundColor $(if ($emailSent) { "Green" } else { if ($EMAIL_CONFIG.Enabled) { "Red" } else { "Yellow" } })
if ($EMAIL_CONFIG.Enabled -and $emailSent) {
    Write-Host "  Recipient: $($EMAIL_CONFIG.RecipientEmail)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "System time at completion: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "=================================================="

# Exit with Python's exit code
exit $exitCode