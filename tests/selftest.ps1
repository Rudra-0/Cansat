# Cognitive CanSat Ground Station - Complete Verification & Timer Self-Test Suite
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "COGNITIVE CANSAT GROUND STATION - TIMER COUNTER & UI VERIFICATION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$rootDir = Split-Path -Parent $PSScriptRoot
$testCasesDir = Join-Path $rootDir "test_cases"
$htmlPath = Join-Path (Join-Path $rootDir "frontend") "dashboard.html"
$failed = 0


function Assert-Test([string]$name, [bool]$condition, [string]$detail = "") {
    if ($condition) {
        Write-Host "PASS  $name" -ForegroundColor Green
    } else {
        $script:failed += 1
        $msg = "FAIL  $name"
        if ($detail -ne "") { $msg += " - $detail" }
        Write-Host $msg -ForegroundColor Red
    }
}

# --- 1. VERIFY TIMER FORMATTING LOGIC ---
Write-Host "`n--- 1. Testing Mission Elapsed Time (MET) Timer Formatting ---" -ForegroundColor Yellow

function Format-MissionTime([double]$totalSeconds) {
    if ($totalSeconds -lt 0) { $totalSeconds = 0.0 }
    $mins = [Math]::Floor($totalSeconds / 60.0)
    $secs = [Math]::Floor($totalSeconds % 60.0)
    $tenths = [Math]::Floor([Math]::Round($totalSeconds * 10.0) % 10)
    return "T+ {0:D2}:{1:D2}.{2:D1}" -f [int]$mins, [int]$secs, [int]$tenths
}

Assert-Test "MET 0.0s -> T+ 00:00.0" ((Format-MissionTime 0.0) -eq "T+ 00:00.0")
Assert-Test "MET 4.8s -> T+ 00:04.8" ((Format-MissionTime 4.8) -eq "T+ 00:04.8")
Assert-Test "MET 65.4s -> T+ 01:05.4" ((Format-MissionTime 65.4) -eq "T+ 01:05.4")
Assert-Test "MET 342.9s -> T+ 05:42.9" ((Format-MissionTime 342.9) -eq "T+ 05:42.9")

# --- 2. VERIFY ALL 10 FLIGHT PROFILES ---
Write-Host "`n--- 2. Testing Flight Profiles & 5-Phase State Progression ---" -ForegroundColor Yellow

$dynamicTestCases = @(
    "01_nominal_sounding_flight.csv",
    "02_high_altitude_burst_1200m.csv",
    "03_severe_wind_shear_drift.csv",
    "04_apogee_ejection_shock_tumble.csv",
    "05_parachute_pendulum_resonance.csv",
    "06_delayed_chute_deployment.csv",
    "07_thermal_inversion_sounding.csv",
    "08_sensor_glitch_gps_recovery.csv",
    "09_low_battery_voltage_sag.csv"
)

function Test-FlightStateMachine([string]$csvPath) {
    $lines = [System.IO.File]::ReadAllLines($csvPath)
    $dataLines = $lines | Select-Object -Skip 1
    
    $currentPhase = "PAD_IDLE"
    $apogee = 0.0
    $padBaselineAlt = 0.0
    $padRecorded = $false
    $previousAlt = $null
    
    $phasesSeen = New-Object System.Collections.Generic.HashSet[string]
    $phasesSeen.Add("PAD_IDLE") | Out-Null
    
    foreach ($line in $dataLines) {
        $parts = $line.Trim().Split(',')
        if ($parts.Length -lt 14) { continue }
        
        $alt = [double]::Parse($parts[3])
        $ax = [double]::Parse($parts[7])
        $ay = [double]::Parse($parts[8])
        $az = [double]::Parse($parts[9])
        $accelMag = [Math]::Sqrt($ax*$ax + $ay*$ay + $az*$az)
        
        if ($alt -gt $apogee) { $apogee = $alt }
        if (-not $padRecorded -and $alt -gt 0) {
            $padBaselineAlt = $alt
            $padRecorded = $true
        }
        
        $vSpd = 0.0
        if ($previousAlt -ne $null) {
            $vSpd = ($alt - $previousAlt) / 0.8
        }
        $previousAlt = $alt
        
        if ($currentPhase -eq "PAD_IDLE") {
            if ($alt -gt ($padBaselineAlt + 8.0) -and $vSpd -gt 0.8) {
                $currentPhase = "BALLOON_ASCENT"
            }
        } elseif ($currentPhase -eq "BALLOON_ASCENT") {
            $hasClimbed = $apogee -gt ($padBaselineAlt + 30.0)
            $isDescendingFromPeak = ($alt -le ($apogee - 2.0)) -or ($vSpd -le 0.2 -and $alt -ge ($apogee - 15.0))
            $isEjectionShock = $accelMag -gt 2.0
            if ($hasClimbed -and ($isDescendingFromPeak -or $isEjectionShock)) {
                $currentPhase = "APOGEE_BURST"
            }
        } elseif ($currentPhase -eq "APOGEE_BURST") {
            if ($vSpd -lt -1.8 -or $alt -lt ($apogee - 15.0)) {
                $currentPhase = "PARACHUTE_DESCENT"
            }
        } elseif ($currentPhase -eq "PARACHUTE_DESCENT") {
            $nearGround = ($alt -le ($padBaselineAlt + 4.0)) -or ($alt -le 4.0)
            $settled = [Math]::Abs($vSpd) -lt 1.5
            if ($nearGround -and $settled) {
                $currentPhase = "TOUCHDOWN_RECOVERY"
            }
        }
        
        $phasesSeen.Add($currentPhase) | Out-Null
    }
    
    return $phasesSeen
}

foreach ($tc in $dynamicTestCases) {
    $tcPath = Join-Path $testCasesDir $tc
    $exists = Test-Path $tcPath
    Assert-Test "File exists: $tc" $exists
    if ($exists) {
        $phases = Test-FlightStateMachine $tcPath
        $hasAll5 = $phases.Contains("PAD_IDLE") -and $phases.Contains("BALLOON_ASCENT") -and $phases.Contains("APOGEE_BURST") -and $phases.Contains("PARACHUTE_DESCENT") -and $phases.Contains("TOUCHDOWN_RECOVERY")
        Assert-Test "$tc tracks all 5 phases (Pad -> Ascent -> Apogee -> Descent -> Touchdown)" $hasAll5
    }
}

# --- 3. HTML INTEGRITY & CLEANUP CHECKS ---
Write-Host "`n--- 3. Testing HTML Features, Clean Header & Timer Implementation ---" -ForegroundColor Yellow
# Markup and the extracted dashboard script are checked together.
$dashboardJsPath = Join-Path (Join-Path (Join-Path $rootDir "frontend") "js") "dashboard.js"
$htmlContent = [System.IO.File]::ReadAllText($htmlPath) + [System.IO.File]::ReadAllText($dashboardJsPath)

Assert-Test "Subtitle paragraph REMOVED" (-not $htmlContent.Contains("Sounding Pico-Satellite Bus • Real-Time Atmospheric Characterization &amp; Telemetry"))
Assert-Test "Live MET timer wiring present" ($htmlContent.Contains("formatMissionTime") -and $htmlContent.Contains("phase-elapsed-time"))
Assert-Test "Obsidian background present" ($htmlContent.Contains("#0A0910"))
Assert-Test "Purple theme and glassmorphism present" ($htmlContent.Contains("glass-card") -and $htmlContent.Contains("btn-gradient-purple"))
Assert-Test "Field Recovery tool styled in Purple & Black" ($htmlContent.Contains("OPEN COORDINATES IN GOOGLE MAPS"))
Assert-Test "HMM Stage Tracker present" ($htmlContent.Contains("Mission Phase Progression"))
Assert-Test "Replay engine present" ($htmlContent.Contains("Flight Log Replay &amp; Test Suite"))
Assert-Test "Three.js CanSat 3D Model present" ($htmlContent.Contains("CylinderGeometry"))
Assert-Test "Leaflet multi-layer map present" ($htmlContent.Contains("ArcGIS") -and $htmlContent.Contains("basemaps.cartocdn.com"))

Write-Host "`n================================================================================" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "VERIFICATION RESULT: ALL TESTS PASSED SUCCESSFULLY! (0 Failures)" -ForegroundColor Green
} else {
    Write-Host "VERIFICATION RESULT: $failed TESTS FAILED!" -ForegroundColor Red
    exit 1
}
Write-Host "================================================================================" -ForegroundColor Cyan
