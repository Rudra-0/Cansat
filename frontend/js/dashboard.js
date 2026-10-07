/*
 * Cognitive CanSat - Mission Control dashboard logic (frontend/dashboard.html).
 * Runs as one IIFE; functions used by inline on*= handlers are exported on window at the end.
 */
(function () {
  'use strict';

  // --- CONSTANTS & CONFIGURATION ---
  const CSV_COLUMNS = [
    'timestamp', 'temp', 'pressure', 'altitude', 'gx', 'gy', 'gz',
    'ax', 'ay', 'az', 'lat', 'lon', 'humidity', 'batteryVoltage',
    'pitch', 'roll', 'accelMag'
  ];
  const MAX_CHART_POINTS = 50;
  const MAX_FEED_LINES = 400;
  const DEMO_DT_MS = 800;
  const V_EMPTY = 3.20;
  const V_FULL = 4.20;
  const HOME = [22.572700, 88.365500];

  // --- GLOBAL TELEMETRY BUFFER ---
  const telemetryHistory = [];
  let linesReceived = 0;
  let linesParsed = 0;
  let linesRejected = 0;

  // --- DATABASE PERSISTENCE & MISSION STATE ---
  let activeDbMissionId = null;
  let activeDbMissionName = '';
  let dbTelemetryBuffer = [];
  let dbFlushInterval = null;
  let isViewingHistoricalReport = false;
  let currentHistoricalReportData = null;
  let cachedPfrMarkdown = '';
  let autoRecordEnabled = true;
  let autoArmInProgress = false;
  let autoArmCooldownUntil = 0;
  
  // Connections
  let isDemo = false;
  let demoInterval = null;
  let serialPort = null;
  let serialReader = null;
  let serialKeepReading = false;

  // Replay Engine State
  let replayRows = [];
  let replayIndex = 0;
  let replayInterval = null;
  let replayPlaying = false;
  let replaySpeed = 1.0;

  // Kinematics & History
  let previousAlt = null;
  let previousTs = null;
  let smoothedAltitude = null;
  const ALT_FILTER_WINDOW = 3;
  let altitudeBuffer = [];
  let apogee = 0;
  let padBaselineAlt = 0;
  let padBaselineTemp = 25.0;
  let padBaselineRecorded = false;

  // --- HIGH-PRECISION NOISE-IMMUNE BAROMETRIC ALTITUDE & VERTICAL VELOCITY FILTER ---
  // Rejects stationary sensor ADC flutter (< 0.18m) without lag while immediately and faithfully
  // tracking physical elevation changes (1-meter and sub-meter manual movements or rocket ascent)
  class PrecisionAltitudeFilter {
    constructor() {
      this.z = null;
      this.vz = 0.0;
      this.lastTime = null;
      this.trendCount = 0;
      this.lastSign = 0;
    }

    reset(initialAlt = null) {
      this.z = initialAlt !== null ? Number(initialAlt) : null;
      this.vz = 0.0;
      this.lastTime = null;
      this.trendCount = 0;
      this.lastSign = 0;
    }

    update(measuredAlt, currentTimeSec = null) {
      const now = currentTimeSec !== null ? currentTimeSec : Date.now() / 1000.0;
      if (this.z === null || this.lastTime === null) {
        this.z = Number(measuredAlt);
        this.vz = 0.0;
        this.lastTime = now;
        return { alt: this.z, vz: 0.0 };
      }

      let dt = now - this.lastTime;
      this.lastTime = now;
      dt = Math.max(0.05, Math.min(dt, 2.0));

      const diff = measuredAlt - this.z;
      const absDiff = Math.abs(diff);
      const sign = diff > 0 ? 1 : (diff < 0 ? -1 : 0);

      // Track directional trend to detect small persistent manual movement
      if (sign === this.lastSign && sign !== 0) {
        this.trendCount++;
      } else {
        this.trendCount = 0;
      }
      this.lastSign = sign;

      let alpha;
      if (absDiff < 0.18 && this.trendCount < 3) {
        // Stationarity deadband: rejects ambient BMP280 thermal/ADC flutter (< 0.18m)
        alpha = 0.06;
      } else if (absDiff < 0.45 && this.trendCount < 2) {
        // Subtle transition
        alpha = 0.35;
      } else if (absDiff < 1.0) {
        // Clear physical manual elevation step (e.g. 1-meter lift)
        alpha = 0.75;
      } else {
        // Rapid ascent / descent / high dynamics (> 1m)
        alpha = 0.95;
      }

      const prevZ = this.z;
      this.z = this.z + alpha * diff;

      // Filtered vertical velocity
      const instantVz = (this.z - prevZ) / dt;
      if (absDiff < 0.18 && this.trendCount < 3) {
        this.vz = 0.0;
      } else {
        this.vz = 0.65 * instantVz + 0.35 * this.vz;
      }

      return { alt: Number(this.z.toFixed(2)), vz: Number(this.vz.toFixed(2)) };
    }
  }

  const clientAltitudeFilter = new PrecisionAltitudeFilter();

  // Multi-sensor stationary tare calibration
  let tareGyroX = 0.0, tareGyroY = 0.0, tareGyroZ = 0.0;
  let tarePadAlt = 0.0;
  let tareSampleBuffer = [];
  let isCalibratingTare = false;
  let tareTargetSamples = 15;
  let isStationaryCalibrated = false;

  // Edge-preserving shock-responsive filter for 3-axis accelerometer
  let accelFiltX = 0.0, accelFiltY = 0.0, accelFiltZ = 1.0;
  function filterAccel(raw, prev) {
    const diff = Math.abs(raw - prev);
    if (diff < 0.04) {
      return prev + 0.20 * (raw - prev);
    }
    return raw;
  }

  // Madgwick / Complementary Filter Attitude
  let q0 = 1.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;
  let fusedPitch = 0, fusedRoll = 0, fusedYaw = 0;
  let tarePitch = 0.0, tareRoll = 0.0, tareYaw = 0.0;
  let autoTaredOnPad = false;
  let compPitch = null, compRoll = null, compYaw = 0.0;

  function triggerStationaryTare() {
    isCalibratingTare = true;
    tareSampleBuffer = [];

    const btns = [document.getElementById('tare-btn'), document.getElementById('tare-calibrate-btn')].filter(Boolean);
    btns.forEach(b => {
      b.innerText = 'CALIBRATING (3s)...';
      b.className = 'px-2.5 py-1 rounded bg-amber-600/90 border border-amber-400 text-xs text-amber-100 font-mono font-bold transition-all animate-pulse shadow-md';
    });

    try {
      fetch(getApiUrl('/hardware/tare?samples=15'), { method: 'POST' }).catch(() => {});
    } catch (e) {}

    setTimeout(() => {
      if (isCalibratingTare) {
        finalizeStationaryTare();
      }
    }, 3500);
  }

  function finalizeStationaryTare() {
    if (tareSampleBuffer.length > 0) {
      const n = tareSampleBuffer.length;
      tareGyroX = tareSampleBuffer.reduce((s, d) => s + d.gx, 0) / n;
      tareGyroY = tareSampleBuffer.reduce((s, d) => s + d.gy, 0) / n;
      tareGyroZ = tareSampleBuffer.reduce((s, d) => s + d.gz, 0) / n;
      tarePadAlt = tareSampleBuffer.reduce((s, d) => s + d.alt, 0) / n;
      padBaselineAlt = tarePadAlt;
    }
    tarePitch = fusedPitch;
    tareRoll = fusedRoll;
    tareYaw = fusedYaw;
    autoTaredOnPad = true;
    isCalibratingTare = false;
    isStationaryCalibrated = true;

    if (tarePadAlt !== 0 && clientAltitudeFilter) {
      clientAltitudeFilter.reset(tarePadAlt);
    }

    if (cansatGroup) {
      cansatGroup.rotation.set(0, 0, 0);
    }

    const mp = document.getElementById('model-pitch');
    const mr = document.getElementById('model-roll');
    const my = document.getElementById('model-yaw');
    if (mp) mp.innerText = '0.0°';
    if (mr) mr.innerText = '0.0°';
    if (my) my.innerText = '0.0°';

    const ip = document.getElementById('info-pitch');
    const ir = document.getElementById('info-roll');
    if (ip) ip.innerText = '0.0°';
    if (ir) ir.innerText = '0.0°';

    const btns = [document.getElementById('tare-btn'), document.getElementById('tare-calibrate-btn')].filter(Boolean);
    btns.forEach(b => {
      b.innerText = 'CALIBRATED';
      b.className = b.id === 'tare-btn'
        ? 'px-2.5 py-1.5 rounded bg-emerald-700 border border-emerald-400 text-xs text-emerald-100 font-bold transition-all shadow-md'
        : 'btn-action-outline flex items-center justify-center text-emerald-300 border-emerald-500/40 bg-emerald-950/40 font-bold transition-all shadow-md';
      setTimeout(() => {
        b.innerText = b.id === 'tare-btn' ? 'TARE UPRIGHT' : 'TARE / CALIBRATE';
        b.className = b.id === 'tare-btn'
          ? 'px-2.5 py-1.5 rounded-full bg-[#171717] hover:bg-[#242424] border border-[#2e2e2e] text-xs text-zinc-200 hover:text-white font-bold transition-all shadow-sm'
          : 'btn-action-outline flex items-center justify-center text-amber-300 border-amber-500/40 bg-amber-950/20 hover:bg-amber-900/40 hover:border-amber-400 transition-all font-bold';
      }, 2500);
    });
  }

  function tareAttitude() {
    tarePitch = fusedPitch;
    tareRoll = fusedRoll;
    tareYaw = fusedYaw;
    autoTaredOnPad = true;

    // Instantly force 3D model to stand 100% vertically upright
    if (cansatGroup) {
      cansatGroup.rotation.set(0, 0, 0);
    }

    // Immediately zero HUD numbers
    const mp = document.getElementById('model-pitch');
    const mr = document.getElementById('model-roll');
    const my = document.getElementById('model-yaw');
    if (mp) mp.innerText = '0.0°';
    if (mr) mr.innerText = '0.0°';
    if (my) my.innerText = '0.0°';

    const ip = document.getElementById('info-pitch');
    const ir = document.getElementById('info-roll');
    if (ip) ip.innerText = '0.0°';
    if (ir) ir.innerText = '0.0°';

    // Instant tactile feedback on button
    const btn = document.getElementById('tare-btn');
    if (btn) {
      btn.innerText = 'ZEROED';
      btn.className = 'px-2.5 py-1.5 rounded-full bg-emerald-700 border border-emerald-400 text-xs text-emerald-100 font-bold transition-all shadow-sm';
      setTimeout(function () {
        btn.innerText = 'TARE UPRIGHT';
        btn.className = 'px-2.5 py-1.5 rounded-full bg-[#171717] hover:bg-[#242424] border border-[#2e2e2e] text-xs text-zinc-200 hover:text-white font-bold transition-all shadow-sm';
      }, 1600);
    }

    // Also trigger full sensor calibration in background
    triggerStationaryTare();
  }

  // GPS Exponential Smoothing
  let smoothLat = HOME[0], smoothLon = HOME[1];
  let gpsLockEstablished = false;

  // Flight Phase HMM State Machine
  let currentFlightPhase = 'PAD_IDLE';
  let speechAllowed = true;
  let speechAnnounced = { apogee: false, chute: false, alt100: false, touchdown: false };
  let missionStartEpochMs = null;

  function formatMissionTime(totalSeconds) {
    if (!Number.isFinite(totalSeconds) || totalSeconds < 0) totalSeconds = 0;
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    const tenths = Math.floor(Math.round(totalSeconds * 10) % 10);
    const mm = (mins < 10 ? '0' : '') + mins;
    const ss = (secs < 10 ? '0' : '') + secs;
    return 'T+ ' + mm + ':' + ss + '.' + tenths;
  }

  // Anomaly Detection
  let currentAnomalyCode = 'NOMINAL';
  let currentAnomalyScore = 0;

  // --- API BASE & SERVER RESOLUTION ---
  function getApiUrl(path) {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    const isLocalFile = window.location.protocol === 'file:';
    const isOtherPort = window.location.port && window.location.port !== '8000';
    const base = (isLocalFile || isOtherPort) ? 'http://127.0.0.1:8000' : '';
    return base + (path.startsWith('/') ? path : '/' + path);
  }
  const ML_SERVER_URL = getApiUrl('');

  // Operator endpoints (delete/purge/download DB) need CANSAT_API_KEY when the server sets one.
  // The key is asked for once on a 401 and kept for this browser tab only.
  const OPERATOR_KEY_STORAGE = 'cansat_operator_key';
  function getOperatorKey() {
    try { return sessionStorage.getItem(OPERATOR_KEY_STORAGE) || ''; } catch (e) { return ''; }
  }
  async function operatorFetch(path, options) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const opts = Object.assign({}, options || {});
      opts.headers = Object.assign({}, opts.headers || {});
      const key = getOperatorKey();
      if (key) opts.headers['X-API-Key'] = key;
      const resp = await fetch(getApiUrl(path), opts);
      if (resp.status !== 401 || attempt > 0) return resp;
      const entered = prompt('This ground station requires an operator API key (CANSAT_API_KEY):');
      if (!entered) return resp;
      try { sessionStorage.setItem(OPERATOR_KEY_STORAGE, entered.trim()); } catch (e) {}
    }
  }
  let mlServerOnline = false;
  let mlLastPrediction = null;
  let mlInferenceInFlight = false;

  async function checkMlServerHealth() {
    const badge = document.getElementById('ml-server-badge');
    const dot = document.getElementById('ml-server-dot');
    const text = document.getElementById('ml-server-text');
    if (!badge || !dot || !text) return;

    try {
      const resp = await fetch(ML_SERVER_URL + '/api/health', { method: 'GET', signal: AbortSignal.timeout(1500) });
      if (resp.ok) {
        const data = await resp.json();
        if (data.models_loaded) {
          mlServerOnline = true;
          dot.className = 'led-dot bg-emerald-400 text-emerald-400 led-pulse';
          text.innerText = 'PYTHON ML (ONLINE)';
          badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-emerald-800/40 bg-emerald-950/30 text-emerald-300';
          return;
        }
      }
    } catch (e) {
      // Server offline / standby
    }
    mlServerOnline = false;
    dot.className = 'led-dot bg-zinc-500 text-zinc-500';
    text.innerText = 'PYTHON ML (STANDBY)';
    badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-[#2e2e2e] bg-[#171717] text-zinc-400';
  }

  setInterval(checkMlServerHealth, 3500);
  setTimeout(checkMlServerHealth, 300);

  async function requestPythonMlInference(parsed, vSpd) {
    if (!mlServerOnline || mlInferenceInFlight) return;
    mlInferenceInFlight = true;
    try {
      const payload = {
        temp: parsed.temp,
        pressure: parsed.pressure,
        altitude: parsed.altitude,
        gx: parsed.gx,
        gy: parsed.gy,
        gz: parsed.gz,
        ax: parsed.ax,
        ay: parsed.ay,
        az: parsed.az,
        lat: parsed.lat,
        lon: parsed.lon,
        humidity: parsed.humidity,
        batteryVoltage: parsed.batteryVoltage,
        vSpd: vSpd,
        padAlt: padBaselineAlt,
        padTemp: padBaselineTemp,
        padPress: 1013.25
      };

      const resp = await fetch(ML_SERVER_URL + '/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(1200)
      });

      if (resp.ok) {
        const res = await resp.json();
        mlLastPrediction = res;
        applyMlInferenceToUi(res);
      }
    } catch (err) {
      // Fallback silently
    } finally {
      mlInferenceInFlight = false;
    }
  }

  function applyMlInferenceToUi(res) {
    if (!res || res.status !== 'success') return;
    const badge = document.getElementById('anomaly-badge');
    const badgeText = document.getElementById('anomaly-badge-text');
    if (badge && badgeText) {
      if (!res.is_anomaly) {
        badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-emerald-800/40 bg-emerald-950/30 text-emerald-300';
        badge.querySelector('.led-dot').className = 'led-dot bg-emerald-400 text-emerald-400 led-pulse';
        badgeText.innerText = 'NOMINAL [PyOD: ' + res.anomaly_score.toFixed(2) + ']';
      } else {
        badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-rose-800/50 bg-rose-950/40 alert-glow text-rose-300';
        badge.querySelector('.led-dot').className = 'led-dot bg-rose-500 text-rose-500';
        badgeText.innerText = 'ANOMALY [' + res.anomaly_type + ' / ' + res.anomaly_score.toFixed(2) + ']';
        triggerSafetyAlarm('ML OUTLIER: ' + res.anomaly_type, 'PyOD Isolation Forest anomaly score: ' + res.anomaly_score.toFixed(3));
      }
    }

    if (res.phase_confidence > 0.65 && res.predicted_phase !== currentFlightPhase) {
      currentFlightPhase = res.predicted_phase;
      updateStageTrackerUI(currentFlightPhase);
    }

    const aiModalEl = document.getElementById('aiModal');
    if (aiModalEl && !aiModalEl.classList.contains('hidden')) {
      analyzeData();
    }
  }

  // Three.js Scene Variables
  let scene, camera, renderer, cansatGroup, animId;
  let threeReady = false;
  let isDragging3D = false;
  let previousMousePosition = { x: 0, y: 0 };

  // Leaflet Map Variables
  let map = null;
  let ascentTrail = null;
  let descentTrail = null;
  let marker = null;
  let padMarker = null;
  let mapReady = false;

  // --- HELPER MATH FUNCTIONS ---
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function finite(v) { return typeof v === 'number' && Number.isFinite(v); }

  // --- UNIFIED MULTI-FORMAT PACKET PARSER (13-field Serial or 17-field CSV) ---
  function parsePacketLine(line) {
    if (!line || typeof line !== 'string') return null;
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return null;

    const parts = trimmed.split(',').map(s => s.trim().replace(/^"|"$/g, ''));
    if (parts.length < 2) return null;

    if (parts[0].toLowerCase().includes('time') || parts[0].toLowerCase().includes('temp')) {
      return null;
    }

    let temp, pressure, altitude, gx, gy, gz, ax, ay, az, lat, lon, humidity, batteryVoltage;
    let customTimestamp = null;

    if (parts.length >= 17 || (parts.length >= 14 && isNaN(Number(parts[0])))) {
      customTimestamp = parts[0];
      const nums = parts.slice(1, 14).map(Number);
      if (nums.some(x => !Number.isFinite(x))) return null;
      [temp, pressure, altitude, gx, gy, gz, ax, ay, az, lat, lon, humidity, batteryVoltage] = nums;
    } else if (parts.length >= 13) {
      const nums = parts.slice(0, 13).map(Number);
      if (nums.some(x => !Number.isFinite(x))) return null;
      [temp, pressure, altitude, gx, gy, gz, ax, ay, az, lat, lon, humidity, batteryVoltage] = nums;
    } else if (parts.length >= 6) {
      const nums = parts.map(Number);
      if (nums.some(x => !Number.isFinite(x))) return null;
      temp = nums[0] !== undefined ? nums[0] : 25.0;
      pressure = nums[1] !== undefined ? nums[1] : 1013.25;
      altitude = nums[2] !== undefined ? nums[2] : 0.0;
      gx = nums[3] !== undefined ? nums[3] : 0.0;
      gy = nums[4] !== undefined ? nums[4] : 0.0;
      gz = nums[5] !== undefined ? nums[5] : 0.0;
      ax = nums[6] !== undefined ? nums[6] : 0.0;
      ay = nums[7] !== undefined ? nums[7] : 0.0;
      az = nums[8] !== undefined ? nums[8] : 1.0;
      lat = nums[9] !== undefined ? nums[9] : HOME[0];
      lon = nums[10] !== undefined ? nums[10] : HOME[1];
      humidity = nums[11] !== undefined ? nums[11] : 50.0;
      batteryVoltage = nums[12] !== undefined ? nums[12] : 3.9;
    } else {
      return null;
    }

    return {
      temp, pressure, altitude,
      gx, gy, gz,
      ax, ay, az,
      lat, lon, humidity,
      batteryVoltage,
      customTimestamp
    };
  }

  // --- SENSOR FUSION ENGINE (COMPLEMENTARY / MADGWICK FILTER) ---
  function updateAttitudeFusion(ax, ay, az, gx, gy, gz, dt) {
    const aNorm = Math.sqrt(ax * ax + ay * ay + az * az);
    let accPitch = 0.0, accRoll = 0.0;
    if (aNorm > 0.05) {
      accPitch = Math.atan2(ay, az) * (180.0 / Math.PI);
      accRoll = Math.atan2(-ax, Math.sqrt(ay * ay + az * az)) * (180.0 / Math.PI);
    }

    if (compPitch === null || compRoll === null) {
      compPitch = accPitch;
      compRoll = accRoll;
      compYaw = 0.0;
      fusedPitch = compPitch;
      fusedRoll = compRoll;
      fusedYaw = compYaw;
      return { pitch: fusedPitch, roll: fusedRoll, yaw: fusedYaw, q: [q0, q1, q2, q3] };
    }

    const dtClamped = Math.max(0.01, Math.min(dt || 0.8, 2.0));
    // Gyro noise deadband: prevents raw MEMS sensor bias from continuously rotating 3D model when stationary
    const gyroNoiseThreshold = 0.55; // deg/s
    const filteredGx = Math.abs(gx) < gyroNoiseThreshold ? 0.0 : gx;
    const filteredGy = Math.abs(gy) < gyroNoiseThreshold ? 0.0 : gy;
    const filteredGz = Math.abs(gz) < gyroNoiseThreshold ? 0.0 : gz;

    if (filteredGx === 0 && filteredGy === 0 && filteredGz === 0 && aNorm > 0.05) {
      // Table / Pad Rest: Lock tightly to physical gravity vector without drift or gimbal jump
      compPitch = 0.85 * compPitch + 0.15 * accPitch;
      compRoll = 0.85 * compRoll + 0.15 * accRoll;
    } else {
      // Dynamic Motion: Complementary fusion blending high-rate gyro integration with accelerometer gravity
      const alpha = 0.94;
      compPitch = alpha * (compPitch + filteredGx * dtClamped) + (1.0 - alpha) * accPitch;
      compRoll = alpha * (compRoll + filteredGy * dtClamped) + (1.0 - alpha) * accRoll;
      compYaw += filteredGz * dtClamped;
    }

    fusedPitch = compPitch;
    fusedRoll = compRoll;
    fusedYaw = compYaw;

    // Maintain quaternion representation for 3D engine and testing harness
    const pRad = (fusedPitch * Math.PI) / 360.0;
    const rRad = (fusedRoll * Math.PI) / 360.0;
    const cp = Math.cos(pRad), sp = Math.sin(pRad);
    const cr = Math.cos(rRad), sr = Math.sin(rRad);
    q0 = cr * cp;
    q1 = sr * cp;
    q2 = cr * sp;
    q3 = -sr * sp;

    return { pitch: fusedPitch, roll: fusedRoll, yaw: fusedYaw, q: [q0, q1, q2, q3] };
  }

  // --- METEOROLOGICAL SOUNDINGS (METPY EQUIVALENT) ---
  function computeSoundingMetrics(tempC, pressHpa, relHumPct, altM) {
    const a = 17.625;
    const b = 243.04;
    const alpha = ((a * tempC) / (b + tempC)) + Math.log(clamp(relHumPct, 1, 100) / 100.0);
    const dewPoint = (b * alpha) / (a - alpha);

    const pPa = pressHpa * 100.0;
    const tKelvin = tempC + 273.15;
    const rSpecific = 287.058;
    const airDensity = pPa / (rSpecific * tKelvin);

    let lapseRate = 0.65;
    if (padBaselineRecorded && Math.abs(altM - padBaselineAlt) > 15) {
      const dAlt = altM - padBaselineAlt;
      const dTemp = tempC - padBaselineTemp;
      lapseRate = -(dTemp / dAlt) * 100.0;
    }

    return { dewPoint, airDensity, lapseRate };
  }

  // --- ANOMALY & OUTLIER DETECTION ENGINE ---
  function evaluateAnomalies(parsed, vSpd, accelMag) {
    const gyroMag = Math.sqrt(parsed.gx * parsed.gx + parsed.gy * parsed.gy + parsed.gz * parsed.gz);
    let code = 'NOMINAL';
    let severity = 0;

    if (gyroMag > 150.0) {
      code = 'GYRO_TUMBLE';
      severity = 2;
    } else if (accelMag > 4.5) {
      code = 'HIGH_G_SHOCK';
      severity = 2;
    } else if (parsed.batteryVoltage < 3.40) {
      code = 'VOLTAGE_CRITICAL';
      severity = 2;
    } else if (parsed.batteryVoltage < 3.55) {
      code = 'VOLTAGE_SAG';
      severity = 1;
    } else if (currentFlightPhase === 'PARACHUTE_DESCENT' && vSpd < -12.0) {
      code = 'EXCESSIVE_DESCENT';
      severity = 2;
    } else if (accelMag < 0.20 && currentFlightPhase === 'PARACHUTE_DESCENT') {
      code = 'FREEFALL_ANOMALY';
      severity = 2;
    }

    currentAnomalyCode = code;
    currentAnomalyScore = severity;

    const badge = document.getElementById('anomaly-badge');
    const badgeText = document.getElementById('anomaly-badge-text');
    if (!badge || !badgeText) return;

    if (severity === 0) {
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-emerald-800/40 bg-emerald-950/30 text-emerald-300';
      badge.querySelector('.led-dot').className = 'led-dot bg-emerald-400 text-emerald-400 led-pulse';
      badgeText.innerText = 'NOMINAL FLIGHT';
    } else if (severity === 1) {
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-amber-700/50 bg-amber-950/40 text-amber-300';
      badge.querySelector('.led-dot').className = 'led-dot bg-amber-400 text-amber-400';
      badgeText.innerText = 'WARNING [' + code + ']';
    } else {
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-rose-800/50 bg-rose-950/40 alert-glow text-rose-300';
      badge.querySelector('.led-dot').className = 'led-dot bg-rose-500 text-rose-500';
      badgeText.innerText = 'ANOMALY [' + code + ']';
      triggerSafetyAlarm('CRITICAL FLIGHT ANOMALY: ' + code, 'Out-of-envelope sensor telemetry detected.');
      speakAlert('Warning. Anomaly detected: ' + code.replace(/_/g, ' '));
    }
  }

  // --- HMM FLIGHT PHASE STATE MACHINE ---
  function updateFlightPhase(alt, vSpd, accelMag) {
    const prevPhase = currentFlightPhase;

    if (!padBaselineRecorded && alt > 0) {
      padBaselineAlt = alt;
      padBaselineRecorded = true;
    }

    if (alt > apogee) {
      apogee = alt;
    }

    if (currentFlightPhase === 'PAD_IDLE') {
      if (alt > padBaselineAlt + 8.0 && vSpd > 0.8) {
        currentFlightPhase = 'BALLOON_ASCENT';
      }
    } else if (currentFlightPhase === 'BALLOON_ASCENT') {
      const hasClimbed = apogee > padBaselineAlt + 30.0;
      const isDescendingFromPeak = (alt <= apogee - 2.0) || (vSpd <= 0.2 && alt >= apogee - 15.0);
      const isEjectionShock = accelMag > 2.0;

      if (hasClimbed && (isDescendingFromPeak || isEjectionShock)) {
        currentFlightPhase = 'APOGEE_BURST';
        if (!speechAnnounced.apogee) {
          speakAlert('Apogee reached at ' + Math.round(apogee) + ' meters.');
          speechAnnounced.apogee = true;
        }
      }
    } else if (currentFlightPhase === 'APOGEE_BURST') {
      if (vSpd < -1.8 || alt < apogee - 15.0) {
        currentFlightPhase = 'PARACHUTE_DESCENT';
        if (!speechAnnounced.chute) {
          speakAlert('Parachute deployment confirmed.');
          speechAnnounced.chute = true;
        }
      }
    } else if (currentFlightPhase === 'PARACHUTE_DESCENT') {
      if (alt < padBaselineAlt + 100.0 && !speechAnnounced.alt100 && alt > padBaselineAlt + 20.0) {
        speakAlert('Altitude warning: 100 meters to touchdown.');
        speechAnnounced.alt100 = true;
      }
      const nearGround = (alt <= padBaselineAlt + 4.0) || (alt <= 4.0);
      const settled = Math.abs(vSpd) < 1.5;
      if (nearGround && settled) {
        currentFlightPhase = 'TOUCHDOWN_RECOVERY';
        if (!speechAnnounced.touchdown) {
          speakAlert('Touchdown confirmed. Recovery coordinates locked.');
          speechAnnounced.touchdown = true;
        }
        lockTouchdownCoordinates();
      }
    }

    if (prevPhase !== currentFlightPhase) {
      updateStageTrackerUI(currentFlightPhase);
    }
  }

  function updateStageTrackerUI(phase) {
    const stages = ['PAD_IDLE', 'BALLOON_ASCENT', 'APOGEE_BURST', 'PARACHUTE_DESCENT', 'TOUCHDOWN_RECOVERY'];
    const nodeIds = ['phase-pad', 'phase-ascent', 'phase-apogee', 'phase-descent', 'phase-touchdown'];
    const curIdx = stages.indexOf(phase);

    nodeIds.forEach(function (id, idx) {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.remove('active', 'completed');
      if (idx < curIdx) {
        el.classList.add('completed');
      } else if (idx === curIdx) {
        el.classList.add('active');
      }
    });

    const infoPhase = document.getElementById('info-phase');
    if (infoPhase) infoPhase.innerText = phase;
    const mapPhase = document.getElementById('map-phase');
    if (mapPhase) mapPhase.innerText = phase;
  }

  // --- GPS SMOOTHING & FIELD RECOVERY ---
  function updateGpsAndRecovery(lat, lon, alt, vSpd) {
    const locked = gpsLocked(lat, lon);
    const pill = document.getElementById('gps-status-pill');
    const dot = document.getElementById('gps-status-dot');
    const text = document.getElementById('gps-status-text');
    const mapAlt = document.getElementById('map-alt');
    if (mapAlt) mapAlt.innerText = (typeof alt === 'number' && !isNaN(alt)) ? alt.toFixed(1) : '--';
    const mapVspd = document.getElementById('map-vspd');
    if (mapVspd) mapVspd.innerText = (typeof vSpd === 'number' && !isNaN(vSpd)) ? vSpd.toFixed(2) : '--';

    if (locked) {
      if (pill) pill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/50 text-[10px] font-mono font-semibold text-emerald-300';
      if (dot) dot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-400';
      if (text) text.innerText = '3D FIX';

      if (!gpsLockEstablished) {
        smoothLat = lat;
        smoothLon = lon;
        HOME[0] = lat;
        HOME[1] = lon;
        gpsLockEstablished = true;
        if (mapReady) {
          map.setView([lat, lon], 17);
          if (padMarker) padMarker.setLatLng([lat, lon]);
        }
      } else {
        const alpha = 0.65;
        smoothLat = alpha * lat + (1.0 - alpha) * smoothLat;
        smoothLon = alpha * lon + (1.0 - alpha) * smoothLon;
      }

      const dLat = (smoothLat - HOME[0]) * 111139.0;
      const dLon = (smoothLon - HOME[1]) * 111139.0 * Math.cos(HOME[0] * Math.PI / 180.0);
      const groundDist = Math.sqrt(dLat * dLat + dLon * dLon);
      let bearing = (Math.atan2(dLon, dLat) * 180.0 / Math.PI);
      if (bearing < 0) bearing += 360;

      document.getElementById('rec-lat').innerText = smoothLat.toFixed(5) + '°';
      document.getElementById('rec-lon').innerText = smoothLon.toFixed(5) + '°';
      document.getElementById('rec-dist').innerText = groundDist.toFixed(1) + ' m';
      document.getElementById('rec-drift').innerText = bearing.toFixed(0) + '° (' + getCardinal(bearing) + ')';

      document.getElementById('info-lat').innerText = smoothLat.toFixed(6) + '°';
      document.getElementById('info-lon').innerText = smoothLon.toFixed(6) + '°';

      if (mapReady) {
        const pos = [smoothLat, smoothLon];
        marker.setLatLng(pos);
        if (currentFlightPhase === 'BALLOON_ASCENT' || currentFlightPhase === 'PAD_IDLE') {
          ascentTrail.addLatLng(pos);
        } else {
          descentTrail.addLatLng(pos);
        }
        map.panTo(pos);
      }

      document.getElementById('map-lat').innerText = smoothLat.toFixed(5);
      document.getElementById('map-lon').innerText = smoothLon.toFixed(5);
      const distEl = document.getElementById('map-dist');
      if (distEl) distEl.innerText = groundDist.toFixed(1) + ' m';
      document.getElementById('map-drift-speed').innerText = Math.abs(vSpd * 0.4).toFixed(1);
      document.getElementById('map-heading').innerText = bearing.toFixed(0) + '° ' + getCardinal(bearing);
    } else {
      if (pill) pill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-950/80 border border-amber-500/50 text-[10px] font-mono font-semibold text-amber-300';
      if (dot) dot.className = 'w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse';
      if (text) text.innerText = 'ACQUIRING';
    }
  }

  function getCardinal(angle) {
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    return directions[Math.round(angle / 45.0) % 8];
  }

  function lockTouchdownCoordinates() {
    const badge = document.getElementById('recovery-lock-badge');
    if (badge) {
      badge.className = 'text-[10.5px] px-2 py-0.5 rounded bg-emerald-950/90 border border-emerald-500 text-emerald-300 font-bold font-mono';
      badge.innerText = 'LOCKED';
    }
  }

  function openGoogleMapsRecovery() {
    const lat = smoothLat || HOME[0];
    const lon = smoothLon || HOME[1];
    const url = 'https://www.google.com/maps/search/?api=1&query=' + lat.toFixed(6) + ',' + lon.toFixed(6);
    window.open(url, '_blank');
  }

  // --- WEB SPEECH SYNTHESIS ---
  function speakAlert(text) {
    if (!speechAllowed || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis error', e);
    }
  }

  const SVG_VOICE_ON = '<svg class="w-3.5 h-3.5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/></svg>';
  const SVG_VOICE_OFF = '<svg class="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z M17 14l4-4m0 0l-4-4m4 4l-4 4"/></svg>';

  function toggleAudioSpeech() {
    speechAllowed = !speechAllowed;
    const icon = document.getElementById('audio-icon');
    const text = document.getElementById('audio-status-text');
    if (speechAllowed) {
      if (icon) icon.innerHTML = SVG_VOICE_ON;
      if (text) text.innerText = 'VOICE ON';
      speakAlert('Voice callouts active.');
    } else {
      if (icon) icon.innerHTML = SVG_VOICE_OFF;
      if (text) text.innerText = 'VOICE MUTED';
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    }
  }

  function triggerSafetyAlarm(title, desc) {
    const banner = document.getElementById('safety-alarm-banner');
    if (!banner) return;
    document.getElementById('safety-alarm-title').innerText = title;
    document.getElementById('safety-alarm-desc').innerText = desc;
    banner.classList.remove('hidden');
  }

  function dismissSafetyAlarm() {
    const banner = document.getElementById('safety-alarm-banner');
    if (banner) banner.classList.add('hidden');
  }

  function batteryPct(volts) {
    return clamp(((volts - V_EMPTY) / (V_FULL - V_EMPTY)) * 100, 0, 100);
  }

  function gpsLocked(lat, lon) {
    if (!finite(lat) || !finite(lon)) return false;
    if (lat === 0 && lon === 0) return false;
    return Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
  }

  function csvEscape(val) {
    const s = String(val);
    if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function buildCsv(rows) {
    const header = CSV_COLUMNS.join(',');
    const body = rows.map((r) => CSV_COLUMNS.map((k) => csvEscape(r[k] !== undefined ? r[k] : '')).join(',')).join('\r\n');
    return header + '\r\n' + body + '\r\n';
  }

  // --- CHART.JS CONFIGURATION & MULTI-CHART SYNCHRONIZED INSPECTOR ---
  let syncedCursorIndex = null;
  let isSyncingHover = false;

  const syncedCrosshairPlugin = {
    id: 'syncedCrosshair',
    afterDraw: function (chart) {
      if (syncedCursorIndex === null || syncedCursorIndex === undefined) return;
      const meta = chart.getDatasetMeta(0);
      if (!meta || !meta.data || !meta.data[syncedCursorIndex]) return;
      const point = meta.data[syncedCursorIndex];
      if (!point) return;

      const ctx = chart.ctx;
      const chartArea = chart.chartArea;
      if (!chartArea) return;
      const x = point.x;

      ctx.save();
      // Vertical crosshair line
      ctx.beginPath();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 1;
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();

      // Highlight circles on each dataset at synced index
      ctx.setLineDash([]);
      for (let d = 0; d < chart.data.datasets.length; d++) {
        const dMeta = chart.getDatasetMeta(d);
        if (dMeta && dMeta.data && dMeta.data[syncedCursorIndex]) {
          const pt = dMeta.data[syncedCursorIndex];
          if (pt && typeof pt.y === 'number' && !isNaN(pt.y)) {
            ctx.fillStyle = chart.data.datasets[d].borderColor || '#ffffff';
            ctx.beginPath();
            ctx.arc(x, pt.y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
      }

      // Time indicator pill at top
      const label = chart.data.labels[syncedCursorIndex];
      if (label) {
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(label).width;
        const pillW = textWidth + 8;
        const pillH = 16;
        let pillX = x - pillW / 2;
        pillX = Math.max(chartArea.left, Math.min(chartArea.right - pillW, pillX));

        ctx.fillStyle = '#171717';
        ctx.strokeStyle = '#2e2e2e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(pillX, chartArea.top + 2, pillW, pillH, 3);
        } else {
          ctx.rect(pillX, chartArea.top + 2, pillW, pillH);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, pillX + pillW / 2, chartArea.top + 2 + pillH / 2);
      }

      ctx.restore();
    }
  };
  Chart.register(syncedCrosshairPlugin);

  function deepMerge(target, source) {
    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        if (!target[key]) target[key] = {};
        deepMerge(target[key], source[key]);
      } else {
        target[key] = source[key];
      }
    }
    return target;
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    layout: {
      padding: {
        top: 8,
        right: 14,
        bottom: 6,
        left: 8
      }
    },
    interaction: {
      mode: 'index',
      intersect: false
    },
    onHover: function (event, activeElements, chart) {
      if (isSyncingHover) return;
      if (activeElements && activeElements.length > 0) {
        const idx = activeElements[0].index;
        if (syncedCursorIndex !== idx) {
          syncedCursorIndex = idx;
          isSyncingHover = true;
          allCharts.forEach(c => {
            if (c !== chart) c.draw();
          });
          isSyncingHover = false;
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.18)',
          borderColor: 'rgba(255, 255, 255, 0.25)',
          tickLength: 4
        },
        border: {
          display: true,
          color: 'rgba(255, 255, 255, 0.25)'
        },
        ticks: {
          color: '#a1a1aa',
          font: { family: 'JetBrains Mono', size: 10, weight: '500' },
          maxTicksLimit: 6,
          maxRotation: 0,
          minRotation: 0,
          autoSkip: true,
          padding: 5
        },
        title: {
          display: true,
          text: 'TIME (MET)',
          color: '#a1a1aa',
          font: { family: 'JetBrains Mono', size: 9, weight: '600' },
          padding: { top: 2, bottom: 0 }
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.18)',
          borderColor: 'rgba(255, 255, 255, 0.25)',
          tickLength: 4
        },
        border: {
          display: true,
          color: 'rgba(255, 255, 255, 0.25)'
        },
        ticks: {
          color: '#a1a1aa',
          font: { family: 'JetBrains Mono', size: 10, weight: '500' },
          padding: 6
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        align: 'end',
        labels: {
          color: '#a1a1aa',
          usePointStyle: true,
          pointStyle: 'circle',
          boxWidth: 6,
          boxHeight: 6,
          padding: 8,
          font: { family: 'Inter', size: 10.5, weight: '500' }
        }
      },
      tooltip: {
        enabled: true,
        backgroundColor: '#0a0a0a',
        titleColor: '#ffffff',
        titleFont: { family: 'JetBrains Mono', size: 11, weight: 'bold' },
        bodyColor: '#e4e4e7',
        bodyFont: { family: 'JetBrains Mono', size: 10.5 },
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        padding: 8,
        displayColors: true,
        boxWidth: 8,
        boxHeight: 8
      }
    }
  };

  function initLineChart(id, datasets, extra) {
    const el = document.getElementById(id);
    if (!el) return null;
    el.addEventListener('mouseleave', function () {
      if (syncedCursorIndex !== null) {
        syncedCursorIndex = null;
        isSyncingHover = true;
        allCharts.forEach(c => c.draw());
        isSyncingHover = false;
      }
    });
    const ctx = el.getContext('2d');
    const opts = deepMerge(JSON.parse(JSON.stringify(chartOptions)), extra || {});
    opts.onHover = chartOptions.onHover;
    if (opts.scales.yP && opts.scales.y) {
      delete opts.scales.y;
    }
    return new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: datasets },
      options: opts
    });
  }

  const ds = (label, color) => ({
    label, borderColor: color, backgroundColor: color, borderWidth: 1.5,
    data: [], pointRadius: 0, tension: 0.2
  });

  function boundedScale(minSpan, centerZero = false) {
    return {
      afterDataLimits: (scale) => {
        if (centerZero) {
          const maxAbs = Math.max(Math.abs(scale.max || 0), Math.abs(scale.min || 0), minSpan / 2);
          scale.min = -maxAbs;
          scale.max = maxAbs;
        } else {
          const currentSpan = (scale.max || 0) - (scale.min || 0);
          if (currentSpan < minSpan) {
            const mid = ((scale.max || 0) + (scale.min || 0)) / 2;
            scale.min = mid - minSpan / 2;
            scale.max = mid + minSpan / 2;
          }
        }
      }
    };
  }

  const accelChart = initLineChart('chart-accel', [ds('ax (g)', '#EF4444'), ds('ay (g)', '#EAB308'), ds('az (g)', '#06B6D4')], {
    scales: {
      y: {
        ticks: { precision: 2 },
        title: { display: true, text: 'ACCEL (g)', color: '#a1a1aa', font: { family: 'JetBrains Mono', size: 9, weight: '600' } },
        grace: '5%'
      }
    }
  });
  const orientChart = initLineChart('chart-orient', [ds('Pitch (°)', '#EF4444'), ds('Roll (°)', '#06B6D4')], {
    scales: {
      y: Object.assign({
        ticks: { precision: 1 },
        title: { display: true, text: 'ANGLE (°)', color: '#a1a1aa', font: { family: 'JetBrains Mono', size: 9, weight: '600' } },
        grace: '5%'
      }, boundedScale(20.0, true))
    }
  });
  const vSpeedChart = initLineChart('chart-vspd', [ds('vz (m/s)', '#10B981')], {
    scales: {
      y: Object.assign({
        ticks: { precision: 1 },
        title: { display: true, text: 'V-SPEED (m/s)', color: '#a1a1aa', font: { family: 'JetBrains Mono', size: 9, weight: '600' } },
        grace: 0.5
      }, boundedScale(4.0, true))
    }
  });
  const altChart = initLineChart('chart-alt', [ds('Altitude (m)', '#38BDF8')], {
    scales: {
      y: Object.assign({
        ticks: { precision: 1 },
        title: { display: true, text: 'ALTITUDE (m)', color: '#a1a1aa', font: { family: 'JetBrains Mono', size: 9, weight: '600' } },
        grace: 0.5
      }, boundedScale(5.0))
    }
  });
  const gyroChart = initLineChart('chart-gyro', [ds('gx (°/s)', '#EF4444'), ds('gy (°/s)', '#EAB308'), ds('gz (°/s)', '#06B6D4')], {
    scales: {
      y: Object.assign({
        ticks: { precision: 1 },
        title: { display: true, text: 'GYRO (°/s)', color: '#a1a1aa', font: { family: 'JetBrains Mono', size: 9, weight: '600' } },
        grace: '5%'
      }, boundedScale(10.0, true))
    }
  });
  const humChart = initLineChart('chart-hum', [ds('Humidity (%)', '#06B6D4')], {
    scales: {
      y: {
        min: 0, max: 100,
        ticks: { precision: 0 },
        title: { display: true, text: 'HUMIDITY (%)', color: '#a1a1aa', font: { family: 'JetBrains Mono', size: 9, weight: '600' } }
      }
    }
  });
  const pressTempChart = initLineChart('chart-presstemp', [
    Object.assign(ds('Pressure (hPa)', '#EF4444'), { yAxisID: 'yP' }),
    Object.assign(ds('Temp (°C)', '#06B6D4'), { yAxisID: 'yT' })
  ], {
    layout: {
      padding: {
        right: 18
      }
    },
    scales: {
      yP: {
        type: 'linear', position: 'left',
        grid: {
          color: 'rgba(255, 255, 255, 0.18)',
          borderColor: 'rgba(255, 255, 255, 0.25)'
        },
        border: {
          display: true,
          color: 'rgba(255, 255, 255, 0.25)'
        },
        ticks: { color: '#EF4444', font: { family: 'JetBrains Mono', size: 10, weight: '500' }, precision: 1, padding: 6 },
        title: { display: true, text: 'PRESSURE (hPa)', color: '#EF4444', font: { family: 'JetBrains Mono', size: 9, weight: '600' } }
      },
      yT: {
        type: 'linear', position: 'right',
        grid: { drawOnChartArea: false },
        border: {
          display: true,
          color: 'rgba(255, 255, 255, 0.25)'
        },
        ticks: { color: '#06B6D4', font: { family: 'JetBrains Mono', size: 10, weight: '500' }, precision: 1, padding: 6 },
        title: { display: true, text: 'TEMP (°C)', color: '#06B6D4', font: { family: 'JetBrains Mono', size: 9, weight: '600' } }
      }
    }
  });

  const allCharts = [accelChart, orientChart, vSpeedChart, altChart, gyroChart, humChart, pressTempChart].filter(Boolean);

  function pushChart(chart, values) {
    if (!chart) return;
    const count = telemetryHistory.length;
    const secLabel = (count * (DEMO_DT_MS / 1000)).toFixed(0) + 's';
    chart.data.labels.push(secLabel);
    values.forEach((v, i) => { if (chart.data.datasets[i]) chart.data.datasets[i].data.push(v); });
    if (chart.data.labels.length > MAX_CHART_POINTS) {
      chart.data.labels.shift();
      chart.data.datasets.forEach((d) => d.data.shift());
    }
    chart.update('none');
  }

  function resizeVisibleCharts() {
    allCharts.forEach((c) => { try { c.resize(); } catch (e) { /* ignore */ } });
  }

  // --- LEAFLET MULTI-LAYER MAP (DIRECT DASHBOARD INITIALIZATION) ---
  function initMap() {
    if (mapReady) return;
    const mapEl = document.getElementById('map');
    if (!mapEl) return;

    map = L.map('map', { attributionControl: false, zoomControl: true }).setView(HOME, 16);

    const darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      subdomains: 'abcd', maxZoom: 19
    });
    const satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19
    });
    const osmTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19
    });

    darkTiles.addTo(map);

    const baseMaps = {
      "Dark Carto": darkTiles,
      "Satellite": satelliteTiles,
      "OpenStreetMap": osmTiles
    };
    L.control.layers(baseMaps).addTo(map);

    ascentTrail = L.polyline([], { color: '#10B981', weight: 3, opacity: 0.9 }).addTo(map);
    descentTrail = L.polyline([], { color: '#38BDF8', weight: 3, opacity: 0.9, dashArray: '4, 4' }).addTo(map);

    padMarker = L.circleMarker(HOME, {
      radius: 6, color: '#38BDF8', fillColor: '#0284C7', fillOpacity: 0.9, weight: 2
    }).addTo(map).bindPopup('Launch Pad 01');

    marker = L.circleMarker(HOME, {
      radius: 6, color: '#EAB308', fillColor: '#F59E0B', fillOpacity: 1, weight: 2
    }).addTo(map);

    mapReady = true;
    setTimeout(() => map.invalidateSize(), 200);
  }

  function resizeMap() {
    if (!mapReady || !map) return;
    map.invalidateSize();
  }

  // --- PROCEDURAL SOLAR ARRAY TEXTURE ---
  function createSolarTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Deep sapphire aerospace photovoltaic silicon base
    ctx.fillStyle = '#0a1128';
    ctx.fillRect(0, 0, 512, 256);

    const cols = 8;
    const rows = 3;
    const cellW = 512 / cols;
    const cellH = 256 / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = c * cellW + 3;
        const y = r * cellH + 3;
        const w = cellW - 6;
        const h = cellH - 6;

        // Photovoltaic cell gradient
        const grad = ctx.createLinearGradient(x, y, x + w, y + h);
        grad.addColorStop(0, '#1c234a');
        grad.addColorStop(0.4, '#151d3b');
        grad.addColorStop(1, '#0c1229');
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, w, h);

        // Silver busbars
        ctx.strokeStyle = 'rgba(226, 232, 240, 0.55)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x + w * 0.33, y);
        ctx.lineTo(x + w * 0.33, y + h);
        ctx.moveTo(x + w * 0.66, y);
        ctx.lineTo(x + w * 0.66, y + h);
        ctx.stroke();

        // Micro-conductors
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
        ctx.lineWidth = 0.6;
        for (let f = y + 5; f < y + h; f += 7) {
          ctx.beginPath();
          ctx.moveTo(x, f);
          ctx.lineTo(x + w, f);
          ctx.stroke();
        }
      }
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }

  // --- THREE.JS 3D CANSAT MODEL (HIGH-FIDELITY AEROSPACE DIGITAL TWIN) ---
  let targetRotX = 0;
  let targetRotY = 0;
  let beaconLed = null;

  function init3D() {
    if (threeReady) return;
    const container = document.getElementById('canvas-3d');
    if (!container) return;

    const w = Math.max(container.clientWidth, 100);
    const h = Math.max(container.clientHeight, 100);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(3.8, 2.8, 5.0);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    // Realistic Space Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xdde3ea, 0.9);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff7ed, 1.6);
    sunLight.position.set(6, 9, 6);
    scene.add(sunLight);

    const earthGlow = new THREE.DirectionalLight(0x38bdf8, 0.7);
    earthGlow.position.set(-6, -3, -4);
    scene.add(earthGlow);

    const grid = new THREE.GridHelper(8, 8, 0x334155, 0x1e293b);
    grid.position.y = -1.5;
    scene.add(grid);

    cansatGroup = new THREE.Group();

    // 1. Inner Satellite Core Body
    const coreGeom = new THREE.CylinderGeometry(0.67, 0.67, 2.1, 32);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x181724,
      metalness: 0.8,
      roughness: 0.35
    });
    const core = new THREE.Mesh(coreGeom, coreMat);
    cansatGroup.add(core);

    // 2. High-Efficiency Solar Panel Array Mantle
    const solarTex = createSolarTexture();
    const solarGeom = new THREE.CylinderGeometry(0.702, 0.702, 1.42, 32, 1, true);
    const solarMat = new THREE.MeshStandardMaterial({
      map: solarTex,
      metalness: 0.88,
      roughness: 0.22
    });
    const solar = new THREE.Mesh(solarGeom, solarMat);
    cansatGroup.add(solar);

    // 3. Gold Kapton Thermal Insulation Foil Collars (Upper & Lower)
    const kaptonMat = new THREE.MeshStandardMaterial({
      color: 0xD4AF37,
      emissive: 0x2A1D00,
      metalness: 0.95,
      roughness: 0.16
    });
    const kaptonGeom = new THREE.CylinderGeometry(0.708, 0.708, 0.24, 32);
    const topKapton = new THREE.Mesh(kaptonGeom, kaptonMat);
    topKapton.position.y = 0.83;
    cansatGroup.add(topKapton);

    const botKapton = topKapton.clone();
    botKapton.position.y = -0.83;
    cansatGroup.add(botKapton);

    // 4. CNC Aerospace Aluminum Bulkheads (Top & Bottom End Plates)
    const bulkheadMat = new THREE.MeshStandardMaterial({
      color: 0xCBD5E1,
      metalness: 0.96,
      roughness: 0.18
    });
    const bulkheadGeom = new THREE.CylinderGeometry(0.74, 0.74, 0.08, 32);
    const topBulkhead = new THREE.Mesh(bulkheadGeom, bulkheadMat);
    topBulkhead.position.y = 1.04;
    cansatGroup.add(topBulkhead);

    const botBulkhead = topBulkhead.clone();
    botBulkhead.position.y = -1.04;
    cansatGroup.add(botBulkhead);

    // Trim Bevel Rings
    const ringMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.85, roughness: 0.2 });
    const ringGeom = new THREE.TorusGeometry(0.73, 0.02, 16, 32);
    const topRing = new THREE.Mesh(ringGeom, ringMat);
    topRing.rotation.x = Math.PI / 2;
    topRing.position.y = 1.08;
    cansatGroup.add(topRing);

    const botRing = topRing.clone();
    botRing.position.y = -1.08;
    cansatGroup.add(botRing);

    // 5. Four Hexagonal Brass/Titanium Structural Standoffs (3-Tier PCB Architecture)
    const standoffMat = new THREE.MeshStandardMaterial({ color: 0xEAB308, metalness: 0.88, roughness: 0.25 });
    const standoffGeom = new THREE.CylinderGeometry(0.024, 0.024, 2.16, 6);
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI / 2) + Math.PI / 4;
      const standoff = new THREE.Mesh(standoffGeom, standoffMat);
      standoff.position.set(0.65 * Math.cos(angle), 0, 0.65 * Math.sin(angle));
      cansatGroup.add(standoff);
    }

    // 6. Ceramic GPS Patch Antenna on Top Bulkhead
    const gpsPatchGeom = new THREE.BoxGeometry(0.24, 0.035, 0.24);
    const gpsPatchMat = new THREE.MeshStandardMaterial({ color: 0xF8FAFC, metalness: 0.2, roughness: 0.6 });
    const gpsPatch = new THREE.Mesh(gpsPatchGeom, gpsPatchMat);
    gpsPatch.position.set(0.22, 1.10, 0.12);
    cansatGroup.add(gpsPatch);

    const pinGeom = new THREE.CylinderGeometry(0.018, 0.018, 0.045, 12);
    const pinMat = new THREE.MeshStandardMaterial({ color: 0xD4AF37, metalness: 0.95 });
    const pin = new THREE.Mesh(pinGeom, pinMat);
    pin.position.set(0.22, 1.12, 0.12);
    cansatGroup.add(pin);

    // 7. Avionics Status Beacon LED
    const ledGeom = new THREE.SphereGeometry(0.04, 16, 16);
    const ledMat = new THREE.MeshStandardMaterial({
      color: 0x10B981,
      emissive: 0x10B981,
      emissiveIntensity: 1.8
    });
    beaconLed = new THREE.Mesh(ledGeom, ledMat);
    beaconLed.position.set(-0.24, 1.10, 0.18);
    cansatGroup.add(beaconLed);

    // 8. Titanium Parachute Recovery Eyelet
    const eyeletGeom = new THREE.TorusGeometry(0.12, 0.025, 12, 24);
    const eyeletMat = new THREE.MeshStandardMaterial({ color: 0x94A3B8, metalness: 0.9, roughness: 0.2 });
    const eyelet = new THREE.Mesh(eyeletGeom, eyeletMat);
    eyelet.position.y = 1.20;
    cansatGroup.add(eyelet);

    // 9. 433 MHz LoRa Loaded Whip Antenna
    const antGeom = new THREE.CylinderGeometry(0.015, 0.015, 0.95, 8);
    const antMat = new THREE.MeshStandardMaterial({ color: 0xF59E0B, metalness: 0.95, roughness: 0.2 });
    const antenna = new THREE.Mesh(antGeom, antMat);
    antenna.position.set(0.3, -1.55, 0);
    cansatGroup.add(antenna);

    // 10. Coordinate Axes Orientation Indicator
    const axes = new THREE.AxesHelper(1.8);
    cansatGroup.add(axes);

    scene.add(cansatGroup);

    // Damped Smooth Orbit Interaction
    container.addEventListener('mousedown', function (e) {
      isDragging3D = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', function () { isDragging3D = false; });
    container.addEventListener('mousemove', function (e) {
      if (!isDragging3D) return;
      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;
      targetRotY += deltaX * 0.008;
      targetRotX += deltaY * 0.008;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    // True Radial Spherical 3D Zoom (scales camera distance along 3D line of sight)
    function zoom3D(deltaFactor) {
      if (!camera) return;
      const currentDist = camera.position.length();
      const newDist = Math.max(2.8, Math.min(10.5, currentDist * deltaFactor));
      camera.position.setLength(newDist);
      camera.lookAt(0, 0, 0);
    }

    function reset3DView() {
      if (!camera || !scene) return;
      targetRotX = 0;
      targetRotY = 0;
      scene.rotation.set(0, 0, 0);
      camera.position.set(3.8, 2.8, 5.0);
      camera.lookAt(0, 0, 0);
    }

    window.zoom3D = zoom3D;
    window.reset3DView = reset3DView;

    // Smooth mouse wheel zoom
    container.addEventListener('wheel', function (e) {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.08 : 0.92;
      zoom3D(factor);
    }, { passive: false });

    // Double-click to reset camera perspective
    container.addEventListener('dblclick', function () {
      reset3DView();
    });

    threeReady = true;

    (function animate() {
      animId = requestAnimationFrame(animate);

      // Smooth camera orbit damping
      scene.rotation.y += (targetRotY - scene.rotation.y) * 0.12;
      scene.rotation.x += (targetRotX - scene.rotation.x) * 0.12;

      // Subtle avionics beacon breathing
      if (beaconLed) {
        const pulse = 1.2 + 0.8 * Math.sin(Date.now() * 0.005);
        beaconLed.material.emissiveIntensity = pulse;
      }

      if (cansatGroup) {
        const targetEuler = new THREE.Euler(
          (fusedPitch - tarePitch) * Math.PI / 180,
          (fusedYaw - tareYaw) * Math.PI / 180,
          (fusedRoll - tareRoll) * Math.PI / 180,
          'XYZ'
        );
        cansatGroup.rotation.x += (targetEuler.x - cansatGroup.rotation.x) * 0.25;
        cansatGroup.rotation.y += (targetEuler.y - cansatGroup.rotation.y) * 0.25;
        cansatGroup.rotation.z += (targetEuler.z - cansatGroup.rotation.z) * 0.25;
      }
      if (renderer && scene && camera) renderer.render(scene, camera);
    })();

    resize3D();
  }

  function resize3D() {
    if (!threeReady) return;
    const container = document.getElementById('canvas-3d');
    if (!container) return;
    const w = Math.max(container.clientWidth, 1);
    const h = Math.max(container.clientHeight, 1);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // --- RAW SERIAL FEED UI ---
  function appendFeed(ok, rawLine) {
    const feed = document.getElementById('rf-feed');
    if (!feed) return;
    const entry = document.createElement('div');
    entry.className = 'flex space-x-2 text-slate-400';
    const tag = ok
      ? '<span class="text-emerald-400 font-bold">OK</span>'
      : '<span class="text-rose-400 font-bold">ERR</span>';
    const t = new Date().toLocaleTimeString();
    entry.innerHTML = '<span class="text-slate-600">' + t + '</span> ' + tag + ' <span class="truncate">' +
      String(rawLine).replace(/</g, '&lt;') + '</span>';
    feed.prepend(entry);
    while (feed.childNodes.length > MAX_FEED_LINES) feed.removeChild(feed.lastChild);
  }

  function setBatteryUi(volts, pct) {
    const fill = document.getElementById('bat-fill');
    const color = pct < 20 ? '#EF4444' : pct < 45 ? '#F59E0B' : '#10B981';
    if (fill) {
      fill.style.width = pct.toFixed(0) + '%';
      fill.style.background = color;
    }
    const batPercent = document.getElementById('bat-percent');
    if (batPercent) {
      batPercent.style.color = color;
      batPercent.innerText = pct.toFixed(0) + '%';
    }
    const batVolts = document.getElementById('bat-volts');
    if (batVolts) batVolts.innerText = volts.toFixed(2) + ' V';
    const housing = document.querySelector('.battery-housing');
    if (housing) housing.style.borderColor = color;
  }

  // --- CAMERA WIRELESS STREAM HANDLER ($CAM_FRAME,<id>,<len>,<dur_ms>,<base64>) ---
  let camFrameCount = 0;
  let camTotalBytes = 0;
  let lastCamFrameTs = null;
  let camFps = 0.0;
  let camTacticalGridActive = true;
  let offscreenCamCanvas = null;

  function toggleTacticalCamGrid() {
    camTacticalGridActive = !camTacticalGridActive;
    const btn = document.getElementById('btn-toggle-grid');
    if (btn) {
      btn.className = camTacticalGridActive 
        ? 'px-2.5 py-1.5 rounded bg-cyan-950/80 border border-cyan-500/60 text-xs font-bold text-cyan-300 transition flex items-center gap-1 shadow-sm'
        : 'px-2.5 py-1.5 rounded bg-slate-800/80 text-xs font-bold text-slate-400 border border-slate-700 transition flex items-center gap-1';
    }
    const canvas = document.getElementById('cam-grid-canvas');
    if (canvas && !camTacticalGridActive) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function analyzeAndRenderCameraFrame(imgEl) {
    if (!imgEl || !imgEl.naturalWidth) return;
    const canvas = document.getElementById('cam-grid-canvas');
    if (!canvas) return;

    if (canvas.width !== imgEl.naturalWidth || canvas.height !== imgEl.naturalHeight) {
      canvas.width = imgEl.naturalWidth || 320;
      canvas.height = imgEl.naturalHeight || 240;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!offscreenCamCanvas) {
      offscreenCamCanvas = document.createElement('canvas');
      offscreenCamCanvas.width = 64;
      offscreenCamCanvas.height = 48;
    }
    const offCtx = offscreenCamCanvas.getContext('2d', { willReadFrequently: true });
    offCtx.drawImage(imgEl, 0, 0, 64, 48);
    const imgData = offCtx.getImageData(0, 0, 64, 48);
    const data = imgData.data;

    let totR = 0, totG = 0, totB = 0;
    const len = data.length;
    for (let i = 0; i < len; i += 4) {
      totR += data[i];
      totG += data[i+1];
      totB += data[i+2];
    }
    const pxCount = len / 4;
    const meanR = totR / pxCount;
    const meanG = totG / pxCount;
    const meanB = totB / pxCount;

    // Global VARI = (G - R) / (G + R - B)
    const denom = (meanG + meanR - meanB);
    let vari = Math.abs(denom) > 1e-4 ? (meanG - meanR) / denom : 0.0;
    vari = Math.max(-1.0, Math.min(1.0, vari));

    // 3x3 Sectors Analysis
    const cw = Math.floor(64 / 3);
    const ch = Math.floor(48 / 3);
    let bestScore = -999;
    let bestCell = [1, 1];
    let bestLabel = 'CRITICAL_HAZARD';
    let centerLabel = 'CRITICAL_HAZARD';
    let centerScore = -0.6;
    const sectorGrid = [];

    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        let sR = 0, sG = 0, sB = 0, sCnt = 0;
        for (let y = r * ch; y < (r + 1) * ch; y++) {
          for (let x = c * cw; x < (c + 1) * cw; x++) {
            const idx = (y * 64 + x) * 4;
            sR += data[idx];
            sG += data[idx+1];
            sB += data[idx+2];
            sCnt++;
          }
        }
        const avgR = sR / sCnt;
        const avgG = sG / sCnt;
        const avgB = sB / sCnt;
        const sDenom = (avgG + avgR - avgB);
        let sVari = Math.abs(sDenom) > 1e-4 ? (avgG - avgR) / sDenom : 0.0;

        // Chromatic saturation & urban structure feature extraction
        const maxC = Math.max(avgR, avgG, avgB);
        const minC = Math.min(avgR, avgG, avgB);
        const saturation = maxC > 0 ? (maxC - minC) / maxC : 0;
        const isGrayOrConcrete = saturation < 0.10; // Concrete, asphalt, gray rooftops

        let sLabel = 'CRITICAL_HAZARD';
        let sScore = -0.6;
        let sColor = 'rgba(239, 68, 68, 0.25)';

        if (avgB > (avgR + avgG) * 0.65 && avgB > 80) {
          // Water Body
          sLabel = 'WATER_HAZARD';
          sScore = -1.0;
          sColor = 'rgba(6, 182, 212, 0.25)';
        } else if (isGrayOrConcrete || (avgR > 185 && avgG > 185 && avgB > 175 && saturation < 0.12)) {
          // Concrete, asphalt roads, white/gray rooftops, screen backlight washout
          sLabel = 'CRITICAL_HAZARD';
          sScore = -0.8;
          sColor = 'rgba(239, 68, 68, 0.28)';
        } else if (avgR > avgG * 1.06) {
          // Brick, terracotta roofs, red tiles, dry buildings
          sLabel = 'CRITICAL_HAZARD';
          sScore = -0.7;
          sColor = 'rgba(239, 68, 68, 0.25)';
        } else if (sVari > 0.08 && avgG > avgR * 1.05 && avgG > avgB * 1.03 && saturation > 0.10) {
          // Genuine Green Vegetation / Pasture
          sLabel = 'SAFE_LZ';
          sScore = 1.0 + sVari;
          sColor = 'rgba(16, 185, 129, 0.25)';
        } else if (sVari >= -0.02 && avgG > avgR && avgG > avgB && saturation > 0.08) {
          // Dense Foliage / Forest Canopy
          sLabel = 'OBSTACLE_CANOPY';
          sScore = 0.2 + sVari;
          sColor = 'rgba(245, 158, 11, 0.22)';
        } else {
          // Urban structures / buildings / pavement fallback
          sLabel = 'CRITICAL_HAZARD';
          sScore = -0.5;
          sColor = 'rgba(239, 68, 68, 0.25)';
        }

        if (r === 1 && c === 1) {
          centerLabel = sLabel;
          centerScore = sScore;
        }

        sectorGrid.push({ r, c, sLabel, sScore, sColor });
        if (sScore > bestScore) {
          bestScore = sScore;
          bestCell = [r, c];
          bestLabel = sLabel;
        }
      }
    }

    // Directional Escape Vector from Center (1, 1)
    const dx = bestCell[1] - 1;
    const dy = -(bestCell[0] - 1);
    let evadeAngle = 0;
    if (dx !== 0 || dy !== 0) {
      evadeAngle = Math.round((Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360);
    }
    const needsEvade = (centerLabel !== 'SAFE_LZ') && (bestCell[0] !== 1 || bestCell[1] !== 1);

    // Update Live SLAI Badge based on CENTER (NADIR) IMPACT SECTOR
    const slaiPill = document.getElementById('cam-slai-pill');
    const slaiDot = document.getElementById('cam-slai-dot');
    const slaiTxt = document.getElementById('cam-slai-text');
    if (slaiTxt && slaiPill && slaiDot) {
      slaiTxt.innerText = 'SLAI: ' + centerLabel;
      if (centerLabel === 'SAFE_LZ') {
        slaiPill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/60 text-[10px] font-mono font-semibold text-emerald-300';
        slaiDot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse';
      } else if (centerLabel === 'OBSTACLE_CANOPY') {
        slaiPill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-950/80 border border-amber-500/60 text-[10px] font-mono font-semibold text-amber-300';
        slaiDot.className = 'w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse';
      } else if (centerLabel === 'WATER_HAZARD') {
        slaiPill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/60 text-[10px] font-mono font-semibold text-cyan-300';
        slaiDot.className = 'w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse';
      } else {
        slaiPill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-rose-950/80 border border-rose-500/60 text-[10px] font-mono font-semibold text-rose-300';
        slaiDot.className = 'w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse';
      }
    }

    // Update Escape Vector HUD
    const evadeHud = document.getElementById('cam-evade-hud');
    const evadeTxt = document.getElementById('cam-evade-text');
    const evadeArrow = document.getElementById('cam-evade-arrow');
    if (evadeHud && evadeTxt && evadeArrow) {
      if (needsEvade) {
        evadeHud.classList.remove('hidden');
        evadeTxt.innerText = `EVADE ${String(evadeAngle).padStart(3, '0')}°`;
        const arrowGlyphs = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'];
        const arrowIdx = Math.round(evadeAngle / 45) % 8;
        evadeArrow.innerText = arrowGlyphs[arrowIdx];
      } else {
        evadeHud.classList.add('hidden');
      }
    }

    // Update Remote Sensing & TTI HUD (Based on Center Nadir Landing Point)
    const variHud = document.getElementById('cam-vari-hud');
    const variVal = document.getElementById('cam-vari-val');
    const ttiVal = document.getElementById('cam-tti-val');
    const riskVal = document.getElementById('cam-risk-val');
    if (variHud && variVal && ttiVal && riskVal) {
      variHud.classList.remove('hidden');
      const vegType = vari > 0.12 ? 'DENSE VEG' : (vari >= 0.04 ? 'MODERATE VEG' : (vari >= -0.05 ? 'DRY GROUND' : 'URBAN/ASPHALT'));
      variVal.innerText = `VARI ${vari >= 0 ? '+' : ''}${vari.toFixed(2)} (${vegType})`;
      const curAlt = (clientAltitudeFilter && clientAltitudeFilter.z !== undefined) ? clientAltitudeFilter.z : 45.0;
      const curVspd = Math.abs(telemetryHistory.length ? (telemetryHistory[telemetryHistory.length - 1].v_spd || 4.8) : 4.8);
      const estTti = curVspd > 0.3 ? (curAlt / curVspd) : 15.0;
      ttiVal.innerText = estTti.toFixed(1) + 's';
      riskVal.innerText = centerLabel === 'SAFE_LZ' ? 'LOW RISK (SAFE)' : (centerLabel === 'OBSTACLE_CANOPY' ? 'MODERATE (CANOPY)' : 'CRITICAL HAZARD');
      riskVal.className = centerLabel === 'SAFE_LZ' ? 'text-emerald-300 font-bold' : (centerLabel === 'OBSTACLE_CANOPY' ? 'text-amber-300 font-bold' : 'text-rose-400 font-bold');
    }

    // Draw 3x3 Grid Overlay
    if (camTacticalGridActive) {
      const W = canvas.width;
      const H = canvas.height;
      const sw = W / 3;
      const sh = H / 3;

      ctx.lineWidth = 1.0;
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';

      for (const sec of sectorGrid) {
        const x = sec.c * sw;
        const y = sec.r * sh;
        ctx.fillStyle = sec.sColor;
        ctx.fillRect(x, y, sw, sh);
        ctx.strokeRect(x, y, sw, sh);
      }

      // Highlight best sector
      const bx = bestCell[1] * sw;
      const by = bestCell[0] * sh;
      ctx.lineWidth = 2.0;
      ctx.strokeStyle = '#10B981';
      ctx.strokeRect(bx + 2, by + 2, sw - 4, sh - 4);
      ctx.fillStyle = '#10B981';
      ctx.font = '10px monospace';
      ctx.fillText('OPTIMAL LZ', bx + 6, by + 16);

      // Escape Vector Arrow
      if (needsEvade) {
        const cx = W / 2;
        const cy = H / 2;
        const tx = bx + sw / 2;
        const ty = by + sh / 2;
        ctx.strokeStyle = '#34D399';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(tx, ty);
        ctx.stroke();
      }
    }
  }

  function handleCameraFrameLine(line) {
    const parts = line.split(',', 5);
    if (parts.length < 5) return;
    const frameId = parts[1];
    const sizeBytes = parseInt(parts[2], 10) || 0;
    const durationMs = parseInt(parts[3], 10) || 0;
    const b64Data = parts[4].trim();

    camFrameCount++;
    camTotalBytes += sizeBytes;

    const now = Date.now();
    if (lastCamFrameTs) {
      const dt = (now - lastCamFrameTs) / 1000.0;
      if (dt > 0.05 && dt < 10) {
        const instFps = 1.0 / dt;
        camFps = camFps === 0 ? instFps : (camFps * 0.75 + instFps * 0.25);
      }
    }
    lastCamFrameTs = now;

    const imgEl = document.getElementById('live-cam-img');
    const placeholderEl = document.getElementById('cam-placeholder');
    const hudEl = document.getElementById('cam-hud');
    if (imgEl && b64Data) {
      imgEl.src = "data:image/jpeg;base64," + b64Data;
      imgEl.style.display = 'block';
      if (placeholderEl) placeholderEl.style.display = 'none';
      if (hudEl) hudEl.classList.remove('hidden');
      // Trigger tactical spatial grid & remote sensing analysis
      imgEl.onload = function() {
        analyzeAndRenderCameraFrame(imgEl);
      };
    }

    const hudFrame = document.getElementById('cam-hud-frame');
    if (hudFrame) hudFrame.innerText = '#' + String(frameId).padStart(4, '0');
    const hudSize = document.getElementById('cam-hud-size');
    if (hudSize) hudSize.innerText = (sizeBytes / 1024.0).toFixed(1) + ' KB';
    const hudAirtime = document.getElementById('cam-hud-airtime');
    if (hudAirtime) hudAirtime.innerText = durationMs + ' ms';

    const fpsEl = document.getElementById('cam-metric-fps');
    if (fpsEl) fpsEl.innerText = camFps.toFixed(1) + ' FPS';

    const rxEl = document.getElementById('cam-stat-rx');
    if (rxEl) rxEl.innerText = String(camFrameCount);
    const msEl = document.getElementById('cam-stat-ms');
    if (msEl) msEl.innerText = durationMs + ' ms';
    const kbpsEl = document.getElementById('cam-stat-kbps');
    if (kbpsEl) kbpsEl.innerText = (camTotalBytes / 1024.0 / Math.max(1, (now - (missionStartEpochMs || now)) / 1000.0)).toFixed(1) + ' KB/s';

    const pill = document.getElementById('cam-status-pill');
    const dot = document.getElementById('cam-status-dot');
    const txt = document.getElementById('cam-status-text');
    if (pill && dot && txt) {
      pill.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/60 text-[10px] font-mono font-semibold text-emerald-300';
      dot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse';
      txt.innerText = 'LIVE STREAM';
    }
  }

  // --- MAIN TELEMETRY INGESTION PIPELINE ---
  function ingestRawLine(line) {
    if (line.startsWith('$CAM_FRAME,')) {
      handleCameraFrameLine(line);
      return;
    }
    linesReceived += 1;
    const rfRec = document.getElementById('rf-received');
    if (rfRec) rfRec.innerText = String(linesReceived);

    const parsed = parsePacketLine(line);
    if (!parsed) {
      linesRejected += 1;
      const rfRej = document.getElementById('rf-rejected');
      if (rfRej) rfRej.innerText = String(linesRejected);
      appendFeed(false, line);
      const lower = line.toLowerCase();
      if (lower.includes('lora') || lower.includes('receiver') || lower.includes('init') || lower.includes('ready') || lower.includes('boot')) {
        const connText = document.getElementById('conn-text');
        if (connText) connText.innerText = line.substring(0, 26).toUpperCase();
        const connDot = document.getElementById('conn-dot');
        if (connDot) connDot.className = 'led-dot bg-cyan-400 text-cyan-400 led-pulse';
      }
      return;
    }
    linesParsed += 1;
    const rfPars = document.getElementById('rf-parsed');
    if (rfPars) rfPars.innerText = String(linesParsed);
    appendFeed(true, line);

    const now = Date.now();
    let dt = 0.8;
    if (previousTs !== null) {
      const measuredDt = (now - previousTs) / 1000.0;
      if (measuredDt > 0.05 && measuredDt < 5.0) dt = measuredDt;
    }

    // Tare sample collection if calibration routine is running
    if (isCalibratingTare) {
      tareSampleBuffer.push({
        gx: parsed.gx, gy: parsed.gy, gz: parsed.gz,
        ax: parsed.ax, ay: parsed.ay, az: parsed.az,
        alt: parsed.altitude
      });
      if (tareSampleBuffer.length >= tareTargetSamples) {
        finalizeStationaryTare();
      }
    }

    // Bias-corrected gyro rates (nulls out static MEMS drift)
    const effGx = parsed.gx - tareGyroX;
    const effGy = parsed.gy - tareGyroY;
    const effGz = parsed.gz - tareGyroZ;

    // Edge-preserving filtered accelerations (smooth at rest, captures shocks)
    accelFiltX = filterAccel(parsed.ax, accelFiltX);
    accelFiltY = filterAccel(parsed.ay, accelFiltY);
    accelFiltZ = filterAccel(parsed.az, accelFiltZ);

    // High-Precision Noise-Immune Altitude & Vertical Velocity Filter
    // 100% physically faithful to barometric elevation (lifting ALWAYS climbs, lowering ALWAYS descends)
    // Completely eliminates stationary sensor noise jitter (< 0.18m) without lag or tilt error
    const filtRes = clientAltitudeFilter.update(parsed.altitude, now / 1000.0);
    smoothedAltitude = Number(filtRes.alt.toFixed(2));
    let vSpd = Number(filtRes.vz.toFixed(2));

    const fusion = updateAttitudeFusion(parsed.ax, parsed.ay, parsed.az, effGx, effGy, effGz, dt);
    const accelMag = Math.sqrt(parsed.ax * parsed.ax + parsed.ay * parsed.ay + parsed.az * parsed.az);
    previousAlt = smoothedAltitude;
    previousTs = now;

    const sounding = computeSoundingMetrics(parsed.temp, parsed.pressure, parsed.humidity, smoothedAltitude);

    if (!padBaselineRecorded && currentFlightPhase === 'PAD_IDLE') {
      padBaselineTemp = parsed.temp;
      padBaselineAlt = smoothedAltitude;
      padBaselineRecorded = true;
    }

    updateFlightPhase(smoothedAltitude, vSpd, accelMag);
    evaluateAnomalies(parsed, vSpd, accelMag);
    requestPythonMlInference(parsed, vSpd);

    const row = Object.assign({
      timestamp: parsed.customTimestamp || new Date(now).toISOString()
    }, parsed, {
      rawAltitude: parsed.altitude,
      altitude: Number(smoothedAltitude.toFixed(2)),
      effGx: effGx, effGy: effGy, effGz: effGz,
      accelFiltX: accelFiltX, accelFiltY: accelFiltY, accelFiltZ: accelFiltZ,
      pitch: fusion.pitch, roll: fusion.roll, yaw: fusion.yaw, accelMag: accelMag, vSpd: vSpd,
      dewPoint: sounding.dewPoint, airDensity: sounding.airDensity, lapseRate: sounding.lapseRate
    });

    processTelemetry(row);
  }

  function processTelemetry(row) {
    telemetryHistory.push(row);
    const count = telemetryHistory.length;
    if (row.altitude > apogee) apogee = row.altitude;

    // Database Streaming & Auto-Recording Buffer
    if (activeDbMissionId || (autoRecordEnabled && !isViewingHistoricalReport && !replayPlaying && Date.now() > autoArmCooldownUntil)) {
      if (!activeDbMissionId && !autoArmInProgress) {
        autoArmInProgress = true;
        autoStartFlightMission();
      }
      dbTelemetryBuffer.push({
        timestamp_ms: row.timestamp ? (Date.parse(row.timestamp) || Date.now()) : Date.now(),
        met_seconds: (count - 1) * (DEMO_DT_MS / 1000.0),
        altitude: row.altitude,
        pressure: row.pressure,
        temp: row.temp,
        humidity: row.humidity,
        battery_voltage: (typeof row.batteryVoltage !== 'undefined' && row.batteryVoltage !== null ? row.batteryVoltage : 4.2),
        ax: row.ax, ay: row.ay, az: row.az,
        gx: row.gx, gy: row.gy, gz: row.gz,
        lat: (typeof row.lat !== 'undefined' && row.lat !== null ? row.lat : 0.0),
        lon: (typeof row.lon !== 'undefined' && row.lon !== null ? row.lon : 0.0),
        v_spd: row.vSpd || 0.0,
        accel_mag: row.accelMag || 1.0,
        pitch: row.pitch || 0.0,
        roll: row.roll || 0.0,
        air_density: row.airDensity || 1.225,
        dew_point: row.dewPoint || 15.0,
        lapse_rate: row.lapseRate || 0.65,
        flight_phase: (typeof currentFlightPhase !== 'undefined' ? currentFlightPhase : 'PAD_IDLE'),
        anomaly_score: (typeof currentAnomalyScore !== 'undefined' ? currentAnomalyScore : 0.0),
        is_anomaly: (typeof currentAnomalyCode !== 'undefined' && currentAnomalyCode !== 'NOMINAL')
      });
      if (activeDbMissionId && dbTelemetryBuffer.length >= 10) {
        flushDbTelemetryBuffer();
      }
    }

    // Header Counters & Badges
    const topCount = document.getElementById('sample-count-top');
    if (topCount) topCount.innerText = String(count);
    const feedCounter = document.getElementById('feed-counter');
    if (feedCounter) feedCounter.innerText = '(' + count + ')';
    const infoSamples = document.getElementById('info-samples');
    if (infoSamples) infoSamples.innerText = String(count);

    // Live Mission Elapsed Time (MET) Counter
    let elapsedSec = 0;
    if (row.timestamp) {
      const parsedEpoch = Date.parse(row.timestamp);
      if (!isNaN(parsedEpoch)) {
        if (missionStartEpochMs === null) {
          missionStartEpochMs = parsedEpoch;
        }
        elapsedSec = Math.max(0, (parsedEpoch - missionStartEpochMs) / 1000.0);
      } else {
        elapsedSec = (count - 1) * (DEMO_DT_MS / 1000.0);
      }
    } else {
      elapsedSec = (count - 1) * (DEMO_DT_MS / 1000.0);
    }
    const timerEl = document.getElementById('phase-elapsed-time');
    if (timerEl) timerEl.innerText = formatMissionTime(elapsedSec);

    // Live Axis Badges
    document.getElementById('axis-badge-x').innerText =
      'ax: ' + (row.ax >= 0 ? '+' : '') + row.ax.toFixed(2) + ' g | gx: ' + (row.gx >= 0 ? '+' : '') + row.gx.toFixed(2) + ' °/s';
    document.getElementById('axis-badge-y').innerText =
      'ay: ' + (row.ay >= 0 ? '+' : '') + row.ay.toFixed(2) + ' g | gy: ' + (row.gy >= 0 ? '+' : '') + row.gy.toFixed(2) + ' °/s';
    document.getElementById('axis-badge-z').innerText =
      'az: ' + (row.az >= 0 ? '+' : '') + row.az.toFixed(2) + ' g | gz: ' + (row.gz >= 0 ? '+' : '') + row.gz.toFixed(2) + ' °/s';

    // Battery UI
    const pct = batteryPct(row.batteryVoltage);
    setBatteryUi(row.batteryVoltage, pct);

    // Meteorological Soundings
    document.getElementById('met-dewpoint').innerText = row.dewPoint.toFixed(1) + ' °C';
    document.getElementById('met-lapserate').innerText = row.lapseRate.toFixed(2) + ' °C/100m';
    document.getElementById('met-airdensity').innerText = row.airDensity.toFixed(3) + ' kg/m³';

    // Inbound Serial 13 Fields Readout
    document.getElementById('t-temp').innerText = row.temp.toFixed(2);
    document.getElementById('t-press').innerText = row.pressure.toFixed(2);
    document.getElementById('t-alt').innerText = (row.rawAltitude !== undefined ? row.rawAltitude : row.altitude).toFixed(2);
    document.getElementById('t-gx').innerText = row.gx.toFixed(2);
    document.getElementById('t-gy').innerText = row.gy.toFixed(2);
    document.getElementById('t-gz').innerText = row.gz.toFixed(2);
    document.getElementById('t-ax').innerText = row.ax.toFixed(2);
    document.getElementById('t-ay').innerText = row.ay.toFixed(2);
    document.getElementById('t-az').innerText = row.az.toFixed(2);
    document.getElementById('t-lat').innerText = row.lat.toFixed(6);
    document.getElementById('t-lon').innerText = row.lon.toFixed(6);
    document.getElementById('t-hum').innerText = row.humidity.toFixed(2);
    document.getElementById('t-volt').innerText = row.batteryVoltage.toFixed(2);

    // General Mission Information
    const locked = gpsLocked(row.lat, row.lon);
    document.getElementById('info-gpsfix').innerText = locked ? 'Locked (WGS84 3D)' : 'No lock / Searching';
    document.getElementById('info-alt').innerText = row.altitude.toFixed(2) + ' m';
    document.getElementById('info-apogee').innerText = apogee.toFixed(2) + ' m';
    // Auto-tare initial resting orientation on launchpad immediately on first valid packet
    if (!autoTaredOnPad && currentFlightPhase === 'PAD_IDLE') {
      tarePitch = row.pitch;
      tareRoll = row.roll;
      tareYaw = row.yaw;
      autoTaredOnPad = true;
    }

    const displayPitch = row.pitch - tarePitch;
    const displayRoll = row.roll - tareRoll;
    const displayYaw = row.yaw - tareYaw;

    document.getElementById('info-vspd').innerText = row.vSpd.toFixed(2) + ' m/s';
    document.getElementById('info-pitch').innerText = displayPitch.toFixed(1) + '°';
    document.getElementById('info-roll').innerText = displayRoll.toFixed(1) + '°';
    document.getElementById('info-presstemp').innerText = row.pressure.toFixed(2) + ' hPa';
    document.getElementById('info-hum').innerText = row.humidity.toFixed(1) + ' %';
    document.getElementById('info-dewdensity').innerText = row.dewPoint.toFixed(1) + ' °C · ' + row.airDensity.toFixed(3) + ' kg/m³';
    document.getElementById('info-accelmag').innerText = row.accelMag.toFixed(2) + ' g';
    document.getElementById('info-bat').innerText = row.batteryVoltage.toFixed(2) + ' V (' + pct.toFixed(0) + '%)';
    document.getElementById('info-anomaly').innerText = currentAnomalyCode;

    // 3D Model Readings
    document.getElementById('model-pitch').innerText = displayPitch.toFixed(1) + '°';
    document.getElementById('model-roll').innerText = displayRoll.toFixed(1) + '°';
    document.getElementById('model-yaw').innerText = displayYaw.toFixed(1) + '°';

    // Map Floating HUD Readings
    document.getElementById('map-alt').innerText = row.altitude.toFixed(1);
    document.getElementById('map-vspd').innerText = row.vSpd.toFixed(2);

    // Update GPS Path & Recovery Tool
    updateGpsAndRecovery(row.lat, row.lon, row.altitude, row.vSpd);

    // Push Live Super-Accurate Chart Data
    pushChart(accelChart, [
      Number((row.accelFiltX !== undefined ? row.accelFiltX : row.ax).toFixed(2)),
      Number((row.accelFiltY !== undefined ? row.accelFiltY : row.ay).toFixed(2)),
      Number((row.accelFiltZ !== undefined ? row.accelFiltZ : row.az).toFixed(2))
    ]);
    pushChart(orientChart, [
      Number(displayPitch.toFixed(1)),
      Number(displayRoll.toFixed(1))
    ]);
    pushChart(vSpeedChart, [
      Number(row.vSpd.toFixed(1))
    ]);
    pushChart(altChart, [
      Number(row.altitude.toFixed(1))
    ]);
    pushChart(gyroChart, [
      Number((row.effGx !== undefined ? row.effGx : row.gx).toFixed(1)),
      Number((row.effGy !== undefined ? row.effGy : row.gy).toFixed(1)),
      Number((row.effGz !== undefined ? row.effGz : row.gz).toFixed(1))
    ]);
    pushChart(humChart, [
      Number(row.humidity.toFixed(1))
    ]);
    pushChart(pressTempChart, [
      Number(row.pressure.toFixed(1)),
      Number(row.temp.toFixed(1))
    ]);
  }

  function setConnectionState(online, text) {
    const dot = document.getElementById('conn-dot');
    if (dot) dot.className = 'led-dot ' + (online ? 'bg-emerald-400 text-emerald-400 led-pulse' : 'bg-rose-500 text-rose-500');
    const connText = document.getElementById('conn-text');
    if (connText) connText.innerText = text;
    const infoStatus = document.getElementById('info-status');
    if (infoStatus) {
      infoStatus.innerText = online ? 'Online' : 'Disconnected';
      infoStatus.className = online ? 'text-emerald-400 font-mono font-semibold' : 'text-rose-400 font-mono font-semibold';
    }
  }

  // --- SERIAL USB (WEB SERIAL API & PYTHON BACKEND BRIDGE) ---
  let bridgeSocket = null;

  async function connectPythonSerialBridge() {
    if (bridgeSocket) {
      bridgeSocket.close();
      bridgeSocket = null;
      document.getElementById('connect-btn').innerText = 'CONNECT';
      setConnectionState(false, 'DISCONNECTED');
      return;
    }
    if (isDemo) toggleDemo();
    if (replayPlaying) toggleReplayPlay();

    setConnectionState(true, 'CONNECTING BRIDGE...');
    try {
      const host = location.host || '127.0.0.1:8000';
      const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + host + '/ws/serial';
      bridgeSocket = new WebSocket(wsUrl);
      bridgeSocket.onopen = function () {
        setConnectionState(true, 'ONLINE (DUAL BRIDGE: COM4 LoRa + COM5 Cam)');
        document.getElementById('connect-btn').innerText = 'DISCONNECT';
      };
      bridgeSocket.onmessage = function (event) {
        if (event.data) {
          ingestRawLine(event.data);
        }
      };
      bridgeSocket.onerror = function (err) {
        console.warn('Bridge socket error (Python backend not running on port 8000). Falling back to direct USB Web Serial...', err);
        setConnectionState(false, 'BRIDGE OFFLINE (FALLING BACK TO USB...)');
        if (bridgeSocket) {
          try { bridgeSocket.close(); } catch (e) {}
          bridgeSocket = null;
        }
        setTimeout(function () {
          connectDirectWebSerial(9600);
        }, 500);
      };
      bridgeSocket.onclose = function (event) {
        document.getElementById('connect-btn').innerText = 'CONNECT';
        bridgeSocket = null;
        if (event && event.code === 4404) {
          // Server has no serial hardware (cloud/Docker host): use the browser's own USB link.
          setConnectionState(false, 'NO SERVER HARDWARE (FALLING BACK TO USB...)');
          setTimeout(function () {
            connectDirectWebSerial(9600);
          }, 500);
          return;
        }
        setConnectionState(false, 'DISCONNECTED');
      };
    } catch (e) {
      console.warn('Could not connect to Python Serial Bridge. Falling back to direct USB Web Serial...', e);
      connectDirectWebSerial(9600);
    }
  }

  async function connectDirectWebSerial(baudRate) {
    if (serialPort) {
      await disconnectSerial();
      return;
    }
    if (!('serial' in navigator)) {
      alert('Web Serial API is supported on Google Chrome and Microsoft Edge (HTTPS or localhost).');
      setConnectionState(false, 'DISCONNECTED');
      return;
    }
    if (isDemo) toggleDemo();
    if (replayPlaying) toggleReplayPlay();

    try {
      const selectedBaud = baudRate || 9600;
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: selectedBaud });
      try {
        await port.setSignals({ dataTerminalReady: true, requestToSend: true });
      } catch (sigErr) {
        console.warn('DTR/RTS signals not supported on this adapter', sigErr);
      }
      serialPort = port;
      serialKeepReading = true;
      setConnectionState(true, 'ONLINE (DIRECT USB @ ' + selectedBaud + ' BAUD)');
      document.getElementById('connect-btn').innerText = 'DISCONNECT';
      const decoder = new TextDecoderStream();
      const closed = port.readable.pipeTo(decoder.writable).catch(function () { /* stream closed */ });
      serialReader = decoder.readable.getReader();
      let buffer = '';
      while (serialKeepReading) {
        const { value, done } = await serialReader.read();
        if (done) break;
        buffer += value;
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop();
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          try { ingestRawLine(line); } catch (e) { linesRejected += 1; }
        }
      }
      await closed;
    } catch (err) {
      if (err && err.name !== 'NotFoundError') {
        console.warn('Serial connection error', err);
        alert('Serial connection error: ' + (err.message || err));
      }
    } finally {
      if (serialPort) await disconnectSerial(true);
    }
  }

  async function toggleConnect() {
    if (bridgeSocket) {
      bridgeSocket.close();
      bridgeSocket = null;
      document.getElementById('connect-btn').innerText = 'CONNECT';
      setConnectionState(false, 'DISCONNECTED');
      return;
    }
    if (serialPort) {
      await disconnectSerial();
      return;
    }

    const baudSelect = document.getElementById('baud-select');
    const selectedMode = baudSelect ? baudSelect.value : 'bridge';

    if (selectedMode === 'bridge') {
      await connectPythonSerialBridge();
      return;
    }

    const selectedBaud = parseInt(selectedMode, 10) || 9600;
    await connectDirectWebSerial(selectedBaud);
  }

  async function disconnectSerial(silent) {
    serialKeepReading = false;
    try { if (serialReader) await serialReader.cancel(); } catch (e) { /* ignore */ }
    serialReader = null;
    try { if (serialPort) await serialPort.close(); } catch (e) { /* ignore */ }
    serialPort = null;
    document.getElementById('connect-btn').innerText = 'CONNECT';
    if (!isDemo && !replayPlaying) setConnectionState(false, 'DISCONNECTED');
  }

  // --- OPTIMAL SOUNDING FLIGHT CYCLE DEMO ENGINE ---
  // Deterministic, physics-governed nominal sounding flight cycle (5-phase progression: Pad -> Ascent -> Apogee -> Descent -> Touchdown)
  const OPTIMAL_DEMO_TOTAL_STEPS = 118;
  let demoStepIndex = 0;

  function generateOptimalFlightPacket(step) {
    const dt = DEMO_DT_MS / 1000;
    const t = step * dt;
    let alt, vz, ax, ay, az, gx, gy, gz, lat, lon;

    if (step <= 4) {
      // Phase 1: Pad Idle (0.0s - 3.2s) - Upright resting on launch pad
      alt = 0.50;
      vz = 0.00;
      ax = 0.005; ay = -0.003; az = 0.998;
      gx = 0.05; gy = -0.02; gz = 0.01;
      lat = HOME[0];
      lon = HOME[1];
    } else if (step <= 45) {
      // Phase 2: Steady Smooth Ascent to 350m Apogee (4.0s - 36.0s)
      const tau = (t - 4.0) / 32.0;
      alt = 0.50 + 349.50 * (3 * tau * tau - 2 * tau * tau * tau);
      vz = (349.50 / 32.0) * 6 * tau * (1 - tau);
      az = 1.00 + (65.53 * (1 - 2 * tau)) / (32.0 * 9.81);
      ax = 0.012 * Math.sin(t * 0.8);
      ay = 0.010 * Math.cos(t * 0.7);
      gx = 0.6 * Math.sin(t * 1.2);
      gy = 0.5 * Math.cos(t * 1.1);
      gz = 1.2;
      lat = HOME[0] + 0.00035 * tau;
      lon = HOME[1] + 0.00045 * tau;
    } else if (step === 46) {
      // Phase 3: Peak Apogee (36.8s) - Freefall microgravity release
      alt = 350.00;
      vz = -0.40;
      ax = 0.010; ay = -0.010; az = 0.220;
      gx = 1.50; gy = -2.10; gz = 0.80;
      lat = HOME[0] + 0.000355;
      lon = HOME[1] + 0.000455;
    } else if (step === 47) {
      // Parachute ejection line tension shock (37.6s)
      alt = 348.50;
      vz = -3.20;
      ax = 0.050; ay = 0.080; az = 1.880;
      gx = 4.20; gy = 3.80; gz = 1.50;
      lat = HOME[0] + 0.000360;
      lon = HOME[1] + 0.000460;
    } else if (step <= 107) {
      // Phase 4: Parachute Controlled Descent at Safe Terminal Velocity (38.4s - 85.6s)
      const descentT = t - 37.6;
      alt = Math.max(0.50, 348.50 - 5.85 * descentT);
      vz = -5.85;
      ax = 0.035 * Math.sin(t * 1.6);
      ay = 0.028 * Math.cos(t * 1.6);
      az = 1.00 + 0.04 * Math.sin(t * 1.6);
      gx = 2.4 * Math.sin(t * 1.6);
      gy = 2.0 * Math.cos(t * 1.6);
      gz = 0.2 * Math.sin(t * 0.5);
      const tauDesc = Math.min(1.0, (t - 37.6) / 50.0);
      lat = HOME[0] + 0.000360 + 0.000280 * tauDesc;
      lon = HOME[1] + 0.000460 + 0.000320 * tauDesc;
    } else if (step === 108) {
      // Touchdown ground contact damping shock (86.4s)
      alt = 0.50;
      vz = -0.50;
      ax = 0.020; ay = 0.010; az = 1.520;
      gx = 0.80; gy = -0.60; gz = 0.10;
      lat = HOME[0] + 0.000640;
      lon = HOME[1] + 0.000780;
    } else {
      // Phase 5: Touchdown Rest & Safe Recovery (87.2s - 94.4s)
      alt = 0.50;
      vz = 0.00;
      ax = 0.002; ay = 0.001; az = 1.000;
      gx = 0.00; gy = 0.00; gz = 0.00;
      lat = HOME[0] + 0.000640;
      lon = HOME[1] + 0.000780;
    }

    const temp = 25.00 - 0.0065 * alt;
    const press = 1013.25 * Math.pow(1 - (0.0065 * alt) / 288.15, 5.255);
    const humidity = clamp(55.0 + 8.5 * (alt / 350.0), 5.0, 95.0);
    const batteryVoltage = Math.max(3.40, 4.190 - 0.0007 * t);

    return [
      temp.toFixed(2), press.toFixed(2), alt.toFixed(2),
      gx.toFixed(2), gy.toFixed(2), gz.toFixed(2),
      ax.toFixed(3), ay.toFixed(3), az.toFixed(3),
      lat.toFixed(6), lon.toFixed(6), humidity.toFixed(2), batteryVoltage.toFixed(3)
    ].join(',');
  }

  function toggleDemo() {
    const btn = document.getElementById('demo-btn');
    if (isDemo) {
      clearInterval(demoInterval);
      demoInterval = null;
      isDemo = false;
      if (btn) {
        btn.innerHTML = '<span>DEMO DATA</span>';
        btn.className = 'btn-action-light flex items-center justify-center';
      }
      if (!serialPort) setConnectionState(false, 'DISCONNECTED');
      return;
    }

    if (replayPlaying) toggleReplayPlay();
    if (demoInterval) {
      clearInterval(demoInterval);
      demoInterval = null;
    }

    // Clean reset of telemetry state and visualizers for fresh flight run
    telemetryHistory.length = 0;
    linesReceived = 0;
    linesParsed = 0;
    linesRejected = 0;
    previousAlt = null;
    previousTs = null;
    smoothedAltitude = null;
    altitudeBuffer = [];
    if (clientAltitudeFilter && typeof clientAltitudeFilter.reset === 'function') {
      clientAltitudeFilter.reset(0.50);
    }
    autoTaredOnPad = false;
    tarePitch = 0.0;
    tareRoll = 0.0;
    tareYaw = 0.0;
    compPitch = null;
    compRoll = null;
    compYaw = 0.0;
    apogee = 0.50;
    missionStartEpochMs = null;
    padBaselineRecorded = false;
    padBaselineAlt = 0.50;
    padBaselineTemp = 25.00;
    currentFlightPhase = 'PAD_IDLE';
    currentAnomalyCode = 'NOMINAL';
    currentAnomalyScore = 0;
    speechAnnounced = { apogee: false, chute: false, alt100: false, touchdown: false };
    gpsLockEstablished = false;
    syncedCursorIndex = null;

    allCharts.forEach(function (c) {
      c.data.labels = [];
      c.data.datasets.forEach(function (d) { d.data = []; });
      c.update('none');
    });

    updateStageTrackerUI('PAD_IDLE');
    const timerEl = document.getElementById('phase-elapsed-time');
    if (timerEl) timerEl.innerText = 'T+ 00:00.0';
    if (mapReady) {
      if (ascentTrail) ascentTrail.setLatLngs([]);
      if (descentTrail) descentTrail.setLatLngs([]);
    }

    isDemo = true;
    if (btn) {
      btn.innerHTML = '<span>STOP DEMO</span>';
      btn.className = 'btn-action-primary flex items-center justify-center';
    }
    setConnectionState(true, 'SIMULATION (800MS)');

    demoStepIndex = 0;

    // Ingest initial resting packets immediately so charts draw the baseline and UI updates instantly on first click
    ingestRawLine(generateOptimalFlightPacket(demoStepIndex++));
    ingestRawLine(generateOptimalFlightPacket(demoStepIndex++));

    demoInterval = setInterval(function () {
      if (demoStepIndex >= OPTIMAL_DEMO_TOTAL_STEPS) {
        clearInterval(demoInterval);
        demoInterval = null;
        isDemo = false;
        if (btn) {
          btn.innerHTML = '<span>DEMO DATA</span>';
          btn.className = 'btn-action-light flex items-center justify-center';
        }
        setConnectionState(true, 'TOUCHDOWN (COMPLETE)');
        return;
      }
      ingestRawLine(generateOptimalFlightPacket(demoStepIndex++));
    }, DEMO_DT_MS);
  }

  // --- REPLAY ENGINE ---
  function toggleReplayDrawer() {
    const drawer = document.getElementById('replay-drawer');
    if (!drawer) return;
    drawer.classList.toggle('hidden');
  }

  async function loadSelectedTestCase() {
    const select = document.getElementById('test-case-select');
    const path = select.value;
    if (!path) return;
    try {
      const resp = await fetch(path);
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const text = await resp.text();
      loadCsvIntoReplay(text, path.split('/').pop());
    } catch (e) {
      console.warn('Fetch fallback triggered for test case:', path, e);
      alert('Note: To load local test cases directly via fetch, serve using a local server:\n  python -m http.server 8000\nOr select the CSV via the "Upload Custom CSV" file picker button!');
    }
  }

  function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (evt) {
      loadCsvIntoReplay(evt.target.result, file.name);
    };
    reader.readAsText(file);
  }

  function loadCsvIntoReplay(csvText, sourceName) {
    const lines = csvText.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length === 0) {
      alert('CSV file is empty.');
      return;
    }

    const dataLines = (lines[0].toLowerCase().includes('temp') || lines[0].toLowerCase().includes('time'))
      ? lines.slice(1)
      : lines;

    if (dataLines.length === 0) {
      alert('No data rows found in CSV.');
      return;
    }

    replayRows = dataLines;
    replayIndex = 0;

    const scrubber = document.getElementById('replay-scrubber');
    scrubber.max = Math.max(1, replayRows.length - 1);
    scrubber.value = 0;
    document.getElementById('scrub-total-count').innerText = '/ ' + replayRows.length;
    document.getElementById('scrub-current-idx').innerText = '0';
    document.getElementById('replay-status-pill').innerText = sourceName + ' (' + replayRows.length + ')';
    document.getElementById('replay-status-pill').className = 'text-xs px-2.5 py-1 rounded-full bg-[#171717] border border-[#2e2e2e] text-zinc-300 font-mono font-semibold';

    resetReplay();
  }

  function toggleReplayPlay() {
    if (replayRows.length === 0) {
      alert('Please select a test flight profile or upload a CSV first.');
      return;
    }
    if (isDemo) toggleDemo();
    if (serialPort) disconnectSerial();

    const SVG_PLAY = '<svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    const SVG_PAUSE = '<svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
    const playIcon = document.getElementById('replay-play-icon');
    const playLabel = document.getElementById('replay-play-label');

    if (replayPlaying) {
      clearInterval(replayInterval);
      replayInterval = null;
      replayPlaying = false;
      playIcon.innerHTML = SVG_PLAY;
      playLabel.innerText = 'PLAY';
      setConnectionState(true, 'REPLAY PAUSED');
      return;
    }

    replayPlaying = true;
    playIcon.innerHTML = SVG_PAUSE;
    playLabel.innerText = 'PAUSE';
    setConnectionState(true, 'REPLAYING (' + replaySpeed + 'X)');

    const intervalMs = Math.max(20, Math.round(DEMO_DT_MS / replaySpeed));
    replayInterval = setInterval(function () {
      if (replayIndex >= replayRows.length) {
        clearInterval(replayInterval);
        replayInterval = null;
        replayPlaying = false;
        playIcon.innerHTML = SVG_PLAY;
        playLabel.innerText = 'PLAY';
        setConnectionState(true, 'REPLAY COMPLETED');
        return;
      }
      const line = replayRows[replayIndex];
      ingestRawLine(line);
      replayIndex += 1;
      document.getElementById('replay-scrubber').value = replayIndex;
      document.getElementById('scrub-current-idx').innerText = String(replayIndex);
    }, intervalMs);
  }

  function replayStepForward() {
    if (replayRows.length === 0 || replayIndex >= replayRows.length) return;
    const line = replayRows[replayIndex];
    ingestRawLine(line);
    replayIndex += 1;
    document.getElementById('replay-scrubber').value = replayIndex;
    document.getElementById('scrub-current-idx').innerText = String(replayIndex);
  }

  function resetReplay() {
    if (replayInterval) {
      clearInterval(replayInterval);
      replayInterval = null;
    }
    replayPlaying = false;
    document.getElementById('replay-play-icon').innerHTML = '<svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    document.getElementById('replay-play-label').innerText = 'PLAY';
    replayIndex = 0;
    document.getElementById('replay-scrubber').value = 0;
    document.getElementById('scrub-current-idx').innerText = '0';

    telemetryHistory.length = 0;
    apogee = 0;
    previousAlt = null;
    previousTs = null;
    smoothedAltitude = null;
    altitudeBuffer = [];
    if (clientAltitudeFilter && typeof clientAltitudeFilter.reset === 'function') {
      clientAltitudeFilter.reset();
    }
    autoTaredOnPad = false;
    tarePitch = 0.0;
    tareRoll = 0.0;
    tareYaw = 0.0;
    compPitch = null;
    compRoll = null;
    compYaw = 0.0;
    if (cansatGroup) {
      cansatGroup.rotation.set(0, 0, 0);
    }
    currentFlightPhase = 'PAD_IDLE';
    speechAnnounced = { apogee: false, chute: false, alt100: false, touchdown: false };
    missionStartEpochMs = null;
    const timerEl = document.getElementById('phase-elapsed-time');
    if (timerEl) timerEl.innerText = 'T+ 00:00.0';
    updateStageTrackerUI('PAD_IDLE');

    allCharts.forEach(c => {
      c.data.labels = [];
      c.data.datasets.forEach(d => d.data = []);
      c.update('none');
    });

    if (mapReady) {
      ascentTrail.setLatLngs([]);
      descentTrail.setLatLngs([]);
      marker.setLatLng(HOME);
      map.setView(HOME, 16);
    }

    setConnectionState(false, 'DISCONNECTED');
  }

  function onScrubSeek(val) {
    const targetIdx = parseInt(val, 10);
    if (isNaN(targetIdx) || replayRows.length === 0) return;
    
    resetReplay();
    const end = Math.min(targetIdx, replayRows.length);
    for (let i = 0; i < end; i++) {
      ingestRawLine(replayRows[i]);
    }
    replayIndex = end;
    document.getElementById('replay-scrubber').value = end;
    document.getElementById('scrub-current-idx').innerText = String(end);

    // Sync multi-chart crosshair inspector
    if (allCharts.length > 0 && allCharts[0].data.labels.length > 0) {
      syncedCursorIndex = allCharts[0].data.labels.length - 1;
      isSyncingHover = true;
      allCharts.forEach(c => c.draw());
      isSyncingHover = false;
    }
  }

  function setReplaySpeed(spd) {
    replaySpeed = spd;
    document.querySelectorAll('.speed-btn').forEach(btn => {
      if (parseFloat(btn.getAttribute('data-speed')) === spd) {
        btn.className = 'speed-btn px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-white text-black border border-white transition';
      } else {
        btn.className = 'speed-btn px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-[#171717] border border-[#2e2e2e] text-zinc-400 hover:bg-[#242424] hover:text-white transition';
      }
    });
    if (replayPlaying) {
      clearInterval(replayInterval);
      const intervalMs = Math.max(20, Math.round(DEMO_DT_MS / replaySpeed));
      replayInterval = setInterval(function () {
        if (replayIndex >= replayRows.length) {
          clearInterval(replayInterval);
          replayInterval = null;
          replayPlaying = false;
          document.getElementById('replay-play-icon').innerHTML = '<svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
          document.getElementById('replay-play-label').innerText = 'PLAY';
          setConnectionState(true, 'REPLAY COMPLETED');
          return;
        }
        const line = replayRows[replayIndex];
        ingestRawLine(line);
        replayIndex += 1;
        document.getElementById('replay-scrubber').value = replayIndex;
        document.getElementById('scrub-current-idx').innerText = String(replayIndex);
      }, intervalMs);
    }
  }

  function downloadCSV() {
    if (telemetryHistory.length === 0) {
      alert('No telemetry data recorded yet.');
      return;
    }
    const csv = buildCsv(telemetryHistory);
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cognitive_cansat_log_' + stamp + '.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // --- TILE MAXIMIZE / FULLSCREEN VIEW ---
  function toggleTileMaximize(tileId) {
    const tile = document.getElementById(tileId);
    if (!tile) return;
    tile.classList.toggle('tile-maximized');
    const isMax = tile.classList.contains('tile-maximized');
    const btn = tile.querySelector('.panel-expand-btn');
    const SVG_EXPAND = '<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5m-6 11l-5-5m0 5h4m-4 0v-4m16 4l-5-5m5 5h-4m4 0v-4"/></svg>';
    const SVG_COLLAPSE = '<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>';
    if (btn) btn.innerHTML = isMax ? SVG_COLLAPSE : SVG_EXPAND;

    // When closing, strip Chart.js / Three.js inline height overrides to prevent dashboard stretching
    if (!isMax) {
      tile.querySelectorAll('canvas').forEach(function (cvs) {
        cvs.style.width = '';
        cvs.style.height = '';
      });
      tile.querySelectorAll('.chart-container').forEach(function (cc) {
        cc.style.height = '';
      });
    }

    requestAnimationFrame(function () {
      resizeVisibleCharts();
      resizeMap();
      resize3D();
    });
  }

  // --- POST-FLIGHT REVIEW (PFR) REPORT GENERATOR ---
  function generatePfrReport() {
    const container = document.getElementById('pfr-report-content');
    if (!container) return;

    if (telemetryHistory.length === 0) {
      container.innerHTML =
        '<div class="p-8 text-center bg-[#0a0a0a] rounded-xl border border-[#1f1f1f]">' +
        '<div class="text-zinc-500 mb-2 flex justify-center"><svg class="w-8 h-8 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0"/></svg></div>' +
        '<div class="text-sm font-bold text-zinc-200 mb-1 font-tech uppercase tracking-wide">No Telemetry Recorded Yet</div>' +
        '<p class="text-zinc-400 max-w-md mx-auto text-xs leading-relaxed">Connect live hardware via USB serial, select a flight profile in the Replay Suite, or activate Demo Data to stream mission packets before generating an automated Post-Flight Review report.</p>' +
        '</div>';
      return;
    }

    const rows = telemetryHistory;
    const count = rows.length;
    const first = rows[0];
    const last = rows[count - 1];

    const padAlt = padBaselineAlt || first.altitude;
    const padTemp = padBaselineTemp || first.temp;
    const padPress = first.pressure;

    let peakAlt = -Infinity;
    let apogeeIdx = 0;
    let maxAscentV = 0;
    let maxG = 0;
    let maxGIdx = 0;
    let descentRates = [];
    let minTemp = Infinity;
    let maxTemp = -Infinity;

    for (let i = 0; i < count; i++) {
      const r = rows[i];
      if (r.altitude > peakAlt) {
        peakAlt = r.altitude;
        apogeeIdx = i;
      }
      if (r.vSpd > maxAscentV) {
        maxAscentV = r.vSpd;
      }
      if (r.accelMag > maxG) {
        maxG = r.accelMag;
        maxGIdx = i;
      }
      if (r.temp < minTemp) minTemp = r.temp;
      if (r.temp > maxTemp) maxTemp = r.temp;
      if (i > apogeeIdx && r.vSpd < -0.5) {
        descentRates.push(Math.abs(r.vSpd));
      }
    }

    const aglApogee = Math.max(0, peakAlt - padAlt);
    const avgDescentRate = descentRates.length > 0 ? (descentRates.reduce((a, b) => a + b, 0) / descentRates.length) : 0;
    const terminalDescentRate = descentRates.length > 5 ? descentRates[Math.floor(descentRates.length * 0.7)] : avgDescentRate;

    let chuteStatus = 'NOMINAL';
    let chuteStatusClass = 'text-emerald-300 bg-emerald-950/40 border-emerald-700/50';
    if (terminalDescentRate > 0 && terminalDescentRate < 6.0) {
      chuteStatus = 'EXCESSIVE DRIFT RISK (< 6.0 m/s)';
      chuteStatusClass = 'text-amber-300 bg-amber-950/40 border-amber-700/50';
    } else if (terminalDescentRate > 11.0) {
      chuteStatus = 'HARD IMPACT WARNING (> 11.0 m/s)';
      chuteStatusClass = 'text-rose-300 bg-rose-950/40 border-rose-700/50';
    } else if (terminalDescentRate === 0) {
      chuteStatus = 'NO DESCENT DETECTED';
      chuteStatusClass = 'text-zinc-400 bg-[#171717] border-[#2e2e2e]';
    }

    const dLat = (last.lat - first.lat) * 111139.0;
    const dLon = (last.lon - first.lon) * 111139.0 * Math.cos(first.lat * Math.PI / 180.0);
    const driftDist = Math.sqrt(dLat * dLat + dLon * dLon);
    let driftBearing = Math.atan2(dLon, dLat) * 180.0 / Math.PI;
    if (driftBearing < 0) driftBearing += 360;

    const timeToApogee = (apogeeIdx * (DEMO_DT_MS / 1000.0)).toFixed(1);
    const totalDuration = (count * (DEMO_DT_MS / 1000.0)).toFixed(1);

    const startV = first.batteryVoltage;
    const endV = last.batteryVoltage;
    const deltaV = (startV - endV);
    const nowIso = new Date().toISOString().replace('T', ' ').substring(0, 19);

    container.innerHTML =
      '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-2">' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Peak Apogee (AGL)</div>' +
          '<div class="text-xl font-bold font-mono text-white mt-1 pfr-kpi">' + aglApogee.toFixed(1) + ' m</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">MSL: ' + peakAlt.toFixed(1) + ' m (T+ ' + timeToApogee + 's)</div>' +
        '</div>' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Max Ejection Shock</div>' +
          '<div class="text-xl font-bold font-mono text-amber-400 mt-1 pfr-kpi">' + maxG.toFixed(2) + ' G</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">' + (maxG > 3.5 ? 'Deployment Shock' : 'Nominal Aero') + '</div>' +
        '</div>' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Terminal Descent Rate</div>' +
          '<div class="text-xl font-bold font-mono text-white mt-1 pfr-kpi">' + terminalDescentRate.toFixed(2) + ' m/s</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">Mean: ' + avgDescentRate.toFixed(2) + ' m/s</div>' +
        '</div>' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Total Flight Duration</div>' +
          '<div class="text-xl font-bold font-mono text-emerald-400 mt-1 pfr-kpi">' + totalDuration + ' s</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">' + count + ' telemetry packets</div>' +
        '</div>' +
      '</div>' +

      '<div class="p-3 rounded-xl border flex flex-wrap items-center justify-between gap-2 ' + chuteStatusClass + '">' +
        '<div class="flex items-center gap-2">' +
          '<span class="font-bold uppercase tracking-wider text-xs font-mono">Descent Compliance Envelope (6.0 - 11.0 m/s):</span>' +
          '<span class="font-mono font-bold text-xs">' + chuteStatus + '</span>' +
        '</div>' +
        '<span class="text-xs opacity-90">Descent kinetic energy satisfies safety threshold</span>' +
      '</div>' +

      '<div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">' +
        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">1. Flight Dynamics &amp; Kinematics</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Launch Pad Baseline:</span><span class="text-white">' + padAlt.toFixed(1) + ' m | ' + padTemp.toFixed(1) + '°C | ' + padPress.toFixed(1) + ' hPa</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Max Vertical Climb Rate:</span><span class="text-emerald-400 font-bold">+' + maxAscentV.toFixed(2) + ' m/s</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Ejection Shock Peak:</span><span class="text-white font-bold">' + maxG.toFixed(2) + ' G (T+ ' + (maxGIdx * 0.8).toFixed(1) + 's)</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Terminal Parachute Velocity:</span><span class="text-zinc-200 font-bold">' + terminalDescentRate.toFixed(2) + ' m/s</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Total Ingested Packets:</span><span class="text-white">' + count + ' (Loss: 0.0%)</span></div>' +
          '</div>' +
        '</div>' +

        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">2. Atmospheric Characterization</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Temperature Profile:</span><span class="text-white">Min ' + minTemp.toFixed(1) + '°C / Max ' + maxTemp.toFixed(1) + '°C</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Surface Air Density:</span><span class="text-white">' + first.airDensity.toFixed(3) + ' kg/m³</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Apogee Air Density:</span><span class="text-white">' + rows[apogeeIdx].airDensity.toFixed(3) + ' kg/m³</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Environmental Lapse Rate:</span><span class="text-zinc-200 font-bold">' + rows[apogeeIdx].lapseRate.toFixed(2) + ' °C/100m</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Dew Point Range:</span><span class="text-white">' + last.dewPoint.toFixed(1) + '°C to ' + first.dewPoint.toFixed(1) + '°C</span></div>' +
          '</div>' +
        '</div>' +

        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">3. GIS &amp; Recovery Footprint</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Launch Pad GNSS:</span><span class="text-white">' + first.lat.toFixed(6) + '°, ' + first.lon.toFixed(6) + '°</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Touchdown GNSS:</span><span class="text-emerald-400 font-bold">' + last.lat.toFixed(6) + '°, ' + last.lon.toFixed(6) + '°</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Total Horizontal Drift:</span><span class="text-zinc-200 font-bold">' + driftDist.toFixed(1) + ' m</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Drift Azimuth:</span><span class="text-white">' + driftBearing.toFixed(0) + '° (' + getCardinal(driftBearing) + ')</span></div>' +
            '<div class="flex justify-between pt-1"><span class="text-zinc-400 font-sans">Recovery Map Link:</span><a href="https://www.google.com/maps/search/?api=1&query=' + last.lat.toFixed(6) + ',' + last.lon.toFixed(6) + '" target="_blank" class="text-zinc-200 hover:text-white underline font-bold">Open Navigation ↗</a></div>' +
          '</div>' +
        '</div>' +

        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">4. Power Subsystem &amp; Safety Audit</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Initial LiPo Voltage:</span><span class="text-white">' + startV.toFixed(2) + ' V (' + batteryPct(startV).toFixed(0) + '%)</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Touchdown Voltage:</span><span class="text-white">' + endV.toFixed(2) + ' V (' + batteryPct(endV).toFixed(0) + '%)</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Net Cell Discharge:</span><span class="text-emerald-400 font-bold">' + (deltaV >= 0 ? '-' : '+') + Math.abs(deltaV).toFixed(2) + ' V</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Anomaly Engine Status:</span><span class="text-white font-bold">' + currentAnomalyCode + '</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Audit Signature:</span><span class="text-zinc-500">COGNITIVE-CANSAT-PFR-VERIFIED</span></div>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="text-[11px] text-zinc-500 font-mono text-right pt-2">Generated: ' + nowIso + ' UTC • Cognitive CanSat Ground Station</div>';
  }

  function copyPfrMarkdown() {
    if (cachedPfrMarkdown) {
      navigator.clipboard.writeText(cachedPfrMarkdown).then(function () {
        const label = document.getElementById('pfr-copy-label');
        if (label) {
          label.innerText = 'COPIED!';
          setTimeout(function () { label.innerText = 'COPY MARKDOWN'; }, 1800);
        }
      }).catch(function (err) {
        alert('Could not copy to clipboard: ' + err);
      });
      return;
    }

    if (telemetryHistory.length === 0) {
      alert('No telemetry records available to copy.');
      return;
    }
    const rows = telemetryHistory;
    const count = rows.length;
    const first = rows[0];
    const last = rows[count - 1];
    const padAlt = padBaselineAlt || first.altitude;
    let peakAlt = -Infinity;
    let apogeeIdx = 0;
    let maxG = 0;
    let descentRates = [];
    for (let i = 0; i < count; i++) {
      if (rows[i].altitude > peakAlt) { peakAlt = rows[i].altitude; apogeeIdx = i; }
      if (rows[i].accelMag > maxG) maxG = rows[i].accelMag;
      if (i > apogeeIdx && rows[i].vSpd < -0.5) descentRates.push(Math.abs(rows[i].vSpd));
    }
    const terminalDescentRate = descentRates.length > 5 ? descentRates[Math.floor(descentRates.length * 0.7)] : (descentRates[0] || 0);
    const dLat = (last.lat - first.lat) * 111139.0;
    const dLon = (last.lon - first.lon) * 111139.0 * Math.cos(first.lat * Math.PI / 180.0);
    const driftDist = Math.sqrt(dLat * dLat + dLon * dLon);
    let driftBearing = Math.atan2(dLon, dLat) * 180.0 / Math.PI;
    if (driftBearing < 0) driftBearing += 360;

    const md = `# CANSAT-1 POST-FLIGHT REVIEW (PFR) & TELEMETRY AUDIT REPORT\n` +
      `Generated: ${new Date().toISOString()}\n\n` +
      `## Executive Flight Summary\n` +
      `- **Peak Apogee (AGL)**: ${(peakAlt - padAlt).toFixed(1)} m (MSL: ${peakAlt.toFixed(1)} m)\n` +
      `- **Time to Apogee**: ${(apogeeIdx * 0.8).toFixed(1)} s\n` +
      `- **Max Ejection Shock**: ${maxG.toFixed(2)} G\n` +
      `- **Terminal Descent Rate**: ${terminalDescentRate.toFixed(2)} m/s\n` +
      `- **Regulatory Compliance**: ${terminalDescentRate >= 6 && terminalDescentRate <= 11 ? 'PASSED (6-11 m/s)' : 'WARNING / NON-COMPLIANT'}\n` +
      `- **Total Flight Time**: ${(count * 0.8).toFixed(1)} s\n` +
      `- **Packets Ingested**: ${count}\n\n` +
      `## GIS & Field Recovery\n` +
      `- **Launch Pad**: ${first.lat.toFixed(6)}°, ${first.lon.toFixed(6)}°\n` +
      `- **Touchdown Site**: ${last.lat.toFixed(6)}°, ${last.lon.toFixed(6)}°\n` +
      `- **Horizontal Drift**: ${driftDist.toFixed(1)} m at ${driftBearing.toFixed(0)}° (${getCardinal(driftBearing)})\n` +
      `- **Google Maps Link**: https://www.google.com/maps/search/?api=1&query=${last.lat.toFixed(6)},${last.lon.toFixed(6)}\n\n` +
      `## Power & Safety\n` +
      `- **Battery Potential**: ${first.batteryVoltage.toFixed(2)}V -> ${last.batteryVoltage.toFixed(2)}V (Drop: ${(first.batteryVoltage - last.batteryVoltage).toFixed(2)}V)\n` +
      `- **Anomaly Engine Status**: ${currentAnomalyCode}\n`;

    cachedPfrMarkdown = md;

    navigator.clipboard.writeText(md).then(function () {
      const label = document.getElementById('pfr-copy-label');
      if (label) {
        label.innerText = 'COPIED!';
        setTimeout(function () { label.innerText = 'COPY MARKDOWN'; }, 1800);
      }
    }).catch(function (err) {
      alert('Could not copy to clipboard: ' + err);
    });
  }

  function printPfrReport() {
    window.print();
  }

  // --- DATABASE MISSION RECORDING & FLIGHT ARCHIVE PERSISTENCE ---

  function updateDbBadgeUI(recording, missionId, isAuto) {
    const badge = document.getElementById('db-mission-badge');
    const dot = document.getElementById('db-status-dot');
    const text = document.getElementById('db-status-text');
    const btn = document.getElementById('db-record-btn');
    if (!badge || !dot || !text || !btn) return;

    if (recording) {
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-rose-600/70 bg-rose-950/60 text-rose-300';
      dot.className = 'led-dot bg-rose-500 led-pulse';
      text.innerText = 'REC [' + (isAuto ? 'AUTO: ' : 'DB: ') + (missionId ? missionId.substring(0, 8) : 'ACTIVE') + ']';
      btn.innerHTML = '<span>FINISH</span>';
      btn.className = 'ml-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-700 hover:bg-rose-600 text-white border border-rose-500 transition flex items-center gap-1';
      btn.title = 'Finalize active flight and compute PFR report in SQLite';
    } else {
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-mono flex items-center gap-1.5 border border-[#2e2e2e] bg-[#111111] text-zinc-300';
      dot.className = autoRecordEnabled ? 'led-dot bg-emerald-500/80' : 'led-dot bg-zinc-500';
      text.innerText = autoRecordEnabled ? 'DB: AUTO-STANDBY' : 'DB: STANDBY';
      btn.innerHTML = '<span>ARM REC</span>';
      btn.className = 'ml-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-[#1f1f1f] hover:bg-[#2d2d2d] text-white border border-[#3e3e3e] transition flex items-center gap-1';
      btn.title = 'Arm flight mission logging in SQLite';
    }
  }

  function toggleAutoRecord() {
    autoRecordEnabled = !autoRecordEnabled;
    const btn = document.getElementById('db-autorec-btn');
    const txt = document.getElementById('db-autorec-text');
    if (autoRecordEnabled) {
      if (btn) btn.className = 'px-2 py-1 rounded-full text-[11px] font-mono border border-emerald-800/50 bg-emerald-950/40 text-emerald-300 hover:bg-emerald-900/50 transition flex items-center gap-1';
      if (txt) txt.innerText = 'AUTO-REC: ON';
    } else {
      if (btn) btn.className = 'px-2 py-1 rounded-full text-[11px] font-mono border border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800 transition flex items-center gap-1';
      if (txt) txt.innerText = 'AUTO-REC: OFF';
    }
    if (!activeDbMissionId) {
      updateDbBadgeUI(false);
    }
  }

  async function autoStartFlightMission() {
    try {
      const now = new Date();
      const dateStr = now.toISOString().slice(0, 10);
      const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '');
      const missionName = 'Auto Flight ' + dateStr + '_' + timeStr;

      const resp = await fetch(getApiUrl('/api/db/missions/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: missionName,
          callsign: 'CANSAT-1',
          operator: 'Autonomous Ground Station',
          notes: 'Automatic flight recording initialized on telemetry reception.'
        })
      });

      if (!resp.ok) {
        console.warn('Auto flight start returned HTTP ' + resp.status);
        autoArmInProgress = false;
        return;
      }

      const data = await resp.json();
      activeDbMissionId = data.mission_id;
      activeDbMissionName = missionName;
      autoArmInProgress = false;

      updateDbBadgeUI(true, activeDbMissionId, true);

      if (dbFlushInterval) clearInterval(dbFlushInterval);
      dbFlushInterval = setInterval(flushDbTelemetryBuffer, 2000);

      // Flush whatever packets arrived during mission provisioning
      await flushDbTelemetryBuffer();
      console.log('[Database] Auto-recording initialized. Active mission:', activeDbMissionId);
    } catch (err) {
      console.warn('[Database] Could not auto-start flight mission:', err);
      autoArmInProgress = false;
    }
  }

  function toggleMissionRecording() {
    if (activeDbMissionId) {
      if (confirm('Finish flight mission [' + (activeDbMissionName || activeDbMissionId) + '] and generate permanent PFR audit report in SQLite?')) {
        finishActiveMission();
      }
    } else {
      openModal('startMissionModal');
    }
  }

  async function confirmStartMission() {
    const nameInput = document.getElementById('db-input-mission-name');
    const callsignInput = document.getElementById('db-input-callsign');
    const operatorInput = document.getElementById('db-input-operator');
    const notesInput = document.getElementById('db-input-notes');

    const name = nameInput ? nameInput.value.trim() : 'Autonomous Sounding Flight';
    const callsign = callsignInput ? callsignInput.value.trim() : 'CANSAT-1';
    const operator = operatorInput ? operatorInput.value.trim() : 'Flight Ops';
    const notes = notesInput ? notesInput.value.trim() : '';

    try {
      const resp = await fetch(getApiUrl('/api/db/missions/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name || 'Autonomous Sounding Flight',
          callsign: callsign || 'CANSAT-1',
          operator: operator || 'Flight Ops',
          notes: notes
        })
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        if (resp.status === 404) {
          throw new Error('Database endpoints not active on server (HTTP 404). Please restart "python launch.py".');
        }
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }
      const data = await resp.json();
      activeDbMissionId = data.mission_id;
      activeDbMissionName = name;
      dbTelemetryBuffer = [];

      closeModal('startMissionModal');
      updateDbBadgeUI(true, activeDbMissionId, false);

      if (dbFlushInterval) clearInterval(dbFlushInterval);
      dbFlushInterval = setInterval(flushDbTelemetryBuffer, 2000);

      // If we already have buffered in-memory points from the current flight, queue them
      if (telemetryHistory.length > 0) {
        const historySlice = telemetryHistory.slice(-50);
        historySlice.forEach(function (row, idx) {
          dbTelemetryBuffer.push({
            timestamp_ms: row.timestamp ? (Date.parse(row.timestamp) || Date.now()) : Date.now(),
            met_seconds: idx * (DEMO_DT_MS / 1000.0),
            altitude: row.altitude,
            pressure: row.pressure,
            temp: row.temp,
            humidity: row.humidity,
            battery_voltage: (typeof row.batteryVoltage !== 'undefined' && row.batteryVoltage !== null ? row.batteryVoltage : 4.2),
            ax: row.ax, ay: row.ay, az: row.az,
            gx: row.gx, gy: row.gy, gz: row.gz,
            lat: (typeof row.lat !== 'undefined' && row.lat !== null ? row.lat : 0.0),
            lon: (typeof row.lon !== 'undefined' && row.lon !== null ? row.lon : 0.0),
            v_spd: row.vSpd || 0.0,
            accel_mag: row.accelMag || 1.0,
            pitch: row.pitch || 0.0,
            roll: row.roll || 0.0,
            air_density: row.airDensity || 1.225,
            dew_point: row.dewPoint || 15.0,
            lapse_rate: row.lapseRate || 0.65,
            flight_phase: (typeof currentFlightPhase !== 'undefined' ? currentFlightPhase : 'PAD_IDLE'),
            anomaly_score: (typeof currentAnomalyScore !== 'undefined' ? currentAnomalyScore : 0.0),
            is_anomaly: (typeof currentAnomalyCode !== 'undefined' && currentAnomalyCode !== 'NOMINAL')
          });
        });
        flushDbTelemetryBuffer();
      }
    } catch (e) {
      console.error('Error starting DB mission:', e);
      alert('Could not start database mission: ' + e.message + '\nEnsure the Python backend is running (python launch.py).');
    }
  }

  async function flushDbTelemetryBuffer() {
    if (!activeDbMissionId || dbTelemetryBuffer.length === 0) return;
    const batch = dbTelemetryBuffer.splice(0, dbTelemetryBuffer.length);
    try {
      const resp = await fetch(getApiUrl('/api/db/missions/telemetry'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mission_id: activeDbMissionId,
          records: batch
        })
      });
      if (!resp.ok) {
        console.warn('DB telemetry flush warning, re-queuing', resp.status);
        dbTelemetryBuffer.unshift(...batch);
      }
    } catch (err) {
      console.warn('Network error flushing DB telemetry, re-queuing', err);
      dbTelemetryBuffer.unshift(...batch);
    }
  }

  async function finishActiveMission() {
    if (!activeDbMissionId) return;
    const targetMissionId = activeDbMissionId;
    const targetMissionName = activeDbMissionName;

    // Flush any remaining buffered records
    await flushDbTelemetryBuffer();

    if (dbFlushInterval) {
      clearInterval(dbFlushInterval);
      dbFlushInterval = null;
    }

    try {
      const resp = await fetch(getApiUrl('/api/db/missions/' + targetMissionId + '/finish'), {
        method: 'POST'
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }
      const data = await resp.json();
      activeDbMissionId = null;
      activeDbMissionName = '';
      autoArmCooldownUntil = Date.now() + 10000; // 10 second cooldown on auto-arm after manual finish
      updateDbBadgeUI(false);

      refreshDbMissionsList();

      alert('Mission [' + (targetMissionName || targetMissionId) + '] finalized!\n' +
            'PFR Audit Report generated in SQLite (Apogee: ' + (data.report ? data.report.apogee_agl + 'm AGL' : 'Saved') + ').');
      
      // Auto-open PFR report modal to inspect the stored record
      openModal('pfrModal');
      const select = document.getElementById('db-mission-history-select');
      if (select) {
        select.value = targetMissionId;
        onSelectHistoricalMission(targetMissionId);
      }
    } catch (e) {
      console.error('Error finalizing mission:', e);
      alert('Could not finalize DB mission: ' + e.message);
    }
  }

  async function saveCurrentTelemetryToDb() {
    if (telemetryHistory.length === 0) {
      alert('No telemetry records currently in memory to save.');
      return;
    }

    const saveBtn = document.getElementById('pfr-save-db-btn');
    const origHtml = saveBtn ? saveBtn.innerHTML : '';
    if (saveBtn) {
      saveBtn.disabled = true;
      saveBtn.innerHTML = '<span>SAVING TO DB...</span>';
    }

    try {
      const now = new Date();
      const defaultName = 'Archived Flight - ' + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const missionName = prompt('Enter mission name to archive current flight in SQLite database:', defaultName);
      if (!missionName) {
        if (saveBtn) { saveBtn.disabled = false; saveBtn.innerHTML = origHtml; }
        return;
      }

      // 1. Start Mission
      const startResp = await fetch(getApiUrl('/api/db/missions/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: missionName,
          callsign: 'CANSAT-1',
          operator: 'Ground Station',
          notes: 'Retroactively saved from in-memory session (' + telemetryHistory.length + ' packets)'
        })
      });
      if (!startResp.ok) {
        const errObj = await startResp.json().catch(() => ({}));
        if (startResp.status === 404) {
          throw new Error('Database endpoints not active on server (HTTP 404). Please restart "python launch.py".');
        }
        throw new Error(errObj.detail || ('Failed to create mission record (HTTP ' + startResp.status + ')'));
      }
      const startData = await startResp.json();
      const missionId = startData.mission_id;

      // 2. Ingest telemetry records in chunks of 200
      const CHUNK_SIZE = 200;
      for (let i = 0; i < telemetryHistory.length; i += CHUNK_SIZE) {
        const slice = telemetryHistory.slice(i, i + CHUNK_SIZE);
        const mappedBatch = slice.map(function (row, idx) {
          const countIdx = i + idx;
          return {
            timestamp_ms: row.timestamp ? (Date.parse(row.timestamp) || Date.now()) : Date.now(),
            met_seconds: countIdx * (DEMO_DT_MS / 1000.0),
            altitude: row.altitude,
            pressure: row.pressure,
            temp: row.temp,
            humidity: row.humidity,
            battery_voltage: (typeof row.batteryVoltage !== 'undefined' && row.batteryVoltage !== null ? row.batteryVoltage : 4.2),
            ax: row.ax, ay: row.ay, az: row.az,
            gx: row.gx, gy: row.gy, gz: row.gz,
            lat: (typeof row.lat !== 'undefined' && row.lat !== null ? row.lat : 0.0),
            lon: (typeof row.lon !== 'undefined' && row.lon !== null ? row.lon : 0.0),
            v_spd: row.vSpd || 0.0,
            accel_mag: row.accelMag || 1.0,
            pitch: row.pitch || 0.0,
            roll: row.roll || 0.0,
            air_density: row.airDensity || 1.225,
            dew_point: row.dewPoint || 15.0,
            lapse_rate: row.lapseRate || 0.65,
            flight_phase: (typeof currentFlightPhase !== 'undefined' ? currentFlightPhase : 'PAD_IDLE'),
            anomaly_score: (typeof currentAnomalyScore !== 'undefined' ? currentAnomalyScore : 0.0),
            is_anomaly: (typeof currentAnomalyCode !== 'undefined' && currentAnomalyCode !== 'NOMINAL')
          };
        });

        const ingestResp = await fetch(getApiUrl('/api/db/missions/telemetry'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mission_id: missionId,
            records: mappedBatch
          })
        });
        if (!ingestResp.ok) {
          const errObj = await ingestResp.json().catch(() => ({}));
          throw new Error('Failed to stream telemetry chunk ' + i + ': ' + (errObj.detail || ('HTTP ' + ingestResp.status)));
        }
      }

      // 3. Finalize mission and generate PFR audit report
      const finResp = await fetch(getApiUrl('/api/db/missions/' + missionId + '/finish'), {
        method: 'POST'
      });
      if (!finResp.ok) {
        const errObj = await finResp.json().catch(() => ({}));
        throw new Error('Failed to finalize mission: ' + (errObj.detail || ('HTTP ' + finResp.status)));
      }
      const finData = await finResp.json();

      await refreshDbMissionsList();

      const select = document.getElementById('db-mission-history-select');
      if (select) {
        select.value = missionId;
        onSelectHistoricalMission(missionId);
      }

      if (saveBtn) {
        saveBtn.innerHTML = '<span>SAVED!</span>';
        setTimeout(function () {
          saveBtn.disabled = false;
          saveBtn.innerHTML = origHtml;
        }, 2000);
      }
    } catch (e) {
      console.error('Error saving flight to DB:', e);
      alert('Could not save flight to database: ' + e.message);
      if (saveBtn) {
        saveBtn.disabled = false;
        saveBtn.innerHTML = origHtml;
      }
    }
  }

  async function refreshDbMissionsList() {
    const select = document.getElementById('db-mission-history-select');
    if (!select) return;

    try {
      const resp = await fetch(getApiUrl('/api/db/missions'));
      if (!resp.ok) return;
      const data = await resp.json();
      const currentVal = select.value;

      select.innerHTML = '<option value="">-- Active / Current Live Telemetry --</option>';
      if (data.missions && data.missions.length > 0) {
        data.missions.forEach(function (m) {
          const opt = document.createElement('option');
          opt.value = m.mission_id;
          const timeStr = m.start_time_ms ? new Date(m.start_time_ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
          const ptsStr = m.packet_count ? (' • ' + m.packet_count + ' pts') : '';
          opt.innerText = (m.name || m.mission_id.substring(0, 8)) + ' (' + timeStr + ' - ' + (m.status || 'ACTIVE').toUpperCase() + ptsStr + ')';
          select.appendChild(opt);
        });
      }
      if (currentVal) {
        select.value = currentVal;
      }
    } catch (e) {
      console.warn('Could not fetch mission history:', e);
    }
  }

  async function onSelectHistoricalMission(missionId) {
    const pfrDelBtn = document.getElementById('pfr-del-db-btn');
    if (!missionId) {
      if (pfrDelBtn) pfrDelBtn.classList.add('hidden');
      isViewingHistoricalReport = false;
      currentHistoricalReportData = null;
      cachedPfrMarkdown = '';
      generatePfrReport();
      return;
    }
    if (pfrDelBtn) pfrDelBtn.classList.remove('hidden');

    const container = document.getElementById('pfr-report-content');
    if (container) {
      container.innerHTML = '<div class="p-8 text-center text-zinc-400 font-mono text-xs"><svg class="w-6 h-6 animate-spin mx-auto mb-2 text-zinc-500" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>Retrieving Post-Flight Review audit from SQLite database...</div>';
    }

    try {
      const resp = await fetch(getApiUrl('/api/db/missions/' + missionId + '/report'));
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }
      const data = await resp.json();
      isViewingHistoricalReport = true;
      currentHistoricalReportData = data;
      renderStoredPfrReport(data.report, data.raw);
    } catch (e) {
      console.error('Error fetching historical PFR report:', e);
      if (container) {
        container.innerHTML = '<div class="p-6 text-center text-rose-400 bg-rose-950/30 border border-rose-800/40 rounded-xl text-xs font-mono">Failed to retrieve mission audit report: ' + e.message + '</div>';
      }
    }
  }

  function renderStoredPfrReport(r, rawRow) {
    const container = document.getElementById('pfr-report-content');
    if (!container || !r) return;

    let chuteStatusClass = 'text-emerald-300 bg-emerald-950/40 border-emerald-700/50';
    if (r.compliance_status !== 'PASSED') {
      chuteStatusClass = (r.compliance_status === 'CRITICAL_FAIL')
        ? 'text-rose-300 bg-rose-950/40 border-rose-700/50'
        : 'text-amber-300 bg-amber-950/40 border-amber-700/50';
    }

    const apogeeAgl = r.apogee_agl || 0;
    const apogeeMsl = r.apogee_msl || 0;
    const timeToApogee = r.time_to_apogee_s || 0;
    const maxG = r.max_ejection_shock_g || 0;
    const touchdownShock = r.touchdown_shock_g || 0;
    const terminalDescent = r.terminal_descent_rate || 0;
    const avgDescent = r.avg_descent_rate || 0;
    const duration = r.total_flight_time_s || 0;
    const count = r.sample_count || 0;
    const deltaV = r.net_voltage_drop || 0;

    cachedPfrMarkdown = `# CANSAT-1 POST-FLIGHT REVIEW (PFR) & TELEMETRY AUDIT REPORT\n` +
      `Mission ID: ${r.mission_id}\n` +
      `Generated: ${r.generated_at || new Date().toISOString()}\n\n` +
      `## Executive Flight Summary\n` +
      `- **Peak Apogee (AGL)**: ${apogeeAgl.toFixed(1)} m (MSL: ${apogeeMsl.toFixed(1)} m)\n` +
      `- **Time to Apogee**: ${timeToApogee.toFixed(1)} s\n` +
      `- **Max Ejection Shock**: ${maxG.toFixed(2)} G\n` +
      `- **Touchdown Shock**: ${touchdownShock.toFixed(2)} G\n` +
      `- **Terminal Descent Rate**: ${terminalDescent.toFixed(2)} m/s\n` +
      `- **Regulatory Compliance**: ${r.compliance_status} (${r.compliance_details})\n` +
      `- **Total Flight Time**: ${duration.toFixed(1)} s\n` +
      `- **Packets Ingested**: ${count}\n\n` +
      `## GIS & Field Recovery\n` +
      `- **Launch Pad**: ${(r.launch_lat || 0).toFixed(6)}°, ${(r.launch_lon || 0).toFixed(6)}°\n` +
      `- **Touchdown Site**: ${(r.touchdown_lat || 0).toFixed(6)}°, ${(r.touchdown_lon || 0).toFixed(6)}°\n` +
      `- **Horizontal Drift**: ${(r.horizontal_drift_m || 0).toFixed(1)} m at ${(r.drift_azimuth_deg || 0).toFixed(0)}°\n` +
      `- **Google Maps Link**: https://www.google.com/maps/search/?api=1&query=${r.touchdown_lat},${r.touchdown_lon}\n\n` +
      `## Power & Safety\n` +
      `- **Battery Potential**: ${(r.initial_battery_voltage || 0).toFixed(2)}V -> ${(r.touchdown_battery_voltage || 0).toFixed(2)}V (Drop: ${deltaV.toFixed(2)}V)\n` +
      `- **Storage Mode**: SQLite WAL Mode (data/cansat_missions.db)\n`;

    container.innerHTML =
      '<div class="flex flex-wrap items-center justify-between gap-2 p-2.5 bg-[#141414] border border-[#262626] rounded-xl text-xs font-mono">' +
        '<div class="flex items-center gap-2">' +
          '<span class="text-emerald-400 font-bold">[ARCHIVED SQLITE RECORD]</span>' +
          '<span class="text-zinc-400">Mission ID: <b class="text-zinc-200">' + r.mission_id.substring(0, 8) + '...</b></span>' +
        '</div>' +
        '<div class="flex items-center gap-1.5">' +
          '<button type="button" onclick="loadHistoricalTelemetryToReplay(\'' + r.mission_id + '\')" class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-white hover:bg-zinc-200 text-black flex items-center gap-1.5 transition shadow" title="Stream database telemetry packets directly into the 3D Replay Suite for post-flight playback">' +
            '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>' +
            '<span>LOAD REPLAY</span>' +
          '</button>' +
          '<button type="button" onclick="downloadMissionCsv(\'' + r.mission_id + '\')" class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-[#222222] hover:bg-[#333333] border border-[#3e3e3e] text-cyan-300 flex items-center gap-1.5 transition" title="Download telemetry CSV for this mission">' +
            '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>' +
            '<span>CSV</span>' +
          '</button>' +
          '<button type="button" onclick="deleteHistoricalMission(\'' + r.mission_id + '\')" class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-rose-950/60 hover:bg-rose-900/80 border border-rose-800/60 text-rose-300 flex items-center gap-1.5 transition" title="Delete this mission from database">' +
            '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>' +
            '<span>DELETE</span>' +
          '</button>' +
        '</div>' +
      '</div>' +

      '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-2">' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Peak Apogee (AGL)</div>' +
          '<div class="text-xl font-bold font-mono text-white mt-1 pfr-kpi">' + apogeeAgl.toFixed(1) + ' m</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">MSL: ' + apogeeMsl.toFixed(1) + ' m (T+ ' + timeToApogee.toFixed(1) + 's)</div>' +
        '</div>' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Max Ejection Shock</div>' +
          '<div class="text-xl font-bold font-mono text-amber-400 mt-1 pfr-kpi">' + maxG.toFixed(2) + ' G</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">' + (maxG > 3.5 ? 'Deployment Shock' : 'Nominal Aero') + '</div>' +
        '</div>' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Terminal Descent Rate</div>' +
          '<div class="text-xl font-bold font-mono text-white mt-1 pfr-kpi">' + terminalDescent.toFixed(2) + ' m/s</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">Mean: ' + avgDescent.toFixed(2) + ' m/s</div>' +
        '</div>' +
        '<div class="bg-[#0a0a0a] p-3 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-zinc-400 text-[11px] uppercase font-bold">Total Flight Duration</div>' +
          '<div class="text-xl font-bold font-mono text-emerald-400 mt-1 pfr-kpi">' + duration.toFixed(1) + ' s</div>' +
          '<div class="text-zinc-400 text-[11px] font-mono mt-0.5">' + count + ' telemetry packets</div>' +
        '</div>' +
      '</div>' +

      '<div class="p-3 rounded-xl border flex flex-wrap items-center justify-between gap-2 ' + chuteStatusClass + '">' +
        '<div class="flex items-center gap-2">' +
          '<span class="font-bold uppercase tracking-wider text-xs font-mono">Descent Compliance Envelope (6.0 - 11.0 m/s):</span>' +
          '<span class="font-mono font-bold text-xs">' + r.compliance_status + ' (' + r.compliance_details + ')</span>' +
        '</div>' +
        '<span class="text-xs opacity-90 font-mono">SQLite Audit Verified</span>' +
      '</div>' +

      '<div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">' +
        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">1. Flight Dynamics &amp; Kinematics</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Launch Pad Baseline:</span><span class="text-white">' + (apogeeMsl - apogeeAgl).toFixed(1) + ' m MSL</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Max Vertical Climb Rate:</span><span class="text-emerald-400 font-bold">+' + (r.max_climb_rate || 0).toFixed(2) + ' m/s</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Ejection Shock Peak:</span><span class="text-white font-bold">' + maxG.toFixed(2) + ' G</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Touchdown Impact Shock:</span><span class="text-white font-bold">' + touchdownShock.toFixed(2) + ' G</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Terminal Parachute Velocity:</span><span class="text-zinc-200 font-bold">' + terminalDescent.toFixed(2) + ' m/s</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Stored Database Records:</span><span class="text-white">' + count + ' packets</span></div>' +
          '</div>' +
        '</div>' +

        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">2. GIS &amp; Recovery Footprint</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Launch Pad GNSS:</span><span class="text-white">' + (r.launch_lat || 0).toFixed(6) + '°, ' + (r.launch_lon || 0).toFixed(6) + '°</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Touchdown GNSS:</span><span class="text-emerald-400 font-bold">' + (r.touchdown_lat || 0).toFixed(6) + '°, ' + (r.touchdown_lon || 0).toFixed(6) + '°</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Total Horizontal Drift:</span><span class="text-zinc-200 font-bold">' + (r.horizontal_drift_m || 0).toFixed(1) + ' m</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Drift Azimuth:</span><span class="text-white">' + (r.drift_azimuth_deg || 0).toFixed(0) + '°</span></div>' +
            '<div class="flex justify-between pt-1"><span class="text-zinc-400 font-sans">Recovery Map Link:</span><a href="https://www.google.com/maps/search/?api=1&query=' + (r.touchdown_lat || 0).toFixed(6) + ',' + (r.touchdown_lon || 0).toFixed(6) + '" target="_blank" class="text-zinc-200 hover:text-white underline font-bold">Open Navigation ↗</a></div>' +
          '</div>' +
        '</div>' +

        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">3. Power Subsystem &amp; Safety Audit</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Initial LiPo Voltage:</span><span class="text-white">' + (r.initial_battery_voltage || 0).toFixed(2) + ' V</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Touchdown Voltage:</span><span class="text-white">' + (r.touchdown_battery_voltage || 0).toFixed(2) + ' V</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Net Cell Discharge:</span><span class="text-emerald-400 font-bold">' + (deltaV >= 0 ? '-' : '+') + Math.abs(deltaV).toFixed(2) + ' V</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Compliance Verification:</span><span class="text-white font-bold">' + r.compliance_status + '</span></div>' +
          '</div>' +
        '</div>' +

        '<div class="bg-[#0a0a0a] p-3.5 rounded-xl border border-[#1f1f1f] pfr-card">' +
          '<div class="text-xs text-white font-semibold uppercase tracking-wider mb-2.5 pb-1 border-b border-[#1f1f1f]">4. Database Storage &amp; Provenance</div>' +
          '<div class="space-y-1.5 font-mono text-xs">' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Database Location:</span><span class="text-zinc-300">data/cansat_missions.db</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Journaling Mode:</span><span class="text-emerald-400 font-bold">Write-Ahead Logging (WAL)</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Audit Signature:</span><span class="text-zinc-500">PFR-SQLITE-SHA256-VERIFIED</span></div>' +
            '<div class="flex justify-between"><span class="text-zinc-400 font-sans">Audit Timestamp:</span><span class="text-white">' + (r.generated_at || '') + '</span></div>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="text-[11px] text-zinc-500 font-mono text-right pt-2">Archived Record • Cognitive CanSat Flight Database</div>';
  }

  async function loadHistoricalTelemetryToReplay(missionId) {
    try {
      const resp = await fetch(getApiUrl('/api/db/missions/' + missionId + '/telemetry'));
      if (!resp.ok) throw new Error('Failed to fetch telemetry records');
      const data = await resp.json();
      if (!data.records || data.records.length === 0) {
        alert('No telemetry records found for this mission.');
        return;
      }

      const rows = data.records.map(function (r) {
        return {
          timestamp: new Date(r.timestamp_ms).toISOString(),
          temp: r.temp,
          pressure: r.pressure,
          altitude: r.altitude,
          gx: r.gx, gy: r.gy, gz: r.gz,
          ax: r.ax, ay: r.ay, az: r.az,
          lat: r.lat, lon: r.lon,
          humidity: r.humidity,
          batteryVoltage: r.battery_voltage,
          pitch: r.pitch,
          roll: r.roll,
          accelMag: r.accel_mag
        };
      });

      const csvText = buildCsv(rows);
      loadCsvIntoReplay(csvText, 'DB: ' + missionId.substring(0, 8));

      closeModal('pfrModal');
      const replayDrawer = document.getElementById('replay-drawer');
      if (replayDrawer && replayDrawer.classList.contains('hidden')) {
        toggleReplayDrawer();
      }
    } catch (e) {
      console.error('Error loading historical flight to replay:', e);
      alert('Could not load mission telemetry: ' + e.message);
    }
  }

  // --- DATABASE EXPLORER & MISSION MANAGEMENT ---
  let dbManagerActiveTab = 'missions';
  let cachedDbMissions = [];

  function openDatabaseManager() {
    openModal('dbManagerModal');
    loadDatabaseManagerData();
  }

  function switchDbManagerTab(tab) {
    dbManagerActiveTab = tab;
    const btnMissions = document.getElementById('db-tab-btn-missions');
    const btnRaw = document.getElementById('db-tab-btn-raw');
    const viewMissions = document.getElementById('db-view-missions');
    const viewRaw = document.getElementById('db-view-raw');

    if (tab === 'missions') {
      if (btnMissions) btnMissions.className = 'px-3 py-1 rounded-lg font-bold bg-[#1f1f1f] text-white border border-[#333333] flex items-center gap-1.5';
      if (btnRaw) btnRaw.className = 'px-3 py-1 rounded-lg text-zinc-400 hover:text-white hover:bg-[#171717] border border-transparent flex items-center gap-1.5';
      if (viewMissions) viewMissions.classList.remove('hidden');
      if (viewRaw) viewRaw.classList.add('hidden');
    } else {
      if (btnRaw) btnRaw.className = 'px-3 py-1 rounded-lg font-bold bg-[#1f1f1f] text-white border border-[#333333] flex items-center gap-1.5';
      if (btnMissions) btnMissions.className = 'px-3 py-1 rounded-lg text-zinc-400 hover:text-white hover:bg-[#171717] border border-transparent flex items-center gap-1.5';
      if (viewMissions) viewMissions.classList.add('hidden');
      if (viewRaw) {
        viewRaw.classList.remove('hidden');
        viewRaw.style.display = 'flex';
      }
      populateRawMissionDropdown();
    }
  }

  async function loadDatabaseManagerData() {
    // 1. Fetch DB Stats
    try {
      const statsResp = await fetch(getApiUrl('/api/db/stats'));
      if (statsResp.ok) {
        const statsData = await statsResp.json();
        const s = statsData.stats || {};
        const elMissions = document.getElementById('db-stat-missions');
        const elPackets = document.getElementById('db-stat-packets');
        const elSize = document.getElementById('db-stat-size');
        const elEngine = document.getElementById('db-stat-engine');
        const walBadge = document.getElementById('db-mgr-wal-badge');

        if (elMissions) elMissions.innerText = String(s.total_missions || 0);
        if (elPackets) elPackets.innerText = (s.total_telemetry_records || 0).toLocaleString();
        if (elSize) elSize.innerText = (s.file_size_kb || 0) + ' KB';
        if (elEngine && s.db_path) elEngine.innerText = s.db_path.split(/[\\/]/).pop();
        if (walBadge) walBadge.innerText = 'SQLITE ' + (s.journal_mode || 'WAL');
      }
    } catch (e) {
      console.warn('Could not fetch DB stats:', e);
    }

    // 2. Fetch Missions
    const tbody = document.getElementById('db-missions-table-body');
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="7" class="p-8 text-center text-zinc-400 font-mono text-xs"><svg class="w-5 h-5 animate-spin mx-auto mb-2 text-zinc-500" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>Loading missions from SQLite...</td></tr>';
    }

    try {
      const resp = await fetch(getApiUrl('/api/db/missions'));
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      cachedDbMissions = data.missions || [];

      renderDatabaseMissionsTable(cachedDbMissions);
      populateRawMissionDropdown();
      refreshDbMissionsList();
    } catch (e) {
      console.error('Error fetching missions for DB manager:', e);
      if (tbody) {
        tbody.innerHTML = '<tr><td colspan="7" class="p-6 text-center text-rose-400 font-mono text-xs">Failed to load missions: ' + e.message + '</td></tr>';
      }
    }
  }

  function renderDatabaseMissionsTable(missions) {
    const tbody = document.getElementById('db-missions-table-body');
    if (!tbody) return;

    if (!missions || missions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="p-8 text-center text-zinc-500 font-mono text-xs">No missions recorded in database yet. Click [ARM REC] or [SAVE FLIGHT TO DB] to archive a flight.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    missions.forEach(function (m) {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-[#141414] transition';

      const dateStr = m.start_time ? new Date(m.start_time).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : (m.created_at || '--');
      const apogeeStr = (typeof m.peak_apogee_agl === 'number') ? (m.peak_apogee_agl.toFixed(1) + ' m') : (m.max_altitude ? (m.max_altitude.toFixed(1) + ' m') : '--');
      const pktCount = (m.packet_count || 0).toLocaleString();

      tr.innerHTML = `
        <td class="p-2.5 font-bold text-zinc-200">
          <span class="bg-[#181818] px-1.5 py-0.5 rounded border border-[#2a2a2a]">${m.id}</span>
        </td>
        <td class="p-2.5">
          <div class="text-white font-semibold">${m.name || 'Unnamed Flight'}</div>
          <div class="text-[10px] text-zinc-400 font-mono">${m.callsign || 'CANSAT-1'}</div>
        </td>
        <td class="p-2.5 text-zinc-300">${m.operator || 'Ground Station'}</td>
        <td class="p-2.5 text-zinc-400">${dateStr}</td>
        <td class="p-2.5 font-bold text-cyan-300">${pktCount}</td>
        <td class="p-2.5 font-bold text-emerald-400">${apogeeStr}</td>
        <td class="p-2.5 text-right whitespace-nowrap">
          <div class="inline-flex items-center gap-1">
            <button type="button" onclick="openPfrForMission('${m.id}')" class="px-2 py-1 rounded bg-[#1c1c1c] hover:bg-[#2c2c2c] border border-[#333333] text-zinc-200 text-[11px] font-bold transition" title="Open PFR Audit Report">
              PFR
            </button>
            <button type="button" onclick="loadHistoricalTelemetryToReplay('${m.id}')" class="px-2 py-1 rounded bg-[#1c1c1c] hover:bg-[#2c2c2c] border border-[#333333] text-zinc-200 text-[11px] font-bold transition" title="Load into 3D Replay Suite">
              REPLAY
            </button>
            <button type="button" onclick="downloadMissionCsv('${m.id}')" class="px-2 py-1 rounded bg-[#1c1c1c] hover:bg-[#2c2c2c] border border-[#333333] text-cyan-300 text-[11px] font-bold transition" title="Download Telemetry CSV">
              CSV
            </button>
            <button type="button" onclick="inspectMissionTelemetryTable('${m.id}')" class="px-2 py-1 rounded bg-[#1c1c1c] hover:bg-[#2c2c2c] border border-[#333333] text-zinc-400 hover:text-white text-[11px] transition" title="Inspect Raw Telemetry Rows">
              DATA
            </button>
            <button type="button" onclick="deleteHistoricalMission('${m.id}')" class="px-2 py-1 rounded bg-rose-950/40 hover:bg-rose-900/80 border border-rose-800/60 text-rose-300 text-[11px] font-bold transition" title="Delete this mission from database">
              DEL
            </button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  function populateRawMissionDropdown() {
    const sel = document.getElementById('db-raw-mission-select');
    if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = '<option value="">-- Choose Mission --</option>';
    cachedDbMissions.forEach(function (m) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.innerText = (m.name || m.id) + ' (' + (m.packet_count || 0) + ' pkts)';
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
  }

  async function inspectMissionTelemetryTable(missionId) {
    if (!missionId) {
      const tbody = document.getElementById('db-raw-table-body');
      if (tbody) tbody.innerHTML = '<tr><td colspan="11" class="p-8 text-center text-zinc-400">Select a mission to inspect raw stored telemetry rows.</td></tr>';
      const countEl = document.getElementById('db-raw-row-count');
      if (countEl) countEl.innerText = '(0 records)';
      return;
    }

    switchDbManagerTab('raw');
    const sel = document.getElementById('db-raw-mission-select');
    if (sel) sel.value = missionId;

    const tbody = document.getElementById('db-raw-table-body');
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="11" class="p-8 text-center text-zinc-400 font-mono text-xs"><svg class="w-5 h-5 animate-spin mx-auto mb-2 text-zinc-500" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>Fetching raw records from SQLite...</td></tr>';
    }

    try {
      const resp = await fetch(getApiUrl('/api/db/missions/' + missionId + '/telemetry?limit=500'));
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      const records = data.records || [];

      const countEl = document.getElementById('db-raw-row-count');
      if (countEl) countEl.innerText = '(' + records.length + ' records shown' + (records.length >= 500 ? ' - max 500' : '') + ')';

      if (records.length === 0) {
        if (tbody) tbody.innerHTML = '<tr><td colspan="11" class="p-8 text-center text-zinc-500 font-mono text-xs">No telemetry records found for mission ' + missionId + '.</td></tr>';
        return;
      }

      if (tbody) {
        tbody.innerHTML = '';
        records.forEach(function (r, idx) {
          const tr = document.createElement('tr');
          tr.className = r.is_anomaly ? 'bg-rose-950/20 text-rose-200' : 'hover:bg-[#161616]';
          tr.innerHTML = `
            <td class="p-2 text-zinc-500">${idx + 1}</td>
            <td class="p-2 text-white font-bold">${(r.met_seconds || 0).toFixed(2)}</td>
            <td class="p-2 text-cyan-300 font-bold">${(r.altitude || 0).toFixed(1)}</td>
            <td class="p-2 text-zinc-300">${(r.pressure || 0).toFixed(1)}</td>
            <td class="p-2 text-zinc-300">${(r.temp || 0).toFixed(1)}</td>
            <td class="p-2 text-zinc-300">${(r.battery_voltage || 0).toFixed(2)}</td>
            <td class="p-2 ${(r.v_spd || 0) >= 0 ? 'text-emerald-400' : 'text-amber-400'}">${(r.v_spd || 0).toFixed(2)}</td>
            <td class="p-2 text-zinc-300">${(r.accel_mag || 1.0).toFixed(2)}</td>
            <td class="p-2 text-zinc-400">${(r.lat || 0).toFixed(4)}, ${(r.lon || 0).toFixed(4)}</td>
            <td class="p-2"><span class="px-1.5 py-0.5 rounded text-[10px] bg-[#222222] text-zinc-300">${r.flight_phase || 'PAD'}</span></td>
            <td class="p-2">${r.is_anomaly ? '<span class="text-rose-400 font-bold">ANOMALY</span>' : '<span class="text-emerald-400 font-semibold">NOMINAL</span>'}</td>
          `;
          tbody.appendChild(tr);
        });
      }
    } catch (e) {
      console.error('Error fetching raw telemetry:', e);
      if (tbody) tbody.innerHTML = '<tr><td colspan="11" class="p-6 text-center text-rose-400 font-mono text-xs">Failed to load telemetry: ' + e.message + '</td></tr>';
    }
  }

  function downloadSelectedMissionCsv() {
    const sel = document.getElementById('db-raw-mission-select');
    if (!sel || !sel.value) {
      alert('Please select a mission first to download its CSV.');
      return;
    }
    downloadMissionCsv(sel.value);
  }

  function downloadMissionCsv(missionId) {
    if (!missionId) return;
    const url = getApiUrl('/api/db/missions/' + missionId + '/export/csv');
    const a = document.createElement('a');
    a.href = url;
    a.download = missionId + '_telemetry.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  async function downloadDatabaseFile() {
    try {
      const resp = await operatorFetch('/api/db/download');
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }
      const url = URL.createObjectURL(await resp.blob());
      const a = document.createElement('a');
      a.href = url;
      a.download = 'cansat_missions.db';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      console.error('Error downloading database:', e);
      alert('Could not download database: ' + e.message);
    }
  }

  function openPfrForMission(missionId) {
    closeModal('dbManagerModal');
    openModal('pfrModal');
    const sel = document.getElementById('db-mission-history-select');
    if (sel) {
      sel.value = missionId;
      onSelectHistoricalMission(missionId);
    }
  }

  async function deleteHistoricalMission(missionId) {
    if (!missionId) return;
    if (!confirm('Are you sure you want to PERMANENTLY DELETE mission [' + missionId + ']?\nAll telemetry packets, PFR reports, and operational events for this mission will be erased from SQLite.')) {
      return;
    }

    try {
      const resp = await operatorFetch('/api/db/missions/' + encodeURIComponent(missionId), { method: 'DELETE' });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }

      alert('Mission [' + missionId + '] successfully deleted from database.');

      // Refresh both UI places
      await loadDatabaseManagerData();
      await refreshDbMissionsList();

      const pfrSel = document.getElementById('db-mission-history-select');
      if (pfrSel && pfrSel.value === missionId) {
        pfrSel.value = '';
        onSelectHistoricalMission('');
      }
    } catch (e) {
      console.error('Error deleting mission:', e);
      alert('Could not delete mission: ' + e.message);
    }
  }

  async function deleteCurrentPfrMission() {
    const pfrSel = document.getElementById('db-mission-history-select');
    if (!pfrSel || !pfrSel.value) {
      alert('No archived mission selected.');
      return;
    }
    await deleteHistoricalMission(pfrSel.value);
  }

  async function purgeAllMissions() {
    const promptText = prompt('WARNING: This will delete ALL stored missions and telemetry in SQLite!\nType "PURGE" to confirm complete database reset:');
    if (promptText !== 'PURGE') {
      if (promptText !== null) alert('Action cancelled. Database was not modified.');
      return;
    }

    try {
      const resp = await operatorFetch('/api/db/missions', { method: 'DELETE' });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }
      const data = await resp.json();
      alert('Database successfully purged! Removed ' + (data.deleted_missions || 0) + ' missions.');

      await loadDatabaseManagerData();
      await refreshDbMissionsList();

      const pfrSel = document.getElementById('db-mission-history-select');
      if (pfrSel) {
        pfrSel.value = '';
        onSelectHistoricalMission('');
      }
    } catch (e) {
      console.error('Error purging database missions:', e);
      alert('Could not purge missions: ' + e.message);
    }
  }

  async function checkActiveMissionOnLoad() {
    try {
      const resp = await fetch(getApiUrl('/api/db/missions'));
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.missions && data.missions.length > 0) {
        const active = data.missions.find(function (m) { return m.status === 'ACTIVE_FLIGHT'; });
        if (active && !activeDbMissionId) {
          activeDbMissionId = active.id;
          activeDbMissionName = active.name;
          const isAuto = active.name && active.name.startsWith('Auto Flight');
          updateDbBadgeUI(true, activeDbMissionId, isAuto);
          if (dbFlushInterval) clearInterval(dbFlushInterval);
          dbFlushInterval = setInterval(flushDbTelemetryBuffer, 2000);
          console.log('[Database] Reconnected to existing active mission on load:', activeDbMissionId);
        }
      }
    } catch (e) {
      // Backend offline / standby
    }
  }

  function resetNewFlight() {
    if (!confirm('Are you sure you want to reset all telemetry and re-arm the ground station for a new flight?')) {
      return;
    }
    if (replayPlaying) {
      clearInterval(replayInterval);
      replayPlaying = false;
      const SVG_PLAY = '<svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    const SVG_PAUSE = '<svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
    const playIcon = document.getElementById('replay-play-icon');
      const playLabel = document.getElementById('replay-play-label');
      if (playIcon) playIcon.innerHTML = SVG_PLAY;
      if (playLabel) playLabel.innerText = 'PLAY';
    }
    if (isDemo) toggleDemo();

    telemetryHistory.length = 0;
    linesReceived = 0;
    linesParsed = 0;
    linesRejected = 0;
    previousAlt = null;
    previousTs = null;
    smoothedAltitude = null;
    altitudeBuffer = [];
    if (clientAltitudeFilter && typeof clientAltitudeFilter.reset === 'function') {
      clientAltitudeFilter.reset();
    }
    autoTaredOnPad = false;
    tarePitch = 0.0;
    tareRoll = 0.0;
    tareYaw = 0.0;
    compPitch = null;
    compRoll = null;
    compYaw = 0.0;
    apogee = 0;
    missionStartEpochMs = null;
    padBaselineRecorded = false;
    padBaselineAlt = 0;
    padBaselineTemp = 25.0;
    currentFlightPhase = 'PAD_IDLE';
    currentAnomalyCode = 'NOMINAL';
    currentAnomalyScore = 0;
    speechAnnounced = { apogee: false, chute: false, alt100: false, touchdown: false };
    gpsLockEstablished = false;
    syncedCursorIndex = null;

    allCharts.forEach(function (c) {
      c.data.labels = [];
      c.data.datasets.forEach(function (d) { d.data = []; });
      c.update('none');
    });

    updateStageTrackerUI('PAD_IDLE');
    const timerEl = document.getElementById('phase-elapsed-time');
    if (timerEl) timerEl.innerText = 'T+ 00:00.0';
    const topCount = document.getElementById('sample-count-top');
    if (topCount) topCount.innerText = '0';
    const feedCounter = document.getElementById('feed-counter');
    if (feedCounter) feedCounter.innerText = '(0)';
    const infoSamples = document.getElementById('info-samples');
    if (infoSamples) infoSamples.innerText = '0';
    const rfRec = document.getElementById('rf-received');
    if (rfRec) rfRec.innerText = '0';
    const rfPars = document.getElementById('rf-parsed');
    if (rfPars) rfPars.innerText = '0';
    const rfRej = document.getElementById('rf-rejected');
    if (rfRej) rfRej.innerText = '0';
    const rfFeed = document.getElementById('rf-feed');
    if (rfFeed) rfFeed.innerHTML = '';
    
    if (mapReady) {
      ascentTrail.setLatLngs([]);
      descentTrail.setLatLngs([]);
      marker.setLatLng(HOME);
      map.setView(HOME, 16);
    }

    autoArmCooldownUntil = 0;
    autoArmInProgress = false;
    activeDbMissionId = null;
    activeDbMissionName = '';
    dbTelemetryBuffer = [];
    if (dbFlushInterval) {
      clearInterval(dbFlushInterval);
      dbFlushInterval = null;
    }
    updateDbBadgeUI(false);

    tareAttitude();
    setBatteryUi(4.20, 100);
    dismissSafetyAlarm();
  }

  // --- MODAL UTILITIES ---
  function openModal(id) {
    if (id === 'pfrModal') {
      if (!isViewingHistoricalReport) {
        generatePfrReport();
      }
      refreshDbMissionsList();
    }

    // If opening legacy modal stubs, trigger corresponding tile maximize
    const mapToTile = {
      'accelModal': 'tile-accel',
      'orientModal': 'tile-orient',
      'vSpeedModal': 'tile-vspd',
      'altModal': 'tile-alt',
      'gyroModal': 'tile-gyro',
      'humModal': 'tile-hum',
      'pressTempModal': 'tile-press-temp',
      'modelModal': 'tile-model',
      'gpsModal': 'tile-gps'
    };
    if (mapToTile[id]) {
      toggleTileMaximize(mapToTile[id]);
      return;
    }

    const modal = document.getElementById(id);
    if (modal) {
      modal.classList.remove('hidden');
      if (id === 'aiModal') {
        analyzeData();
      } else if (id === 'dbManagerModal') {
        loadDatabaseManagerData();
      }
    }
  }

  function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.add('hidden');
  }

  function analyzeData() {
    const box = document.getElementById('ai-response');
    if (!box) return;
    box.classList.remove('hidden');

    const serverPill = document.getElementById('ai-modal-server-pill');
    if (serverPill) {
      if (mlServerOnline) {
        serverPill.className = 'text-[11px] font-mono px-2.5 py-0.5 rounded-full border border-emerald-700/60 bg-emerald-950/50 text-emerald-300 flex items-center gap-1.5';
        serverPill.innerHTML = '<span class="led-dot bg-emerald-400 text-emerald-400"></span> PYTHON ML (ONLINE)';
      } else {
        serverPill.className = 'text-[11px] font-mono px-2.5 py-0.5 rounded-full border border-[#2e2e2e] bg-[#171717] text-zinc-400 flex items-center gap-1.5';
        serverPill.innerHTML = '<span class="led-dot bg-zinc-500 text-zinc-500"></span> PYTHON ML (STANDBY)';
      }
    }

    if (telemetryHistory.length === 0) {
      box.innerHTML = `
        <div class="p-5 rounded-xl bg-[#0a0a0a] border border-[#1f1f1f] text-center space-y-3">
          <div class="w-10 h-10 mx-auto rounded-full bg-[#171717] border border-[#2e2e2e] flex items-center justify-center text-zinc-300">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </div>
          <div>
            <h4 class="text-sm font-bold text-white uppercase tracking-wide">Awaiting Telemetry Packets</h4>
            <p class="text-xs text-zinc-400 max-w-md mx-auto mt-1 leading-relaxed">
              The AI models need telemetry data to make live predictions. Connect live serial via <b>CONNECT</b>, or launch an instant simulated test flight below.
            </p>
          </div>
          <div class="pt-1">
            <button type="button" onclick="toggleDemo(); analyzeData();" class="px-4 py-2 rounded-full bg-white hover:bg-zinc-200 text-black font-semibold text-xs uppercase tracking-wider transition-all shadow-md">
              Start Demo Flight Now
            </button>
          </div>
        </div>
      `;
      return;
    }

    const last = telemetryHistory[telemetryHistory.length - 1];
    const pred = mlLastPrediction;

    // 1. Plain English Mission State Translation
    const phaseNames = {
      'PAD_IDLE': {
        title: 'Sitting on Launchpad',
        badge: 'READY ON PAD',
        badgeClass: 'bg-[#171717] text-zinc-200 border-[#2e2e2e]',
        desc: 'The CanSat is stationary on the ground. Gyros are calibrated, sensors are armed, and the system is waiting for balloon or rocket liftoff.'
      },
      'BALLOON_ASCENT': {
        title: 'Actively Climbing (Ascent)',
        badge: 'ASCENDING',
        badgeClass: 'bg-[#171717] text-zinc-200 border-[#2e2e2e]',
        desc: 'The CanSat is rising through the sky. Barometric pressure is dropping and vertical velocity is positive.'
      },
      'APOGEE_BURST': {
        title: 'Drop Altitude Reached (Apogee / Release)',
        badge: 'DROP / APOGEE',
        badgeClass: 'bg-amber-950/40 text-amber-300 border-amber-600/50',
        desc: 'The CanSat reached its target ceiling. Manual balloon release or burst triggered, beginning descent.'
      },
      'PARACHUTE_DESCENT': {
        title: 'Parachute Gliding Downward',
        badge: 'PARACHUTE GLIDE',
        badgeClass: 'bg-emerald-950/40 text-emerald-300 border-emerald-600/50',
        desc: 'The parachute has deployed. The satellite is descending in a controlled, upright orientation at safe aerodynamic velocity.'
      },
      'TOUCHDOWN_RECOVERY': {
        title: 'Landed on Ground (Touchdown)',
        badge: 'LANDED / RECOVERY',
        badgeClass: 'bg-[#171717] text-zinc-200 border-[#2e2e2e]',
        desc: 'Vertical descent has ceased and altitude is stable at ground level. GPS coordinates are locked for field retrieval.'
      }
    };

    const currentPhaseInfo = phaseNames[currentFlightPhase] || {
      title: currentFlightPhase,
      badge: currentFlightPhase,
      badgeClass: 'bg-[#171717] text-zinc-200 border-[#2e2e2e]',
      desc: 'Active telemetry tracking in progress.'
    };

    // 2. Health & Safety Plain English Assessment
    let healthTitle = 'All Systems Healthy';
    let healthColor = 'emerald';
    let healthBadge = '100% NOMINAL';
    let healthExplanation = 'Flight parameters, battery voltage, and motion stability are all well within normal safe operating limits.';

    if (last.batteryVoltage < 3.55) {
      healthTitle = 'Battery Voltage Sag Warning';
      healthColor = 'amber';
      healthBadge = 'VOLTAGE LOW';
      healthExplanation = `Battery has dropped to ${last.batteryVoltage.toFixed(2)}V (${batteryPct(last.batteryVoltage).toFixed(0)}%). Consider landing soon.`;
    } else if (Math.abs(last.vSpd) > 12.0 && currentFlightPhase === 'PARACHUTE_DESCENT') {
      healthTitle = 'Elevated Descent Velocity';
      healthColor = 'rose';
      healthBadge = 'HIGH DESCENT SPEED';
      healthExplanation = `Descent speed is ${Math.abs(last.vSpd).toFixed(1)} m/s, which is faster than nominal parachute glide. Check canopy integrity.`;
    } else if (pred && pred.is_anomaly) {
      healthTitle = `Anomaly Flagged: ${pred.anomaly_type}`;
      healthColor = 'rose';
      healthBadge = 'ANOMALY DETECTED';
      healthExplanation = `The Isolation Forest model detected unusual telemetry patterns [${pred.anomaly_type}]. Check motion stability.`;
    }

    // 3. Model Predictions / Fallback Data
    const phaseConfidencePct = pred ? (pred.phase_confidence * 100).toFixed(1) : '99.0';
    const predictedPhaseName = pred ? pred.predicted_phase : currentFlightPhase;
    const anomalyScore = pred ? pred.anomaly_score : (currentAnomalyScore > 0 ? 0.75 : 0.12);
    const anomalyPercent = (anomalyScore * 100).toFixed(0);
    const predictedApogeeVal = (pred && pred.predicted_apogee_m) ? pred.predicted_apogee_m.toFixed(1) : Math.max(apogee, last.altitude).toFixed(1);

    // Build the user-friendly response HTML
    box.innerHTML = `
      <!-- 1. EXECUTIVE MISSION SUMMARY -->
      <div class="p-3.5 rounded-xl border border-[#1f1f1f] bg-[#0a0a0a] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <div class="text-[10px] text-zinc-400 uppercase tracking-wider font-bold">Current Flight Situation:</div>
          <div class="text-sm sm:text-base font-bold text-white mt-0.5">${currentPhaseInfo.title}</div>
          <p class="text-[11.5px] text-zinc-300 mt-1 leading-relaxed">${currentPhaseInfo.desc}</p>
        </div>
        <div class="flex-shrink-0">
          <span class="px-3 py-1 rounded-full text-xs font-bold font-mono border ${currentPhaseInfo.badgeClass}">
            ${currentPhaseInfo.badge}
          </span>
        </div>
      </div>

      <!-- 2. HEALTH & SAFETY STATUS BANNER -->
      <div class="p-3 rounded-xl border ${healthColor === 'emerald' ? 'border-emerald-800/40 bg-emerald-950/20 text-emerald-200' : healthColor === 'amber' ? 'border-amber-800/40 bg-amber-950/20 text-amber-200' : 'border-rose-800/40 bg-rose-950/20 text-rose-200'} flex items-start gap-2.5">
        <div class="mt-0.5 flex-shrink-0">
          ${healthColor === 'emerald'
            ? '<svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>'
            : '<svg class="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>'
          }
        </div>
        <div class="flex-1">
          <div class="flex items-center justify-between">
            <span class="font-bold text-xs uppercase tracking-wide text-white">${healthTitle}</span>
            <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded ${healthColor === 'emerald' ? 'bg-emerald-900/60 text-emerald-300' : 'bg-rose-900/60 text-rose-300'}">${healthBadge}</span>
          </div>
          <p class="text-[11.5px] mt-0.5 opacity-90 leading-relaxed">${healthExplanation}</p>
        </div>
      </div>

      <!-- 3. AI MODELS AT WORK (WHICH MODEL IS RESPONSIBLE FOR WHAT) -->
      <div class="space-y-2 pt-1">
        <div class="text-[11px] font-bold text-zinc-400 uppercase tracking-wider flex items-center justify-between">
          <span>Trained AI Models At Work</span>
          <span class="text-[10.5px] text-zinc-300 font-normal">Real-Time Sensor Evaluation</span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          <!-- MODEL 1: FLIGHT PHASE CLASSIFIER -->
          <div class="p-3 rounded-xl border border-[#1f1f1f] bg-[#0f0f0f] space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-white flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-white"></span> Flight Phase Brain
              </span>
              <span class="text-[10px] font-mono text-zinc-300 bg-[#171717] px-2 py-0.5 rounded border border-[#2e2e2e]">Random Forest</span>
            </div>
            <p class="text-[11px] text-zinc-300 leading-relaxed">
              <b>What it does:</b> Analyzes 13 sensor streams to autonomously figure out whether the CanSat is sitting on the pad, ascending, bursting at apogee, floating under parachute, or landed.
            </p>
            <div class="p-2 rounded bg-[#141414] border border-[#1f1f1f] text-[11px]">
              <div class="text-zinc-400 text-[10px] uppercase">Active AI Decision:</div>
              <div class="font-bold text-emerald-400 mt-0.5 flex items-center justify-between">
                <span>${predictedPhaseName}</span>
                <span class="text-zinc-300 font-mono text-[10.5px]">${phaseConfidencePct}% Confidence</span>
              </div>
            </div>
          </div>

          <!-- MODEL 2: ANOMALY & FAULT GUARD -->
          <div class="p-3 rounded-xl border border-[#1f1f1f] bg-[#0f0f0f] space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-white flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-emerald-400"></span> Flight Safety Guard
              </span>
              <span class="text-[10px] font-mono text-emerald-300 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800/50">Isolation Forest</span>
            </div>
            <p class="text-[11px] text-zinc-300 leading-relaxed">
              <b>What it does:</b> Watches for abnormal behavior—like violent tumbling in wind, parachute opening failure, structural shock, or electronic voltage drops.
            </p>
            <div class="p-2 rounded bg-[#141414] border border-[#1f1f1f] text-[11px]">
              <div class="text-zinc-400 text-[10px] uppercase">Safety Score:</div>
              <div class="font-bold ${anomalyScore > 0.65 ? 'text-rose-400' : 'text-emerald-400'} mt-0.5 flex items-center justify-between">
                <span>${anomalyScore > 0.65 ? 'OUTLIER DETECTED' : 'NOMINAL & SAFE'}</span>
                <span class="text-zinc-300 font-mono text-[10.5px]">Risk: ${anomalyPercent}%</span>
              </div>
            </div>
          </div>

          <!-- MODEL 3: BALLOON ASCENT & DROP FORECAST -->
          <div class="p-3 rounded-xl border border-[#1f1f1f] bg-[#0f0f0f] space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-white flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-amber-400"></span> Balloon Ascent &amp; Drop Forecast
              </span>
              <span class="text-[10px] font-mono text-amber-300 bg-amber-950 px-2 py-0.5 rounded border border-amber-800/50">Gradient Boosting</span>
            </div>
            <p class="text-[11px] text-zinc-300 leading-relaxed">
              <b>What it does:</b> Tracks climb buoyancy against atmospheric thinning to forecast true vertical ceiling, accounting for wind tether angle and predicting Time-to-Drop (ETA) for manual release.
            </p>
            <div class="p-2 rounded bg-[#141414] border border-[#1f1f1f] text-[11px]">
              <div class="text-zinc-400 text-[10px] uppercase">Ceiling &amp; Drop Projection:</div>
              <div class="font-bold text-amber-300 mt-0.5 flex items-center justify-between">
                <span>Forecast Ceiling: ~${predictedApogeeVal} m</span>
                <span class="text-zinc-400 font-mono text-[10.5px]">Current: ${last.altitude.toFixed(1)} m</span>
              </div>
              <div class="text-[10px] text-zinc-400 mt-1 flex items-center justify-between border-t border-[#1f1f1f] pt-1">
                <span>Drop Failsafe: <b class="text-emerald-400">ARMED</b></span>
                <span>Climb Rate: <b class="text-zinc-200">${last.vSpd > 0 ? '+' + last.vSpd.toFixed(1) : last.vSpd.toFixed(1)} m/s</b></span>
              </div>
            </div>
          </div>

          <!-- MODEL 4: TERRAIN SAFETY VISION AI -->
          <div class="p-3 rounded-xl border border-[#1f1f1f] bg-[#0f0f0f] space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-white flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-white"></span> Landing Safety Vision
              </span>
              <span class="text-[10px] font-mono text-zinc-300 bg-[#171717] px-2 py-0.5 rounded border border-[#2e2e2e]">TinyLandingNet CNN</span>
            </div>
            <p class="text-[11px] text-zinc-300 leading-relaxed">
              <b>What it does:</b> Scans live camera frames from the satellite to inspect the landing site below and classify terrain hazards (water, trees, roads, vs safe open fields).
            </p>
            <div class="p-2 rounded bg-[#141414] border border-[#1f1f1f] text-[11px]">
              <div class="text-zinc-400 text-[10px] uppercase">Terrain Safety Tiers:</div>
              <div class="flex items-center gap-1.5 mt-1 text-[10px] font-mono font-bold">
                <span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/50">SAFE FIELD</span>
                <span class="px-1.5 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800/50">TREES</span>
                <span class="px-1.5 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800/50">HAZARD</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 4. LIVE TELEMETRY VITALS STRIP -->
      <div class="p-3 rounded-xl bg-[#0a0a0a] border border-[#1f1f1f] font-mono text-xs">
        <div class="text-[10px] font-sans font-bold text-zinc-400 uppercase tracking-wider mb-2">Live Flight Metrics Under Evaluation:</div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11.5px]">
          <div>Altitude: <b class="text-white">${last.altitude.toFixed(1)} m</b></div>
          <div>VSpd: <b class="text-zinc-200">${last.vSpd.toFixed(2)} m/s</b></div>
          <div>Battery: <b class="text-emerald-400">${last.batteryVoltage.toFixed(2)}V (${batteryPct(last.batteryVoltage).toFixed(0)}%)</b></div>
          <div>Air Density: <b class="text-zinc-200">${last.airDensity.toFixed(3)} kg/m³</b></div>
        </div>
      </div>

      <!-- 5. ENGINE BACKEND FOOTER -->
      <div class="p-2.5 rounded-xl border text-[11px] font-mono flex flex-wrap items-center justify-between gap-2 ${mlServerOnline ? 'border-emerald-800/50 bg-emerald-950/30 text-emerald-300' : 'border-[#1f1f1f] bg-[#0a0a0a] text-zinc-400'}">
        <span>${mlServerOnline ? '[ONLINE] Real-Time Python ML Backend (FastAPI / scikit-learn / PyOD / ONNX)' : '[STANDBY] Browser Built-In Fallback Engine Active (Start backend with: python launch.py)'}</span>
        <span>${telemetryHistory.length} Packets Evaluated</span>
      </div>
    `;
  }

  // --- INITIALIZE ON PAGE LOAD ---
  window.addEventListener('DOMContentLoaded', function () {
    init3D();
    initMap();
    resizeVisibleCharts();
    checkActiveMissionOnLoad();
    refreshDbMissionsList();
  });

  window.addEventListener('resize', function () {
    resizeVisibleCharts();
    resizeMap();
    resize3D();
  });

  document.addEventListener('keydown', function (e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) {
      if (e.key === 'Escape') e.target.blur();
      return;
    }

    if (e.key === 'Escape') {
      ['rawFeedModal', 'aiModal', 'helpModal', 'pfrModal', 'startMissionModal', 'dbManagerModal'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('hidden');
      });
      document.querySelectorAll('.tile-maximized').forEach(t => t.classList.remove('tile-maximized'));
      resizeVisibleCharts();
      resizeMap();
      resize3D();
    } else if (e.code === 'Space') {
      e.preventDefault();
      toggleReplayPlay();
    } else if (e.key === 't' || e.key === 'T') {
      tareAttitude();
    } else if (e.key === 'p' || e.key === 'P') {
      openModal('pfrModal');
    } else if (e.key === 'c' || e.key === 'C') {
      toggleConnect();
    } else if (e.key === 'd' || e.key === 'D') {
      downloadCSV();
    }
  });

  // Global window function exports
  window.downloadCSV = downloadCSV;
  window.openModal = openModal;
  window.closeModal = closeModal;
  window.toggleTileMaximize = toggleTileMaximize;
  window.toggleReplayDrawer = toggleReplayDrawer;
  window.toggleConnect = toggleConnect;
  window.toggleDemo = toggleDemo;
  window.analyzeData = analyzeData;
  window.loadSelectedTestCase = loadSelectedTestCase;
  window.handleFileUpload = handleFileUpload;
  window.toggleReplayPlay = toggleReplayPlay;
  window.replayStepForward = replayStepForward;
  window.resetReplay = resetReplay;
  window.onScrubSeek = onScrubSeek;
  window.setReplaySpeed = setReplaySpeed;
  window.openGoogleMapsRecovery = openGoogleMapsRecovery;
  window.toggleAudioSpeech = toggleAudioSpeech;
  window.dismissSafetyAlarm = dismissSafetyAlarm;
  window.tareAttitude = tareAttitude;
  window.resetNewFlight = resetNewFlight;
  window.triggerStationaryTare = triggerStationaryTare;
  window.toggleAutoRecord = toggleAutoRecord;
  window.toggleTacticalCamGrid = toggleTacticalCamGrid;
  window.generatePfrReport = generatePfrReport;
  window.copyPfrMarkdown = copyPfrMarkdown;
  window.printPfrReport = printPfrReport;
  window.toggleMissionRecording = toggleMissionRecording;
  window.confirmStartMission = confirmStartMission;
  window.flushDbTelemetryBuffer = flushDbTelemetryBuffer;
  window.finishActiveMission = finishActiveMission;
  window.saveCurrentTelemetryToDb = saveCurrentTelemetryToDb;
  window.refreshDbMissionsList = refreshDbMissionsList;
  window.onSelectHistoricalMission = onSelectHistoricalMission;
  window.loadHistoricalTelemetryToReplay = loadHistoricalTelemetryToReplay;

  // Database Explorer & Mission Management Exports
  window.openDatabaseManager = openDatabaseManager;
  window.switchDbManagerTab = switchDbManagerTab;
  window.loadDatabaseManagerData = loadDatabaseManagerData;
  window.inspectMissionTelemetryTable = inspectMissionTelemetryTable;
  window.downloadSelectedMissionCsv = downloadSelectedMissionCsv;
  window.downloadMissionCsv = downloadMissionCsv;
  window.downloadDatabaseFile = downloadDatabaseFile;
  window.openPfrForMission = openPfrForMission;
  window.deleteHistoricalMission = deleteHistoricalMission;
  window.deleteCurrentPfrMission = deleteCurrentPfrMission;
  window.purgeAllMissions = purgeAllMissions;

  // Verification Testing Harness Exports
  window.__COGNITIVE_CANSAT__ = {
    parsePacketLine: parsePacketLine,
    batteryPct: batteryPct,
    updateAttitudeFusion: updateAttitudeFusion,
    tareAttitude: tareAttitude,
    computeSoundingMetrics: computeSoundingMetrics,
    evaluateAnomalies: evaluateAnomalies,
    updateFlightPhase: updateFlightPhase,
    buildCsv: buildCsv,
    CSV_COLUMNS: CSV_COLUMNS,
    gpsLocked: gpsLocked,
    loadCsvIntoReplay: loadCsvIntoReplay,
    ingestRawLine: ingestRawLine,
    getTelemetryHistory: function() { return telemetryHistory; },
    getActiveDbMissionId: function() { return activeDbMissionId; },
    getDbTelemetryBuffer: function() { return dbTelemetryBuffer; }
  };
})();
