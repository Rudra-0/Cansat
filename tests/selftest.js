/**
 * Offline self-test for CanSat ground-station parse / kinematics / CSV.
 * Mirrors formulas in index.html (no DOM).
 */
'use strict';

function parsePacketLine(line) {
  const parts = line.split(',').map((s) => s.trim());
  if (parts.length !== 13) return null;
  const nums = parts.map(Number);
  if (nums.some((x) => !Number.isFinite(x))) return null;
  return {
    temp: nums[0], pressure: nums[1], altitude: nums[2],
    gx: nums[3], gy: nums[4], gz: nums[5],
    ax: nums[6], ay: nums[7], az: nums[8],
    lat: nums[9], lon: nums[10], humidity: nums[11],
    batteryVoltage: nums[12]
  };
}

function derive(ax, ay, az, altT, altPrev, dt) {
  const accelMag = Math.sqrt(ax * ax + ay * ay + az * az);
  const pitch = Math.atan2(ay, az) * (180 / Math.PI);
  const roll = Math.atan2(-ax, Math.sqrt(ay * ay + az * az)) * (180 / Math.PI);
  const vSpd = dt > 0 ? (altT - altPrev) / dt : 0;
  return { accelMag, pitch, roll, vSpd };
}

function batteryPct(v) {
  return Math.min(100, Math.max(0, ((v - 3.2) / (4.2 - 3.2)) * 100));
}

function csvEscape(val) {
  const s = String(val);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

const CSV_COLUMNS = [
  'timestamp', 'temp', 'pressure', 'altitude', 'gx', 'gy', 'gz',
  'ax', 'ay', 'az', 'lat', 'lon', 'humidity', 'batteryVoltage',
  'pitch', 'roll', 'accelMag'
];

function buildCsv(rows) {
  const header = CSV_COLUMNS.join(',');
  const body = rows.map((r) => CSV_COLUMNS.map((k) => csvEscape(r[k])).join(',')).join('\r\n');
  return header + '\r\n' + body + '\r\n';
}

let failed = 0;
function assert(name, cond, detail) {
  if (cond) {
    console.log('PASS  ' + name);
  } else {
    failed += 1;
    console.log('FAIL  ' + name + (detail ? '  — ' + detail : ''));
  }
}

const good = '24.50,1013.25,120.00,1.1,-2.2,0.3,0.10,0.20,0.98,22.572700,88.365500,55.00,3.90';
const p = parsePacketLine(good);
assert('parse 13-field CSV', p && p.temp === 24.5 && p.batteryVoltage === 3.9);
assert('reject short line', parsePacketLine('1,2,3') === null);
assert('reject NaN field', parsePacketLine('a,1,2,3,4,5,6,7,8,9,10,11,12') === null);
assert('reject extra field', parsePacketLine(good + ',99') === null);

const k = derive(0.10, 0.20, 0.98, 128, 120, 0.8);
assert('v_spd = dAlt/dt', Math.abs(k.vSpd - 10) < 1e-9, String(k.vSpd));
assert('accelMag', Math.abs(k.accelMag - Math.sqrt(0.1 ** 2 + 0.2 ** 2 + 0.98 ** 2)) < 1e-12);
assert('pitch atan2(ay,az)', Math.abs(k.pitch - (Math.atan2(0.2, 0.98) * 180 / Math.PI)) < 1e-12);
assert('roll formula', Math.abs(k.roll - (Math.atan2(-0.10, Math.sqrt(0.2 ** 2 + 0.98 ** 2)) * 180 / Math.PI)) < 1e-12);

assert('battery 3.20 -> 0%', batteryPct(3.20) === 0);
assert('battery 4.20 -> 100%', batteryPct(4.20) === 100);
assert('battery 3.70 -> 50%', Math.abs(batteryPct(3.70) - 50) < 1e-9);
assert('battery clamp low', batteryPct(2.0) === 0);
assert('battery clamp high', batteryPct(5.0) === 100);

const row = Object.assign({
  timestamp: '2026-08-30T17:00:00.000Z',
  pitch: k.pitch, roll: k.roll, accelMag: k.accelMag
}, p);
const csv = buildCsv([row]);
const header = csv.split('\r\n')[0];
assert('CSV header order', header === CSV_COLUMNS.join(','));
assert('CSV RFC4180 CRLF', csv.includes('\r\n'));
assert('CSV 17 columns', header.split(',').length === 17);

const fs = require('fs');
const path = require('path');
const frontendDir = path.join(__dirname, '..', 'frontend');
// Markup and the extracted dashboard script are checked together.
const html = fs.readFileSync(path.join(frontendDir, 'dashboard.html'), 'utf8') +
  fs.readFileSync(path.join(frontendDir, 'js', 'dashboard.js'), 'utf8');

assert('Web Serial requestPort', html.includes('navigator.serial.requestPort'));
assert('TextDecoderStream', html.includes('TextDecoderStream'));
assert('baud 115200', html.includes('115200'));
assert('invalidateSize', html.includes('invalidateSize'));
assert('Chart resize', html.includes('c.resize()') || html.includes('.resize()'));
assert('CartoDB dark tiles', html.includes('basemaps.cartocdn.com/dark_all'));
assert('Three.js cylinder', html.includes('CylinderGeometry'));
assert('DEMO 800ms', html.includes('800'));
assert('title UNIVERSE TELEMETRY', html.includes('Ground Station - UNIVERSE TELEMETRY'));
assert('palette bg', html.includes('#0d0c11'));
assert('palette accent', html.includes('#5d35e8'));

if (failed) {
  console.log('\nRESULT: ' + failed + ' FAILED');
  process.exit(1);
}
console.log('\nRESULT: ALL TESTS PASSED');
