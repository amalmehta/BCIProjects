// Re-implements the phosphene renderer and the intensity baseline in plain JavaScript from
// electrodes.json / baseline_encoder.json alone, and checks the exported test cases.
// Usage: node tests/js/render_check.mjs <site_export_dir>
import { readFileSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2];
const el = JSON.parse(readFileSync(join(dir, "electrodes.json"), "utf8"));
const bl = JSON.parse(readFileSync(join(dir, "baseline_encoder.json"), "utf8"));
const c = el.constants;
const fov = el.field_of_view_deg;

function phosphenes(amps) {
  return el.electrodes.map((e, n) => {
    const I = e.alive ? amps[n] : 0;
    const ieff = Math.max(0, (I * 1e-6 - c.rheobase_a) * c.pulse_width_s * c.frequency_hz);
    const A = ieff * c.activation_gain_s;
    const b = A > e.threshold_activation ? 1 / (1 + Math.exp(-c.brightness_slope * (A - c.a50))) : 0;
    const r = Math.sqrt(Math.max(I, 0) * 1e-6 / c.current_spread_a_per_mm2 + 1e-12);
    const M = c.k * (1 / (e.ecc_deg + c.a) - 1 / (e.ecc_deg + c.b));
    const s = Math.sqrt((r / M) ** 2 + c.sigma_floor_deg ** 2);
    return { b, s, x: e.x_deg, y: e.y_deg };
  });
}

function pixel(ph, R, row, col) {
  const x = (col + 0.5) / R * fov - fov / 2;
  const y = fov / 2 - (row + 0.5) / R * fov;
  let v = 0;
  for (const p of ph) if (p.b > 0) v += p.b * Math.exp(-((x - p.x) ** 2 + (y - p.y) ** 2) / (2 * p.s * p.s));
  return Math.min(1, Math.max(0, v));
}

let fails = 0;
const tc = el.test_case;
const ph = phosphenes(tc.amplitudes_ua);
for (const px of tc.pixels) {
  const got = pixel(ph, tc.resolution, px.row, px.col);
  if (Math.abs(got - px.value) > 2e-3) { fails++; console.log("render mismatch", px, got); }
}

// Intensity baseline on the documented test image.
const R = el.reference_resolution, P = bl.params;
const img = [];
for (let r = 0; r < R; r++) { img.push([]); for (let q = 0; q < R; q++)
  img[r].push(Math.min(1, Math.max(0, 0.6 * q / (R - 1) + 0.4 * (((q - 40) ** 2 + (r - 24) ** 2) < 100 ? 1 : 0)))); }
const k = [-3, -2, -1, 0, 1, 2, 3].map(t => Math.exp(-t * t / (2 * P.blur_sigma_px ** 2)));
const ks = k.reduce((a, b) => a + b); for (let i = 0; i < 7; i++) k[i] /= ks;
const refl = i => (i < 0 ? -i : i >= R ? 2 * (R - 1) - i : i);
const h = img.map(row => row.map((_, q) => k.reduce((acc, w, i) => acc + w * row[refl(q + i - 3)], 0)));
const bimg = h.map((row, r) => row.map((_, q) => k.reduce((acc, w, i) => acc + w * h[refl(r + i - 3)][q], 0)));
let amps = el.electrodes.map(e => {
  const col = Math.min(Math.max((e.nominal_x_deg + fov / 2) / fov * R - 0.5, 0), R - 1 - 1e-4);
  const row = Math.min(Math.max((fov / 2 - e.nominal_y_deg) / fov * R - 0.5, 0), R - 1 - 1e-4);
  const c0 = Math.floor(col), r0 = Math.floor(row), fc = col - c0, fr = row - r0;
  const c1 = Math.min(c0 + 1, R - 1), r1 = Math.min(r0 + 1, R - 1);
  let v = bimg[r0][c0] * (1 - fr) * (1 - fc) + bimg[r0][c1] * (1 - fr) * fc + bimg[r1][c0] * fr * (1 - fc) + bimg[r1][c1] * fr * fc;
  v = Math.min(1, Math.max(0, v));
  const t = Math.min(e.threshold_ua, P.max_current_ua);
  return (v > P.cutoff ? t + v * (P.max_current_ua - t) : 0) * e.alive;
});
const kth = [...amps].sort((a, b) => b - a)[P.max_active_electrodes - 1];
amps = amps.map(a => (a >= kth ? Math.min(a, P.max_current_ua) : 0));
for (const [i, val] of Object.entries(bl.test_case.expected_amplitudes_ua)) {
  if (Math.abs(amps[+i] - val) > 0.05) { fails++; console.log("baseline mismatch", i, amps[+i], val); }
}
console.log(fails === 0 ? `OK: ${tc.pixels.length} render pixels and ${Object.keys(bl.test_case.expected_amplitudes_ua).length} baseline amplitudes match` : `FAILED: ${fails}`);
process.exit(fails === 0 ? 0 : 1);
