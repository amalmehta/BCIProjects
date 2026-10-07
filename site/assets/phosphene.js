// Browser port of the Blindsight phosphene renderer and intensity-sampling baseline.
// Follows the exact formulas in data/blindsight/electrodes.json and baseline_encoder.json
// (same maths as projects/blindsight/tests/js/render_check.mjs).

export class PhospheneSim {
  constructor(electrodesJson, baselineJson) {
    this.el = electrodesJson;
    this.bl = baselineJson;
    this.c = electrodesJson.constants;
    this.fov = electrodesJson.field_of_view_deg;
    this.R = electrodesJson.reference_resolution;
    this.alive = this.el.electrodes.map(e => e.alive);
  }

  /** Mark a random fraction of electrodes dead (deterministic for a given seed), on top of the patient's own dead ones. */
  setFailures(fraction, seed = 7) {
    let s = seed >>> 0;
    const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    this.alive = this.el.electrodes.map(e => (e.alive && rand() >= fraction ? 1 : 0));
  }

  /** Safety layer: clamp, zero dead electrodes, keep the top-k amplitudes. */
  safety(amps) {
    const { max_current_ua: cap, max_active_electrodes: k } = this.el.safety;
    let a = Array.from(amps, (v, n) => (this.alive[n] ? Math.min(Math.max(v, 0), cap) : 0));
    const kth = [...a].sort((x, y) => y - x)[k - 1];
    return a.map(v => (v >= kth && v > 0 ? v : 0));
  }

  /** Per-electrode brightness and size from amplitudes (µA). */
  phosphenes(amps) {
    const c = this.c;
    return this.el.electrodes.map((e, n) => {
      const I = this.alive[n] ? amps[n] : 0;
      const ieff = Math.max(0, (I * 1e-6 - c.rheobase_a) * c.pulse_width_s * c.frequency_hz);
      const A = ieff * c.activation_gain_s;
      const b = A > e.threshold_activation ? 1 / (1 + Math.exp(-c.brightness_slope * (A - c.a50))) : 0;
      const r = Math.sqrt(Math.max(I, 0) * 1e-6 / c.current_spread_a_per_mm2 + 1e-12);
      const M = c.k * (1 / (e.ecc_deg + c.a) - 1 / (e.ecc_deg + c.b));
      const s = Math.sqrt((r / M) ** 2 + c.sigma_floor_deg ** 2);
      return { b, s, x: e.x_deg, y: e.y_deg };
    });
  }

  /** Render the percept into an R×R Float32Array (row 0 = top). Only touches pixels within 4σ of each phosphene. */
  render(amps, R) {
    const out = new Float32Array(R * R);
    const fov = this.fov, px = fov / R;
    for (const p of this.phosphenes(amps)) {
      if (p.b <= 0) continue;
      const rad = 4 * p.s, inv = 1 / (2 * p.s * p.s);
      const c0 = Math.max(0, Math.floor((p.x - rad + fov / 2) / px)), c1 = Math.min(R - 1, Math.ceil((p.x + rad + fov / 2) / px));
      const r0 = Math.max(0, Math.floor((fov / 2 - p.y - rad) / px)), r1 = Math.min(R - 1, Math.ceil((fov / 2 - p.y + rad) / px));
      for (let i = r0; i <= r1; i++) {
        const dy = fov / 2 - (i + 0.5) * px - p.y;
        for (let j = c0; j <= c1; j++) {
          const dx = (j + 0.5) * px - fov / 2 - p.x;
          out[i * R + j] += p.b * Math.exp(-(dx * dx + dy * dy) * inv);
        }
      }
    }
    for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i]);
    return out;
  }

  /** Intensity-sampling baseline on an R×R grayscale Float32Array in [0,1]. Returns amplitudes (µA) after safety. */
  baseline(img) {
    const R = this.R, P = this.bl.params, fov = this.fov;
    const k = [-3, -2, -1, 0, 1, 2, 3].map(t => Math.exp(-t * t / (2 * P.blur_sigma_px ** 2)));
    const ks = k.reduce((a, b) => a + b);
    for (let i = 0; i < 7; i++) k[i] /= ks;
    const refl = i => (i < 0 ? -i : i >= R ? 2 * (R - 1) - i : i);
    const h = new Float32Array(R * R), b = new Float32Array(R * R);
    for (let r = 0; r < R; r++) for (let q = 0; q < R; q++) {
      let acc = 0; for (let i = 0; i < 7; i++) acc += k[i] * img[r * R + refl(q + i - 3)]; h[r * R + q] = acc;
    }
    for (let r = 0; r < R; r++) for (let q = 0; q < R; q++) {
      let acc = 0; for (let i = 0; i < 7; i++) acc += k[i] * h[refl(r + i - 3) * R + q]; b[r * R + q] = acc;
    }
    const amps = this.el.electrodes.map(e => {
      const col = Math.min(Math.max((e.nominal_x_deg + fov / 2) / fov * R - 0.5, 0), R - 1 - 1e-4);
      const row = Math.min(Math.max((fov / 2 - e.nominal_y_deg) / fov * R - 0.5, 0), R - 1 - 1e-4);
      const c0 = Math.floor(col), r0 = Math.floor(row), fc = col - c0, fr = row - r0;
      const c1 = Math.min(c0 + 1, R - 1), r1 = Math.min(r0 + 1, R - 1);
      let v = b[r0 * R + c0] * (1 - fr) * (1 - fc) + b[r0 * R + c1] * (1 - fr) * fc + b[r1 * R + c0] * fr * (1 - fc) + b[r1 * R + c1] * fr * fc;
      v = Math.min(1, Math.max(0, v));
      const t = Math.min(e.threshold_ua, P.max_current_ua);
      return v > P.cutoff ? t + v * (P.max_current_ua - t) : 0;
    });
    return this.safety(amps);
  }
}

/** Grab a source (canvas/img/video) as R×R luma in [0,1], centre-cropped to square. */
export function toGray(source, R) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  const s = Math.min(sw, sh);
  const cv = toGray._cv || (toGray._cv = document.createElement("canvas"));
  cv.width = cv.height = R;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, (sw - s) / 2, (sh - s) / 2, s, s, 0, 0, R, R);
  const d = ctx.getImageData(0, 0, R, R).data;
  const out = new Float32Array(R * R);
  for (let i = 0; i < R * R; i++) out[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
  return out;
}

/** Paint an R×R Float32Array onto a canvas, with an optional warm phosphene tint. */
export function paint(canvas, arr, R, tint = [255, 244, 214]) {
  const off = paint._cv || (paint._cv = document.createElement("canvas"));
  off.width = off.height = R;
  const octx = off.getContext("2d");
  const im = octx.createImageData(R, R);
  for (let i = 0; i < R * R; i++) {
    const v = Math.max(0, Math.min(1, arr[i]));
    im.data[4 * i] = v * tint[0]; im.data[4 * i + 1] = v * tint[1]; im.data[4 * i + 2] = v * tint[2]; im.data[4 * i + 3] = 255;
  }
  octx.putImageData(im, 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = R < canvas.width / 2 ? false : true;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}
