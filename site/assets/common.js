// Shared helpers for every page: nav, theme, data loading, tables, canvas setup.

export const PROJECTS = [
  { slug: "telepathy", page: "telepathy.html", name: "Telepathy", color: "--telepathy" },
  { slug: "speech", page: "speech.html", name: "Speech", color: "--speech" },
  { slug: "robotic_arm", page: "robotic-arm.html", name: "Assisted Robotic Arm", color: "--arm" },
  { slug: "blindsight", page: "blindsight.html", name: "Blindsight", color: "--blindsight" },
];

function storedTheme() {
  try { return localStorage.getItem("theme"); } catch { return null; }
}

export function initTheme() {
  const t = storedTheme();
  if (t) document.documentElement.dataset.theme = t;
}

export function renderNav(active) {
  const nav = document.createElement("nav");
  nav.className = "nav";
  nav.innerHTML = `<div class="wrap">
    <a class="brand" href="index.html">BCI Portfolio</a>
    ${PROJECTS.map(p => `<a class="link ${p.slug === active ? "active" : ""}" href="${p.page}">${p.name}</a>`).join("")}
    <span class="spacer"></span>
    <button class="theme-btn" type="button" aria-label="Toggle light/dark theme">Theme</button>
  </div>`;
  document.body.prepend(nav);
  nav.querySelector(".theme-btn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme ||
      (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {}
    window.dispatchEvent(new Event("themechange"));
  });
}

export function renderFooter() {
  const f = document.createElement("footer");
  f.innerHTML = `<div class="wrap">
    <p>Amal Mehta · Independent portfolio projects.</p>
    <p class="disclaimer">Not affiliated with or endorsed by Neuralink. Product names are used only to describe the
    problem area each project studies. Closed-loop results come from simulated users, not human participants.</p>
  </div>`;
  document.body.append(f);
}

export async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

/** Read a CSS custom property, e.g. css("--accent"). */
export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Size a canvas for the device pixel ratio; returns a 2D context in CSS pixels. */
export function setupCanvas(canvas, aspect) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = aspect ? Math.round(w / aspect) : canvas.clientHeight;
  canvas.style.height = h + "px";
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/** Render rows (array of objects) into a <table>. cols: [{key, label, fmt?, num?}]. bestKey highlights the max (or min) row. */
export function renderTable(el, rows, cols, { bestKey, lowerIsBetter = false } = {}) {
  let best = null;
  if (bestKey) {
    const vals = rows.map(r => r[bestKey]).filter(v => typeof v === "number");
    best = lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
  }
  el.innerHTML = `<div class="table-wrap"><table><thead><tr>${cols.map(c =>
    `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("")}</tr></thead><tbody>${rows.map(r =>
    `<tr class="${bestKey && r[bestKey] === best ? "best" : ""}">${cols.map(c => {
      const v = r[c.key];
      const s = v == null ? "—" : c.fmt ? c.fmt(v, r) : v;
      return `<td class="${c.num ? "num" : ""}">${s}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>`;
}

export const fmt = {
  f2: v => Number(v).toFixed(2),
  f3: v => Number(v).toFixed(3),
  pct: v => (100 * v).toFixed(1) + "%",
  pm: (m, s) => s == null ? Number(m).toFixed(3) : `${Number(m).toFixed(3)} ± ${Number(s).toFixed(3)}`,
};

export function page(active) {
  initTheme();
  renderNav(active);
  renderFooter();
}
