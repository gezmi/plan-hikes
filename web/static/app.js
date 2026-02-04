// Israel Hiking Transit Planner — frontend
"use strict";

// ── Map setup ──────────────────────────────────────────────────────
const map = L.map("map").setView([31.5, 35.0], 8);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
  maxZoom: 18,
}).addTo(map);

// Layer groups for results
const trailLayers = L.layerGroup().addTo(map);
const markerLayers = L.layerGroup().addTo(map);

// Trail color map
const TRAIL_COLORS = {
  red: "#d32f2f",
  blue: "#1976d2",
  green: "#388e3c",
  black: "#333333",
  orange: "#f57c00",
  purple: "#7b1fa2",
};

// Sparkline characters
const SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588";

function sparkline(values, width) {
  width = width || 30;
  if (!values || values.length < 2) return "";
  const n = values.length;
  let resampled;
  if (n > width) {
    resampled = [];
    for (let i = 0; i < width; i++) {
      const start = Math.floor(i * n / width);
      const end = Math.floor((i + 1) * n / width);
      let sum = 0;
      for (let j = start; j < end; j++) sum += values[j];
      resampled.push(sum / (end - start));
    }
  } else if (n < width) {
    resampled = [];
    for (let i = 0; i < width; i++) {
      const frac = i * (n - 1) / (width - 1);
      const lo = Math.floor(frac);
      const hi = Math.min(lo + 1, n - 1);
      const t = frac - lo;
      resampled.push(values[lo] * (1 - t) + values[hi] * t);
    }
  } else {
    resampled = values.slice();
  }

  const lo = Math.min(...resampled);
  const hi = Math.max(...resampled);
  const span = hi !== lo ? hi - lo : 1;
  return resampled.map(v => {
    const idx = Math.round((v - lo) / span * (SPARK_CHARS.length - 1));
    return SPARK_CHARS[Math.max(0, Math.min(idx, SPARK_CHARS.length - 1))];
  }).join("");
}

// ── Load cities ────────────────────────────────────────────────────
async function loadCities() {
  try {
    const res = await fetch("/api/cities");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const select = document.getElementById("origins");
    data.cities.forEach(city => {
      const opt = document.createElement("option");
      opt.value = city.name;
      opt.textContent = city.name;
      select.appendChild(opt);
    });
    // Pre-select first city
    if (select.options.length > 0) {
      select.options[0].selected = true;
    }
  } catch (err) {
    console.error("Failed to load cities:", err);
    document.getElementById("status").textContent = "Failed to load city list. Server may be starting up — try refreshing.";
    document.getElementById("status").className = "error";
  }
}

// ── Set default date ───────────────────────────────────────────────
function setDefaultDate() {
  const input = document.getElementById("date");
  const today = new Date();
  // Find next valid day (skip Saturday)
  let target = new Date(today);
  target.setDate(target.getDate() + 1);
  if (target.getDay() === 6) target.setDate(target.getDate() + 1); // skip Saturday
  input.value = target.toISOString().split("T")[0];
}

// ── Build request ──────────────────────────────────────────────────
function buildRequest() {
  const select = document.getElementById("origins");
  const origins = Array.from(select.selectedOptions).map(o => o.value);
  if (origins.length === 0) {
    throw new Error("Select at least one origin city.");
  }

  const date = document.getElementById("date").value;
  if (!date) throw new Error("Select a date.");

  const maxResults = parseInt(document.getElementById("max-results").value) || 10;
  const minHike = parseFloat(document.getElementById("min-hike").value) || 1.0;
  const maxWalk = parseInt(document.getElementById("max-walk").value) || 1000;

  // Filters
  const colors = Array.from(document.querySelectorAll(".color-chips input:checked")).map(i => i.value);
  const minDist = parseFloat(document.getElementById("min-distance").value) || null;
  const maxDist = parseFloat(document.getElementById("max-distance").value) || null;
  const maxElev = parseFloat(document.getElementById("max-elevation").value) || null;

  const trailType = document.querySelector('input[name="trail-type"]:checked').value;

  return {
    origins,
    date,
    max_results: maxResults,
    min_hike_hours: minHike,
    max_walk_m: maxWalk,
    colors: colors.length > 0 ? colors : null,
    min_distance_km: minDist,
    max_distance_km: maxDist,
    loop_only: trailType === "loop",
    linear_only: trailType === "linear",
    max_elevation_gain_m: maxElev,
  };
}

// ── Render results ─────────────────────────────────────────────────
let activePlanIdx = null;

function ratioClass(ratio) {
  if (ratio >= 0.5) return "ratio-high";
  if (ratio >= 0.3) return "ratio-medium";
  return "ratio-low";
}

function colorDots(colors) {
  return colors.map(c => {
    const hex = TRAIL_COLORS[c] || "#888";
    return `<span class="color-dot" style="background:${hex}"></span>`;
  }).join("");
}

function formatLeg(leg) {
  return `
    <div class="leg-row">
      <span class="bus-badge">${leg.line}</span>
      <span>${leg.departure} ${leg.from_stop} &rarr; ${leg.arrival} ${leg.to_stop}</span>
      <span class="label">(${leg.duration_min} min)</span>
    </div>
  `;
}

function renderPlan(plan, globalIdx) {
  const t = plan.trail;
  const h = plan.hiking;
  const pct = Math.round(plan.hiking_ratio * 100);
  const typeLabel = h.is_through_hike ? "Through" : (t.is_loop ? "Loop" : "Out & Back");
  const walkMin = Math.round(h.walk_to_trail_m / 1000 / 4.5 * 60);

  // Elevation info
  let elevInfo = "";
  if (t.elevation_gain_m > 0 || t.elevation_loss_m > 0) {
    elevInfo = ` | +${Math.round(t.elevation_gain_m)}m / -${Math.round(t.elevation_loss_m)}m`;
  }

  // Sparkline
  let sparkHtml = "";
  if (plan.elevation_profile && plan.elevation_profile.length >= 2) {
    const spark = sparkline(plan.elevation_profile);
    sparkHtml = `
      <div style="margin-top:4px">
        <span class="elev-label">${Math.round(t.min_elevation_m)}m</span>
        <span class="elevation-spark">${spark}</span>
        <span class="elev-label">${Math.round(t.max_elevation_m)}m</span>
      </div>
    `;
  }

  return `
    <div class="plan-card" data-idx="${globalIdx}" onclick="selectPlan(${globalIdx})">
      <div class="card-header">
        <span class="rank">#${plan.rank}</span>
        <span class="trail-name">${t.name}</span>
        <span class="ratio-badge ${ratioClass(plan.hiking_ratio)}">${pct}%</span>
      </div>
      <div class="card-body">
        <div class="info-row">
          <span>${t.distance_km} km | ${typeLabel} | ${colorDots(t.colors)}${elevInfo}</span>
        </div>
        <div class="info-row">
          <span>${plan.departure} &rarr; ${plan.arrival}</span>
          <span class="label">${plan.total_hours}h total, ${h.hours}h hiking</span>
        </div>
        ${sparkHtml}
      </div>
      <div class="card-details">
        <div class="section-label outbound">Outbound</div>
        ${plan.outbound.map(formatLeg).join("")}
        <div class="leg-row">Walk ${Math.round(h.walk_to_trail_m)}m to trail (${walkMin} min)</div>

        <div class="section-label hiking">Hiking</div>
        <div>${h.start} - ${h.end} | ~${h.distance_km} km | ${h.hours}h</div>
        ${h.is_through_hike && h.exit_stop ? `<div>${h.entry_stop} &rarr; ${h.exit_stop}</div>` : ""}

        <div class="section-label return">Return</div>
        ${h.is_through_hike
          ? `<div class="leg-row">Walk ${Math.round(h.walk_from_trail_m)}m from trail</div>`
          : `<div class="leg-row">Walk ${Math.round(h.walk_to_trail_m)}m back to stop</div>`
        }
        ${plan.return_legs.map(formatLeg).join("")}

        <div style="margin-top:6px;color:#888;font-size:0.9em">
          Deadline: ${plan.deadline} | Home by: ${plan.arrival}
        </div>
        ${plan.warnings.map(w => `<div class="warning-row">${w}</div>`).join("")}
        <div class="links-row" style="margin-top:6px">
          ${plan.links.osm ? `<a href="${plan.links.osm}" target="_blank">OSM</a>` : ""}
          <a href="${plan.links.google_maps}" target="_blank">Google Maps</a>
          <a href="${plan.links.israel_hiking}" target="_blank">Israel Hiking</a>
          ${plan.links.directions ? `<a href="${plan.links.directions}" target="_blank">Directions</a>` : ""}
        </div>
      </div>
    </div>
  `;
}

// Store all plans globally for map interaction
let allPlans = [];

function renderResults(data) {
  const container = document.getElementById("results");
  allPlans = [];
  let html = "";
  let globalIdx = 0;

  for (const originResult of data.results) {
    if (data.results.length > 1) {
      const count = originResult.plans.length;
      html += `<div class="origin-header">${originResult.origin} (${count} results)</div>`;
    }

    for (const plan of originResult.plans) {
      allPlans.push(plan);
      html += renderPlan(plan, globalIdx);
      globalIdx++;
    }

    if (originResult.plans.length === 0) {
      html += `<div style="color:#888;padding:8px 0">No hikes found from ${originResult.origin}.</div>`;
    }
  }

  container.innerHTML = html;
}

// ── Map interaction ────────────────────────────────────────────────
function showAllTrails() {
  trailLayers.clearLayers();
  markerLayers.clearLayers();

  if (allPlans.length === 0) return;

  const bounds = L.latLngBounds();

  allPlans.forEach((plan, idx) => {
    const coords = plan.trail.geometry;
    if (!coords || coords.length < 2) return;

    const color = plan.trail.colors.length > 0
      ? (TRAIL_COLORS[plan.trail.colors[0]] || "#888")
      : "#888";

    const line = L.polyline(coords, {
      color: color,
      weight: 3,
      opacity: 0.6,
    }).addTo(trailLayers);

    line.on("click", () => selectPlan(idx));
    coords.forEach(c => bounds.extend(c));

    // Entry marker
    const marker = L.circleMarker(
      [plan.access_point_lat, plan.access_point_lon],
      { radius: 5, color: color, fillColor: color, fillOpacity: 0.8 }
    ).addTo(markerLayers);
    marker.bindTooltip(`#${plan.rank} ${plan.trail.name}`);
  });

  if (bounds.isValid()) {
    map.fitBounds(bounds, { padding: [30, 30] });
  }
}

function selectPlan(idx) {
  // Deselect previous
  document.querySelectorAll(".plan-card.active").forEach(el => el.classList.remove("active"));

  // Select new
  const card = document.querySelector(`.plan-card[data-idx="${idx}"]`);
  if (card) {
    card.classList.add("active");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  activePlanIdx = idx;

  // Highlight on map
  trailLayers.clearLayers();
  markerLayers.clearLayers();

  const plan = allPlans[idx];
  if (!plan) return;

  const coords = plan.trail.geometry;
  if (!coords || coords.length < 2) return;

  const color = plan.trail.colors.length > 0
    ? (TRAIL_COLORS[plan.trail.colors[0]] || "#2196f3")
    : "#2196f3";

  // Dim all other trails
  allPlans.forEach((p, i) => {
    if (i === idx) return;
    const c = p.trail.geometry;
    if (!c || c.length < 2) return;
    L.polyline(c, { color: "#ccc", weight: 2, opacity: 0.4 }).addTo(trailLayers);
  });

  // Highlight selected trail
  const line = L.polyline(coords, { color: color, weight: 5, opacity: 0.9 }).addTo(trailLayers);

  // Entry marker
  L.marker([plan.access_point_lat, plan.access_point_lon])
    .bindPopup(`<b>${plan.trail.name}</b><br>Entry: ${plan.hiking.entry_stop}`)
    .addTo(markerLayers)
    .openPopup();

  map.fitBounds(line.getBounds(), { padding: [50, 50] });
}

// Make selectPlan available globally
window.selectPlan = selectPlan;

// ── Form submit ────────────────────────────────────────────────────
document.getElementById("query-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const status = document.getElementById("status");
  const btn = document.getElementById("search-btn");
  const results = document.getElementById("results");

  let req;
  try {
    req = buildRequest();
  } catch (err) {
    status.textContent = err.message;
    status.className = "error";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Searching...";
  status.textContent = "Loading data and planning hikes...";
  status.className = "";
  results.innerHTML = "";
  trailLayers.clearLayers();
  markerLayers.clearLayers();

  try {
    const res = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        detail = err.detail || detail;
      } catch (_) {
        // Response body wasn't JSON — use the status text
        detail = `HTTP ${res.status}: ${res.statusText}`;
      }
      throw new Error(detail);
    }

    const data = await res.json();
    const totalPlans = data.results.reduce((s, r) => s + r.plans.length, 0);
    status.textContent = `Found ${totalPlans} hiking plans. Deadline: ${data.deadline}`;

    renderResults(data);
    showAllTrails();
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    status.className = "error";
  } finally {
    btn.disabled = false;
    btn.textContent = "Search Hikes";
  }
});

// ── Init ───────────────────────────────────────────────────────────
loadCities();
setDefaultDate();
