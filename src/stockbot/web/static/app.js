"use strict";

const REFRESH_MS = 15000; // near real-time polling

const fmtMoney = (v) =>
  v == null ? "—" : v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const fmtPct = (v) => (v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%");
const fmtNum = (v) => (v == null ? "—" : v.toLocaleString("en-US", { maximumFractionDigits: 2 }));
const fmtTime = (ms) => new Date(ms).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

const signClass = (v) => (v == null ? "" : v >= 0 ? "pos" : "neg");

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

let chart;
function renderChart(series) {
  const withSpy = series.filter((p) => p.spy_price != null);
  const eq0 = series.length ? series[0].equity : null;
  const spy0 = withSpy.length ? withSpy[0].spy_price : null;

  const labels = series.map((p) => fmtTime(p.ts));
  const botPct = series.map((p) => (eq0 ? (p.equity / eq0 - 1) * 100 : 0));
  const spyPct = series.map((p) => (spy0 && p.spy_price != null ? (p.spy_price / spy0 - 1) * 100 : null));

  const data = {
    labels,
    datasets: [
      { label: "Bot", data: botPct, borderColor: "#58a6ff", backgroundColor: "rgba(88,166,255,.1)", fill: true, tension: 0.15, pointRadius: 0, borderWidth: 2 },
      { label: "SPY", data: spyPct, borderColor: "#8b949e", borderDash: [5, 4], fill: false, tension: 0.15, pointRadius: 0, borderWidth: 1.5 },
    ],
  };
  const opts = {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { ticks: { color: "#8b949e", maxTicksLimit: 8 }, grid: { color: "#2a323c" } },
      y: { ticks: { color: "#8b949e", callback: (v) => v + "%" }, grid: { color: "#2a323c" } },
    },
    plugins: { legend: { labels: { color: "#e6edf3" } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y == null ? "—" : c.parsed.y.toFixed(2) + "%"}` } } },
  };
  if (chart) {
    chart.data = data;
    chart.options = opts;
    chart.update("none");
  } else {
    chart = new Chart(document.getElementById("chart"), { type: "line", data, options: opts });
  }
}

function renderPositions(rows) {
  document.getElementById("posCount").textContent = rows.length;
  const tb = document.querySelector("#positions tbody");
  tb.innerHTML = rows.length
    ? rows
        .map(
          (p) => `<tr>
            <td>${p.symbol}</td>
            <td>${fmtNum(p.qty)}</td>
            <td>${fmtMoney(p.avg_price)}</td>
            <td>${fmtMoney(p.market_value)}</td>
            <td class="${signClass(p.unrealized_pnl)}">${p.unrealized_pnl == null ? "—" : fmtMoney(p.unrealized_pnl)}</td>
          </tr>`
        )
        .join("")
    : `<tr><td colspan="5" class="muted">No open positions</td></tr>`;
}

function renderTrades(rows) {
  const tb = document.querySelector("#trades tbody");
  tb.innerHTML = rows.length
    ? rows
        .map(
          (t) => `<tr>
            <td>${fmtTime(t.ts)}</td>
            <td class="${t.side.toLowerCase() === "buy" ? "buy" : "sell"}">${t.side.toUpperCase()}</td>
            <td>${t.symbol}</td>
            <td>${fmtNum(t.qty)}</td>
            <td>${fmtMoney(t.price)}</td>
          </tr>`
        )
        .join("")
    : `<tr><td colspan="5" class="muted">No trades yet</td></tr>`;
}

function renderSummary(s) {
  document.getElementById("title").textContent = s.label || "Random Bot";
  document.getElementById("equity").textContent = fmtMoney(s.equity);
  document.getElementById("cash").textContent = fmtMoney(s.cash);

  const botEl = document.getElementById("botReturn");
  botEl.textContent = fmtPct(s.bot_return);
  botEl.className = "value " + signClass(s.bot_return);

  const spyEl = document.getElementById("spyReturn");
  spyEl.textContent = fmtPct(s.spy_return);
  spyEl.className = "value " + signClass(s.spy_return);

  const alpha = s.bot_return != null && s.spy_return != null ? s.bot_return - s.spy_return : null;
  const alphaEl = document.getElementById("alpha");
  alphaEl.textContent = fmtPct(alpha);
  alphaEl.className = "value " + signClass(alpha);

  document.getElementById("updated").textContent = s.last_ts ? "updated " + fmtTime(s.last_ts) : "no data yet";
}

async function refresh() {
  try {
    const [summary, equity, positions, trades] = await Promise.all([
      getJSON("/api/summary"),
      getJSON("/api/equity"),
      getJSON("/api/positions"),
      getJSON("/api/trades?limit=100"),
    ]);
    renderSummary(summary);
    renderChart(equity);
    renderPositions(positions);
    renderTrades(trades);
  } catch (e) {
    document.getElementById("updated").textContent = "connection error — retrying";
    console.error(e);
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
