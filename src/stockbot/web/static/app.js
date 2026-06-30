"use strict";

const REFRESH_MS = 15000; // near real-time polling

const fmtMoney = (v) =>
  v == null ? "—" : v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const fmtPct = (v) => (v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%");
const fmtNum = (v) => (v == null ? "—" : v.toLocaleString("en-US", { maximumFractionDigits: 2 }));
const fmtTime = (ms) => new Date(ms).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const fmtDate = (ms) => (ms == null ? "—" : new Date(ms).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" }));

const signClass = (v) => (v == null ? "" : v >= 0 ? "pos" : "neg");

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

// ---- chart theming ----
const CHART = { accent: "#4f8cff", muted: "#8b97a8", grid: "rgba(255,255,255,0.05)", text: "#cbd5e1" };

function areaFill(alphaTop) {
  return (c) => {
    const area = c.chart.chartArea;
    if (!area) return "rgba(79,140,255,0)";
    const g = c.chart.ctx.createLinearGradient(0, area.top, 0, area.bottom);
    g.addColorStop(0, `rgba(79,140,255,${alphaTop})`);
    g.addColorStop(1, "rgba(79,140,255,0)");
    return g;
  };
}

function chartOpts(yTick, tipLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { ticks: { color: CHART.muted, maxTicksLimit: 7, font: { size: 11 } }, grid: { display: false } },
      y: { ticks: { color: CHART.muted, callback: yTick, font: { size: 11 } }, grid: { color: CHART.grid }, border: { display: false } },
    },
    plugins: {
      legend: { labels: { color: CHART.text, usePointStyle: true, pointStyle: "line", boxWidth: 22 } },
      tooltip: {
        backgroundColor: "#0d1320", borderColor: "#283650", borderWidth: 1, padding: 10,
        titleColor: "#e8eef6", bodyColor: "#cbd5e1", cornerRadius: 8,
        callbacks: { label: tipLabel },
      },
    },
  };
}

const lineDataset = (label, data, opts = {}) => ({
  label, data, borderColor: CHART.accent, backgroundColor: areaFill(0.28),
  fill: true, tension: 0.25, pointRadius: 0, pointHoverRadius: 4, borderWidth: 2, ...opts,
});
const spyDataset = (label, data) => ({
  label, data, borderColor: CHART.muted, backgroundColor: "transparent",
  borderDash: [5, 4], fill: false, tension: 0.25, pointRadius: 0, borderWidth: 1.5,
});

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
    datasets: [lineDataset("Bot", botPct), spyDataset("SPY", spyPct)],
  };
  const opts = chartOpts(
    (v) => v + "%",
    (c) => `${c.dataset.label}: ${c.parsed.y == null ? "—" : (c.parsed.y >= 0 ? "+" : "") + c.parsed.y.toFixed(2) + "%"}`
  );
  if (chart) {
    chart.data = data;
    chart.options = opts;
    chart.update("none");
  } else {
    chart = new Chart(document.getElementById("chart"), { type: "line", data, options: opts });
  }
}

let valueChart;
function renderValueChart(series) {
  // Raw dollars: actual portfolio value vs investing the same starting capital
  // in SPY at inception (start_equity * spy_t / spy_0).
  const withSpy = series.filter((p) => p.spy_price != null);
  const eq0 = series.length ? series[0].equity : null;
  const spy0 = withSpy.length ? withSpy[0].spy_price : null;

  const labels = series.map((p) => fmtTime(p.ts));
  const botVal = series.map((p) => p.equity);
  const spyVal = series.map((p) => (eq0 && spy0 && p.spy_price != null ? eq0 * (p.spy_price / spy0) : null));

  const data = {
    labels,
    datasets: [lineDataset("Portfolio", botVal), spyDataset("SPY equivalent", spyVal)],
  };
  const opts = chartOpts(
    (v) => fmtMoney(v),
    (c) => `${c.dataset.label}: ${c.parsed.y == null ? "—" : fmtMoney(c.parsed.y)}`
  );
  if (valueChart) {
    valueChart.data = data;
    valueChart.options = opts;
    valueChart.update("none");
  } else {
    valueChart = new Chart(document.getElementById("chartValue"), { type: "line", data, options: opts });
  }
}

function renderPositions(rows) {
  document.getElementById("posCount").textContent = rows.length;
  const tb = document.querySelector("#positions tbody");
  tb.innerHTML = rows.length
    ? rows
        .map((p) => {
          const acq =
            p.acquired_last_ts && p.acquired_last_ts !== p.acquired_first_ts
              ? `${fmtTime(p.acquired_first_ts)} – ${fmtTime(p.acquired_last_ts)}`
              : fmtTime(p.acquired_first_ts);
          const lots = p.n_lots > 1 ? ` <span class="muted">(${p.n_lots} lots)</span>` : "";
          const pricePerShare = p.qty ? p.market_value / p.qty : null;
          return `<tr>
            <td>${p.symbol}</td>
            <td>${fmtNum(p.qty)}</td>
            <td>${fmtMoney(p.cost_basis_per_share)}</td>
            <td>${fmtMoney(pricePerShare)}</td>
            <td>${acq}${lots}</td>
            <td>${fmtMoney(p.market_value)}</td>
            <td class="${signClass(p.unrealized_pnl)}">${p.unrealized_pnl == null ? "—" : fmtMoney(p.unrealized_pnl)}</td>
            <td class="${signClass(p.unrealized_pnl_pct)}">${fmtPct(p.unrealized_pnl_pct)}</td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="8" class="muted">No open positions</td></tr>`;
}

function renderTrades(rows) {
  const tb = document.querySelector("#trades tbody");
  tb.innerHTML = rows.length
    ? rows
        .map((t) => {
          const isSell = t.side.toLowerCase() === "sell";
          const pnl =
            isSell && t.realized_pnl != null
              ? `<span class="${signClass(t.realized_pnl)}">${fmtMoney(t.realized_pnl)} ${
                  t.realized_pnl_pct != null ? `(${fmtPct(t.realized_pnl_pct)})` : ""
                }</span>`
              : '<span class="muted">—</span>';
          return `<tr>
            <td>${fmtTime(t.ts)}</td>
            <td><span class="${isSell ? "sell" : "buy"}">${t.side.toUpperCase()}</span></td>
            <td>${t.symbol}</td>
            <td>${fmtNum(t.qty)}</td>
            <td>${fmtMoney(t.price)}</td>
            <td>${fmtMoney(t.qty * t.price)}</td>
            <td>${pnl}</td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="7" class="muted">No trades yet</td></tr>`;
}

function renderLeaderboard(lb) {
  const row = (t) => `<tr>
      <td>${fmtTime(t.ts)}</td>
      <td>${t.symbol}</td>
      <td>${fmtNum(t.qty)}</td>
      <td>${fmtMoney(t.price)}</td>
      <td class="${signClass(t.realized_pnl)}">${fmtMoney(t.realized_pnl)}</td>
      <td class="${signClass(t.realized_pnl_pct)}">${fmtPct(t.realized_pnl_pct)}</td>
    </tr>`;
  const fill = (id, rows) => {
    document.querySelector(`#${id} tbody`).innerHTML = rows.length
      ? rows.map(row).join("")
      : `<tr><td colspan="6" class="muted">No closed trades yet</td></tr>`;
  };
  fill("best", lb.best || []);
  fill("worst", lb.worst || []);
}

// Set text and color (pos/neg) without disturbing the element's base classes.
function setColored(id, text, sign) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.classList.remove("pos", "neg");
  const c = signClass(sign);
  if (c) el.classList.add(c);
}

function renderSummary(s) {
  document.getElementById("title").textContent = s.label || "Random Bot";
  document.getElementById("equity").textContent = fmtMoney(s.equity);
  document.getElementById("cash").textContent = fmtMoney(s.cash);

  setColored("botReturn", fmtPct(s.bot_return), s.bot_return);
  setColored("spyReturn", fmtPct(s.spy_return), s.spy_return);

  const alpha = s.bot_return != null && s.spy_return != null ? s.bot_return - s.spy_return : null;
  setColored("alpha", fmtPct(alpha), alpha);
  setColored("realized", s.realized_pnl == null ? "—" : fmtMoney(s.realized_pnl), s.realized_pnl);
  setColored("unrealized", s.unrealized_pnl == null ? "—" : fmtMoney(s.unrealized_pnl), s.unrealized_pnl);
  setColored("totalPnl", s.total_pnl == null ? "—" : fmtMoney(s.total_pnl), s.total_pnl);

  document.getElementById("winRate").textContent =
    s.win_rate == null ? "—" : `${(s.win_rate * 100).toFixed(0)}% (${s.n_wins}/${s.n_wins + s.n_losses})`;
  document.getElementById("fees").textContent = s.fees_total == null ? "—" : fmtMoney(s.fees_total);
  document.getElementById("updated").textContent = s.last_ts ? "updated " + fmtTime(s.last_ts) : "no data yet";
}

function renderPeriods(periods) {
  const el = document.getElementById("periods");
  if (!periods || !periods.length) {
    el.innerHTML = `<div class="muted">No data yet</div>`;
    return;
  }
  el.innerHTML = periods
    .map(
      (p) => `<div class="period">
        <div class="pk">${p.label}</div>
        <div class="pv ${signClass(p.bot)}">${fmtPct(p.bot)}</div>
        <div class="pspy">SPY <b class="${signClass(p.spy)}">${fmtPct(p.spy)}</b></div>
      </div>`
    )
    .join("");
}

const fmtRatio = (v) => (v == null ? "—" : v.toFixed(2));
const fmtPctPlain = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const fmtDays = (ms) => {
  if (ms == null) return "—";
  const d = ms / 86400000;
  if (d >= 1) return d.toFixed(1) + "d";
  const h = ms / 3600000;
  if (h >= 1) return h.toFixed(1) + "h";
  return Math.round(ms / 60000) + "m";
};
function setStat(id, text, sign) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = "v" + (sign == null ? "" : " " + signClass(sign));
}

function renderStats(s) {
  setStat("sTotalRet", fmtPct(s.total_pnl_pct), s.total_pnl_pct);
  setStat("sMdd", fmtPct(s.max_drawdown), s.max_drawdown);
  setStat("sVol", s.volatility == null ? "—" : (s.volatility * 100).toFixed(1) + "%");
  setStat("sSharpe", fmtRatio(s.sharpe), s.sharpe);
  setStat("sBest", fmtPct(s.best_day), s.best_day);
  setStat("sWorst", fmtPct(s.worst_day), s.worst_day);
  setStat("sPf", fmtRatio(s.profit_factor));
  setStat("sAvgWin", s.avg_win == null ? "—" : fmtMoney(s.avg_win), s.avg_win);
  setStat("sAvgLoss", s.avg_loss == null ? "—" : fmtMoney(s.avg_loss), s.avg_loss);
  setStat("sWinStreak", s.longest_win_streak ?? "—");
  setStat("sLossStreak", s.longest_loss_streak ?? "—");
  setStat("sHold", fmtDays(s.avg_hold_ms));
  setStat("sTpd", s.trades_per_day == null ? "—" : s.trades_per_day.toFixed(1));
  setStat("sInvested", fmtPctPlain(s.invested_pct));
  setStat("sCashPct", fmtPctPlain(s.cash_pct));
  setStat("sLargest", fmtPctPlain(s.largest_position_pct));
}

async function refresh() {
  try {
    const [summary, equity, positions, trades, leaderboard] = await Promise.all([
      getJSON("/api/summary"),
      getJSON("/api/equity"),
      getJSON("/api/positions"),
      getJSON("/api/trades?limit=100"),
      getJSON("/api/leaderboard?n=5"),
    ]);
    renderSummary(summary);
    renderPeriods(summary.period_returns);
    renderStats(summary);
    renderChart(equity);
    renderValueChart(equity);
    renderPositions(positions);
    renderTrades(trades);
    renderLeaderboard(leaderboard);
    document.getElementById("statusDot").classList.remove("err");
  } catch (e) {
    document.getElementById("updated").textContent = "connection error — retrying";
    document.getElementById("statusDot").classList.add("err");
    console.error(e);
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
