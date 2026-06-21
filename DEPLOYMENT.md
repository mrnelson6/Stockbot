# Deploying the Random Bot + Public Dashboard

This is a **two-process** deployment on your Linux box:

- **Bot** (`scripts/random_portfolio.py --execute --record-db ...`) — trades a separate
  Alpaca account and writes equity/positions/trades to a SQLite file.
- **Web** (`scripts/serve_web.py`) — a read-only FastAPI site that reads that same SQLite
  file and serves the public dashboard.

They share one SQLite file (WAL mode = safe concurrent read/write). The web process holds
**no Alpaca credentials** and exposes **no write endpoints**.

---

## 0. Getting the code onto the box

You're developing on a different machine, so the cleanest path (the one you suggested) is
**Git**: push from here, clone/pull on the box. It also makes future updates a one-liner
(`git pull && systemctl restart ...`).

On your dev machine:
```bash
git add -A
git commit -m "Random bot + dashboard"
git push            # to GitHub (or any remote the box can reach)
```

On the Linux box:
```bash
git clone <your-repo-url> stockbot
cd stockbot
```

> Pure SSH (editing files directly on the box) also works, but Git gives you versioned,
> repeatable deploys and easy rollback. `.env` and `data/` are gitignored, so your keys and
> the database never get committed — you create those fresh on the box (below).

To ship an update later: `git pull` on the box, then restart the services.

---

## 1. One-time setup on the box

```bash
# Python 3.11+ and a venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[web]"     # installs the bot + FastAPI/uvicorn

# Credentials — create .env (NOT committed). Use a SECOND Alpaca paper account
# so the random bot is isolated from anything else.
cp .env.example .env
nano .env                   # set RANDOM_ALPACA_API_KEY / RANDOM_ALPACA_SECRET_KEY,
                            # RANDOM_ALPACA_PAPER=true
mkdir -p data
```

Quick sanity check (no orders, no DB):
```bash
python scripts/random_portfolio.py --universe-source static --universe-size 10
```

---

## 2. Run it manually first (verify before daemonizing)

Terminal 1 — the bot (paper trading + recording):
```bash
source .venv/bin/activate
python scripts/random_portfolio.py \
    --execute \
    --record-db ./data/dashboard.db \
    --label "Random Bot" \
    --poll-interval 30 \
    --trade-prob 0.2
```

Terminal 2 — the web server (bound to localhost; Caddy will expose it):
```bash
source .venv/bin/activate
python scripts/serve_web.py --db ./data/dashboard.db --host 127.0.0.1 --port 8000
```

Visit `http://<box-ip>:8000` (or tunnel) to confirm the dashboard populates. The bot only
records during market hours when `--execute` is on, so you may need to wait for the open.

---

## 3. Run as services (systemd)

Create `/etc/systemd/system/stockbot-random.service` (adjust `User` and paths):

```ini
[Unit]
Description=Random trading bot
After=network-online.target
Wants=network-online.target

[Service]
User=youruser
WorkingDirectory=/home/youruser/stockbot
ExecStart=/home/youruser/stockbot/.venv/bin/python scripts/random_portfolio.py \
    --execute --record-db /home/youruser/stockbot/data/dashboard.db \
    --label "Random Bot" --poll-interval 30 --trade-prob 0.2
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/stockbot-web.service`:

```ini
[Unit]
Description=Random bot dashboard
After=network-online.target

[Service]
User=youruser
WorkingDirectory=/home/youruser/stockbot
Environment=STOCKBOT_WEB_DB=/home/youruser/stockbot/data/dashboard.db
ExecStart=/home/youruser/stockbot/.venv/bin/python scripts/serve_web.py --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable both:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stockbot-random stockbot-web
sudo systemctl status stockbot-random stockbot-web
journalctl -u stockbot-random -f      # tail logs
```

---

## 4. HTTPS on your domain (Caddy)

The web server binds to `127.0.0.1:8000`; **don't expose that port directly.** Put
[Caddy](https://caddyserver.com) in front — it gets and renews a Let's Encrypt cert
automatically.

1. Point your domain's DNS **A record** at the box's public IP. If your IP is dynamic, use a
   dynamic-DNS updater. Open inbound **80** and **443** on the box/router firewall (you do
   NOT need to open 8000).
2. Install Caddy, then `/etc/caddy/Caddyfile`:

```
yourdomain.com {
    reverse_proxy 127.0.0.1:8000
}
```

3. `sudo systemctl reload caddy`

Visit `https://yourdomain.com` — done. Caddy terminates TLS and proxies to the dashboard.

---

## 5. Updating later

```bash
cd stockbot
git pull
source .venv/bin/activate
pip install -e ".[web]"        # only if deps changed
sudo systemctl restart stockbot-random stockbot-web
```

---

## Notes & safety

- **Paper first.** Keep `RANDOM_ALPACA_PAPER=true` until you're confident. Live trading
  means real losses on a deliberately random strategy.
- **The site is fully public and read-only** — that's the intent. It serves only derived
  data (equity curve, positions, trades). No keys, no order endpoints.
- **One DB file, two processes** is fine thanks to WAL. If you ever run multiple bots, give
  each its own `--record-db` file and its own web service/subdomain.
- **Market hours:** equity is recorded each tick while `--execute` is on and the market is
  open; overnight the curve is flat by design.
