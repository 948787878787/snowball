# Snowball 

> **Language / 語言**: **English** | [繁體中文](./README.zh-TW.md)

An open-source **All-Agents Skill** & Python quantitative research workbench for **global equity multi-factor ranking (v5)**, **point-in-time backtesting**, and **buy-only retail cash-flow rebalancing** — covering **US, Europe, Japan, Hong Kong, Taiwan, and custom global tickers**.

Designed to work seamlessly across **Antigravity (AGY)**, **Claude Code**, **OpenAI Codex CLI**, **Cursor**, **Windsurf**, **Cline**, and **OpenCode**.

> **Disclaimer**: This repository is strictly for quantitative research, backtesting methodology, and educational use. It is not licensed financial advice.

---

## Global Market Coverage (2,780+ Built-In Equities, Infinitely Extensible)

Every regional market ships as an editable JSON database (`skills/snowball/resources/universes/*.json`) covering **13 standardized industries × ≥ 20 listed companies per industry** — including **1,000 US** and **1,000 Taiwan** official listed companies — and users can freely append more tickers, bulk-import CSVs, or add brand-new markets (`KR`, `AU`, `CA`, `IN`, etc.) via `manage_universe.py`:

| Market Flag | Exchanges / Scope | Stocks / Industries | Benchmark | Cost Model |
|---|---|---|---|---|
| `--market US` | NYSE / NASDAQ top 1,000 operating companies | **1,000** (13 × ≥30) | `SPY` | `0.10%` |
| `--market TW` | TWSE / TPEx top 1,000 listed companies (`.TW` / `.TWO`) | **1,000** (13 × ≥20) | `^TWII` | `0.585%` |
| `--market EU` | Euronext / XETRA / LSE / SIX / Nordic | **260** (13 × 20) | `^STOXX50E` | `0.30%` |
| `--market JP` | Tokyo Stock Exchange (`.T`) | **260** (13 × 20) | `^N225` | `0.20%` |
| `--market HK` | Hong Kong Exchange (`.HK`) | **260** (13 × 20) | `^HSI` | `0.45%` |
| `--market GLOBAL` | Auto-aggregated across all regional markets | **2,780+** (13 × ≥115) | `ACWI` | `0.30%` |
| `--tickers ...` / `--universe-file` | Any custom Yahoo Finance tickers or custom JSON | Unlimited | Custom (`ACWI`) | `0.20%` |

```bash
# Inspect all markets & 13 industries
.venv/bin/python skills/snowball/scripts/manage_universe.py --list
.venv/bin/python skills/snowball/scripts/manage_universe.py --market US --show-industries

# Add more tickers anytime (or create a brand-new market like KR)
.venv/bin/python skills/snowball/scripts/manage_universe.py --market US --industry software_internet --add APP,MDB,NET
.venv/bin/python skills/snowball/scripts/manage_universe.py --market KR --name "South Korea (KRX)" --benchmark ^KS11 --currency KRW --industry semiconductors_hardware --add 005930.KS,000660.KS
.venv/bin/python skills/snowball/scripts/manage_universe.py --import-csv ./extra_stocks.csv
```

---

## Features

- **Universal Agent Skill (`skills/snowball/SKILL.md`)**: Follows the open `SKILL.md` progressive-disclosure standard so any AI coding agent can discover and execute the workflow automatically.
- **Global v6 6-Pillar Empirical Multi-Factor Engine (`global_ranker.py` & `backtest_v5.py`)** (also supports `--model v5`):
  - **Momentum Quality (`F_MomQual`, 20%)**: Volatility-Scaled Residual Momentum (Blitz et al. 2011 *JEF*; Barroso & Santa-Clara 2015 *JFE*) + Frog-in-the-Pan Continuous Information smoothness (`FIP`, Da, Gurun, & Warachka 2014 *RFS*).
  - **Full-Group Industry Momentum (`F_IndMom`, 15%)**: 12-week peer-group mean alpha without small-group leave-one-out reversal bias (Moskowitz & Grinblatt 1999 *JF*; Ehsani & Linnainmaa 2022 *JF*).
  - **Capital-Structure-Neutral Multi-Metric Value (`F_NormVal`, 25%)**: Within-peer-group `1/PBR` (financials) or `0.5 × E/P + 0.5 × EBITDA/EV` (non-financials, Loughran & Wellman 2011 *JFQA*, strictly trailing P/E) + `0.5 ×` capped dividend yield (`<= 8.0%`, Boudoukh et al. 2007 *JF*).
  - **Microeconomic Moat & Capital Discipline (`F_Quality`, 20%)**: Lerner excess gross-margin pricing power (`GrossMargin - peer_median`, Lerner 1934 *RES*; De Loecker, Eeckhout, & Unger 2020 *QJE*), within-industry `ROE` & Operating Margin (Novy-Marx 2013 *JFE*; Fama & French 2015 *JFE*), `FCF/EV` (Ball et al. 2016 *JFE*), non-financial liquidity (`CurrentRatio`), minus leverage (`D/E`, Hou, Xue, & Zhang 2015 *RFS*; `QMJ`, Asness, Frazzini, & Pedersen 2019 *RAST*).
  - **Fundamental Growth & Sequoia Rule of 40 (`F_Fund`, 10%)**: Within-industry **Rule of 40** (`Revenue Growth + FCF Margin`, exposing `rule40_%`) combined with Revenue + Earnings growth YoY globally (or point-in-time lagged monthly revenue YoY + acceleration in Taiwan).
  - **Defensive Low-Risk (`F_LowRisk`, 10%)**: Betting Against Beta (Frazzini & Pedersen 2014 *JFE*) + Low Idiosyncratic Volatility (Ang et al. 2006 *JF*).
  - **Macroeconomic Regime Conditioning (`macro_micro.py`, strictly lagged `shift(1)`)**: Tracks Yield Curve (`^TNX - ^IRX`), Credit Spreads (`HYG/IEF`), Volatility (`^VIX`), US Dollar (`DX-Y.NYB`), and Oil (`CL=F`) to smoothly tilt pillar weights within strict bounds across Credit Stress, Rate-Hike/Inflation, and Expansion regimes (Estrella & Mishkin 1998 *REStat*; Gilchrist & Zakrajšek 2012 *AER*; Bruno & Shin 2015 *RES*; Kilian 2009 *AER*).
- **Institutional Audit Harness**: 4 staggered overlapping Jegadeesh-Titman cohorts, true weekly mark-to-market MDD, Newey-West (`lags = 6`) t-stats, Jensen's Alpha vs. equal-weight total return, and a realistic retail odd-lot cash-flow simulator (`0.30%` odd-lot slippage + commission + sell tax).
- **Minimalist Liquid Glass HTML Dashboard (`generate_dashboard.py`)**: Self-contained frosted-glass (`backdrop-filter`) consumer UI with automatic system/host color adaptation and a sleek **Settings popover** featuring **7 rainbow glass tints (Red, Orange, Yellow, Green, Blue, Indigo, Violet)**, **5 languages (English default, 繁體中文, 日本語, 한국어, Deutsch)**, and **6 live-converted display currencies (USD default, TWD, EUR, JPY, HKD, KRW)**, plus `localStorage` persistence, editable monthly contribution & expected annual return rate, next-week market outlook, checklist, editable holdings, and Y1–Y10 compounding calculator.
- **Brier Score Calibration Tracker (`score_predictions.py`)**: Evaluates probabilistic forecasts and reports Hit Rate, Mean Brier Score, and Brier Skill Score (BSS).

---

## Installation for All AI Agents

First, clone the repository:

```bash
git clone https://github.com/948787878787/snowball.git
cd snowball
chmod +x install.sh
```

### Option 1: One-Command Universal Installer (`install.sh`)

| AI Agent / Platform | Global Install (All Projects) | Project-Local Install (Current Workspace) |
|---|---|---|
| **Google Antigravity (AGY)** | `./install.sh antigravity` | `./install.sh antigravity --project` |
| **Claude Code** | `./install.sh claude` | `./install.sh claude --project` |
| **OpenAI Codex CLI** | `./install.sh codex` | `./install.sh codex --project` |
| **Cursor** | `./install.sh cursor` | `./install.sh cursor --project` |
| **Windsurf** | `./install.sh windsurf` | `./install.sh windsurf --project` |
| **OpenCode / Cline / Universal (`.agents/skills`)** | — | `./install.sh universal` |
| **Install All Project Agent Paths at Once** | — | `./install.sh all` |

### Option 2: Manual Copy per Platform

- **Antigravity (AGY)**:
  - Global: `cp -r skills/snowball ~/.gemini/config/skills/`
  - Workspace: `mkdir -p .agents/skills && cp -r skills/snowball .agents/skills/`
- **Claude Code**:
  - Global: `mkdir -p ~/.claude/skills && cp -r skills/snowball ~/.claude/skills/`
  - Project: `mkdir -p .claude/skills && cp -r skills/snowball .claude/skills/`
- **OpenAI Codex CLI**:
  - Global: `mkdir -p ~/.codex/skills && cp -r skills/snowball ~/.codex/skills/`
  - Project: `mkdir -p .codex/skills && cp -r skills/snowball .codex/skills/`
- **Cursor**:
  - Global: `mkdir -p ~/.cursor/skills && cp -r skills/snowball ~/.cursor/skills/`
  - Project: `mkdir -p .cursor/skills && cp -r skills/snowball .cursor/skills/`
- **Windsurf / Cline / OpenCode**:
  - Project: `mkdir -p .agents/skills && cp -r skills/snowball .agents/skills/`

---

## Python Quickstart (Standalone CLI)

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run offline unit test suite (35 tests)
.venv/bin/pytest skills/snowball/scripts

# 3. Rank & backtest ANY global market (US / EU / JP / HK / TW / GLOBAL / custom tickers)
.venv/bin/python skills/snowball/scripts/global_ranker.py --market US --backtest
.venv/bin/python skills/snowball/scripts/global_ranker.py --market GLOBAL --backtest
.venv/bin/python skills/snowball/scripts/global_ranker.py --tickers AAPL,MSFT,GOOGL,ASML.AS,7203.T,2330.TW --benchmark ACWI

# 4. Run full Taiwan point-in-time backtest with FinMind monthly revenue & odd-lot simulator
.venv/bin/python skills/snowball/scripts/backtest_v5.py

# 4. Copy zero-PII portfolio template (gitignored) & run weekly monitor
cp skills/snowball/references/portfolio-template.json ./my_portfolio.json
.venv/bin/python skills/snowball/scripts/monitor_portfolio.py --portfolio ./my_portfolio.json

# 5. Generate interactive Light/Dark HTML dashboard (supports --offline demo mode)
.venv/bin/python skills/snowball/scripts/generate_dashboard.py --portfolio ./my_portfolio.json --output ./snowball_dashboard.html

# 6. Evaluate probabilistic forecasts (Brier Score & BSS)
.venv/bin/python skills/snowball/scripts/score_predictions.py --csv ./my_predictions.csv
```

---

## Out-of-Sample Audit Summary (`2024-01` – `2026-11`)

4 staggered cohorts, true weekly mark-to-market MDD, Newey-West `lags = 6`, long-only Top 20% after `0.585%` round-trip cost, benchmarked against same-universe Equal-Weight Total Return (`EW-84 TR`, MDD `-17.8%`):

| Model | Rank IC (NW6 t) | Net Excess vs EW | Beta | Jensen Alpha (t) | Weekly MDD | Sharpe | 4w Turnover | Avg Sectors |
|---|---|---|---|---|---|---|---|---|
| `v3_smooth4w` | 0.0773 (3.93) | +11.96% | 1.049 | +10.36% (1.52) | -18.1% | 1.88 | 31.9% | 10.1 |
| `v4_income` | 0.0924 (4.57) | +8.74% | 0.952 | +10.31% (1.46) | -15.7% | 1.84 | 38.1% | 9.1 |
| `v5_core` | 0.0950 (5.14) | +16.55% | 1.166 | +11.15% (1.69) | -23.2% | 1.88 | 27.2% | 9.9 |
| **`v5_income`** | **0.0960 (5.63)** | +13.08% | **0.961** | **+14.33% (2.16)** | **-17.4%** | **2.08** | **24.8%** | **11.2** |

> **Survivorship Bias Disclosure**: The 84-stock peer universe is curated ex-post and omits delisted stocks. Expected live net alpha in a point-in-time Top-150 Taiwan universe after odd-lot spreads is realistically **+1.5% to +3.0% / year**. See [`skills/snowball/references/empirical-research-v5.md`](./skills/snowball/references/empirical-research-v5.md) for full details.

## License

MIT License. See [LICENSE](./LICENSE).
