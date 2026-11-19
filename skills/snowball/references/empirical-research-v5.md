# Empirical Quantitative Research: Macro Transmission & Multi-Factor Equity Selection (v1 → v6)

This reference documents the peer-reviewed empirical asset pricing literature, exact mathematical factor definitions, point-in-time anti-look-ahead controls, and audit evolution (`v1` through `v6`) behind the **Snowball** skill.

---

## 1. Peer-Reviewed Empirical Literature Foundations

### A. Macro Shocks & Price Discovery Limits
- **Oil Shock Decomposition** — Kilian (2009, *AER*); Kilian & Park (2009, *IER*); Känzig (2021, *AER*): Crude moves have opposite equity effects depending on whether they stem from supply shocks vs. global demand shocks.
- **High-Frequency Pricing & Fed Information Effect** — Andersen et al. (2003, *AER*); Gürkaynak, Sack, & Swanson (2005, *IJCB*); Nakamura & Steinsson (2018, *QJE*); Bauer & Swanson (2023, *AER*): Public macro surprises are priced within minutes to hours; lagged weekly macro transmission has near-zero cross-sectional equity alpha.
- **Equity Duration & Bank Fragility** — Lettau & Wachter (2007 *JF*, 2011 *RFS*); Gonçalves (2021, *JFE*); Gormsen & Lazarus (2023, *JF*); Flannery & James (1984, *JF*); Drechsler, Savov, & Schnabl (2021, *JF*); Jiang et al. (2023, *JFE*).
- **Global Dollar Financial Channel** — Bruno & Shin (2015, *RES*); Kearns & Patel (2016, *BIS*); Obstfeld & Zhou (2022, *BPEA*).

### B. Momentum Quality & Crash Mitigation
- **Cross-Sectional Momentum** — Jegadeesh & Titman (1993, *JF*): Intermediate-horizon (6–12m, skipping the most recent month) winners outperform losers.
- **Volatility-Scaled Residual Momentum** — Blitz, Huij, & Martens (2011, *Journal of Empirical Finance*): Computing momentum on Beta-adjusted residual returns scaled by residual volatility removes time-varying market Beta and roughly doubles risk-adjusted Sharpe ratios.
- **Momentum Crashes & Volatility Scaling** — Barroso & Santa-Clara (2015, *JFE*); Daniel & Moskowitz (2016, *JFE*): Momentum suffers severe crashes during high-volatility bear-market rebounds; scaling by trailing volatility substantially mitigates crash tail risk.
- **Frog-in-the-Pan Continuous Information (`FIP`)** — Da, Gurun, & Warachka (2014, *Review of Financial Studies*, Vol. 27, No. 7, pp. 2171–2218): Investors underreact to continuous, gradual positive news flow relative to discrete single-day/week jumps. Smooth trends persist far more reliably than lottery-like spikes.
- **Industry & Factor Momentum** — Moskowitz & Grinblatt (1999, *JF*); Ehsani & Linnainmaa (2022, *JF*): Intermediate-horizon industry alpha exhibits strong persistence; computing the full-group mean (including self) prevents small-group leave-one-out mechanical reversal (`-1/(G-1)`).

### C. Capital-Structure-Neutral Value & Quality (Anti-Value-Trap Architecture)
- **Value and Momentum Everywhere** — Asness, Moskowitz, & Pedersen (2013, *JF*): Value and Momentum are negatively correlated across global equity markets; combining them improves Sharpe ratios and lowers drawdowns.
- **Enterprise Multiple (`EBITDA/EV`)** — Loughran & Wellman (2011, *JFQA*, Vol. 46, No. 6, pp. 1629–1650): Enterprise value multiples outperform raw `P/E` across non-financial firms by accounting for capital structure (debt and cash).
- **Industry-Specific Multiples** — Bhojraj & Lee (2002, *JAR*): Financials require Book-to-Price (`1/PBR`) rather than `EBITDA/EV` or raw `E/P`.
- **Payout Yield** — Boudoukh, Michaely, Richardson, & Roberts (2007, *JF*): Cash distributions provide durable valuation signal when extreme yield traps are capped (`<= 8.0%`).
- **Gross Profitability Premium** — Novy-Marx (2013, *JFE*): Profitability is the "other side of Value" — negatively correlated with Value and highly effective at filtering out distressed value traps.
- **5-Factor Profitability (`RMW`) & Cash-Based Profitability** — Fama & French (2015, *JFE*); Ball, Gerakos, Linnainmaa, & Nikolaev (2016, *JFE*, Vol. 121, No. 1, pp. 28–45); Sloan (1996, *TAR*): Cash-flow profitability (`FCF Yield`) beats accrual-heavy earnings.
- **Quality Minus Junk (`QMJ`)** — Asness, Frazzini, & Pedersen (2019, *Review of Accounting Studies*): Combining profitability (`ROE`, operating margin, gross margin), cash-flow generation, and balance-sheet safety (low non-financial leverage) delivers persistent global alpha and downside protection.

### D. Defensive Low-Risk & Retail Implementation Frictions
- **Betting Against Beta (`BAB`)** — Frazzini & Pedersen (2014, *JFE*): Leverage-constrained investors bid up high-Beta stocks, leaving low-Beta equities with superior risk-adjusted alpha.
- **Idiosyncratic Volatility (`IVOL`) & Lottery Effects** — Ang, Hodrick, Xing, & Zhang (2006, *JF*); Bali, Cakici, & Whitelaw (2011, *JFE*): High-IVOL and lottery-spike stocks systematically underperform globally.
- **Transaction Costs & Rebalancing Hysteresis** — Novy-Marx & Velikov (2016, *RFS*): Real-world commissions, taxes, and odd-lot spreads destroy weekly high-turnover signals; 4-week staggered cadence and cash-flow rebalancing preserve net alpha.
- **Disposition Effect** — Shefrin & Statman (1985, *JF*); Odean (1998, *JF*).

---

## 2. Exact Mathematical Specification (v6 6-Pillar Empirical Model)

Let $r_{i,t}$ denote weekly total return of stock $i$ at Friday close $t$ (converted into USD via historical FX in multi-currency `GLOBAL` runs), $r_{m,t}$ the benchmark weekly return, and $\hat{\beta}_{i,t-1}$ the lagged 52-week rolling market Beta (`shift(1)`, min 26 weeks). Weekly Beta-adjusted residual return is:

$$\hat{\varepsilon}_{i,t} = r_{i,t} - \hat{\beta}_{i,t-1}\,r_{m,t}$$

All cross-sectional z-scores $z_{\text{cs}}(\cdot)$ and within-industry z-scores $z_{\text{ind}}(\cdot)$ are winsorized to $[-3, +3]$.

### Pillar 1 — Momentum Quality (`F_MomQual`, Weight = `0.20`)
Combines **Volatility-Scaled Residual Momentum** (Blitz et al. 2011) and **Frog-in-the-Pan Continuous Information** (Da et al. 2014) over the 26-week window skipping the most recent 4 weeks ($k \in \{4,\dots,25\}$):

$$F_{\text{ResMom,Vol}}(i,t) = z_{\text{cs}}\!\left(\frac{\sum_{k=4}^{25}\hat{\varepsilon}_{i,t-k}}{\hat{\sigma}_{\varepsilon,i,t-4:t-25}\sqrt{22}}\right)$$

$$\text{ID}_{i,t} = \operatorname{sgn}(\text{PRET}_{i,t-4:t-25}) \cdot \left(\%\text{neg weeks}_{i,t-4:t-25} - \%\text{pos weeks}_{i,t-4:t-25}\right), \qquad F_{\text{FIP}}(i,t) = z_{\text{cs}}(-\text{ID}_{i,t})$$

$$F_{\text{MomQual}}(i,t) = z_{\text{cs}}\!\Big(0.75\,F_{\text{ResMom,Vol}}(i,t) + 0.25\,F_{\text{FIP}}(i,t)\Big)$$

### Pillar 2 — Full-Group Industry Momentum (`F_IndMom`, Weight = `0.15`)
12-week cumulative residual alpha averaged across all members of peer group $G_i$ (inclusive of stock $i$):

$$F_{\text{IndMom}}(i,t) = z_{\text{cs}}\!\left(\frac{1}{|G_i|}\sum_{j \in G_i}\sum_{k=0}^{11}\hat{\varepsilon}_{j,t-k}\right)$$

### Pillar 3 — Multi-Metric Capital-Structure-Neutral Value (`F_NormVal`, Weight = `0.25`)
Strictly uses positive trailing earnings (`trailingEps > 0` and `trailingPE > 0`, never falling back to `forwardPE` for loss-making companies) and capped yields (`E/P <= 15%`, `EBITDA/EV <= 25%`, `DivYield <= 8.0%`):

$$V^{\text{core}}_i = \begin{cases} z_{\text{ind}}(1/\text{PBR}_i) & i \in \text{Financials (PB Groups)} \\ 0.5\,z_{\text{ind}}(\min(\text{E/P}_i, 0.15)) + 0.5\,z_{\text{ind}}(\min(\text{EBITDA/EV}_i, 0.25)) & i \notin \text{Financials} \end{cases}$$

$$F_{\text{NormVal}}(i,t) = z_{\text{cs}}\!\Big(V^{\text{core}}_i + 0.5\,z_{\text{cs}}(\min(\text{DivYield}_i, 8.0\%))\Big)$$

*(For Taiwan cyclical sectors in `backtest_v5.py`, a 5-year rolling point-in-time median is additionally applied before z-scoring.)*

### Pillar 4 — Quality & Cash-Flow Profitability (`F_Quality`, Weight = `0.20`)
Within-industry profitability + cash-flow yield minus non-financial leverage:

$$F_{\text{Quality}}(i,t) = z_{\text{cs}}\!\Big(0.40\,z_{\text{ind}}(\text{ROE}_i) + 0.30\,z_{\text{ind}}(\text{OpMargin}_i) + 0.30\,z_{\text{ind}}(\text{GrossMargin}_i) + 0.25\,z_{\text{cs}}(\text{FCF/EV}_i) - 0.25\,z_{\text{non-fin}}(\text{D/E}_i)\Big)$$

### Pillar 5 — Fundamental Growth (`F_Fund`, Weight = `0.10`)
- Global live ranking (`global_ranker.py`): $F_{\text{Fund}} = z_{\text{cs}}\!\big(0.60\,z_{\text{cs}}(\text{RevYoY}_i) + 0.40\,z_{\text{cs}}(\text{EarnYoY}_i)\big)$ (clipped to $[-50\%, +150\%]$).
- Taiwan point-in-time backtest (`backtest_v5.py`): trailing 3-month revenue YoY + 3-month acceleration strictly lagged to the 11th of the following month.

### Pillar 6 — Defensive Low-Risk (`F_LowRisk`, Weight = `0.10`)
Combines Betting Against Beta and Low Idiosyncratic Volatility (26-week trailing $\hat{\sigma}_{\varepsilon}$):

$$F_{\text{LowRisk}}(i,t) = z_{\text{cs}}\!\Big(-0.5\,z_{\text{cs}}(\hat{\beta}_{i,t-1}) - 0.5\,z_{\text{cs}}(\hat{\sigma}_{\varepsilon,i,t})\Big)$$

### Composite v6 Score & Pure-Price Historical Signal
$$S_{v6}(i,t) = z_{\text{cs}}\!\Big(0.20\,F_{\text{MomQual}} + 0.15\,F_{\text{IndMom}} + 0.25\,F_{\text{NormVal}} + 0.20\,F_{\text{Quality}} + 0.10\,F_{\text{Fund}} + 0.10\,F_{\text{LowRisk}}\Big)$$

For historical global pure-price backtesting (without point-in-time historical financial statement snapshots):

$$S_{\text{price},v6}(i,t) = z_{\text{cs}}\!\Big(0.45\,F_{\text{ResMom,Vol}} + 0.15\,F_{\text{FIP}} + 0.25\,F_{\text{IndMom}} + 0.15\,F_{\text{LowRisk}}\Big)$$

---

## 3. Anti-Look-Ahead & Execution Audit Standards

1. **Next-Trading-Day Open Execution**: Signals computed on Friday close $t$ execute at the first trading day's Open strictly after $t$ (Monday open), never on the same Friday close bar.
2. **Nominal Unadjusted Affordability Filter**: `<= max_price` filters in retail simulation run strictly on unadjusted nominal prices (`raw_close_w`), avoiding future-dividend look-ahead.
3. **Unitized TWR NAV Drawdown**: Retail cash-flow simulations compute MDD on the unitized Time-Weighted Return (TWR) NAV curve (eliminating deposit-dilution artifacts); Core+Satellite portfolios combine exact weekly market values and cash flows before computing joint TWR MDD.
4. **Currency-Consistent Global Beta**: In `GLOBAL` and multi-currency runs, all local-currency stocks are converted to USD via historical Yahoo FX pairs before computing Beta and Residual Momentum against USD `ACWI`.
5. **Versioned TTL Cache**: All disk caches store `.meta.json` sidecars with `schema_version`, `universe_hash`, `created_at`, and dataset-specific TTL (24h daily / 7d monthly revenue).

---

## 4. Taiwan Holdout Audit Benchmark Summary (`2024-01` – `2026-11`, v3–v5 Reference)

Long-only Top 20% equal-weight basket across 4 staggered cohorts, `0.585%` round-trip cost, benchmarked against same-universe Equal-Weight Total Return (`EW-84 TR`, MDD `-17.8%`):

| Model | Rank IC (NW6 t) | Long / Short Gross | Net Excess vs EW (t) | Beta vs EW | Jensen Alpha (t) | True Weekly MDD | Sharpe | 4w Turnover | Avg Sectors (HHI) |
|---|---|---|---|---|---|---|---|---|---|
| `v3_smooth4w` | 0.0773 (3.93) | +11.2% / +19.7% | +11.96% (1.72) | 1.049 | +10.36% (1.52) | -18.1% | 1.88 | 31.9% | 10.1 (0.130) |
| `v4_income` | 0.0924 (4.57) | +12.3% / +14.1% | +8.74% (1.25) | 0.952 | +10.31% (1.46) | -15.7% | 1.84 | 38.1% | 9.1 (0.143) |
| `v5_core` | 0.0950 (5.14) | +14.0% / +19.4% | +16.55% (2.27) | 1.166 | +11.15% (1.69) | -23.2% | 1.88 | 27.2% | 9.9 (0.136) |
| **`v5_income`** | **0.0960 (5.63)** | **+14.6% / +20.9%** | +13.08% (1.97) | **0.961** | **+14.33% (2.16)** | **-17.4%** | **2.08** | **24.8%** | **11.2 (0.116)** |

> **Survivorship Bias Disclosure**: Static current-day stock lists omit delisted companies and include ex-post winners. Pass `--membership-csv` with historical index constituents for survivorship-free backtests. Expected live net alpha after odd-lot spreads in a point-in-time Top-150+ universe is realistically **`+1.5%` to `+3.0%` / year**.
