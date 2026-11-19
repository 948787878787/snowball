# Empirical Quantitative Research: Macro Transmission & Multi-Factor Equity Selection (v1 → v5)

This reference summarizes the academic literature review, point-in-time backtesting evolution (v1 through v5), and independent referee audit behind the `tw-quant-factor-v5` skill.

---

## 1. Empirical Literature Foundations

### A. Macro Shocks & Price Discovery Limits
- **Oil Shock Decomposition** (Kilian 2009 *AER*; Kilian & Park 2009 *IER*; Känzig 2021 *AER*): Oil price moves have opposite effects on stocks depending on whether they stem from crude supply shocks, global aggregate demand shocks, or precautionary demand shocks.
- **High-Frequency Pricing & Fed Information Effect** (Andersen et al. 2003 *AER*; Gürkaynak, Sack, & Swanson 2005 *IJCB*; Nakamura & Steinsson 2018 *QJE*; Bauer & Swanson 2023 *AER*): Public macro releases (CPI, FOMC) are priced into bond yields and FX within minutes to hours; lagged weekly macro transmission has near-zero cross-sectional alpha in equities.
- **Equity Duration & Bank Fragility** (Lettau & Wachter 2007 *JF*, 2011 *RFS*; Gonçalves 2021 *JFE*; Gormsen & Lazarus 2023 *JF*; Flannery & James 1984 *JF*; Drechsler, Savov, & Schnabl 2021 *JF*; Jiang et al. 2023 *JFE*): High-duration growth stocks suffer most when real discount rates rise; commercial banks with sticky deposit franchises benefit from higher rates, while insurers/banks with heavy long-duration bond portfolios face mark-to-market losses.
- **Global Dollar Financial Channel** (Bruno & Shin 2015 *RES*; Kearns & Patel 2016 *BIS*; Obstfeld & Zhou 2022 *BPEA*): A strengthening USD tightens global financial conditions and drives foreign capital outflows from emerging markets.

### B. Cross-Sectional Factor & Portfolio Construction Literature
- **Industry Momentum** (Moskowitz & Grinblatt 1999 *JF*): Intermediate-horizon (3–12 month) industry-level returns exhibit strong positive persistence.
- **Residual Momentum** (Jegadeesh & Titman 1993 *JF*; Blitz, Huij, & Martens 2011 *JEF*): Measuring 6–12 month momentum (skipping the most recent month) on Beta-adjusted residual returns eliminates time-varying market Beta exposure and substantially reduces momentum crash risk (Daniel & Moskowitz 2016 *JFE*).
- **Value and Momentum Everywhere** (Asness, Moskowitz, & Pedersen 2013 *JF*; Bhojraj & Lee 2002 *JAR*): Value and Momentum are negatively correlated across asset classes; combining them improves Sharpe ratios and lowers drawdowns.
- **Transaction Costs & Rebalancing Hysteresis** (Novy-Marx & Velikov 2016 *RFS*): Round-trip costs in Taiwan equities (`0.585%` for round lots, higher for odd lots) destroy high-turnover weekly signals. Moving to a 4-week cadence and using sell-buffers or cash-flow rebalancing preserves net alpha.
- **Disposition Effect** (Shefrin & Statman 1985 *JF*; Odean 1998 *JF*): Retail investors reluctant to realize losses hold losers too long; under a strict zero-realized-loss preference, controlling initial position size, freezing laggards, and trimming overweight winners (`> 2×` target weight) is the key risk-control lever.

---

## 2. Evolution from v1 to v5 (Taiwan 84-Stock, 20-Peer-Group Backtest)

- **Point-in-Time Rules**: Weekly Friday (`W-FRI`) snapshots; monthly revenue lagged to the 11th of the following month (legal filing deadline is the 10th); all rolling time-series statistics strictly use `shift(1)` (zero look-ahead).
- **Split Discipline**:
  - **In-Sample Design Period**: `2020-01` to `2023-12`
  - **Out-of-Sample Holdout Period**: `2024-01` to `2026-11` (note: semi-contaminated because the 84-stock dataset was reused across iterations).

### Key Audit Upgrades Implemented in v5
1. **Replaced raw momentum with Residual Momentum (`F_ResMom_26_4w`)**: Eliminated hidden time-varying Beta exposure.
2. **Replaced leave-one-out small-group industry mean with Full-Group Industry Momentum (`F_IndMom_Full_12w`)**: Fixed the mechanical `-1/(G-1) * r_i` reversal artifact present when peer groups contain only 3–4 stocks.
3. **Cyclical-Normalized & Capped Industry-Neutral Valuation (`F_NormVal`)**:
   - Capped dividend yield at **`8.0%`** and earnings yield at **`15.0%`** (`P/E >= 6.7x`) to prevent cycle-peak value traps (such as 2021–2023 container-shipping dividend spikes).
   - Applied 5-year rolling median normalization to cyclical industries (`container_shipping`, `memory`, `steel`, `petrochemical`, `airline`).
   - Computed within-peer-group z-scores (`1/PBR` for financials, normalized `1/PER` for non-financials) plus `0.5 ×` cross-sectional capped dividend yield.
4. **Dropped short-term high-turnover signals**: Set weights for `F_Macro`, `F_Chain`, `F_Flow`, and `F_PeerRev_4w` to `0`.
5. **Rigorous Evaluation Harness**:
   - 4 staggered overlapping Jegadeesh-Titman cohorts (removing rebalance start-date luck).
   - True weekly mark-to-market Max Drawdown (never skipping intra-month crashes).
   - Explicit Newey-West `lags = 6`, Long-vs-Short spread decomposition, Portfolio Beta, Jensen's Alpha vs. Equal-Weight Total Return (`EW-84 TR`), and Sector HHI.

---

## 3. Out-of-Sample Holdout Audit Summary (`2024-01` to `2026-11`, 4-Cohort Staggered)

Long-only Top 20% equal-weight basket, `0.585%` round-trip cost, benchmarked against same-universe Equal-Weight Total Return (`EW-84 TR`, MDD `-17.8%`):

| Model | Rank IC (NW6 t) | Long / Short Gross | Net Excess vs EW (t) | Beta vs EW | Jensen Alpha (t) | True Weekly MDD | Sharpe | 4w Turnover | Avg Sectors (HHI) |
|---|---|---|---|---|---|---|---|---|---|
| `v3_smooth4w` | 0.0773 (3.93) | +11.2% / +19.7% | +11.96% (1.72) | 1.049 | +10.36% (1.52) | -18.1% | 1.88 | 31.9% | 10.1 (0.130) |
| `v4_income` | 0.0924 (4.57) | +12.3% / +14.1% | +8.74% (1.25) | 0.952 | +10.31% (1.46) | -15.7% | 1.84 | 38.1% | 9.1 (0.143) |
| `v5_core` | 0.0950 (5.14) | +14.0% / +19.4% | +16.55% (2.27) | 1.166 | +11.15% (1.69) | -23.2% | 1.88 | 27.2% | 9.9 (0.136) |
| **`v5_income`** | **0.0960 (5.63)** | **+14.6% / +20.9%** | +13.08% (1.97) | **0.961** | **+14.33% (2.16)** | **-17.4%** | **2.08** | **24.8%** | **11.2 (0.116)** |

---

## 4. Honest Limitations & Realistic Live Expectations

1. **Survivorship Bias**: The 84-stock peer universe was curated in 2026 and therefore omits delisted stocks and includes ex-post winners. While comparing against `EW-84 TR` removes pool-level survivorship bias, factor-survivorship interactions remain.
2. **Expected Live Net Alpha**: In a point-in-time top-150 Taiwan market universe (including delisted names) after odd-lot bid-ask spreads, realistic net out-of-sample excess return over an equal-weight benchmark is estimated at **`+1.5%` to `+3.0%` per year**.
3. **Practical Retail Deployment**: For fixed-monthly-DCA investors with a zero-realized-loss constraint, a **Core (60% Broad/High-Dividend ETF) + Satellite (40% v5 Buy-Only Cash-Flow Rebalancing + Overweight Winner Trimming)** architecture substantially lowers single-stock concentration (`< 8%` max weight) and drawdown compared to 100% single-stock picking.
