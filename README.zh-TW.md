# Snowball（雪球量化選股與現金流再平衡 Skill）

> **Language / 語言**：[English](./README.md) | **繁體中文**

**Snowball** 是一個專為**所有主流 AI 代理人（All-Agents Skill）**設計的開源**全球股市**量化研究套件，整合 **v5 多因子實證選股引擎**、**點對點（Point-in-Time）防偷看回測框架**，以及適合散戶零股的 **「只買不賣現金流再平衡＋超重贏家減碼」** 交易規則與互動式深淺色 HTML 工作台。

支援 **美股（US）、歐股（EU）、日股（JP）、港股（HK）、台股（TW）、跨區全球龍頭（GLOBAL）** 以及任意自訂全球股票代碼，並原生相容於 **Antigravity (AGY)**、**Claude Code**、**OpenAI Codex CLI**、**Cursor**、**Windsurf**、**Cline** 與 **OpenCode**。

> **免責聲明**：本專案僅供量化金融實證研究、回測方法論教學與個人紀律追蹤使用，不構成任何持牌投資建議，所有買賣與部位決策請自行負責。

---

## 支援的全球股市範圍（內建 2,780+ 檔，支援無限擴充）

每個主要市場皆以獨立 JSON 資料庫存放（`skills/snowball/resources/universes/*.json`），涵蓋 **13 大標準產業 × 每產業至少 20 家上市公司**——其中 **美股與台股皆已擴充至 1,000 家官方上市櫃公司**，並支援隨時透過 `manage_universe.py` 新增個股、批次匯入 CSV 或建立全新國家市場（如韓股 `KR`、澳股 `AU`、加股 `CA`）：

| 市場參數 | 涵蓋交易所與範圍 | 內建股數 / 產業數 | 基準指數 | 交易成本模型 |
|---|---|---|---|---|
| `--market US` | 美國 NYSE / NASDAQ 前 1,000 大企業 | **1,000 檔**（13 × ≥30） | `SPY` | `0.10%` |
| `--market TW` | 台灣上市／上櫃前 1,000 大公司（`.TW` / `.TWO`） | **1,000 檔**（13 × ≥20） | `^TWII` | `0.585%` |
| `--market EU` | 歐洲 Euronext / XETRA / LSE / SIX / 北歐 | **260 檔**（13 × 20） | `^STOXX50E` | `0.30%` |
| `--market JP` | 日本東京證券交易所（`.T`） | **260 檔**（13 × 20） | `^N225` | `0.20%` |
| `--market HK` | 香港聯交所（`.HK`） | **260 檔**（13 × 20） | `^HSI` | `0.45%`（含印花稅） |
| `--market GLOBAL` | 自動聚合全部區域市場（跨區同業比較） | **2,780+ 檔**（13 × ≥115） | `ACWI` | `0.30%` |
| `--tickers ...` / `--universe-file` | 任意自訂全球 Yahoo Finance 代碼或外部 JSON | 無上限 | 自訂（預設 `ACWI`） | `0.20%` |

```bash
# 查看所有市場股數與 13 大產業明細
.venv/bin/python skills/snowball/scripts/manage_universe.py --list
.venv/bin/python skills/snowball/scripts/manage_universe.py --market US --show-industries

# 隨時自行擴充現有市場或建立全新市場（例：新增美股或建立韓股 KR）
.venv/bin/python skills/snowball/scripts/manage_universe.py --market US --industry software_internet --add APP,MDB,NET
.venv/bin/python skills/snowball/scripts/manage_universe.py --market KR --name "South Korea (KRX)" --benchmark ^KS11 --currency KRW --industry semiconductors_hardware --add 005930.KS,000660.KS
.venv/bin/python skills/snowball/scripts/manage_universe.py --import-csv ./extra_stocks.csv
```

---

## 核心特色

- **跨平台 All-Agents Skill（`skills/snowball/SKILL.md`）**：採用開放的 `SKILL.md` 漸進式揭露（Progressive Disclosure）結構，讓不同 AI 助理皆可自動發現、閱讀參考文件並執行標準化指令。
- **全球通用 v6 六大支柱實證多因子引擎（`global_ranker.py` 與 `backtest_v5.py`）**（同時相容 `--model v5`）：
  - **動能品質（`F_MomQual`，20%）**：波動率調整殘差動能（Blitz et al. 2011 *JEF*；Barroso & Santa-Clara 2015 *JFE*）結合「溫水煮青蛙」連續資訊平滑度（Frog-in-the-Pan `FIP`，Da, Gurun, & Warachka 2014 *RFS*），獎勵穩健趨勢、過濾暴漲暴跌彩票股。
  - **全組產業動能（`F_IndMom`，15%）**：12 週同業全組平均 alpha（Moskowitz & Grinblatt 1999 *JF*；Ehsani & Linnainmaa 2022 *JF*），消除小組排除自己造成的機械性反轉。
  - **資本結構中性複合價值（`F_NormVal`，25%）**：金融股採組內 `1/PBR`，非金融股採組內 `0.5 × 盈餘殖利率（嚴格 Trailing P/E，上限 15%）+ 0.5 × EBITDA/EV（上限 25%）`（Loughran & Wellman 2011 *JFQA*），搭配 `0.5 ×` 上限 `8.0%` 封頂殖利率（Boudoukh et al. 2007 *JF*）。
  - **個體經濟護城河與資本紀律（`F_Quality`，20%）**：納入勒納超額毛利定價權（`毛利率 − 同業中位數`，Lerner 1934 *RES*；De Loecker, Eeckhout, & Unger 2020 *QJE*）、組內 `ROE` 與營業利益率（Novy-Marx 2013 *JFE*）、自由現金流殖利率 `FCF/EV`（Ball et al. 2016 *JFE*）、非金融流動比率（`CurrentRatio`）並扣除高槓桿負債比（Hou, Xue, & Zhang 2015 *RFS*；`QMJ`，Asness et al. 2019 *RAST*）。
  - **基本面成長（`F_Fund`，10%）**：全球採營收＋盈餘年增率，台股回測採嚴格延後至次月 11 日的點對點月營收動能與加速度。
  - **防禦低風險（`F_LowRisk`，10%）**：對抗高 Beta（Betting Against Beta，Frazzini & Pedersen 2014 *JFE*）結合低特異波動率（Ang et al. 2006 *JF*），降低空頭回撤與動能崩盤風險。
  - **總體經濟情境動態調節（`macro_micro.py`，嚴格滯後 `shift(1)` 防偷看）**：監控美債殖利率曲線（`^TNX - ^IRX`）、高收益信用利差（`HYG/IEF`）、波動率（`^VIX`）、美元流動性（`DX-Y.NYB`）與原油通膨衝擊（`CL=F`），在信用緊縮、升息通膨與景氣擴張三類總經情境間平滑調整六大支柱權重（Estrella & Mishkin 1998 *REStat*；Gilchrist & Zakrajšek 2012 *AER*；Bruno & Shin 2015 *RES*；Kilian 2009 *AER*）。
- **機構級嚴格回測審計**：採用 4 組錯開一週的重疊子組合（Jegadeesh-Titman 4-Cohort）、真實每週盯市最大回撤（MDD）、Newey-West（`lags = 6`）t 檢定、對含息等權大盤的 Jensen's Alpha，以及含 `0.30%` 零股滑價＋手續費＋證交稅的個人現金流模擬器。
- **極簡毛玻璃互動式 HTML 儀表板（`generate_dashboard.py`）**：單一獨立網頁檔，採高斯模糊毛玻璃（Liquid Glass）風格並自動適配系統／環境明暗色調；右上角收斂為單一**設定按鈕**，內建 **紅、橙、黃、綠、藍、靛、紫七彩主題色**、**五國語言切換（English 預設、繁體中文、日本語、한국어、Deutsch）** 與 **六種顯示貨幣即時換算（USD 預設、TWD、EUR、JPY、HKD、KRW）**，整合總投入金額、總資產、可自訂每月投入金額與預估年報酬率、未來一週行情、任務清單、線上持股編輯與第 1–10 年複利試算，並自動記憶於本機瀏覽器。
- **機率預測校準計分器（`score_predictions.py`）**：自動計算預測命中率、平均 Brier Score 與 Brier Skill Score（BSS）。

---

## 各大 AI Agent 安裝方式

先將專案複製到本地：

```bash
git clone https://github.com/948787878787/snowball.git
cd snowball
chmod +x install.sh
```

### 方式一：使用萬用一鍵安裝腳本（`install.sh`）

| AI 平台 / 代理人 | 全域安裝（所有專案生效） | 單一專案安裝（僅目前目錄生效） |
|---|---|---|
| **Google Antigravity (AGY)** | `./install.sh antigravity` | `./install.sh antigravity --project` |
| **Claude Code** | `./install.sh claude` | `./install.sh claude --project` |
| **OpenAI Codex CLI** | `./install.sh codex` | `./install.sh codex --project` |
| **Cursor** | `./install.sh cursor` | `./install.sh cursor --project` |
| **Windsurf** | `./install.sh windsurf` | `./install.sh windsurf --project` |
| **OpenCode / Cline / 通用標準（`.agents/skills`）** | — | `./install.sh universal` |
| **一次安裝至所有專案級目錄（`.agents` / `.claude` / `.cursor`）** | — | `./install.sh all` |

### 方式二：各平台手動複製路徑

- **Antigravity (AGY)**：
  - 全域：`mkdir -p ~/.gemini/config/skills && cp -r skills/snowball ~/.gemini/config/skills/`
  - 專案：`mkdir -p .agents/skills && cp -r skills/snowball .agents/skills/`
- **Claude Code**：
  - 全域：`mkdir -p ~/.claude/skills && cp -r skills/snowball ~/.claude/skills/`
  - 專案：`mkdir -p .claude/skills && cp -r skills/snowball .claude/skills/`
- **OpenAI Codex CLI**：
  - 全域：`mkdir -p ~/.codex/skills && cp -r skills/snowball ~/.codex/skills/`
  - 專案：`mkdir -p .codex/skills && cp -r skills/snowball .codex/skills/`
- **Cursor**：
  - 全域：`mkdir -p ~/.cursor/skills && cp -r skills/snowball ~/.cursor/skills/`
  - 專案：`mkdir -p .cursor/skills && cp -r skills/snowball .cursor/skills/`
- **Windsurf / Cline / OpenCode**：
  - 專案：`mkdir -p .agents/skills && cp -r skills/snowball .agents/skills/`

---

## Python 快速上手（不依賴 AI 也可獨立執行）

```bash
# 1. 建立虛擬環境並安裝套件
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 執行離線單元測試（共 35 項測試）
.venv/bin/pytest skills/snowball/scripts

# 3. 排序與回測全球任意股市（US / EU / JP / HK / TW / GLOBAL / 自訂代碼）
.venv/bin/python skills/snowball/scripts/global_ranker.py --market US --backtest
.venv/bin/python skills/snowball/scripts/global_ranker.py --market GLOBAL --backtest
.venv/bin/python skills/snowball/scripts/global_ranker.py --tickers AAPL,MSFT,GOOGL,ASML.AS,7203.T,2330.TW --benchmark ACWI

# 4. 執行台股深度回測（整合 FinMind 點對點月營收與零股現金流模擬器）
.venv/bin/python skills/snowball/scripts/backtest_v5.py

# 4. 複製零個資持股模板（已列入 .gitignore，絕不會上傳個人資料）並執行每週監控
cp skills/snowball/references/portfolio-template.json ./my_portfolio.json
.venv/bin/python skills/snowball/scripts/monitor_portfolio.py --portfolio ./my_portfolio.json

# 5. 產出深淺色互動式 HTML 工作台（支援 --offline 離線展示模式）
.venv/bin/python skills/snowball/scripts/generate_dashboard.py --portfolio ./my_portfolio.json --output ./snowball_dashboard.html

# 6. 計算預測紀錄命中率與 Brier Score / BSS
.venv/bin/python skills/snowball/scripts/score_predictions.py --csv ./my_predictions.csv
```

---

## 樣本外保留期審計摘要（`2024-01` ～ `2026-11`）

4 組錯開子組合、真實週線 MDD、Newey-West `lags = 6`、只做多前 20% 扣除 `0.585%` 來回交易成本，對照同池含息等權基準（`EW-84 TR`，真實 MDD `-17.8%`）：

| 模型版本 | Rank IC (NW6 t) | 對等權淨超額 | 組合 Beta | Jensen Alpha (t) | 真實週線 MDD | 夏普值 | 月換手率 | 平均產業數 |
|---|---|---|---|---|---|---|---|---|
| `v3_smooth4w` | 0.0773 (3.93) | +11.96% | 1.049 | +10.36% (1.52) | -18.1% | 1.88 | 31.9% | 10.1 |
| `v4_income` | 0.0924 (4.57) | +8.74% | 0.952 | +10.31% (1.46) | -15.7% | 1.84 | 38.1% | 9.1 |
| `v5_core` | 0.0950 (5.14) | +16.55% | 1.166 | +11.15% (1.69) | -23.2% | 1.88 | 27.2% | 9.9 |
| **`v5_income`** | **0.0960 (5.63)** | +13.08% | **0.961** | **+14.33% (2.16)** | **-17.4%** | **2.08** | **24.8%** | **11.2** |

> **倖存者偏誤誠實揭露**：目前 84 檔股票池採 2026 年視角挑選，未納入已下市公司。若擴充至點時性市值前 150 檔（含下市股）並計入零股滑價，合理樣本外淨超額預估約為每年 **`+1.5% ～ +3.0%`**。完整文獻與審計細節請見 [`skills/snowball/references/empirical-research-v5.md`](./skills/snowball/references/empirical-research-v5.md)。

## 授權條款

採用 MIT License，詳見 [LICENSE](./LICENSE)。
