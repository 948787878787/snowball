# Snowball（雪球量化選股與現金流再平衡 Skill）

> **Language / 語言**：[English](./README.md) | **繁體中文**

**Snowball** 是一個專為**所有主流 AI 代理人（All-Agents Skill）**設計的開源台股量化研究套件，整合 **v5 多因子實證選股引擎**、**點對點（Point-in-Time）防偷看回測框架**，以及適合散戶零股的 **「只買不賣現金流再平衡＋超重贏家減碼」** 交易規則與互動式深淺色 HTML 工作台。

原生相容於 **Antigravity (AGY)**、**Claude Code**、**OpenAI Codex CLI**、**Cursor**、**Windsurf**、**Cline** 與 **OpenCode**。

> **免責聲明**：本專案僅供量化金融實證研究、回測方法論教學與個人紀律追蹤使用，不構成任何持牌投資建議，所有買賣與部位決策請自行負責。

---

## 核心特色

- **跨平台 All-Agents Skill（`skills/snowball/SKILL.md`）**：採用開放的 `SKILL.md` 漸進式揭露（Progressive Disclosure）結構，讓不同 AI 助理皆可自動發現、閱讀參考文件並執行標準化指令。
- **v5 實證多因子選股公式（`backtest_v5.py`）**：
  - **殘差動能（`F_ResMom_26_4w`）**：過去 26 週跳過近 4 週的 Beta 調整累積超額報酬（Blitz et al., 2011），去除隱性 Beta 並降低動能崩盤風險。
  - **全組產業動能（`F_IndMom_Full_12w`）**：過去 12 週同業全組平均 alpha（Moskowitz & Grinblatt, 1999），修正小組排除自己造成之 `-1/(G-1)` 機械性反轉漏洞。
  - **點對點月營收動能（`F_FinStmt`）**：近 3 個月營收年增率（YoY）與 3 個月加速度各占 50%，嚴格於次月 11 日才視為市場公開。
  - **殖利率 8% 封頂之同業中性價值（`F_NormVal`）**：組內標準化價值（金融股採 `1/PBR`、非金融採正規化 `1/PER`）搭配 `0.5 ×` 封頂殖利率（殖利率上限 `8.0%`、盈餘殖利率上限 `15%`，景氣循環股取 5 年滾動中位數防範航運／記憶體高峰價值陷阱）。
- **機構級嚴格回測審計**：採用 4 組錯開一週的重疊子組合（Jegadeesh-Titman 4-Cohort）、真實每週盯市最大回撤（MDD）、Newey-West（`lags = 6`）t 檢定、對含息等權大盤的 Jensen's Alpha，以及含 `0.30%` 零股滑價＋手續費＋證交稅的個人現金流模擬器。
- **深色／淺色互動式 HTML 工作台（`generate_dashboard.py`）**：單一獨立網頁檔，支援深淺色切換、`localStorage` 本機持久化、持股線上編輯、每月核心／衛星零股買單計算機與 10 年複利試算。
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

# 2. 執行離線單元測試（共 32 項測試）
.venv/bin/pytest skills/snowball/scripts

# 3. 執行完整 v5 點對點回測（自動抓取 Yahoo Finance 與 FinMind 並快取於本地 cache/）
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
