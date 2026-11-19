"""Extensible Global Multi-Market Equity Universe Registry (1,340+ built-in stocks).

Built-in JSON databases live in `skills/snowball/resources/universes/*.json`:
- US     : 260+ liquid US stocks across 13 industries (>= 20/industry, Benchmark: SPY)
- EU     : 260+ European stocks across 13 industries (>= 20/industry, Benchmark: ^STOXX50E)
- JP     : 260+ Japan stocks across 13 industries (>= 20/industry, Benchmark: ^N225)
- HK     : 260+ Hong Kong stocks across 13 industries (>= 20/industry, Benchmark: ^HSI)
- TW     : 300+ Taiwan stocks across 13 industries (>= 20/industry, Benchmark: ^TWII)
- GLOBAL : Dynamic cross-region aggregation of all regional markets (1,340+ stocks, Benchmark: ACWI)

Extensibility:
Users and AI agents can freely add more stocks to existing markets or add brand-new markets
(e.g., KR, AU, IN, CA) via:
1. `manage_universe.py --market US --industry software_internet --add APP,MDB`
2. Dropping custom JSON files in `resources/custom_universes/` or setting `SNOWBALL_CUSTOM_UNIVERSE_DIR`
3. Passing `--universe-file path/to/my_universe.json` directly to `global_ranker.py`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from universe import PB_GROUPS as TW_LEGACY_PB_GROUPS

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR.parent / "resources"
BUILTIN_UNIVERSES_DIR = RESOURCES_DIR / "universes"
DEFAULT_CUSTOM_UNIVERSES_DIR = RESOURCES_DIR / "custom_universes"

DEFAULT_PB_GROUPS = frozenset({
    "banks",
    "insurance_diversified_fin",
    "us_financials",
    "eu_financials",
    "jp_financials",
    "hk_financials",
    "global_financials",
    *TW_LEGACY_PB_GROUPS,
})

DEFAULT_CYCLICAL_GROUPS = frozenset({
    "transportation_logistics",
    "energy_oil_gas",
    "materials_chemicals_metals",
    "container_shipping",
    "memory",
    "steel",
    "petrochemical",
    "airline",
})

GLOBAL_CONFIG_TEMPLATE = {
    "name": "Global Multi-Region Universe (All Markets)",
    "benchmark": "ACWI",
    "currency": "USD",
    "round_trip_cost": 0.0030,
    "odd_lot_slippage": 0.0020,
}

MARKET_CONFIG: Dict[str, dict] = {}
MARKET_GROUPS: Dict[str, Dict[str, Tuple[str, ...]]] = {}
MARKET_INDUSTRY_LABELS: Dict[str, Dict[str, str]] = {}
GLOBAL_PB_GROUPS = set(DEFAULT_PB_GROUPS)
GLOBAL_CYCLICAL_GROUPS = set(DEFAULT_CYCLICAL_GROUPS)


def _custom_dir() -> Path:
    env_dir = os.environ.get("SNOWBALL_CUSTOM_UNIVERSE_DIR", "").strip()
    return Path(env_dir) if env_dir else DEFAULT_CUSTOM_UNIVERSES_DIR


def _dedup_list(items: Iterable[str]) -> Tuple[str, ...]:
    seen = []
    seen_set = set()
    for raw in items:
        sym = str(raw).strip()
        if sym and sym not in seen_set:
            seen.append(sym)
            seen_set.add(sym)
    return tuple(seen)


def _load_json_into(
    path: Path,
    configs: Dict[str, dict],
    groups_by_mkt: Dict[str, Dict[str, List[str]]],
    labels_by_mkt: Dict[str, Dict[str, str]],
    pb_set: set,
    cyc_set: set,
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mkt = str(data["market"]).upper().strip()
    if mkt == "GLOBAL":
        return

    existing_cfg = configs.get(mkt, {})
    configs[mkt] = {
        "name": data.get("name", existing_cfg.get("name", mkt)),
        "benchmark": data.get("benchmark", existing_cfg.get("benchmark", "ACWI")),
        "currency": data.get("currency", existing_cfg.get("currency", "USD")),
        "round_trip_cost": float(data.get("round_trip_cost", existing_cfg.get("round_trip_cost", 0.0020))),
        "odd_lot_slippage": float(data.get("odd_lot_slippage", existing_cfg.get("odd_lot_slippage", 0.0020))),
    }

    mkt_inds = groups_by_mkt.setdefault(mkt, {})
    for ind, members in data.get("industries", {}).items():
        cur = mkt_inds.setdefault(ind, [])
        cur_set = set(cur)
        for t in members:
            sym = str(t).strip()
            if sym and sym not in cur_set:
                cur.append(sym)
                cur_set.add(sym)

    if isinstance(data.get("industry_labels"), dict):
        labels_by_mkt.setdefault(mkt, {}).update(data["industry_labels"])
    for g in data.get("pb_groups", []):
        pb_set.add(g)
    for g in data.get("cyclical_groups", []):
        cyc_set.add(g)


def reload_universes(
    builtin_dir: Optional[Path] = None,
    custom_dir: Optional[Path] = None,
) -> None:
    """Load all built-in market JSON files and merge any user custom universe files."""
    b_dir = builtin_dir or BUILTIN_UNIVERSES_DIR
    c_dir = custom_dir or _custom_dir()

    configs: Dict[str, dict] = {}
    groups_raw: Dict[str, Dict[str, List[str]]] = {}
    labels_raw: Dict[str, Dict[str, str]] = {}
    pb_set = set(DEFAULT_PB_GROUPS)
    cyc_set = set(DEFAULT_CYCLICAL_GROUPS)

    for folder in (b_dir, c_dir):
        if folder and folder.exists():
            for jf in sorted(folder.glob("*.json")):
                _load_json_into(jf, configs, groups_raw, labels_raw, pb_set, cyc_set)

    # Build GLOBAL by merging all regional markets by industry
    global_inds: Dict[str, List[str]] = {}
    global_labels: Dict[str, str] = {}
    for mkt in sorted(groups_raw.keys()):
        global_labels.update(labels_raw.get(mkt, {}))
        for ind, members in groups_raw[mkt].items():
            cur = global_inds.setdefault(ind, [])
            cur_set = set(cur)
            for sym in members:
                if sym not in cur_set:
                    cur.append(sym)
                    cur_set.add(sym)

    configs["GLOBAL"] = dict(GLOBAL_CONFIG_TEMPLATE)
    groups_raw["GLOBAL"] = global_inds
    labels_raw["GLOBAL"] = global_labels

    MARKET_CONFIG.clear()
    MARKET_CONFIG.update(configs)

    MARKET_GROUPS.clear()
    for mkt, inds in groups_raw.items():
        MARKET_GROUPS[mkt] = {ind: _dedup_list(mems) for ind, mems in inds.items()}

    MARKET_INDUSTRY_LABELS.clear()
    MARKET_INDUSTRY_LABELS.update(labels_raw)

    GLOBAL_PB_GROUPS.clear()
    GLOBAL_PB_GROUPS.update(pb_set)

    GLOBAL_CYCLICAL_GROUPS.clear()
    GLOBAL_CYCLICAL_GROUPS.update(cyc_set)


def list_markets() -> Tuple[str, ...]:
    return tuple(MARKET_CONFIG.keys())


def load_universe_file(path: Path) -> Tuple[dict, Dict[str, Tuple[str, ...]]]:
    """Load a standalone custom JSON universe file for one-off ranking/backtesting."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = {
        "name": data.get("name", path.stem),
        "benchmark": data.get("benchmark", "ACWI"),
        "currency": data.get("currency", "USD"),
        "round_trip_cost": float(data.get("round_trip_cost", 0.0020)),
        "odd_lot_slippage": float(data.get("odd_lot_slippage", 0.0020)),
    }
    groups = {ind: _dedup_list(mems) for ind, mems in data.get("industries", {}).items()}
    return cfg, groups


def get_market_groups(
    market: str,
    industry: Optional[str] = None,
    extra_file: Optional[Path] = None,
) -> Dict[str, Tuple[str, ...]]:
    key = market.upper()
    if key not in MARKET_GROUPS:
        raise ValueError(f"Unsupported market '{market}'. Choose from: {', '.join(MARKET_GROUPS)}")
    groups = {g: tuple(mems) for g, mems in MARKET_GROUPS[key].items()}

    if extra_file is not None:
        _, extra_groups = load_universe_file(extra_file)
        for ind, mems in extra_groups.items():
            merged = list(groups.get(ind, ()))
            for t in mems:
                if t not in merged:
                    merged.append(t)
            groups[ind] = tuple(merged)

    if industry:
        ind_key = industry.strip()
        if ind_key not in groups:
            raise ValueError(
                f"Industry '{ind_key}' not found in {key}. Available: {', '.join(sorted(groups.keys()))}"
            )
        return {ind_key: groups[ind_key]}
    return groups


def get_market_tickers(
    market: str,
    industry: Optional[str] = None,
    extra_file: Optional[Path] = None,
) -> Tuple[str, ...]:
    groups = get_market_groups(market, industry=industry, extra_file=extra_file)
    return _dedup_list(t for mems in groups.values() for t in mems)


def group_map_for(groups: Dict[str, Iterable[str]]) -> Dict[str, str]:
    return {t: g for g, members in groups.items() for t in members}


def add_tickers_to_universe(
    market: str,
    industry: str,
    tickers: Iterable[str],
    target_dir: Optional[Path] = None,
    benchmark: Optional[str] = None,
    currency: Optional[str] = None,
    name: Optional[str] = None,
    round_trip_cost: Optional[float] = None,
) -> Tuple[Path, int, int]:
    """Persistently append tickers to a market/industry JSON database (or create a new market).

    Returns (json_path, added_count, total_market_tickers).
    """
    mkt = market.upper().strip()
    if mkt == "GLOBAL":
        raise ValueError("Add tickers to a specific regional market (e.g., US, EU, JP, HK, TW, KR); GLOBAL aggregates them automatically.")

    out_dir = target_dir or BUILTIN_UNIVERSES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{mkt.lower()}.json"

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        base_cfg = MARKET_CONFIG.get(mkt, {})
        data = {
            "market": mkt,
            "name": name or base_cfg.get("name", f"{mkt} Equities"),
            "benchmark": benchmark or base_cfg.get("benchmark", "ACWI"),
            "currency": currency or base_cfg.get("currency", "USD"),
            "round_trip_cost": float(round_trip_cost if round_trip_cost is not None else base_cfg.get("round_trip_cost", 0.0020)),
            "odd_lot_slippage": float(base_cfg.get("odd_lot_slippage", 0.0020)),
            "pb_groups": ["banks", "insurance_diversified_fin"],
            "cyclical_groups": ["transportation_logistics", "energy_oil_gas", "materials_chemicals_metals"],
            "industry_labels": {},
            "industries": {},
        }

    if name:
        data["name"] = name
    if benchmark:
        data["benchmark"] = benchmark
    if currency:
        data["currency"] = currency
    if round_trip_cost is not None:
        data["round_trip_cost"] = float(round_trip_cost)

    industries = data.setdefault("industries", {})
    existing_all = {t for mems in industries.values() for t in mems}
    target_list = industries.setdefault(industry.strip(), [])

    added = 0
    for raw in tickers:
        sym = str(raw).strip()
        if sym and sym not in existing_all:
            target_list.append(sym)
            existing_all.add(sym)
            added += 1

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reload_universes()
    return json_path, added, len(existing_all)


# Initialize built-in + custom universes on import
reload_universes()
