"""Global multi-market equity universes, peer groups, benchmarks, and transaction cost presets.

Supports:
- US     : 40 liquid US blue chips across 10 sectors (Benchmark: SPY)
- EU     : 18 European leaders across 6 sectors (Benchmark: ^STOXX50E)
- JP     : 16 Japan leaders across 6 sectors (Benchmark: ^N225)
- HK     : 12 Hong Kong leaders across 4 sectors (Benchmark: ^HSI)
- TW     : 84 Taiwan stocks across 20 peer groups (Benchmark: ^TWII, imported from universe.py)
- GLOBAL : Cross-region flagship universe combining liquid leaders across US, EU, JP, HK, TW (Benchmark: ACWI)
"""

from types import MappingProxyType

from universe import PB_GROUPS as TW_PB_GROUPS, PEER_GROUPS as TW_RAW_GROUPS

MARKET_CONFIG = MappingProxyType({
    "US": {
        "name": "United States (NYSE / NASDAQ)",
        "benchmark": "SPY",
        "currency": "USD",
        "round_trip_cost": 0.0010,  # ~10 bps spread/slippage (zero-commission US brokers)
        "odd_lot_slippage": 0.0010,
    },
    "EU": {
        "name": "Europe (Euronext / XETRA / LSE / SIX)",
        "benchmark": "^STOXX50E",
        "currency": "EUR",
        "round_trip_cost": 0.0030,
        "odd_lot_slippage": 0.0020,
    },
    "JP": {
        "name": "Japan (Tokyo Stock Exchange)",
        "benchmark": "^N225",
        "currency": "JPY",
        "round_trip_cost": 0.0020,
        "odd_lot_slippage": 0.0020,
    },
    "HK": {
        "name": "Hong Kong (HKEX)",
        "benchmark": "^HSI",
        "currency": "HKD",
        "round_trip_cost": 0.0045,  # includes HK stamp duty
        "odd_lot_slippage": 0.0030,
    },
    "TW": {
        "name": "Taiwan (TWSE / TPEx)",
        "benchmark": "^TWII",
        "currency": "TWD",
        "round_trip_cost": 0.00585,  # 0.1425%*2 + 0.3% tax
        "odd_lot_slippage": 0.0030,
    },
    "GLOBAL": {
        "name": "Global Multi-Region Flagship",
        "benchmark": "ACWI",
        "currency": "USD",
        "round_trip_cost": 0.0030,
        "odd_lot_slippage": 0.0020,
    },
})

US_GROUPS = MappingProxyType({
    "us_mega_tech": ("AAPL", "MSFT", "GOOGL", "AMZN", "META"),
    "us_semis": ("NVDA", "AVGO", "AMD", "TXN", "QCOM"),
    "us_financials": ("JPM", "BAC", "GS", "BRK-B", "CB"),
    "us_healthcare": ("LLY", "UNH", "JNJ", "ABBV", "MRK"),
    "us_consumer_staples": ("PG", "KO", "PEP", "COST", "WMT"),
    "us_consumer_disc": ("HD", "MCD", "NKE", "LOW"),
    "us_industrials": ("CAT", "GE", "RTX", "HON"),
    "us_energy": ("XOM", "CVX", "COP", "SLB"),
    "us_payments_software": ("V", "MA", "CRM", "ORCL"),
    "us_utilities_telecom": ("NEE", "DUK", "VZ", "T"),
})

EU_GROUPS = MappingProxyType({
    "eu_tech": ("ASML.AS", "SAP.DE", "IFX.DE"),
    "eu_luxury_consumer": ("MC.PA", "OR.PA", "NESN.SW"),
    "eu_healthcare": ("NOVO-B.CO", "AZN.L", "ROG.SW"),
    "eu_industrials": ("SIE.DE", "AIR.PA", "SU.PA"),
    "eu_energy": ("SHEL.L", "TTE.PA", "BP.L"),
    "eu_financials": ("HSBA.L", "ALV.DE", "BNP.PA"),
})

JP_GROUPS = MappingProxyType({
    "jp_tech_auto": ("7203.T", "6758.T", "6861.T", "8035.T"),
    "jp_financials": ("8306.T", "8316.T", "8411.T"),
    "jp_trading_houses": ("8058.T", "8031.T", "8001.T"),
    "jp_telecom_consumer": ("9432.T", "9984.T", "7974.T"),
    "jp_healthcare_ind": ("4502.T", "6501.T", "6367.T"),
})

HK_GROUPS = MappingProxyType({
    "hk_internet": ("0700.HK", "9988.HK", "3690.HK", "9618.HK"),
    "hk_financials": ("0005.HK", "1299.HK", "0939.HK", "3988.HK"),
    "hk_telecom_energy": ("0941.HK", "0883.HK", "0857.HK", "0386.HK"),
})

TW_YAHOO_GROUPS = MappingProxyType({
    g: tuple(f"{sid}.TW" for sid in members) for g, members in TW_RAW_GROUPS.items()
})

GLOBAL_GROUPS = MappingProxyType({
    "global_semis_ai": ("NVDA", "AVGO", "ASML.AS", "2330.TW", "8035.T"),
    "global_mega_platforms": ("MSFT", "GOOGL", "AMZN", "META", "0700.HK"),
    "global_financials": ("JPM", "HSBA.L", "8306.T", "2886.TW", "1299.HK"),
    "global_consumer_defensive": ("PG", "KO", "COST", "NESN.SW", "1216.TW"),
    "global_healthcare": ("LLY", "JNJ", "NOVO-B.CO", "AZN.L", "4502.T"),
    "global_industrials_auto": ("CAT", "SIE.DE", "7203.T", "8058.T", "2308.TW"),
    "global_energy": ("XOM", "CVX", "SHEL.L", "TTE.PA", "0883.HK"),
})

GLOBAL_PB_GROUPS = frozenset({
    "us_financials", "eu_financials", "jp_financials", "hk_financials", "global_financials",
    *TW_PB_GROUPS,
})

GLOBAL_CYCLICAL_GROUPS = frozenset({
    "us_energy", "eu_energy", "hk_telecom_energy", "global_energy",
    "container_shipping", "memory", "steel", "petrochemical", "airline",
})

MARKET_GROUPS = MappingProxyType({
    "US": US_GROUPS,
    "EU": EU_GROUPS,
    "JP": JP_GROUPS,
    "HK": HK_GROUPS,
    "TW": TW_YAHOO_GROUPS,
    "GLOBAL": GLOBAL_GROUPS,
})


def get_market_groups(market: str) -> dict:
    key = market.upper()
    if key not in MARKET_GROUPS:
        raise ValueError(f"Unsupported market '{market}'. Choose from: {', '.join(MARKET_GROUPS)}")
    return dict(MARKET_GROUPS[key])


def get_market_tickers(market: str) -> tuple:
    seen = []
    for members in get_market_groups(market).values():
        for t in members:
            if t not in seen:
                seen.append(t)
    return tuple(seen)


def group_map_for(groups: dict) -> dict:
    return {t: g for g, members in groups.items() for t in members}
