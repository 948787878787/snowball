"""Peer-group universe for the v2 formula backtest (Taiwan listed stocks).

Groups are defined by revenue structure, not only the official industry code.
Leaders are NOT hard-coded: they are chosen point-in-time by trailing turnover
(see features.select_leaders) to avoid using today's knowledge of who is big.
"""

from types import MappingProxyType

PEER_GROUPS = MappingProxyType({
    "foundry": ("2330", "2303", "5347", "6770"),
    "ic_design": ("2454", "3034", "2379", "3035", "3661", "3443"),
    "ai_server_odm": ("2382", "3231", "2356", "6669", "2317", "4938"),
    "power_thermal": ("2308", "2301", "3017", "3324", "2421"),
    "substrate_pcb": ("3037", "8046", "3189", "2383", "2368"),
    "memory": ("2408", "2344", "2337", "3260", "8299"),
    "life_insurance_fhc": ("2881", "2882", "2883", "2887"),
    "bank_fhc": ("2886", "2891", "2892", "2880", "2884", "5880"),
    "electronics_distribution": ("2347", "2414", "3702", "3036", "3209"),
    "container_shipping": ("2603", "2609", "2615"),
    "airline": ("2618", "2610", "2646"),
    "petrochemical": ("1301", "1303", "1326", "6505"),
    "steel": ("2002", "2014", "2027", "2006"),
    "food_retail": ("1216", "2912", "1227", "1210"),
    "telecom": ("2412", "3045", "4904"),
    "cement": ("1101", "1102", "1104"),
    "auto": ("2204", "2201", "2207"),
    "passive_components": ("2327", "2492", "3026", "2375"),
    "networking": ("2345", "3596", "6285", "4906"),
    "optics": ("3008", "3406", "3019"),
})

# Valuation multiple per group (Bhojraj & Lee 2002): financials use P/B.
PB_GROUPS = frozenset({"life_insurance_fhc", "bank_fhc"})

MARKET_INDEX = "^TWII"
US_LEAD_TICKERS = ("^SOX", "SPY")
MACRO_TICKERS = ("^TNX", "DX-Y.NYB", "CL=F")


def all_stock_ids() -> tuple:
    seen = []
    for members in PEER_GROUPS.values():
        for sid in members:
            if sid not in seen:
                seen.append(sid)
    return tuple(seen)


def group_of() -> dict:
    return {sid: g for g, members in PEER_GROUPS.items() for sid in members}
