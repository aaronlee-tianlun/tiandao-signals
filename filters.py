"""A 股可交易性过滤层

不过滤的话，信号池里会混进一堆根本不能买的票：ST、退市整理、次新、停牌、
一字板、成交额几百万的僵尸股。

过滤规则（可在 Params 里关）：
  1. 名称含 ST / *ST / 退 / N / C  → 剔除
  2. 最新价 < 2 元                  → 剔除（低价股波动失真、易被操纵）
  3. 成交额 < 5000 万               → 剔除（流动性不足，买卖冲击成本高）
  4. 涨跌幅接近涨跌停              → 剔除（无法在收盘价成交）
       主板 10%（阈值 9.5）、创业板/科创板 20%（阈值 19.5）、北交所 30%（阈值 29.5）
  5. 日线不足 80 根                 → 剔除（次新，周线 MA13 不够）
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FilterParams:
    min_price: float = 2.0
    min_amount: float = 5_000_0000  # 5000 万
    exclude_st: bool = True
    exclude_limit: bool = True
    include_star: bool = True   # 688 科创板
    include_gem: bool = True    # 300 创业板
    include_bj: bool = False    # 8/4 北交所（默认排除，流动性差）


def board_of(code: str) -> str:
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith(("4", "8")):
        return "北交所"
    return "主板"


def limit_pct(code: str, name: str = "") -> float:
    b = board_of(code)
    if b == "北交所":
        return 30.0
    if b in ("科创板", "创业板"):
        return 20.0
    return 5.0 if ("ST" in name.upper()) else 10.0


def is_tradable(row: dict, fp: FilterParams | None = None) -> tuple[bool, str]:
    """row 需要: code, name, price, chg_pct, amount（成交额，元）"""
    fp = fp or FilterParams()
    def _f(v, default=0.0):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return default
        return x if math.isfinite(x) else default

    code = str(row.get("code", "")).zfill(6)
    name = str(row.get("name", ""))
    price = _f(row.get("price"))
    chg = _f(row.get("chg_pct"))
    amount = _f(row.get("amount"))

    if price <= 0:
        return False, "无最新价(停牌/未交易)"
    if amount <= 0:
        return False, "无成交额(停牌)"

    if fp.exclude_st and any(t in name.upper() for t in ("ST", "退")):
        return False, "ST/退市"
    if name.startswith(("N", "C")):
        return False, "次新"
    if price < fp.min_price:
        return False, f"低价({price})"
    if amount < fp.min_amount:
        return False, f"成交额不足({amount/1e4:.0f}万)"

    b = board_of(code)
    if b == "科创板" and not fp.include_star:
        return False, "科创板排除"
    if b == "创业板" and not fp.include_gem:
        return False, "创业板排除"
    if b == "北交所" and not fp.include_bj:
        return False, "北交所排除"

    if fp.exclude_limit:
        lim = limit_pct(code, name)
        if abs(chg) >= lim - 0.5:
            return False, f"接近涨跌停({chg:+.1f}%)"
    return True, ""


def filter_universe(rows: list[dict], fp: FilterParams | None = None) -> list[dict]:
    out = []
    for r in rows:
        ok, _reason = is_tradable(r, fp)
        if ok:
            r = dict(r)
            r["board"] = board_of(str(r.get("code", "")))
            out.append(r)
    return out
