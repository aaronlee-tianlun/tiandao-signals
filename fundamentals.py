"""基本面快照（akshare 财务摘要）

用途：给「AI 研究委员会」的基本面分析师提供真实数据，而不是让它瞎编。
数据源：ak.stock_financial_abstract（80+ 指标 × 100+ 期，三大报表汇总）
估值：PE = 现价 / 最新每股收益，PB = 现价 / 每股净资产（现价由行情层传入）
"""

from __future__ import annotations

import os
from functools import lru_cache

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

import akshare as ak
import pandas as pd

# 我们关心的指标（对应 stock_financial_abstract 的「指标」列）
WANT = {
    "净资产收益率(ROE)": "roe",
    "毛利率": "gross_margin",
    "销售净利率": "net_margin",
    "资产负债率": "debt_ratio",
    "基本每股收益": "eps",
    "每股净资产": "bps",
    "经营现金流量净额": "ocf",
    "营业利润率": "op_margin",
}


def _to_num(v) -> float | None:
    try:
        f = float(str(v).replace(",", "").strip())
        return f
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=256)
def financial_snapshot(code: str) -> dict:
    """返回最新一期 + 同比变化。取不到就返回空 dict，不抛异常。"""
    try:
        df = ak.stock_financial_abstract(symbol=code)
    except Exception:  # noqa: BLE001
        return {}
    if df is None or df.empty or "指标" not in df.columns:
        return {}

    # 期数列：排除「选项」「指标」，按名称倒序（最新在前）
    period_cols = [c for c in df.columns if c not in ("选项", "指标")]
    if not period_cols:
        return {}
    period_cols = sorted(period_cols, reverse=True)
    latest = period_cols[0]
    # 同比：找去年同一季（如 20260630 → 20250630）
    yoy_col = None
    if len(latest) == 8 and latest.startswith("20"):
        cand = "20" + str(int(latest[:4]) - 1) + latest[4:]
        if cand in period_cols:
            yoy_col = cand

    out: dict = {"period": latest}
    for _, row in df.iterrows():
        name = str(row.get("指标", "")).strip()
        key = WANT.get(name)
        if not key:
            continue
        cur = _to_num(row.get(latest))
        if cur is None:
            continue
        out[key] = round(cur, 4)
        if yoy_col:
            prev = _to_num(row.get(yoy_col))
            if prev not in (None, 0):
                out[f"{key}_yoy"] = round((cur / prev - 1) * 100, 2)
    return out


def valuation(code: str, price: float, fin: dict | None = None) -> dict:
    """PE / PB（用最新一期 EPS、BPS 粗算，年化按季度数放大）"""
    fin = fin or financial_snapshot(code)
    if not fin or price <= 0:
        return {}
    out: dict = {}
    eps = fin.get("eps")
    bps = fin.get("bps")
    period = str(fin.get("period", ""))
    if eps:
        # 季报 EPS 年化：Q1×4、中报×2、三季报×4/3、年报×1
        mult = {"0331": 4, "0630": 2, "0930": 4 / 3, "1231": 1}.get(period[4:8], 1)
        eps_ttm = eps * mult
        if eps_ttm > 0:
            out["pe"] = round(price / eps_ttm, 2)
        out["eps_annualized"] = round(eps_ttm, 3)
    if bps and bps > 0:
        out["pb"] = round(price / bps, 2)
    return out


def describe(fin: dict, val: dict) -> str:
    """给模型/人看的一句话基本面摘要"""
    if not fin:
        return "基本面数据缺失"
    parts = []
    if "roe" in fin:
        parts.append(f"ROE {fin['roe']}%")
    if "gross_margin" in fin:
        parts.append(f"毛利率 {fin['gross_margin']}%")
    if "net_margin" in fin:
        parts.append(f"净利率 {fin['net_margin']}%")
    if "debt_ratio" in fin:
        parts.append(f"资产负债率 {fin['debt_ratio']}%")
    if "roe_yoy" in fin:
        parts.append(f"ROE同比 {fin['roe_yoy']:+.1f}%")
    if val.get("pe"):
        parts.append(f"PE {val['pe']}")
    if val.get("pb"):
        parts.append(f"PB {val['pb']}")
    return f"（{fin.get('period', '')}）" + "，".join(parts) if parts else "基本面数据缺失"
