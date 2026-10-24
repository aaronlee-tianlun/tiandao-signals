"""A 股数据层（多源容错：Sina → 腾讯 → 东财）

为什么三源：东财接口会突然限流（RemoteDisconnected），单源会让整个 App 挂掉。
Sina 最稳（全历史），腾讯快（按区间），东财兜底。
"""

from __future__ import annotations

import datetime as dt
import os
import time
from functools import lru_cache

# 本机有系统级代理（Clash 7897），经常是死的；requests 会从注册表读到它 → ProxyError。
# A 股数据源国内直连即可，这里强制绕过代理。
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
# Windows 上 requests 还会从注册表读系统代理（Clash 常见但端口已死），
# 只 pop 环境变量不够，这里把 getproxies 一起短路掉。
try:
    import requests.utils as _ru
    _ru.getproxies = lambda: {}
except Exception:  # noqa: BLE001
    pass

import akshare as ak
import pandas as pd

KEEP = ["date", "open", "high", "low", "close", "volume"]


def _prefix(symbol: str) -> str:
    s = symbol.strip()
    if s.startswith(("6", "9")):
        return "sh" + s
    if s.startswith(("0", "2", "3")):
        return "sz" + s
    if s.startswith(("4", "8")):
        return "bj" + s
    return s


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df[[c for c in KEEP if c in df.columns]]


def _from_sina(symbol: str, days: int) -> pd.DataFrame:
    df = ak.stock_zh_a_daily(symbol=_prefix(symbol), adjust="qfq")
    return _norm(df).tail(days)


def _from_tx(symbol: str, days: int) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(days * 1.7))
    df = ak.stock_zh_a_hist_tx(
        symbol=_prefix(symbol),
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    return _norm(df).tail(days).reset_index(drop=True)


def _from_em(symbol: str, days: int) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(days * 1.7))
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "成交量": "volume"})
    return _norm(df).tail(days).reset_index(drop=True)


PROVIDERS = (("sina", _from_sina), ("tencent", _from_tx), ("eastmoney", _from_em))


def load_daily(symbol: str, days: int = 700) -> pd.DataFrame:
    """按 sina → tencent → eastmoney 顺序尝试，返回最近 days 根日线。"""
    errors: list[str] = []
    for name, fn in PROVIDERS:
        for attempt in range(2):
            try:
                df = fn(symbol, days)
                if df is not None and len(df) >= 80:
                    df.attrs["provider"] = name
                    return df
                errors.append(f"{name}: 数据不足({0 if df is None else len(df)}行)")
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}#{attempt + 1}: {type(exc).__name__}")
                time.sleep(0.5)
    raise RuntimeError(f"{symbol} 所有数据源均失败 → " + "; ".join(errors))


@lru_cache(maxsize=1)
def code_names() -> dict[str, str]:
    """全市场代码 → 名称（取不到时返回空字典，不阻塞主流程）"""
    try:
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str).str.zfill(6), df["name"].astype(str)))
    except Exception:  # noqa: BLE001
        return {}


def spot_snapshot() -> pd.DataFrame:
    """实时快照（列表页用），失败返回空表。"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率"]]
        return df.rename(columns={"代码": "code", "名称": "name", "最新价": "price",
                                  "涨跌幅": "chg_pct", "成交额": "amount", "换手率": "turnover"})
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
