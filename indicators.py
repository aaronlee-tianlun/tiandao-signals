"""A 股趋势跟踪策略 —— 红绿灯指标引擎（审查修复版 v2）

修复清单（对应团队审查 P0/P1）：
  P0-3 入场判定统一到「四条硬条件」：周线向上 + 日线回踩 + 金叉 + 明显放量，
       不再用 5/6 计数（实测 5/6 的 24 例全部缺「回踩」= 追高信号）
  P0-4 出口统一做有限性守卫，NaN/inf 一律变 None；写盘端 allow_nan=False
  P0-5 金叉窗口必须当前仍 DIF>DEA，否则不算（原实现 6.1% 误报）
  P1-1 周线只用「已完成周」判斜率，未完成周单独输出
  P1-2 量比基准剔除 0 量日（一字板/停牌）
  P1-7 增加数据新鲜度校验（滞后 >10 天判停牌/未更新）
  P1-8 数据不足阈值 80 → 120 根（MACD 需要预热）
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------- 参数

@dataclass
class Params:
    ma_week: int = 13
    ma_stop: int = 21
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    zero_band_pct: float = 1.5
    pullback_min_pct: float = 3.0
    pullback_lookback: int = 20
    volume_ratio_min: float = 2.5     # 明显放量（通达信口径）
    volume_normal_max: float = 1.5
    volume_heavy: float = 5.0
    volume_extreme: float = 10.0
    golden_cross_window: int = 3
    take_profit_pct: float = 15.0
    trailing_drawdown_pct: float = 10.0
    min_bars: int = 120
    max_stale_days: int = 10

    # 入场硬条件（策略 1+2+3 的核心，缺一不可）
    core_keys: tuple[str, ...] = ("weekly_trend", "pullback", "golden_cross", "volume")


def _num(x: Any, nd: int = 4) -> float | None:
    """统一的数值出口：非有限值一律 None，避免 NaN 进 JSON"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def volume_tier(r: float | None, p: Params) -> str:
    if r is None or not math.isfinite(r):
        return "数据缺失"
    if r < 0.8:
        return "缩量"
    if r < p.volume_normal_max:
        return "正常"
    if r < p.volume_ratio_min:
        return "温和放量"
    if r < p.volume_heavy:
        return "明显放量"
    if r < p.volume_extreme:
        return "剧烈放量"
    return "巨量"


# ---------------------------------------------------------------- 指标

def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": (dif - dea) * 2})


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.set_index("date").resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    w["bars"] = df.set_index("date")["close"].resample("W-FRI").count().reindex(w.index)
    w["ma"] = w["close"].rolling(13).mean()
    return w


@dataclass
class Light:
    key: str
    name: str
    passed: bool
    value: dict[str, Any] = field(default_factory=dict)
    rule: str = ""
    advice: str = ""

    @property
    def color(self) -> str:
        return "green" if self.passed else "red"


def evaluate(df: pd.DataFrame, p: Params | None = None) -> dict[str, Any]:
    p = p or Params()
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < p.min_bars:
        raise ValueError(f"数据不足（{len(df)} 根，至少 {p.min_bars} 根）")

    # 数据新鲜度：停牌/数据源未更新时不能当作今天的信号
    last_date = pd.Timestamp(df["date"].iloc[-1]).date()
    lag = (dt.date.today() - last_date).days
    if lag > p.max_stale_days:
        raise ValueError(f"最新K线为 {last_date}，已滞后 {lag} 天（疑似停牌/数据源未更新）")

    close = df["close"]
    last, prev = df.iloc[-1], df.iloc[-2]

    m = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    dif, dea, hist = float(m["dif"].iloc[-1]), float(m["dea"].iloc[-1]), float(m["hist"].iloc[-1])
    dif_pct = dif / float(last["close"]) * 100 if last["close"] else float("nan")

    ma21 = float(close.rolling(p.ma_stop).mean().iloc[-1])
    ma5 = float(close.rolling(5).mean().iloc[-1])

    # ---- 周线：只用已完成周判方向
    w = to_weekly(df)
    last_bar_date = pd.Timestamp(df["date"].iloc[-1]).date()
    incomplete = w.index[-1].date() > last_bar_date   # 本周尚未走完
    base = w.iloc[:-1] if incomplete and len(w) > 2 else w
    w_ma = base["ma"].dropna()
    w_ma_now, w_ma_prev = float(w_ma.iloc[-1]), float(w_ma.iloc[-2])
    w_slope = (w_ma_now / w_ma_prev - 1) * 100

    # ---- 回踩
    hi20 = float(df["high"].iloc[-p.pullback_lookback:].max())
    pullback_pct = (float(last["close"]) / hi20 - 1) * 100

    # ---- 量能：基准剔除 0 量日
    base_vol = df["volume"].iloc[-6:-1]
    base_vol = base_vol[base_vol > 0]
    if len(base_vol) < 3:
        vol_ratio, tier = None, "样本不足"
    else:
        vol_ratio = float(last["volume"] / base_vol.mean())
        tier = volume_tier(vol_ratio, p)

    chg_pct = float(last["close"] / prev["close"] - 1) * 100
    dist_ma21 = (float(last["close"]) / ma21 - 1) * 100

    # ---- 金叉：当前必须仍是多头排列
    days_since_cross = None
    if dif > dea:
        for back in range(p.golden_cross_window):
            i = len(m) - 1 - back
            if m["dif"].iloc[i] > m["dea"].iloc[i] and m["dif"].iloc[i - 1] <= m["dea"].iloc[i - 1]:
                days_since_cross = back
                break

    lights: list[Light] = []

    # 1 周线趋势
    ok1 = bool(w_ma_now > w_ma_prev and float(last["close"]) > w_ma_now)
    lights.append(Light(
        "weekly_trend", "周线趋势(MA13周)", ok1,
        {"MA13周": _num(w_ma_now, 2), "周变化": f"{w_slope:+.2f}%",
         "周收盘": _num(base['close'].iloc[-1], 2)},
        "MA13周 本周 > 上周，且收盘站上 MA13周（只用已完成周）",
        "中期趋势向上，纳入观察池" if ok1 else f"MA13周 周变化 {w_slope:+.2f}%，中期趋势未确认，先不碰",
    ))

    # 2 日线回踩
    ok2 = bool(pullback_pct <= -p.pullback_min_pct and float(last["close"]) > ma21)
    if ok2:
        adv2 = f"已从高点回撤 {abs(pullback_pct):.1f}% 且站在 MA21 上方，等金叉确认"
    elif float(last["close"]) <= ma21:
        adv2 = f"已跌破 MA21（{dist_ma21:+.1f}%），趋势走坏，不接"
    else:
        adv2 = f"距高点仅 {pullback_pct:+.1f}%，没回踩到位，现价追高"
    lights.append(Light(
        "pullback", "日线回踩", ok2,
        {"距20日高点": f"{pullback_pct:+.2f}%", "MA21": _num(ma21, 2), "距MA21": f"{dist_ma21:+.2f}%"},
        f"从 {p.pullback_lookback} 日高点回撤 ≥ {p.pullback_min_pct}%，且未跌破 MA21",
        adv2,
    ))

    # 3 0 轴附近
    ok3 = bool(abs(dif_pct) <= p.zero_band_pct)
    if ok3:
        adv3 = f"DIF {dif_pct:+.2f}%，贴近零轴，处在变盘窗口"
    elif dif_pct > 0:
        adv3 = f"DIF 高于零轴 {dif_pct:.1f}%，已走出一段，不是起涨点"
    else:
        adv3 = f"DIF 低于零轴 {dif_pct:.1f}%，仍在空头区间"
    lights.append(Light(
        "zero_axis", "MACD 贴近 0 轴", ok3,
        {"DIF": _num(dif, 4), "DIF/价": f"{dif_pct:+.2f}%", "DEA": _num(dea, 4)},
        f"|DIF / 收盘价| ≤ {p.zero_band_pct}%",
        adv3,
    ))

    # 4 金叉
    ok4 = days_since_cross is not None
    state = ("今日金叉" if days_since_cross == 0 else f"{days_since_cross} 日前金叉") if ok4 \
        else ("多头排列" if dif > dea else "空头排列")
    if ok4:
        adv4 = f"金叉在 {days_since_cross} 个交易日内且仍多头排列，入场窗口打开"
    elif dif > dea:
        adv4 = "已过金叉窗口，属追高区，等下一次回踩金叉"
    else:
        adv4 = "MACD 空头排列，等 DIF 上穿 DEA"
    lights.append(Light(
        "golden_cross", "MACD 金叉", ok4,
        {"DIF": _num(dif, 4), "DEA": _num(dea, 4), "柱": _num(hist, 4), "状态": state},
        f"DIF 上穿 DEA，发生在 {p.golden_cross_window} 个交易日内，且当前仍 DIF>DEA",
        adv4,
    ))

    # 5 企稳拐头
    ok5 = bool(float(last["close"]) > float(prev["close"]) and float(last["close"]) > ma5)
    adv5 = (f"收阳 {chg_pct:+.2f}% 并站上 MA5，短线企稳"
            if ok5 else f"未收阳或未站上 MA5（MA5={ma5:.2f}），还没企稳")
    lights.append(Light(
        "steady", "企稳拐头", ok5,
        {"收盘": _num(last["close"], 2), "MA5": _num(ma5, 2), "涨跌幅": f"{chg_pct:+.2f}%"},
        "收阳，且收盘站上 MA5",
        adv5,
    ))

    # 6 量能
    ok6 = bool(vol_ratio is not None and vol_ratio >= p.volume_ratio_min)
    warn = "（注意：今日下跌，放量可能是出货）" if chg_pct < 0 else ""
    if vol_ratio is None:
        adv6 = "量能样本不足（近期多日 0 成交），无法判定"
    elif ok6:
        adv6 = f"量比 {vol_ratio:.2f}（{tier}），资金进场{warn}"
    else:
        adv6 = f"量比 {vol_ratio:.2f}（{tier}），量能不够，发令枪还没响{warn}"
    lights.append(Light(
        "volume", "成交量放大", ok6,
        {"量比": _num(vol_ratio, 2), "档位": tier, "今日量(万股)": _num(float(last["volume"]) / 1e4, 1)},
        f"今日量 / 前 5 日均量（剔除 0 量日）≥ {p.volume_ratio_min}",
        adv6,
    ))

    # 7 风控
    ok7 = bool(float(last["close"]) >= ma21)
    adv7 = (f"站在 MA21 上方 {dist_ma21:+.1f}%，可继续持有；浮盈>{p.take_profit_pct:.0f}% 且高点回撤>{p.trailing_drawdown_pct:.0f}% 止盈"
            if ok7 else f"跌破 MA21 {dist_ma21:.1f}%，按纪律离场")
    lights.append(Light(
        "risk", "风控(MA21)", ok7,
        {"距MA21": f"{dist_ma21:+.2f}%", "MA21": _num(ma21, 2), "距20日高点": f"{pullback_pct:+.2f}%"},
        f"收盘 ≥ MA{p.ma_stop} 可持有；跌破离场；浮盈>{p.take_profit_pct:.0f}% 且回撤>{p.trailing_drawdown_pct:.0f}% 止盈",
        adv7,
    ))

    by_key = {x.key: x for x in lights}
    core_missing = [by_key[k].name for k in p.core_keys if not by_key[k].passed]
    core_ok = not core_missing

    entry_keys = list(p.core_keys) + ["steady", "zero_axis"]
    entry_green = sum(1 for x in lights if x.key in entry_keys and x.passed)
    green = sum(1 for x in lights if x.passed)

    # 入场信号 = 四条硬条件 + 站在 MA21 上
    entry_signal = bool(core_ok and ok7)
    if entry_signal:
        summary = "★ 发令枪：周线向上 + 日线回踩 + 金叉 + 明显放量 全部成立，止损放 MA21"
    elif core_ok and not ok7:
        summary = "四条硬条件齐备，但已跌破 MA21，等重新站上再动手"
    elif entry_green >= 3:
        summary = f"观察：缺 {('、'.join(core_missing))}，等信号齐"
    else:
        summary = f"暂不关注：入场仅 {entry_green}/{len(entry_keys)}"

    return {
        "date": str(df["date"].iloc[-1])[:10],
        "close": _num(last["close"], 2),
        "lights": [{"key": x.key, "name": x.name, "color": x.color, "passed": x.passed,
                    "value": x.value, "rule": x.rule, "advice": x.advice} for x in lights],
        "green_count": green,
        "total": len(lights),
        "entry_green": entry_green,
        "entry_total": len(entry_keys),
        "entry_signal": entry_signal,
        "core_missing": core_missing,
        # 给 App 的明确结论：买不买，一句话
        "verdict": "买入" if entry_signal else ("等待" if entry_green >= 3 else "回避"),
        "verdict_reason": summary,
        "summary": summary,
        "provider": df.attrs.get("provider", ""),
        "raw": {
            "close": _num(last["close"]),
            "chg_pct": _num(chg_pct),
            "ma5": _num(ma5),
            "ma21": _num(ma21),
            "ma13w": _num(w_ma_now),
            "dif": _num(dif, 6),
            "dea": _num(dea, 6),
            "vol_ratio": _num(vol_ratio),
            "hi20": _num(hi20),
            "drawdown_from_high": _num(pullback_pct),
            "week_incomplete": bool(incomplete),
        },
    }
