"""全市场策略回测：把七盏灯规则在全部股票上按历史逐日跑一遍，出统计成绩单

和实盘信号同一套规则（与 indicators.py 对齐），但用向量化计算，比逐根 evaluate 快 100 倍。

用法:
  python backtest_market.py --top 200                    # 先小样本验证
  python backtest_market.py --top 5000 --shard 0/4 --out out/bt-0.json
  python backtest_market.py --merge "out/bt-*.json" --out out/backtest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_source import load_daily
from filters import FilterParams, filter_universe
from indicators import Params

OUT_DIR = Path(__file__).parent / "out"



def _py(obj):
    """numpy 标量 -> Python 原生，保证 JSON 可序列化"""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# ---------------------------------------------------------------- 指标（向量化）


def weekly_ma_map(df: pd.DataFrame, n: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """返回每个日线 bar 对应的 (已完成周的 MA_n, 前一周的 MA_n)。
    只用「本周之前」已收完的周，避免用到未完成周（会和实盘口径不一致）。"""
    s = df.set_index("date")["close"]
    wk = s.resample("W-FRI").last().dropna()
    if len(wk) < n + 2:
        return np.full(len(df), np.nan), np.full(len(df), np.nan)
    ma = wk.rolling(n).mean()
    week_end = wk.index.values  # 每周的周五日期

    d = df["date"].values
    # 每个 bar 所属周的周五
    wd = df["date"].dt.day_of_week.values  # 0=周一
    fri = d + (pd.to_timedelta(4 - wd, unit="D"))
    # 已完成周 = 周五 < 本周周五
    pos = np.searchsorted(week_end, fri, side="left")  # 满足 week_end < fri 的个数
    now = np.array([ma.iloc[p - 1] if p >= 1 else np.nan for p in pos], dtype=float)
    prev = np.array([ma.iloc[p - 2] if p >= 2 else np.nan for p in pos], dtype=float)
    return now, prev


def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    vol = df["volume"]

    ema_f = close.ewm(span=p.macd_fast, adjust=False).mean()
    ema_s = close.ewm(span=p.macd_slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=p.macd_signal, adjust=False).mean()

    ma5 = close.rolling(5).mean()
    ma21 = close.rolling(p.ma_stop).mean()
    hi20 = high.rolling(p.pullback_lookback).max()

    # 量比：前 5 日均量（剔除 0 量），不是含当天的
    base = vol.shift(1).rolling(5)
    base_mean = base.apply(lambda x: x[x > 0].mean() if (x > 0).sum() >= 3 else np.nan, raw=False)
    vol_ratio = vol / base_mean

    w_now, w_prev = weekly_ma_map(df, p.ma_week)

    ok1 = (w_now > w_prev) & (close.values > w_now)
    pull = (close / hi20 - 1) * 100
    ok2 = (pull <= -p.pullback_min_pct) & (close > ma21)
    dif_pct = dif / close * 100
    ok3 = dif_pct.abs() <= p.zero_band_pct

    # 金叉：当前 DIF>DEA，且 3 日内曾上穿
    above = dif > dea
    cross = above & (dif.shift(1) <= dea.shift(1))
    ok4 = above & (cross.rolling(p.golden_cross_window, min_periods=1).max().astype(bool))

    ok5 = (close > df["open"]) & (close > ma5)
    ok6 = vol_ratio >= p.volume_ratio_min
    ok7 = close >= ma21

    entry_green = (ok1.astype(int) + ok2 + ok3 + ok4 + ok5 + ok6).astype(int)
    core = ok1 & ok2 & ok4 & ok6
    entry = core & ok7 & (entry_green >= 6)

    out = pd.DataFrame({
        "date": df["date"].values,
        "close": close.values,
        "ma21": ma21.values,
        "entry": entry.values,
        "entry_alt": (core & ok7).values,   # 四条硬条件版（对照用）
        "entry_green": entry_green.values,
    })
    return out


# ---------------------------------------------------------------- 单只回测


def backtest_one(code: str, bars: int, p: Params) -> dict | None:
    df = load_daily(code, days=bars)
    if df is None or len(df) < p.min_bars + 30:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    sig = compute_signals(df, p)

    trades = []
    entry_i = None
    entry_px = peak = 0.0
    equity = [1.0]
    cash = 1.0

    for i in range(p.min_bars, len(sig)):
        row = sig.iloc[i]
        px, ma21 = float(row["close"]), float(row["ma21"])
        if not math.isfinite(ma21):
            equity.append(cash)
            continue

        if entry_i is None:
            if bool(row["entry"]):
                entry_i, entry_px, peak = i, px, px
        else:
            peak = max(peak, px)
            peak_pnl = (peak / entry_px - 1) * 100
            dd = (px / peak - 1) * 100
            reason = None
            if px < ma21:
                reason = "跌破MA21"
            elif peak_pnl >= p.take_profit_pct and dd <= -p.trailing_drawdown_pct:
                reason = "峰值回撤止盈"
            elif i == len(sig) - 1:
                reason = "回测结束"
            if reason and i > entry_i:
                cash *= px / entry_px
                trades.append({
                    "entry_date": str(sig.iloc[entry_i]["date"])[:10],
                    "exit_date": str(row["date"])[:10],
                    "pnl_pct": (px / entry_px - 1) * 100,
                    "hold_days": int(i - entry_i),
                    "reason": reason,
                })
                entry_i = None
            if entry_i is not None:
                equity.append(cash * (px / entry_px))
                continue
        equity.append(cash)

    if entry_i is not None:
        px = float(sig.iloc[-1]["close"])
        cash *= px / entry_px
        trades.append({
            "entry_date": str(sig.iloc[entry_i]["date"])[:10],
            "exit_date": str(sig.iloc[-1]["date"])[:10],
            "pnl_pct": (px / entry_px - 1) * 100,
            "hold_days": int(len(sig) - 1 - entry_i),
            "reason": "回测结束",
        })

    # 净值最大回撤
    peak_eq, mdd = 1.0, 0.0
    for e in equity:
        peak_eq = max(peak_eq, e)
        mdd = min(mdd, (e / peak_eq - 1) * 100)

    days = (sig["date"].iloc[-1] - sig["date"].iloc[0]).days
    years = max(days / 365.0, 0.01)
    total = (cash - 1) * 100
    annual = ((cash ** (1 / years)) - 1) * 100 if cash > 0 else -100

    return {
        "code": code,
        "from": str(sig["date"].iloc[0])[:10],
        "to": str(sig["date"].iloc[-1])[:10],
        "bars": int(len(sig)),
        "trades": trades,
        "total_pct": total,
        "annual_pct": annual,
        "mdd_pct": mdd,
        "buy_hold_pct": (float(sig["close"].iloc[-1]) / float(sig["close"].iloc[0]) - 1) * 100,
    }


# ---------------------------------------------------------------- 汇总


def summarize(results: list[dict]) -> dict:
    all_trades = [t for r in results for t in r["trades"]]
    n = len(all_trades)
    wins = [t for t in all_trades if t["pnl_pct"] > 0]
    losses = [t for t in all_trades if t["pnl_pct"] <= 0]
    gross_w = sum(t["pnl_pct"] for t in wins)
    gross_l = -sum(t["pnl_pct"] for t in losses)

    def pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else None

    ann = [r["annual_pct"] for r in results if r["trades"]] or [0]
    mdd = [r["mdd_pct"] for r in results]

    return {
        "stocks": len(results),
        "total_trades": n,
        "stocks_with_trades": sum(1 for r in results if r["trades"]),
        "win_rate_pct": (100 * len(wins) / n) if n else 0,
        "avg_pnl_pct": (sum(t["pnl_pct"] for t in all_trades) / n) if n else 0,
        "avg_win_pct": (gross_w / len(wins)) if wins else 0,
        "avg_loss_pct": (-gross_l / len(losses)) if losses else 0,
        "profit_factor": (gross_w / gross_l) if gross_l > 0 else None,
        "avg_hold_days": (sum(t["hold_days"] for t in all_trades) / n) if n else 0,
        "median_annual_pct": pct(ann, 50),
        "p25_annual_pct": pct(ann, 25),
        "p75_annual_pct": pct(ann, 75),
        "avg_mdd_pct": (sum(mdd) / len(mdd)) if mdd else 0,
        "worst_mdd_pct": min(mdd) if mdd else 0,
        "beat_buy_hold_ratio": (
            100 * sum(1 for r in results if r["total_pct"] > r["buy_hold_pct"]) / len(results)
        ) if results else 0,
        "exit_reasons": dict(pd.Series([t["reason"] for t in all_trades]).value_counts()) if n else {},
        "best": sorted(results, key=lambda r: -r["total_pct"])[:5],
        "worst": sorted(results, key=lambda r: r["total_pct"])[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--bars", type=int, default=700, help="每只取多少根日线")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="")
    ap.add_argument("--deadline-min", type=int, default=0)
    ap.add_argument("--merge", nargs="*", default=None)
    a = ap.parse_args()

    if a.merge:
        files = []
        for pat in a.merge:
            files.extend(sorted(glob.glob(pat)))
        results = []
        for f in files:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            results.extend(d.get("results") or [])
            print("合并", f, len(d.get("results") or []), "只")
        out = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
               "summary": summarize(results),
               "results": results}
        target = Path(a.out or (OUT_DIR / "backtest.json"))
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(_py(out), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                          encoding="utf-8")
        print("回测汇总 →", target)
        print(json.dumps(_py(out["summary"]), ensure_ascii=False, indent=2)[:2000])
        return

    p = Params()
    fp = FilterParams()

    # 股票池：用当日全市场精简版，没有再退回快照
    codes = []
    lite = OUT_DIR / "all_signals.json"
    if lite.exists():
        d = json.loads(lite.read_text(encoding="utf-8"))
        codes = [(x["code"], x.get("name", ""), x.get("board")) for x in d.get("items") or []]
        print(f"股票池来自 all_signals.json：{len(codes)} 只")
    if not codes:
        import akshare as ak
        df = ak.stock_zh_a_spot()
        df = df.rename(columns={"代码": "code", "名称": "name"})
        df["code"] = df["code"].astype(str).str[-6:]
        codes = list(df[["code", "name"]].itertuples(index=False, name=None))
        print(f"股票池来自实时快照：{len(codes)} 只")

    rows = [{"code": c, "name": n, "board": b or "主板"} for c, n, b in codes][: a.top]
    # 回测不需要实时价格/成交额（历史数据自带），只排掉 ST/退市/次新
    before = len(rows)
    rows = [r for r in rows
            if not any(t in str(r.get("name", "")).upper() for t in ("ST", "退"))
            and not str(r.get("name", "")).startswith(("N", "C"))]
    print(f"排除 ST/退市/次新 {before - len(rows)} 只")
    idx, total = (int(x) for x in a.shard.split("/"))
    if total > 1:
        rows = [r for i, r in enumerate(rows) if i % total == idx]
    print(f"回测 {len(rows)} 只（每只 {a.bars} 根日线）")

    deadline = time.time() + a.deadline_min * 60 if a.deadline_min > 0 else 0
    results, failed = [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        if deadline and time.time() > deadline:
            print(f"⏰ 到时限，已跑 {len(results)} 只")
            break
        try:
            res = backtest_one(r["code"], a.bars, p)
            if res:
                res["name"] = r.get("name", "")
                results.append(res)
        except Exception as exc:  # noqa: BLE001
            failed.append({"code": r["code"], "error": f"{type(exc).__name__}: {str(exc)[:60]}"})
        if i % 50 == 0:
            print(f"  ...{i}/{len(rows)} 用时 {(time.time()-t0)/60:.1f} 分钟")

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "params": {"volume_ratio_min": p.volume_ratio_min, "pullback_min_pct": p.pullback_min_pct,
                   "zero_band_pct": p.zero_band_pct, "take_profit_pct": p.take_profit_pct,
                   "trailing_drawdown_pct": p.trailing_drawdown_pct},
        "universe": len(rows), "done": len(results), "failed": len(failed),
        "summary": summarize(results),
        "results": results,
    }
    target = Path(a.out) if a.out else OUT_DIR / "backtest.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(_py(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                      encoding="utf-8")
    print(f"\n用时 {(time.time()-t0)/60:.1f} 分钟 → {target}")
    print(json.dumps(_py(payload["summary"]), ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
