"""止损方式对比：同一批股票、同一批信号，只改离场规则，看结果差多少

三种离场：
  A 现在：收盘 < MA21 立即卖
  B 缓冲：收盘 < MA21 * (1-3%) 才卖
  C 确认：连续 2 日收盘 < MA21 才卖

用法:
  python compare_stops.py --top 400
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from backtest_market import _py, compute_signals
from data_source import load_daily
from indicators import Params

OUT_DIR = Path(__file__).parent / "out"


def simulate(df, sig, p: Params, mode: str, buffer_pct: float = 3.0) -> dict:
    """在已算好的信号上跑一遍持仓模拟，返回这笔策略的成绩"""
    trades = []
    entry_i = None
    entry_px = peak = 0.0
    cash = 1.0
    equity = [1.0]
    below_days = 0

    for i in range(p.min_bars, len(sig)):
        row = sig.iloc[i]
        px, ma21 = float(row["close"]), float(row["ma21"])
        if not math.isfinite(ma21):
            equity.append(cash)
            continue

        if entry_i is None:
            below_days = 0
            if bool(row["entry"]):
                entry_i, entry_px, peak = i, px, px
        else:
            peak = max(peak, px)
            peak_pnl = (peak / entry_px - 1) * 100
            dd = (px / peak - 1) * 100

            # ---- 三种离场判定
            if mode == "ma21":
                stop = px < ma21
            elif mode == "buffer":
                stop = px < ma21 * (1 - buffer_pct / 100)
            else:  # 2day
                if px < ma21:
                    below_days += 1
                else:
                    below_days = 0
                stop = below_days >= 2

            reason = None
            if stop:
                reason = "止损"
            elif peak_pnl >= p.take_profit_pct and dd <= -p.trailing_drawdown_pct:
                reason = "止盈"
            elif i == len(sig) - 1:
                reason = "结束"

            if reason and i > entry_i:
                cash *= px / entry_px
                trades.append({"pnl": (px / entry_px - 1) * 100, "hold": i - entry_i, "reason": reason})
                entry_i = None
            if entry_i is not None:
                equity.append(cash * (px / entry_px))
                continue
        equity.append(cash)

    if entry_i is not None:
        px = float(sig.iloc[-1]["close"])
        cash *= px / entry_px
        trades.append({"pnl": (px / entry_px - 1) * 100,
                       "hold": len(sig) - 1 - entry_i, "reason": "结束"})

    peak_eq, mdd = 1.0, 0.0
    for e in equity:
        peak_eq = max(peak_eq, e)
        mdd = min(mdd, (e / peak_eq - 1) * 100)

    days = (sig["date"].iloc[-1] - sig["date"].iloc[0]).days
    years = max(days / 365.0, 0.01)
    total = (cash - 1) * 100
    return {
        "total": total,
        "annual": ((cash ** (1 / years)) - 1) * 100 if cash > 0 else -100,
        "mdd": mdd,
        "trades": trades,
    }


def agg(results: list[dict]) -> dict:
    trades = [t for r in results for t in r["trades"]]
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    ann = [r["annual"] for r in results if r["trades"]] or [0]
    return {
        "stocks": len(results),
        "trades": n,
        "win_rate": round(100 * len(wins) / n, 1) if n else 0,
        "avg_pnl": round(sum(t["pnl"] for t in trades) / n, 2) if n else 0,
        "avg_win": round(gw / len(wins), 1) if wins else 0,
        "avg_loss": round(-gl / len(losses), 1) if losses else 0,
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "median_annual": round(float(np.median(ann)), 1),
        "avg_hold": round(sum(t["hold"] for t in trades) / n, 1) if n else 0,
        "avg_mdd": round(sum(r["mdd"] for r in results) / len(results), 1) if results else 0,
        "exit_stop": sum(1 for t in trades if t["reason"] == "止损"),
        "exit_tp": sum(1 for t in trades if t["reason"] == "止盈"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--bars", type=int, default=700)
    a = ap.parse_args()

    p = Params()
    lite = OUT_DIR / "all_signals.json"
    d = json.loads(lite.read_text(encoding="utf-8"))
    codes = [x["code"] for x in d["items"]][: a.top]
    print(f"对比 {len(codes)} 只，三种止损规则跑同一批信号")

    modes = ["ma21", "buffer", "2day"]
    buckets = {m: [] for m in modes}
    done = 0
    for code in codes:
        try:
            df = load_daily(code, days=a.bars)
            if df is None or len(df) < p.min_bars + 30:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            sig = compute_signals(df, p)
            for m in modes:
                buckets[m].append(simulate(df, sig, p, m))
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(codes)}")
        except Exception:  # noqa: BLE001
            continue

    print(f"\n完成 {done} 只\n")
    table = {}
    for m in modes:
        table[m] = agg(buckets[m])

    labels = {"ma21": "A 现在（跌破即卖）", "buffer": "B 缓冲（破3%才卖）", "2day": "C 确认（连2日破）"}
    print(f"{'规则':<18}{'笔数':>6}{'胜率':>8}{'均笔':>8}{'均盈':>8}{'均亏':>8}{'盈亏比':>8}{'年化中位':>10}{'回撤':>8}{'止损退出':>10}")
    for m in modes:
        t = table[m]
        print(f"{labels[m]:<18}{t['trades']:>6}{t['win_rate']:>7}%{t['avg_pnl']:>7}%"
              f"{t['avg_win']:>7}%{t['avg_loss']:>7}%{str(t['pf']):>8}{t['median_annual']:>9}%"
              f"{t['avg_mdd']:>7}%{t['exit_stop']:>10}")

    out = OUT_DIR / "stop_compare.json"
    out.write_text(json.dumps(_py({"stocks": done, "variants": table}), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
