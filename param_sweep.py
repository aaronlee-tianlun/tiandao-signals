"""参数敏感性扫描：量比 / 回踩 / 金叉窗口 的组合对结果的影响

固定股票池、固定行情数据，只改入场参数，看胜率/每笔收益/盈亏比怎么变。
止损固定用用户原版（跌破 MA21 立即卖），不掺入策略改动。

用法: python param_sweep.py --top 300
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from backtest_market import compute_signals
from compare_stops import simulate
from data_source import load_daily
from indicators import Params

OUT_DIR = Path(__file__).parent / "out"

GRID = {
    "volume_ratio_min": [1.5, 1.8, 2.0, 2.5],
    "pullback_min_pct": [3.0, 5.0],
    "golden_cross_window": [3, 5],
}


def agg(results: list[dict]) -> dict:
    trades = [t for r in results for t in r["trades"]]
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win": 0, "avg": 0, "pf": None, "ann": 0, "mdd": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    ann = [r["annual"] for r in results if r["trades"]] or [0]
    return {
        "trades": n,
        "win": round(100 * len(wins) / n, 1),
        "avg": round(sum(t["pnl"] for t in trades) / n, 2),
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "ann": round(float(np.median(ann)), 1),
        "mdd": round(sum(r["mdd"] for r in results) / len(results), 1) if results else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--bars", type=int, default=700)
    a = ap.parse_args()

    lite = OUT_DIR / "all_signals.json"
    codes = [x["code"] for x in json.loads(lite.read_text(encoding="utf-8"))["items"]][: a.top]
    print(f"股票池 {len(codes)} 只，参数组合 {np.prod([len(v) for v in GRID.values()])} 组", flush=True)

    combos = list(itertools.product(*GRID.values()))
    buckets: dict[tuple, list] = {c: [] for c in combos}

    done = 0
    for code in codes:
        try:
            df = load_daily(code, days=a.bars)
            if df is None or len(df) < 150:
                continue
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:  # noqa: BLE001
            continue

        for (vr, pb, gw) in combos:
            p = Params(volume_ratio_min=vr, pullback_min_pct=pb, golden_cross_window=gw)
            try:
                sig = compute_signals(df, p)
                buckets[(vr, pb, gw)].append(simulate(df, sig, p, "ma21"))
            except Exception:  # noqa: BLE001
                continue
        done += 1
        if done % 25 == 0:
            print(f"  ...{done}/{len(codes)}", flush=True)

    print(f"\n完成 {done} 只\n")
    rows = []
    for c in combos:
        r = agg(buckets[c])
        rows.append({"volume_ratio_min": c[0], "pullback_min_pct": c[1],
                     "golden_cross_window": c[2], **r})

    rows.sort(key=lambda r: -(r["avg"] or -999))
    print(f"{'量比':>6}{'回踩%':>7}{'金叉窗':>7}{'笔数':>7}{'胜率':>8}{'每笔':>8}{'盈亏比':>8}{'年化中位':>10}")
    for r in rows:
        print(f"{r['volume_ratio_min']:>6}{r['pullback_min_pct']:>7}{r['golden_cross_window']:>7}"
              f"{r['trades']:>7}{r['win']:>7}%{r['avg']:>7}%{str(r['pf']):>8}{r['ann']:>9}%")

    out = OUT_DIR / "param_sweep.json"
    out.write_text(json.dumps({"stocks": done, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
