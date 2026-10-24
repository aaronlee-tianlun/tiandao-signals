"""每日推荐扫描：全市场 → 过滤 → 七盏灯 → 按入场灯数排序

用法:
  .venv/Scripts/python.exe scan_market.py            # 默认扫成交额前 120 只
  .venv/Scripts/python.exe scan_market.py --top 200 --limit 20
输出:
  D:/Software/quant-app/backend/out/daily_signals.json   （App 直接读这个）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from data_source import load_daily, code_names
from filters import FilterParams, filter_universe
from fundamentals import describe, financial_snapshot, valuation
from indicators import Params, evaluate

OUT_DIR = Path(__file__).parent / "out"


def universe() -> list[dict]:
    """全市场快照。东财 → 新浪 依次尝试。"""
    import akshare as ak

    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={"代码": "code", "名称": "name", "最新价": "price",
                                "涨跌幅": "chg_pct", "成交额": "amount"})
        return df[["code", "name", "price", "chg_pct", "amount"]].to_dict("records")
    except Exception as exc:  # noqa: BLE001
        print(f"东财快照失败({type(exc).__name__})，改用新浪")
    try:
        df = ak.stock_zh_a_spot()
        df = df.rename(columns={"代码": "code", "名称": "name", "最新价": "price",
                                "涨跌幅": "chg_pct", "成交额": "amount"})
        df["code"] = df["code"].astype(str).str[-6:]
        return df[["code", "name", "price", "chg_pct", "amount"]].to_dict("records")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"拿不到全市场快照: {exc}")


def run(top: int, show: int) -> dict:
    p = Params()
    fp = FilterParams()

    rows = universe()
    print(f"全市场 {len(rows)} 只")
    rows = [r for r in rows if r.get("amount")]
    rows.sort(key=lambda r: float(r.get("amount") or 0), reverse=True)
    rows = rows[:top]
    rows = filter_universe(rows, fp)
    print(f"过滤后 {len(rows)} 只（候选池按成交额降序前 {top}）")

    names = code_names()
    results = []
    failed: list[dict] = []
    for i, r in enumerate(rows, 1):
        code = str(r["code"]).zfill(6)
        try:
            df = load_daily(code, days=400)
            res = evaluate(df, p)
            res.update({"code": code, "name": r["name"] or names.get(code, ""), "board": r["board"]})
            # 推荐门槛：只有 6 个入场灯全绿才进推荐列表（用户明确要求，5/6 不推）
            res["recommended"] = bool(res["entry_signal"])
            # 基本面快照（给 AI 委员会的基本面分析师用）
            try:
                fin = financial_snapshot(code)
                val = valuation(code, float(res["close"]), fin)
                res["fundamentals"] = {**fin, **val, "text": describe(fin, val)}
            except Exception:  # noqa: BLE001
                res["fundamentals"] = {"text": "基本面数据缺失"}
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            failed.append({"code": code, "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
            print(f"  [{i}/{len(rows)}] {code} 跳过: {type(exc).__name__}")
            continue
        if i % 10 == 0:
            print(f"  ...{i}/{len(rows)}")

    # 刷新持仓期峰值（止盈判定的基准）
    try:
        from portfolio import update_peak
        for x in results:
            update_peak(x["code"], float(x["close"]))
    except Exception:  # noqa: BLE001
        pass

    results.sort(key=lambda x: (x["entry_green"], x["green_count"]), reverse=True)
    recommended = [x for x in results if x.get("recommended")]
    dates = [x["date"] for x in results if x.get("date")]
    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "trade_date": max(dates) if dates else "",
        "universe": len(rows),
        "scanned": len(results),
        "failed": len(failed),
        "failed_sample": failed[:10],
        "providers": sorted({x.get("provider", "") for x in results if x.get("provider")}),
        "status": "ok" if results and len(results) >= len(rows) * 0.8 else "degraded",
        "recommend_rule": "四条硬条件（周线趋势/回踩/金叉/量能）+ 站上 MA21",
        "recommended_count": len(recommended),
        "items": results,
    }
    OUT_DIR.mkdir(exist_ok=True)
    payload = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)  # 有 NaN 立刻炸在扫描端
    tmp = OUT_DIR / "daily_signals.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, OUT_DIR / "daily_signals.json")

    print(f"\n{'代码':<8}{'名称':<10}{'收盘':>8}  {'入场':<5}{'绿灯':<5}  结论")
    for x in results[:show]:
        print(f"{x['code']:<8}{x['name']:<10}{x['close']:>8.2f}  "
              f"{x['entry_green']}/6   {x['green_count']}/7   {x['summary']}")
    print(f"\n已写入 {OUT_DIR / 'daily_signals.json'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=120, help="候选池大小（按成交额降序）")
    ap.add_argument("--limit", type=int, default=20, help="终端打印条数")
    a = ap.parse_args()
    run(a.top, a.limit)
