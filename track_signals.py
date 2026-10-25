"""信号跟踪复盘：把每天的推荐存下来，之后用真实价格回算 T+N 表现

设计：
  out/track/<YYYYMMDD>.json   当日信号快照（推荐 + 5/6 候选）
  out/track/perf.json         每条信号逐日累积的收益曲线

每次扫描后跑一次 --update：
  1) 用今天的日常信号写快照
  2) 读所有历史快照，用今天 all_signals 的收盘价追加一条"距信号日 N 天的收益"
  3) 汇总 T+5 / T+10 胜率

用法:
  python track_signals.py --update
  python track_signals.py --summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "out"
TRACK_DIR = OUT_DIR / "track"


def _load(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _closes() -> dict[str, float]:
    d = _load(OUT_DIR / "all_signals.json", {})
    out = {}
    for x in d.get("items") or []:
        c, px = x.get("code"), x.get("close")
        if c and isinstance(px, (int, float)):
            out[str(c)] = float(px)
    return out


def record(today_payload: dict) -> None:
    """把今天的信号写成快照"""
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    trade_date = str(today_payload.get("trade_date") or "")
    if not trade_date:
        print("没有 trade_date，跳过快照")
        return

    items = []
    for x in today_payload.get("items") or []:
        eg = int(x.get("entry_green") or 0)
        if x.get("recommended") or eg >= 5:
            items.append({
                "code": str(x.get("code")),
                "name": x.get("name", ""),
                "close": float(x.get("close") or 0),
                "entry_green": eg,
                "recommended": bool(x.get("recommended")),
            })

    p = TRACK_DIR / (trade_date.replace("-", "") + ".json")
    p.write_text(json.dumps({"date": trade_date, "items": items}, ensure_ascii=False,
                            separators=(",", ":")), encoding="utf-8")
    print(f"快照 {p.name}：{len(items)} 条（6/6 {sum(1 for x in items if x['recommended'])} 只）")


def update_perf() -> dict:
    """用今天的收盘价，追加历史信号的最新表现"""
    closes = _closes()
    if not closes:
        print("没有 all_signals.json，跳过复盘")
        return {}

    perf_path = TRACK_DIR / "perf.json"
    perf = _load(perf_path, {"signals": []})
    known = {(s["code"], s["date"]) for s in perf["signals"]}
    today = dt.date.today()

    # 1) 新快照入库
    for f in sorted(TRACK_DIR.glob("2*.json")):
        snap = _load(f, {})
        for it in snap.get("items") or []:
            key = (str(it.get("code")), snap.get("date"))
            if key in known:
                continue
            perf["signals"].append({
                "code": str(it.get("code")),
                "name": it.get("name", ""),
                "date": snap.get("date"),
                "base": float(it.get("close") or 0),
                "entry_green": it.get("entry_green"),
                "recommended": bool(it.get("recommended")),
                "curve": [],
            })
            known.add(key)

    # 2) 每条信号追加当天表现
    for s in perf["signals"]:
        px = closes.get(s["code"])
        base = s.get("base") or 0
        if not px or base <= 0:
            continue
        try:
            d0 = dt.date.fromisoformat(s["date"])
        except Exception:  # noqa: BLE001
            continue
        days = (today - d0).days
        if days < 1:
            continue
        ret = (px / base - 1) * 100
        last = s["curve"][-1] if s["curve"] else None
        if last and last.get("d") == days:
            last["ret"] = round(ret, 2)
        else:
            s["curve"].append({"d": days, "ret": round(ret, 2)})
        s["latest_days"] = days
        s["latest_ret"] = round(ret, 2)

    perf["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    perf_path.write_text(json.dumps(perf, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    return perf


def summarize(perf: dict) -> dict:
    """按持有天数分档统计胜率"""
    def at(s, target):
        best = None
        for pt in s.get("curve") or []:
            if pt["d"] <= target:
                best = pt
            else:
                break
        return best["ret"] if best else None

    out = {}
    # 必须真的持有够天数才算，否则 2 天前的信号会被误算成 T+3
    for label, lo, hi in (("T+3", 3, 6), ("T+5", 5, 9), ("T+10", 10, 15)):
        vals = []
        for s in perf.get("signals") or []:
            v = at(s, hi)
            if v is None:
                continue
            last = s["curve"][-1]
            if last["d"] < lo:
                continue
            vals.append(v)
        if vals:
            win = sum(1 for v in vals if v > 0)
            out[label] = {
                "n": len(vals),
                "win_rate": round(100 * win / len(vals), 1),
                "avg_ret": round(sum(vals) / len(vals), 2),
            }

    recs = [s for s in perf.get("signals") or [] if s.get("recommended") and s.get("latest_ret") is not None]
    if recs:
        out["recommended_latest"] = {
            "n": len(recs),
            "win_rate": round(100 * sum(1 for s in recs if s["latest_ret"] > 0) / len(recs), 1),
            "avg_ret": round(sum(s["latest_ret"] for s in recs) / len(recs), 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    if a.update:
        today = _load(OUT_DIR / "daily_signals.json", {})
        if today:
            record(today)
        perf = update_perf()
        if perf:
            summary = summarize(perf)
            perf["summary"] = summary
            (TRACK_DIR / "perf.json").write_text(
                json.dumps(perf, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print("复盘汇总:", json.dumps(summary, ensure_ascii=False))
    if a.summary:
        perf = _load(TRACK_DIR / "perf.json", {})
        print(json.dumps(summarize(perf), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
