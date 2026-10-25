"""每日推荐扫描：全市场 → 过滤 → 七盏灯 → 按入场灯数排序

设计目标（抗失败）：
  1. 单只票的脏数据（NaN/异常）只丢这一只，绝不废掉整批
  2. 每 CHECKPOINT_EVERY 只落一次盘，中途被杀也有半成品
  3. --deadline-min 到点立即停止，输出 status=partial（宁可发 85% 新数据，不发 0%）
  4. 3 线程并发，全市场从 4 小时压到约 1 小时
  5. 快照失败时用上一交易日的 universe 兜底
  6. --merge 合并多个分片产出

用法:
  python scan_market.py --top 5000                    # 全市场
  python scan_market.py --top 500 --deadline-min 40   # 限时
  python scan_market.py --shard 0/4 --out out/part-0.json
  python scan_market.py --merge "out/part-*.json" --out out/daily_signals.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data_source import load_daily, code_names
from filters import FilterParams, filter_universe
from fundamentals import describe, financial_snapshot, valuation
from indicators import Params, evaluate

OUT_DIR = Path(__file__).parent / "out"
CHECKPOINT_EVERY = 200
WORKERS = 3            # 新浪/腾讯限流考虑，别开太高
PER_STOCK_TIMEOUT = 90  # 单只票最长等待（akshare 内部无 timeout，靠这里兜）


# ------------------------------------------------------------------ 工具


def _clean(obj):
    """把 NaN/Inf 换成 None，保证 JSON 合规"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _safe_dump(obj, path: Path) -> bool:
    """原子写入；序列化失败返回 False（不会留下半个文件）"""
    try:
        payload = json.dumps(_clean(obj), ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  序列化失败: {type(exc).__name__}: {exc}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return True



def _split_payload(out: dict) -> tuple[dict, dict]:
    """拆成两份：daily（入场灯>=4 + 全部推荐，给 App 首页）+ all（全市场精简版）"""
    items = out.get("items") or []
    daily_items = [x for x in items
                   if int(x.get("entry_green") or 0) >= 4 or x.get("recommended")]
    # 全市场统计（在过滤前算，App 直接读，避免用被裁剪的样本当分母）
    dist = {str(i): 0 for i in range(7)}
    light_pass: dict[str, int] = {}
    light_name: dict[str, str] = {}
    for x in items:
        g = int(x.get("entry_green") or 0)
        if 0 <= g <= 6:
            dist[str(g)] += 1
        for lt in x.get("lights") or []:
            k = lt.get("key")
            if not k:
                continue
            light_name[k] = lt.get("name", k)
            light_pass.setdefault(k, 0)
            if lt.get("passed"):
                light_pass[k] += 1

    daily = dict(out)
    daily["items"] = daily_items
    # 带上信号复盘摘要（由 track_signals.py 生成）
    try:
        _perf = json.loads((OUT_DIR / "track" / "perf.json").read_text(encoding="utf-8"))
        if _perf.get("summary"):
            daily["track"] = _perf["summary"]
    except Exception:  # noqa: BLE001
        pass
    daily["stats"] = {
        "entry_dist": dist,
        "light_pass": light_pass,
        "light_name": light_name,
        "light_total": len(items),
        "sample": len(daily_items),
    }
    daily["universe_full"] = out.get("universe")
    daily["scanned_full"] = out.get("scanned")
    daily["filter_note"] = "daily 只含入场灯>=4（含全部6/6推荐）；全市场精简版见 all_signals.json"

    lite = [{
        "code": x.get("code"), "name": x.get("name"), "close": x.get("close"),
        "entry_green": x.get("entry_green"), "green_count": x.get("green_count"),
        "recommended": bool(x.get("recommended")), "summary": x.get("summary", ""),
        "date": x.get("date", ""), "board": x.get("board"),
    } for x in items]
    allsig = {
        "generated_at": out.get("generated_at"), "trade_date": out.get("trade_date"),
        "universe": out.get("universe"), "scanned": out.get("scanned"),
        "failed": out.get("failed"), "status": out.get("status"),
        "recommended_count": out.get("recommended_count"),
        "recommend_rule": out.get("recommend_rule", ""), "lite": True, "items": lite,
    }
    return daily, allsig


def write_outputs(out: dict, target: Path) -> bool:
    """写 daily + all 两个文件；分片输出（part-*.json）保持原样"""
    if target.name != "daily_signals.json":
        return _safe_dump(out, target)
    daily, allsig = _split_payload(out)
    ok1 = _safe_dump(daily, target)
    ok2 = _safe_dump(allsig, target.with_name("all_signals.json"))
    return ok1 and ok2


# ------------------------------------------------------------------ 快照


def universe() -> list[dict]:
    """全市场快照。东财 → 新浪 → 上一交易日缓存 依次尝试。"""
    import akshare as ak

    cache = OUT_DIR / "universe.json"
    for name, fn in (("东财", lambda: ak.stock_zh_a_spot_em()),
                     ("新浪", lambda: ak.stock_zh_a_spot())):
        try:
            df = fn()
            df = df.rename(columns={"代码": "code", "名称": "name", "最新价": "price",
                                    "涨跌幅": "chg_pct", "成交额": "amount"})
            df["code"] = df["code"].astype(str).str[-6:]
            rows = df[["code", "name", "price", "chg_pct", "amount"]].to_dict("records")
            if len(rows) > 1000:
                _safe_dump({"date": dt.date.today().isoformat(), "rows": rows}, cache)
                print(f"{name}快照 {len(rows)} 只")
                return rows
            print(f"{name}快照只拿到 {len(rows)} 只，换下一个源")
        except Exception as exc:  # noqa: BLE001
            print(f"{name}快照失败({type(exc).__name__})")

    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            rows = d.get("rows") or []
            if rows:
                print(f"⚠ 实时快照全挂，用 {d.get('date')} 的缓存（{len(rows)} 只）")
                return rows
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError("拿不到全市场快照，且没有可用缓存")


# ------------------------------------------------------------------ 单只


def scan_one(row: dict, p: Params, names: dict) -> tuple[str, dict]:
    code = str(row["code"]).zfill(6)
    try:
        df = load_daily(code, days=400)
        res = evaluate(df, p)
        res.update({
            "code": code,
            "name": row.get("name") or names.get(code, ""),
            "board": row.get("board"),
        })
        # 用户口径：至少 6 个入场灯全绿才推，5/6 一律不推
        res["recommended"] = bool(res["entry_signal"] and int(res.get("entry_green") or 0) >= 6)
        try:
            fin = financial_snapshot(code)
            val = valuation(code, float(res["close"]), fin)
            res["fundamentals"] = _clean({**fin, **val, "text": describe(fin, val)})
        except Exception:  # noqa: BLE001
            res["fundamentals"] = {"text": "基本面数据缺失"}
        res = _clean(res)
        json.dumps(res, allow_nan=False)  # 自检：不能序列化就丢这一条
        return "ok", res
    except Exception as exc:  # noqa: BLE001
        return "fail", {"code": code, "error": f"{type(exc).__name__}: {str(exc)[:80]}"}


# ------------------------------------------------------------------ 主流程


def run(top: int, show: int, shard: str = "0/1", out_path: str = "",
        deadline_min: int = 0, workers: int = WORKERS) -> dict:
    p = Params()
    fp = FilterParams()

    rows = universe()
    rows = [r for r in rows if r.get("amount")]
    rows.sort(key=lambda r: float(r.get("amount") or 0), reverse=True)
    rows = rows[:top]
    rows = filter_universe(rows, fp)

    idx, total = (int(x) for x in shard.split("/"))
    if total > 1:
        rows = [r for i, r in enumerate(rows) if i % total == idx]
        print(f"分片 {idx}/{total} → {len(rows)} 只")

    names = code_names()
    results: list[dict] = []
    failed: list[dict] = []
    target = Path(out_path) if out_path else OUT_DIR / "daily_signals.json"
    deadline = time.time() + deadline_min * 60 if deadline_min > 0 else 0
    t0 = time.time()
    stopped = False

    def checkpoint():
        _safe_dump({
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "universe": len(rows), "scanned": len(results), "failed": len(failed),
            "items": results, "partial": True,
        }, target.with_suffix(".checkpoint.json"))

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(scan_one, r, p, names): r for r in rows}
    try:
        for done, fut in enumerate(as_completed(futures), 1):
            if deadline and time.time() > deadline:
                stopped = True
                print(f"⏰ 到达时限 {deadline_min} 分钟，停止抓取（已扫 {len(results)} 只）")
                break
            try:
                kind, payload = fut.result(timeout=PER_STOCK_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                r = futures[fut]
                kind, payload = "fail", {"code": str(r.get("code")),
                                         "error": f"timeout/{type(exc).__name__}"}
            if kind == "ok":
                results.append(payload)
            else:
                failed.append(payload)
                if len(failed) <= 20:
                    print(f"  ✗ {payload.get('code')} {payload.get('error')}")
            if done % CHECKPOINT_EVERY == 0:
                checkpoint()
                rate = done / max(time.time() - t0, 1)
                eta = (len(rows) - done) / max(rate, 0.01) / 60
                print(f"  ...{done}/{len(rows)}  失败 {len(failed)}  {rate:.1f}/s  "
                      f"预计还剩 {eta:.0f} 分钟")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results.sort(key=lambda x: (x["entry_green"], x["green_count"]), reverse=True)
    recommended = [x for x in results if x.get("recommended")]
    dates = [x["date"] for x in results if x.get("date")]
    trade_date = max(dates) if dates else ""
    fresh = sum(1 for d in dates if d == trade_date)

    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "trade_date": trade_date,
        "fresh_count": fresh,
        "universe": len(rows),
        "scanned": len(results),
        "failed": len(failed),
        "failed_sample": failed[:10],
        "providers": sorted({x.get("provider", "") for x in results if x.get("provider")}),
        "status": "partial" if stopped else (
            "ok" if results and len(results) >= len(rows) * 0.8 else "degraded"),
        "recommend_rule": "四条硬条件（周线趋势/回踩/金叉/量能）+ 站上 MA21",
        "recommended_count": len(recommended),
        "items": results,
    }
    if not _safe_dump(out, target):
        print("整包写不进，逐条剔除后重试")
        good = []
        for x in results:
            try:
                json.dumps(_clean(x), allow_nan=False)
                good.append(_clean(x))
            except Exception:  # noqa: BLE001
                failed.append({"code": x.get("code"), "error": "序列化失败"})
        out["items"] = good
        out["failed"] = len(failed)
        out["scanned"] = len(good)
        write_outputs(out, target)

    print(f"\n{'代码':<8}{'名称':<10}{'收盘':>8}  {'入场':<6}{'绿灯':<6}结论")
    for x in results[:show]:
        print(f"{x['code']:<8}{x['name']:<10}{x['close']:>8.2f}  "
              f"{x['entry_green']}/6    {x['green_count']}/7    {x['summary']}")
    size = target.stat().st_size / 1048576 if target.exists() else 0
    print(f"\nstatus={out['status']} scanned={len(results)} failed={len(failed)} "
          f"trade_date={trade_date} 用时 {(time.time()-t0)/60:.1f} 分钟 体积 {size:.1f} MiB")
    print(f"已写入 {target}")
    return out


def merge(patterns: list[str], out_path: str) -> dict:
    """合并多个分片产出"""
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"没有匹配到分片: {patterns}")
    items, failed = [], []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        items.extend(d.get("items") or [])
        failed.extend(d.get("failed_sample") or [])
        print(f"  合并 {f}: {len(d.get('items') or [])} 只")
    seen, uniq = set(), []
    for x in items:
        c = x.get("code")
        if c in seen:
            continue
        seen.add(c)
        uniq.append(x)
    uniq.sort(key=lambda x: (x["entry_green"], x["green_count"]), reverse=True)
    dates = [x["date"] for x in uniq if x.get("date")]
    trade_date = max(dates) if dates else ""
    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "trade_date": trade_date,
        "fresh_count": sum(1 for d in dates if d == trade_date),
        "universe": len(uniq) + len(failed),
        "scanned": len(uniq),
        "failed": len(failed),
        "failed_sample": failed[:10],
        "providers": sorted({x.get("provider", "") for x in uniq if x.get("provider")}),
        "status": "ok" if uniq else "degraded",
        "recommend_rule": "四条硬条件（周线趋势/回踩/金叉/量能）+ 站上 MA21",
        "recommended_count": len([x for x in uniq if x.get("recommended")]),
        "items": uniq,
    }
    target = Path(out_path)
    if not write_outputs(out, target):
        raise SystemExit("合并后写入失败")
    print(f"合并完成：{len(uniq)} 只 → {target}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=120, help="候选池大小（按成交额降序）")
    ap.add_argument("--limit", type=int, default=20, help="终端打印条数")
    ap.add_argument("--shard", default="0/1", help="分片 i/n，如 0/4")
    ap.add_argument("--out", default="", help="输出文件（默认 out/daily_signals.json）")
    ap.add_argument("--deadline-min", type=int, default=0, help="总时限（分钟），到点输出 partial")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--merge", nargs="*", default=None, help="合并分片文件")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, a.out or str(OUT_DIR / "daily_signals.json"))
    else:
        run(a.top, a.limit, a.shard, a.out, a.deadline_min, a.workers)
