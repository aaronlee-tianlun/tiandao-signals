"""确定性风控层（纯 Python，无 LLM）

设计原则（来自 ai-hedge-fund 的 risk/limits.py）：
  · 信念层（LLM）负责「提方案」并解释理由
  · 风控层（本文件）负责「做减法」，被削掉的仓位不回补给其他标的
  · 任何一条不满足 → 强制改判，并记录 clamped_by，绝不静默放行

校验项：
  1. 必须给出止损；没有止损的方案直接作废
  2. stop < entry_low ≤ entry_high < target（做多）
  3. 盈亏比 (target - entry_high) / (entry_low - stop) ≥ 2
  4. 止损幅度 (entry_low - stop) / entry_low ≤ 10%（按 A股 T+1 波动给足空间但别太宽）
  5. 入场价必须落在次日涨跌停区间内（跌停价 ≤ entry ≤ 涨停价）
  6. 建议仓位夹到 [0, 单票上限]
  7. 现价已跌破 MA21 → 一律否决
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskLimits:
    min_reward_risk: float = 2.0        # 盈亏比下限
    max_stop_pct: float = 10.0          # 止损幅度上限
    max_position_pct: float = 20.0      # 单票仓位上限
    min_position_pct: float = 2.0       # 仓位太小没意义


@dataclass
class RiskVerdict:
    allowed: bool
    rating: str                        # 买入 / 观望 / 回避
    position_pct: float
    stop: float | None
    entry_low: float | None
    entry_high: float | None
    target: float | None
    reward_risk: float | None
    clamped_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "rating": self.rating,
            "position_pct": round(self.position_pct, 2),
            "stop": self.stop,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "target": self.target,
            "reward_risk": None if self.reward_risk is None else round(self.reward_risk, 2),
            "clamped_by": self.clamped_by,
            "notes": self.notes,
        }


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def price_limits(prev_close: float, limit_pct: float) -> tuple[float, float]:
    """次日涨跌停价（A股按前收盘价四舍五入到分）"""
    up = round(prev_close * (1 + limit_pct / 100), 2)
    down = round(prev_close * (1 - limit_pct / 100), 2)
    return down, up


def clamp_plan(
    plan: dict,
    *,
    prev_close: float,
    limit_pct: float,
    ma21: float,
    close: float,
    limits: RiskLimits | None = None,
) -> RiskVerdict:
    """把 LLM 给的交易计划过一遍硬规则。plan 字段：
    rating / position_pct / entry_low / entry_high / stop / target
    """
    L = limits or RiskLimits()
    clamped: list[str] = []
    notes: list[str] = []

    rating = str(plan.get("rating") or "观望")
    stop = _f(plan.get("stop"))
    entry_low = _f(plan.get("entry_low"))
    entry_high = _f(plan.get("entry_high")) or entry_low
    target = _f(plan.get("target"))
    pos = _f(plan.get("position_pct")) or 0.0

    # 7 先看现价是否已破 MA21 —— 破了直接否决
    if close and ma21 and close < ma21:
        return RiskVerdict(False, "回避", 0, stop, entry_low, entry_high, target, None,
                           ["现价已跌破 MA21，风控否决"], notes)

    # 如果方案本身不是买入，原样返回（不需要校验价位）
    if not rating.startswith(("买", "加")):
        return RiskVerdict(False, rating, 0, stop, entry_low, entry_high, target, None, clamped, notes)

    # 1 必须有止损
    if not stop:
        return RiskVerdict(False, "观望", 0, None, entry_low, entry_high, target, None,
                           ["方案没有止损，作废"], notes)

    if not entry_low:
        entry_low = close
        entry_high = entry_high or close
        clamped.append("未给入场区间，按现价")

    # 2 价格顺序
    if stop >= entry_low:
        return RiskVerdict(False, "观望", 0, stop, entry_low, entry_high, target, None,
                           ["止损价不低于入场价，逻辑错误"], notes)
    if target and target <= entry_high:
        return RiskVerdict(False, "观望", 0, stop, entry_low, entry_high, target, None,
                           ["目标价不高于入场价，逻辑错误"], notes)

    # 3 盈亏比
    rr = None
    if target:
        risk = entry_low - stop
        reward = target - entry_high
        rr = reward / risk if risk > 0 else None
        if rr is None or rr < L.min_reward_risk:
            return RiskVerdict(False, "观望", 0, stop, entry_low, entry_high, target, rr,
                               [f"盈亏比 {rr:.1f} < {L.min_reward_risk}，不值得做" if rr else "无法计算盈亏比"], notes)

    # 4 止损幅度
    stop_pct = (entry_low - stop) / entry_low * 100
    if stop_pct > L.max_stop_pct:
        return RiskVerdict(False, "观望", 0, stop, entry_low, entry_high, target, rr,
                           [f"止损幅度 {stop_pct:.1f}% 超过 {L.max_stop_pct:.0f}% 上限"], notes)

    # 5 入场价落在涨跌停区间
    down, up = price_limits(prev_close, limit_pct)
    if entry_low < down or entry_high > up:
        notes.append(f"入场区间超出次日涨跌停 [{down}, {up}]，可能成交不了")
        clamped.append("入场价越界")
        entry_low = max(min(entry_low, up), down)
        entry_high = max(min(entry_high, up), down)

    # 6 仓位夹紧
    if pos > L.max_position_pct:
        clamped.append(f"仓位 {pos:.0f}% 超上限，削到 {L.max_position_pct:.0f}%")
        pos = L.max_position_pct
    if pos < L.min_position_pct:
        clamped.append(f"仓位 {pos:.0f}% 太小，按 {L.min_position_pct:.0f}% 起")
        pos = L.min_position_pct

    notes.append("T+1：今日买入次日才能卖，止损单无法当日执行")
    return RiskVerdict(True, "买入", pos, stop, entry_low, entry_high, target, rr, clamped, notes)
