from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, write_csv


def price_bond(face: float, coupon_rate: float, years: int, yield_rate: float, spread_bps: float = 0.0) -> float:
    discount = yield_rate + spread_bps / 10000
    cashflows = [face * coupon_rate for _ in range(years - 1)] + [face * (1 + coupon_rate)]
    return sum(cf / ((1 + discount) ** (idx + 1)) for idx, cf in enumerate(cashflows))


def duration(face: float, coupon_rate: float, years: int, yield_rate: float) -> float:
    cashflows = [face * coupon_rate for _ in range(years - 1)] + [face * (1 + coupon_rate)]
    price = price_bond(face, coupon_rate, years, yield_rate)
    weighted = sum((idx + 1) * cf / ((1 + yield_rate) ** (idx + 1)) for idx, cf in enumerate(cashflows))
    return weighted / price


def main() -> None:
    results = ensure_results_dir()
    face = 100.0
    coupon = 0.045
    years = 5
    base_yield = 0.04
    base_price = price_bond(face, coupon, years, base_yield)
    stressed_price = price_bond(face, coupon, years, base_yield, spread_bps=150)
    write_csv(
        results / "bond_pricing_tests.csv",
        [
            {
                "case": "base",
                "price": round(base_price, 6),
                "duration": round(duration(face, coupon, years, base_yield), 6),
                "convexity_proxy": 1.0,
            },
            {
                "case": "spread_plus_150bps",
                "price": round(stressed_price, 6),
                "duration": round(duration(face, coupon, years, base_yield + 0.015), 6),
                "convexity_proxy": 1.0,
            },
        ],
    )
    write_csv(
        results / "financial_risk_eval.csv",
        [
            {
                "scenario": "base",
                "spread_bps": 0,
                "price_change_pct": 0.0,
                "risk_score": 0.2,
            },
            {
                "scenario": "disclosure_stress",
                "spread_bps": 150,
                "price_change_pct": round((stressed_price - base_price) / base_price, 6),
                "risk_score": 0.65,
            },
        ],
    )


if __name__ == "__main__":
    main()
