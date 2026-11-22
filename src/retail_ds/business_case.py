"""3-year business case model (same assumptions as Section 7.2 of the report). All inputs are illustrative."""
from dataclasses import dataclass, field


@dataclass
class Assumptions:
    revenue: float = 500.0          # $M
    gross_margin: float = 0.30
    online_share: float = 0.30
    loyalty_share: float = 0.60
    promo_share: float = 0.08
    discount_rate: float = 0.10
    realisation: tuple = (0.2, 0.6, 1.0)
    # (people, cloud, change) $M per year
    costs: tuple = ((0.95, 0.15, 0.10), (1.05, 0.22, 0.06), (1.10, 0.28, 0.04))
    contingency: float = 0.10


def run_rate_benefits(a: Assumptions) -> dict:
    return {
        "forecasting": a.revenue * 0.004 * a.gross_margin + a.revenue * 0.0025,
        "personalisation": a.revenue * a.online_share * 0.03 * a.gross_margin,
        "promotion": a.revenue * a.promo_share * 0.05,
        "retention": a.revenue * a.loyalty_share * 0.005 * a.gross_margin,
    }


def evaluate(a: Assumptions = Assumptions(), benefit_factor: float = 1.0) -> dict:
    run = sum(run_rate_benefits(a).values())
    benefits = [r * run * benefit_factor for r in a.realisation]
    costs = [sum(c) * (1 + a.contingency) for c in a.costs]
    net = [b - c for b, c in zip(benefits, costs)]
    cum, total = [], 0.0
    for n in net:
        total += n
        cum.append(total)
    npv = sum(n / (1 + a.discount_rate) ** (i + 1) for i, n in enumerate(net))
    payback = next((f"Year {i + 1}" for i, c in enumerate(cum) if c >= 0), "> 3 years")
    return {"run_rate": run, "benefits": benefits, "costs": costs, "net": net,
            "cumulative": cum, "npv": npv, "roi": cum[-1] / sum(costs), "payback": payback}


def breakeven_factor(a: Assumptions = Assumptions()) -> float:
    run = sum(run_rate_benefits(a).values())
    costs = [sum(c) * (1 + a.contingency) for c in a.costs]
    pv_c = sum(c / (1 + a.discount_rate) ** (i + 1) for i, c in enumerate(costs))
    pv_b = sum(r * run / (1 + a.discount_rate) ** (i + 1) for i, r in enumerate(a.realisation))
    return pv_c / pv_b
