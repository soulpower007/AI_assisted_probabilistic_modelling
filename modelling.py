# budget_brain.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import itertools, random, math, statistics

Channel = str
CHANNELS: List[Channel] = ["Google", "Meta", "TikTok", "LinkedIn"]

@dataclass
class Range:
    low: float
    high: float

    def sample(self) -> float:
        # Uniform; swap in triangular or lognormal if you prefer
        return random.uniform(self.low, self.high)

@dataclass
class Priors:
    CPM: Range      # $ per 1000 impressions (positive)
    CTR: Range      # probability in [0,1]
    CVR: Range      # probability in [0,1]
    AOV: Range      # $ average order value / LTV

PriorsByChannel = Dict[Channel, Priors]

@dataclass
class Constraints:
    min_pct: Dict[Channel, float]  # e.g. {"LinkedIn": 0.2}
    max_pct: Dict[Channel, float]  # e.g. {"TikTok": 0.5}

@dataclass
class OutcomeStats:
    p10: float
    p50: float
    p90: float

@dataclass
class AllocationResult:
    allocation: Dict[Channel, float]         # fractions summing to 1.0
    sales_stats: OutcomeStats
    revenue_stats: OutcomeStats
    cac_stats: OutcomeStats                  # CAC over whole allocation
    objective_value: float                   # used for ranking

def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)

def summarize(xs: List[float]) -> OutcomeStats:
    return OutcomeStats(
        p10=percentile(xs, 0.10),
        p50=percentile(xs, 0.50),
        p90=percentile(xs, 0.90),
    )

def generate_allocations(step: float = 0.10,
                         constraints: Optional[Constraints] = None) -> List[Dict[Channel, float]]:
    """
    Grid allocations in 'step' increments that sum to 1.0.
    Applies min/max constraints (fractions).
    """
    n = int(round(1.0 / step))
    allocs = []
    # Represent each channel share as integer ticks that sum to n
    for g, m, t in itertools.product(range(n + 1), repeat=3):
        l = n - (g + m + t)
        if l < 0: 
            continue
        shares = [g, m, t, l]
        frac = [s / n for s in shares]
        cand = dict(zip(CHANNELS, frac))
        if constraints:
            ok = True
            for ch in CHANNELS:
                mn = constraints.min_pct.get(ch, 0.0)
                mx = constraints.max_pct.get(ch, 1.0)
                if not (mn - 1e-9 <= cand[ch] <= mx + 1e-9):
                    ok = False
                    break
            if not ok: 
                continue
        allocs.append(cand)
    return allocs

def simulate_once(budget: float,
                  allocation: Dict[Channel, float],
                  priors: PriorsByChannel) -> Tuple[float, float, float]:
    """
    Returns (sales, revenue, cac_overall).
    Funnel:
      spend S_c -> impressions = (S_c / CPM)*1000
                   -> clicks = impressions * CTR
                   -> sales  = clicks * CVR
                   -> revenue = sales * AOV
    CAC = total_spend / total_sales
    """
    total_sales = 0.0
    total_revenue = 0.0
    for ch, frac in allocation.items():
        S = budget * frac
        p = priors[ch]
        CPM = max(p.CPM.sample(), 1e-6)  # avoid div by zero
        CTR = min(max(p.CTR.sample(), 0.0), 1.0)
        CVR = min(max(p.CVR.sample(), 0.0), 1.0)
        AOV = max(p.AOV.sample(), 0.0)

        impressions = (S / CPM) * 1000.0
        clicks = impressions * CTR
        sales = clicks * CVR
        revenue = sales * AOV

        total_sales += sales
        total_revenue += revenue

    cac = budget / max(total_sales, 1e-9)
    return total_sales, total_revenue, cac

def evaluate_allocation(budget: float,
                        allocation: Dict[Channel, float],
                        priors: PriorsByChannel,
                        sims: int = 500) -> Tuple[OutcomeStats, OutcomeStats, OutcomeStats]:
    sales_samples, rev_samples, cac_samples = [], [], []
    for _ in range(sims):
        s, r, c = simulate_once(budget, allocation, priors)
        sales_samples.append(s)
        rev_samples.append(r)
        cac_samples.append(c)
    return summarize(sales_samples), summarize(rev_samples), summarize(cac_samples)

def optimize(budget: float,
             priors: PriorsByChannel,
             objective: str = "max_revenue",     # "max_sales" | "max_revenue" | "min_cac"
             step: float = 0.10,
             sims: int = 500,
             constraints: Optional[Constraints] = None,
             rng_seed: Optional[int] = 7) -> AllocationResult:
    """
    Grid search + Monte Carlo. Returns the best AllocationResult by the chosen objective.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    candidates = generate_allocations(step=step, constraints=constraints)
    best: Optional[AllocationResult] = None

    for alloc in candidates:
        sales_stats, rev_stats, cac_stats = evaluate_allocation(budget, alloc, priors, sims=sims)
        if objective == "max_sales":
            score = sales_stats.p50
        elif objective == "max_revenue":
            score = rev_stats.p50
        elif objective == "min_cac":
            score = -cac_stats.p50  # lower CAC is better
        else:
            raise ValueError("objective must be one of {'max_sales','max_revenue','min_cac'}")

        result = AllocationResult(
            allocation=alloc,
            sales_stats=sales_stats,
            revenue_stats=rev_stats,
            cac_stats=cac_stats,
            objective_value=score,
        )
        if best is None or result.objective_value > best.objective_value:
            best = result

    assert best is not None
    return best

# ---------- quick demo ----------
if __name__ == "__main__":
    # EXAMPLE priors (replace with Gemini-elicited ranges for your company + keep citations in your app)
    print("priors:")
    priors: PriorsByChannel = {
        "Google":   Priors(CPM=Range(5, 9),   CTR=Range(0.02, 0.05), CVR=Range(0.04, 0.10), AOV=Range(120, 220)),
        "Meta":     Priors(CPM=Range(3, 7),   CTR=Range(0.015,0.04), CVR=Range(0.03, 0.08), AOV=Range(80, 160)),
        "TikTok":   Priors(CPM=Range(2, 6),   CTR=Range(0.02, 0.05), CVR=Range(0.02, 0.06), AOV=Range(60, 120)),
        "LinkedIn": Priors(CPM=Range(8, 14),  CTR=Range(0.005,0.015),CVR=Range(0.08, 0.18), AOV=Range(250, 450)),
    }

    budget = 5000.0
    constraints = Constraints(min_pct={"LinkedIn": 0.20}, max_pct={})  # e.g., LinkedIn ≥ 20%

    best = optimize(
        budget=budget,
        priors=priors,
        objective="max_revenue",   # "max_sales" or "min_cac"
        step=0.10,                 # 10% grid
        sims=800,                  # increase for tighter bands
        constraints=constraints,
        rng_seed=42
    )

    print("Best allocation (fractions):")
    for ch in CHANNELS:
        print(f"  {ch:9s}: {best.allocation[ch]:.2f}  -> ${budget*best.allocation[ch]:.0f}")

    s, r, c = best.sales_stats, best.revenue_stats, best.cac_stats
    print("\nSales  (p10, p50, p90):", f"{s.p10:.1f}, {s.p50:.1f}, {s.p90:.1f}")
    print("Revenue(p10, p50, p90):", f"${r.p10:,.0f}, ${r.p50:,.0f}, ${r.p90:,.0f}")
    print("CAC    (p10, p50, p90):", f"${c.p10:,.2f}, ${c.p50:,.2f}, ${c.p90:,.2f}")
