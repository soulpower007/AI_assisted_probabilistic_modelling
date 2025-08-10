#!/usr/bin/env python3
"""
Marketing Budget Allocator (Monte Carlo + Saturation)
- Priors: CPM (lognormal), CTR/CVR (beta), AOV (lognormal), Quality multiplier
- Response curve: conversions_i(S) = theta_i * (1 - exp(-beta_i * S))
  where early slope at S=0 equals `base_i = (1000/CPM) * CTR * CVR * quality_adj`
  => beta_i = base_i / theta_i
- Revenue_i(S) = conversions_i(S) * AOV_i
- Greedy allocation by marginal revenue dR/dS = base_i * exp(-beta_i * S) * AOV_i
- Constraints: min/max share per channel (e.g., LinkedIn ≥ 20%)

No external deps beyond numpy.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# -----------------------------
# Utility: Beta params from (mean, std)
# -----------------------------
def beta_params(mean: float, std: float) -> Tuple[float, float]:
    mean = np.clip(mean, 1e-6, 1 - 1e-6)
    var = std**2
    # Prevent invalid variance for Beta
    max_var = mean * (1 - mean)
    var = min(var, 0.99 * max_var if max_var > 0 else 1e-12)
    k = (mean * (1 - mean) / var) - 1.0
    alpha = mean * k
    beta = (1 - mean) * k
    # Clamp minimum to avoid numeric issues
    return max(alpha, 1e-3), max(beta, 1e-3)

# -----------------------------
# Channel configuration (assumed priors)
# Means/stds are illustrative; tweak later as needed.
# -----------------------------
@dataclass
class ChannelPriors:
    name: str
    # Lognormal for CPM (per 1000 impressions): log-mean, log-sigma
    cpm_log_mu: float
    cpm_log_sigma: float
    # Beta for CTR & CVR (in 0..1): mean and std to convert into alpha,beta
    ctr_mean: float
    ctr_std: float
    cvr_mean: float
    cvr_std: float
    # Quality multiplier (lognormal around 1.0)
    quality_log_mu: float = 0.0      # ln(1.0)
    quality_log_sigma: float = 0.1   # ~10% sigma
    # Theta cap (max conversions) ~ Lognormal parameters
    theta_log_mu: float = np.log(5000.0)
    theta_log_sigma: float = 0.3
    # AOV per channel (lognormal) — you can also make this global
    aov_log_mu: float = np.log(150.0) - 0.5 * (0.3**2)  # so mean ~150 for sigma=0.3
    aov_log_sigma: float = 0.3

# Defaults (rough, hackathon-friendly)
CHANNEL_PRIORS: Dict[str, ChannelPriors] = {
    "Google": ChannelPriors(
        name="Google",
        cpm_log_mu=np.log(8.0)-0.5*(0.25**2), cpm_log_sigma=0.25,
        ctr_mean=0.030, ctr_std=0.008,
        cvr_mean=0.040, cvr_std=0.008,
        theta_log_mu=np.log(8000.0), theta_log_sigma=0.25
    ),
    "Meta": ChannelPriors(
        name="Meta",
        cpm_log_mu=np.log(6.0)-0.5*(0.25**2), cpm_log_sigma=0.25,
        ctr_mean=0.015, ctr_std=0.005,
        cvr_mean=0.030, cvr_std=0.008,
        theta_log_mu=np.log(6000.0), theta_log_sigma=0.25
    ),
    "TikTok": ChannelPriors(
        name="TikTok",
        cpm_log_mu=np.log(5.5)-0.5*(0.3**2), cpm_log_sigma=0.3,
        ctr_mean=0.012, ctr_std=0.004,
        cvr_mean=0.025, cvr_std=0.008,
        theta_log_mu=np.log(4000.0), theta_log_sigma=0.3
    ),
    "LinkedIn": ChannelPriors(
        name="LinkedIn",
        cpm_log_mu=np.log(12.0)-0.5*(0.25**2), cpm_log_sigma=0.25,
        ctr_mean=0.008, ctr_std=0.002,
        cvr_mean=0.035, cvr_std=0.010,
        theta_log_mu=np.log(3000.0), theta_log_sigma=0.25
    ),
}

# -----------------------------
# Sampling a single world (draw)
# -----------------------------
@dataclass
class ChannelDraw:
    name: str
    cpm: float
    ctr: float
    cvr: float
    aov: float
    theta: float      # conversions cap
    base: float       # initial conversions per $ at S->0
    beta: float       # saturation rate = base / theta

def sample_channel(priors: ChannelPriors, rng: np.random.Generator) -> ChannelDraw:
    # Sample CPM (lognormal)
    cpm = float(rng.lognormal(mean=priors.cpm_log_mu, sigma=priors.cpm_log_sigma))
    # Sample CTR/CVR via Beta
    ctr_a, ctr_b = beta_params(priors.ctr_mean, priors.ctr_std)
    cvr_a, cvr_b = beta_params(priors.cvr_mean, priors.cvr_std)
    ctr = float(rng.beta(ctr_a, ctr_b))
    cvr = float(rng.beta(cvr_a, cvr_b))
    # Quality multiplier (affects CPM inverse, CTR, CVR)
    q = float(rng.lognormal(mean=priors.quality_log_mu, sigma=priors.quality_log_sigma))
    # Apply quality: cheaper CPM, higher CTR/CVR (mildly)
    cpm_eff = cpm / max(q, 1e-6)
    ctr_eff = min(ctr * (q ** 0.5), 0.999999)
    cvr_eff = min(cvr * (q ** 0.5), 0.999999)
    # AOV (lognormal)
    aov = float(rng.lognormal(mean=priors.aov_log_mu, sigma=priors.aov_log_sigma))
    # Theta cap (lognormal)
    theta = float(rng.lognormal(mean=priors.theta_log_mu, sigma=priors.theta_log_sigma))
    # Early slope (conversions per $ at S≈0) from funnel math
    # base = (1000 impressions / $CPM) * CTR * CVR
    base = (1000.0 / max(cpm_eff, 1e-6)) * ctr_eff * cvr_eff
    # Saturation speed to match early slope
    beta = base / max(theta, 1e-9)
    return ChannelDraw(
        name=priors.name, cpm=cpm_eff, ctr=ctr_eff, cvr=cvr_eff,
        aov=aov, theta=theta, base=base, beta=beta
    )

# -----------------------------
# Response + Marginal functions
# -----------------------------
def conversions(draw: ChannelDraw, spend: float) -> float:
    # theta * (1 - exp(-beta S))
    return draw.theta * (1.0 - np.exp(-draw.beta * spend))

def marginal_revenue(draw: ChannelDraw, spend: float) -> float:
    # dR/dS = base * exp(-beta S) * AOV
    return draw.base * np.exp(-draw.beta * spend) * draw.aov

def revenue(draw: ChannelDraw, spend: float) -> float:
    return conversions(draw, spend) * draw.aov

# -----------------------------
# Greedy allocator with constraints
# -----------------------------
@dataclass
class Constraints:
    min_share: Dict[str, float]   # e.g., {"LinkedIn": 0.20}
    max_share: Optional[Dict[str, float]] = None

def allocate_budget(
    total_budget: float,
    draws: Dict[str, ChannelDraw],
    constraints: Constraints,
    step: float = 500.0
) -> Dict[str, float]:
    names = list(draws.keys())
    spend = {n: 0.0 for n in names}

    # Pre-allocate minimum shares
    min_total = 0.0
    for n in names:
        ms = constraints.min_share.get(n, 0.0)
        if ms < 0 or ms > 1:
            raise ValueError(f"Invalid min_share for {n}: {ms}")
        spend[n] = ms * total_budget
        min_total += spend[n]
    if min_total - 1e-6 > total_budget:
        raise ValueError("Sum of min_share exceeds 100% of budget")

    # Compute max caps in $
    max_cap = {n: float("inf") for n in names}
    if constraints.max_share:
        for n in names:
            mx = constraints.max_share.get(n, 1.0)
            max_cap[n] = max(0.0, min(1.0, mx)) * total_budget

    remaining = total_budget - min_total
    if remaining < -1e-6:
        raise ValueError("Negative remaining budget (check min shares)")

    # Greedy increments
    steps = int(np.floor(remaining / step + 1e-9))
    for _ in range(steps):
        # Compute marginal revenue for each channel at current spend
        mr = {}
        for n in names:
            if spend[n] + step <= max_cap[n] + 1e-9:
                mr[n] = marginal_revenue(draws[n], spend[n])
            else:
                mr[n] = -np.inf
        # Choose best
        best = max(mr, key=mr.get)
        if not np.isfinite(mr[best]):
            # all capped: stop early
            break
        spend[best] += step

    # Handle leftover pennies due to rounding
    leftover = total_budget - sum(spend.values())
    if leftover > 1e-6:
        # put leftover where marginal is highest and not capped
        valid = [n for n in names if spend[n] + leftover <= max_cap[n] + 1e-9]
        if valid:
            best = max(valid, key=lambda n: marginal_revenue(draws[n], spend[n]))
            spend[best] += leftover

    return spend

# -----------------------------
# Monte Carlo simulation
# -----------------------------
def run_simulation(
    total_budget: float = 100_000.0,
    n_sims: int = 200,
    seed: int = 42,
    linked_in_min_share: float = 0.20,
    step: float = 500.0
):
    rng = np.random.default_rng(seed)
    names = list(CHANNEL_PRIORS.keys())

    # Storage
    spends = {n: [] for n in names}
    revenues = []
    roases = []

    base_constraints = Constraints(
        min_share={"LinkedIn": linked_in_min_share},
        max_share=None  # add if needed, e.g., {"TikTok": 0.5}
    )

    for _ in range(n_sims):
        # Sample a world
        draws = {n: sample_channel(CHANNEL_PRIORS[n], rng) for n in names}
        # Allocate
        try:
            s = allocate_budget(total_budget, draws, base_constraints, step=step)
        except ValueError:
            # If constraints impossible, fallback to proportional min shares
            s = {n: base_constraints.min_share.get(n, 0.0) * total_budget for n in names}
            # any remaining evenly split
            rem = total_budget - sum(s.values())
            if rem > 0:
                rem_each = rem / len(names)
                for n in names:
                    s[n] += rem_each

        # Compute outcomes
        rev = sum(revenue(draws[n], s[n]) for n in names)
        roas = rev / total_budget if total_budget > 0 else 0.0

        for n in names:
            spends[n].append(s[n])
        revenues.append(rev)
        roases.append(roas)

    # Summaries
    def pct(x, p): return float(np.percentile(x, p))
    
    alloc_summary = {
        n: {
            "median_$": pct(spends[n], 50),
            "p05_$": pct(spends[n], 5),
            "p95_$": pct(spends[n], 95),
            "median_%": pct([v/total_budget*100 for v in spends[n]], 50),
            "p05_%": pct([v/total_budget*100 for v in spends[n]], 5),
            "p95_%": pct([v/total_budget*100 for v in spends[n]], 95),
        }
        for n in names
    }
    revenue_summary = {
        "revenue_median": pct(revenues, 50),
        "revenue_p05": pct(revenues, 5),
        "revenue_p95": pct(revenues, 95),
    }
    roas_summary = {
        "roas_median": pct(roases, 50),
        "roas_p05": pct(roases, 5),
        "roas_p95": pct(roases, 95),
    }

    return alloc_summary, revenue_summary, roas_summary

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    total_budget = 100_000.0
    alloc, rev_sum, roas_sum = run_simulation(
        total_budget=total_budget,
        n_sims=300,          # increase for smoother results
        seed=123,
        linked_in_min_share=0.20,
        step=500.0
    )

    print("=== Recommended Allocation (Median with 5–95% bands) ===")
    for n, s in alloc.items():
        print(
            f"{n:9s}: "
            f"${s['median_$']:,.0f} "
            f"(p05 ${s['p05_$']:,.0f}, p95 ${s['p95_$']:,.0f})  "
            f"{s['median_%']:.1f}% "
            f"(p05 {s['p05_%']:.1f}%, p95 {s['p95_%']:.1f}%)"
        )
    print("\n=== Outcomes ===")
    print(f"Expected Revenue (median): ${rev_sum['revenue_median']:,.0f} "
          f"[{rev_sum['revenue_p05']:,.0f} … {rev_sum['revenue_p95']:,.0f}]")
    print(f"Expected ROAS (median): {roas_sum['roas_median']:.2f} "
          f"[{roas_sum['roas_p05']:.2f} … {roas_sum['roas_p95']:.2f}]")
