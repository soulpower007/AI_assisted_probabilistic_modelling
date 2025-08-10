#!/usr/bin/env python3
"""
Marketing Budget Allocator (Monte Carlo + Saturation + Goal Switches)
- Priors: CPM (lognormal), CTR/CVR (beta), AOV (lognormal), Quality multiplier
- Response curve: conversions_i(S) = theta_i * (1 - exp(-beta_i * S))
  where early slope at S=0 equals base_i = (1000/CPM) * CTR * CVR * quality_adj
  => beta_i = base_i / theta_i   # (fixed)
- Revenue_i(S) = leads_i(S) * aov_effective  # aov_effective = ACV * lead->SQL * SQL->win
- Greedy allocation by a marginal "score" per goal:
    REVENUE:     dR/dS = base_i * exp(-beta_i * S) * aov_effective_i
    CONVERSIONS: dC/dS = base_i * exp(-beta_i * S)
    PROFIT:      per-$ profit increment = gross_margin * (dR/dS) - 1
- Constraints:
    * min/max share per channel (e.g., LinkedIn ≥ 20%)
    * optional ROAS floor: only allocate steps if running ROAS stays ≥ floor
- No external deps beyond numpy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import os
from datetime import datetime

# -----------------------------
# Utility: Beta params from (mean, std)
# -----------------------------
def beta_params(mean: float, std: float) -> Tuple[float, float]:
    mean = np.clip(mean, 1e-6, 1 - 1e-6)
    var = std**2
    max_var = mean * (1 - mean)
    var = min(var, 0.99 * max_var if max_var > 0 else 1e-12)
    k = (mean * (1 - mean) / var) - 1.0
    alpha = mean * k
    beta = (1 - mean) * k
    return max(alpha, 1e-3), max(beta, 1e-3)

# -----------------------------
# Channel configuration (assumed priors)
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
    theta_log_mu: float = np.log(1200.0)
    theta_log_sigma: float = 0.3
    # AOV per channel (lognormal) — can also make this global
    aov_log_mu: float = np.log(150.0) - 0.5 * (0.3**2)  # so mean ~150 for sigma=0.3
    aov_log_sigma: float = 0.3

def load_channel_priors(config_path: str = None) -> Dict[str, ChannelPriors]:
    """Load channel priors from JSON config file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config", "channel_priors.json")
        print(f"Loading channel priors from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        channel_priors = {}
        for channel_name, data in config_data.items():
            channel_priors[channel_name] = ChannelPriors(**data)
        return channel_priors
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using default values")
        return get_default_channel_priors()
    except Exception as e:
        print(f"Error loading config: {e}, using default values")
        return get_default_channel_priors()

def get_default_channel_priors() -> Dict[str, ChannelPriors]:
    """Fallback default channel priors if config file is not available."""
    return {
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

def save_channel_priors(channel_priors: Dict[str, ChannelPriors], config_path: str = None):
    """Save channel priors to JSON config file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config", "channel_priors.json")
    config_data = {}
    for channel_name, priors in channel_priors.items():
        config_data[channel_name] = {
            "name": priors.name,
            "cpm_log_mu": priors.cpm_log_mu,
            "cpm_log_sigma": priors.cpm_log_sigma,
            "ctr_mean": priors.ctr_mean,
            "ctr_std": priors.ctr_std,
            "cvr_mean": priors.cvr_mean,
            "cvr_std": priors.cvr_std,
            "quality_log_mu": priors.quality_log_mu,
            "quality_log_sigma": priors.quality_log_sigma,
            "theta_log_mu": priors.theta_log_mu,
            "theta_log_sigma": priors.theta_log_sigma,
            "aov_log_mu": priors.aov_log_mu,
            "aov_log_sigma": priors.aov_log_sigma
        }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Channel priors saved to {config_path}")

# Load channel priors from config file
CHANNEL_PRIORS: Dict[str, ChannelPriors] = load_channel_priors()

# -----------------------------
# Sampling a single world (draw)
# -----------------------------
@dataclass
class ChannelDraw:
    name: str
    cpm: float
    ctr: float
    cvr: float
    aov_effective: float  # expected $ per LEAD (ACV × lead→SQL × SQL→win)
    aov_sale: float       # ACV per WON deal
    theta: float          # leads cap
    base: float           # initial leads per $ at S->0
    beta: float           # saturation rate = base / theta
    p_lead_to_sql: float  # stored for transparency
    p_sql_to_win: float

def sample_channel(priors: ChannelPriors, rng: np.random.Generator) -> ChannelDraw:
    # Base draws
    cpm = float(rng.lognormal(mean=priors.cpm_log_mu, sigma=priors.cpm_log_sigma))
    ctr_a, ctr_b = beta_params(priors.ctr_mean, priors.ctr_std)
    cvr_a, cvr_b = beta_params(priors.cvr_mean, priors.cvr_std)
    ctr = float(rng.beta(ctr_a, ctr_b))
    cvr = float(rng.beta(cvr_a, cvr_b))
    q = float(rng.lognormal(mean=priors.quality_log_mu, sigma=priors.quality_log_sigma))

    # Quality: cheaper CPM, mild CTR/CVR lift (cap for B2B realism)
    cpm_eff = cpm / max(q ** 0.5, 1e-6)
    ctr_eff = min(ctr * (q ** 0.2), 0.05)  # <= 5%
    cvr_eff = min(cvr * (q ** 0.2), 0.05)  # <= 5% (click->lead)

    # AOV draws
    aov_sale = float(rng.lognormal(mean=priors.aov_log_mu, sigma=priors.aov_log_sigma))

    # Deep funnel (keep both prob and expected-$ per lead)
    p_lead_to_sql = 0.25
    p_sql_to_win  = 0.15
    aov_effective = aov_sale * p_lead_to_sql * p_sql_to_win  # expected $ per LEAD

    # Capacity + base + saturation
    theta = float(rng.lognormal(mean=priors.theta_log_mu, sigma=priors.theta_log_sigma))
    base = (1000.0 / max(cpm_eff, 1e-6)) * ctr_eff * cvr_eff   # leads per $
    beta = base / max(theta, 1e-9)                             # preserve slope

    return ChannelDraw(
        name=priors.name, cpm=cpm_eff, ctr=ctr_eff, cvr=cvr_eff,
        aov_effective=aov_effective, aov_sale=aov_sale,
        theta=theta, base=base, beta=beta,
        p_lead_to_sql=p_lead_to_sql, p_sql_to_win=p_sql_to_win
    )

# -----------------------------
# Response + Marginal functions
# -----------------------------
def conversions(draw: ChannelDraw, spend: float) -> float:
    """Leads produced at spend S."""
    return draw.theta * (1.0 - np.exp(-draw.beta * spend))

def marginal_conversions(draw: ChannelDraw, spend: float) -> float:
    """dLeads/d$ at spend S."""
    return draw.base * np.exp(-draw.beta * spend)

def revenue(draw: ChannelDraw, spend: float) -> float:
    """Expected revenue ($) from LEADS at spend S."""
    return conversions(draw, spend) * draw.aov_effective

def marginal_revenue(draw: ChannelDraw, spend: float) -> float:
    """dRevenue/d$ at spend S."""
    return marginal_conversions(draw, spend) * draw.aov_effective

def wins_from_revenue(draw: ChannelDraw, rev: float) -> float:
    """Convert expected revenue back to expected WON deals using raw ACV."""
    return rev / max(draw.aov_sale, 1e-9)

# -----------------------------
# Goals
# -----------------------------
class GoalType(str, Enum):
    REVENUE = "revenue"
    CONVERSIONS = "conversions"
    PROFIT = "profit"

@dataclass
class GoalConfig:
    goal: GoalType = GoalType.REVENUE
    gross_margin: float = 1.0                # used only for PROFIT; 1.0 = revenue (no COGS)
    roas_floor: Optional[float] = None       # e.g., 2.0 means require ROAS >= 2 during allocation

def marginal_profit(draw: ChannelDraw, spend: float, gross_margin: float) -> float:
    return gross_margin * marginal_revenue(draw, spend) - 1.0

def marginal_score(draw: ChannelDraw, spend: float, goal_cfg: GoalConfig) -> float:
    if goal_cfg.goal == GoalType.REVENUE:
        return marginal_revenue(draw, spend)
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        return marginal_conversions(draw, spend)
    elif goal_cfg.goal == GoalType.PROFIT:
        return marginal_profit(draw, spend, goal_cfg.gross_margin)
    else:
        return marginal_revenue(draw, spend)

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
    step: float = 500.0,
    goal_cfg: GoalConfig = GoalConfig()
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

    steps = int(np.floor(remaining / max(step, 1e-9) + 1e-9))

    # Track totals (for roas_floor guard)
    def total_rev_at_spend_map(spend_map):
        return sum(revenue(draws[n], spend_map[n]) for n in names)

    cur_rev = total_rev_at_spend_map(spend)
    cur_spend = sum(spend.values())

    for _ in range(steps):
        scores = {}
        for n in names:
            if spend[n] + step > max_cap[n] + 1e-9:
                scores[n] = -np.inf
            else:
                sc = marginal_score(draws[n], spend[n], goal_cfg)
                # Optional ROAS floor guard
                if goal_cfg.roas_floor is not None and np.isfinite(sc):
                    delta_rev = marginal_revenue(draws[n], spend[n]) * step
                    new_rev = cur_rev + delta_rev
                    new_spend = cur_spend + step
                    new_roas = new_rev / new_spend if new_spend > 0 else 0.0
                    if new_roas + 1e-12 < goal_cfg.roas_floor:
                        sc = -np.inf
                scores[n] = sc

        best = max(scores, key=scores.get)
        if not np.isfinite(scores[best]):
            break  # nothing admissible (caps or ROAS floor bind)

        # Apply step to best, update totals
        spend[best] += step
        cur_spend += step
        cur_rev += marginal_revenue(draws[best], spend[best] - step) * step

    # Handle leftover pennies due to rounding
    leftover = total_budget - sum(spend.values())
    if leftover > 1e-6:
        valid = []
        for n in names:
            if spend[n] + leftover <= max_cap[n] + 1e-9:
                if goal_cfg.roas_floor is not None:
                    delta_rev = marginal_revenue(draws[n], spend[n]) * leftover
                    new_rev = cur_rev + delta_rev
                    new_spend = cur_spend + leftover
                    if new_rev / new_spend + 1e-12 < goal_cfg.roas_floor:
                        continue
                valid.append(n)
        if valid:
            best = max(valid, key=lambda n: marginal_score(draws[n], spend[n], goal_cfg))
            spend[best] += leftover

    return spend

# -----------------------------
# Monte Carlo simulation
# -----------------------------
def total_conversions(draws: Dict[str, ChannelDraw], spend_map: Dict[str, float]) -> float:
    """Total LEADS given spend_map."""
    return sum(conversions(draws[n], spend_map[n]) for n in spend_map)

def total_wins(draws: Dict[str, ChannelDraw], spend_map: Dict[str, float]) -> float:
    """Total expected WON deals given spend_map."""
    wins = 0.0
    for n, spend in spend_map.items():
        rev_i = revenue(draws[n], spend)
        wins += wins_from_revenue(draws[n], rev_i)
    return wins

def run_simulation(
    total_budget: float = 100_000.0,
    n_sims: int = 200,
    seed: int = 42,
    linked_in_min_share: float = 0.05,
    step: float = 500.0,
    goal_cfg: GoalConfig = GoalConfig(),
    max_share: Optional[Dict[str, float]] = None
):
    rng = np.random.default_rng(seed)
    names = list(CHANNEL_PRIORS.keys())

    spends = {n: [] for n in names}
    revenues, roases, leads_list, wins_list, cpls, cacs_customers = [], [], [], [], [], []

    base_constraints = Constraints(
        min_share={"LinkedIn": linked_in_min_share},
        max_share=max_share
    )

    for _ in range(n_sims):
        # Sample a world
        draws = {n: sample_channel(CHANNEL_PRIORS[n], rng) for n in names}

        # Allocate
        try:
            s = allocate_budget(
                total_budget, draws, base_constraints, step=step, goal_cfg=goal_cfg
            )
        except ValueError:
            # Fallback to proportional min shares
            s = {n: base_constraints.min_share.get(n, 0.0) * total_budget for n in names}
            rem = total_budget - sum(s.values())
            if rem > 0:
                rem_each = rem / len(names)
                for n in names:
                    s[n] += rem_each

        # Outcomes
        rev = sum(revenue(draws[n], s[n]) for n in names)
        leads = total_conversions(draws, s)
        wins = total_wins(draws, s)
        roas = rev / total_budget if total_budget > 0 else 0.0
        cpl  = (total_budget / leads) if leads > 0 else float('inf')
        cac  = (total_budget / wins) if wins  > 0 else float('inf')

        for n in names:
            spends[n].append(s[n])
        revenues.append(rev)
        roases.append(roas)
        leads_list.append(leads)
        wins_list.append(wins)
        cpls.append(cpl)
        cacs_customers.append(cac)

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
    leads_summary = {
        "leads_median": pct(leads_list, 50),
        "leads_p05": pct(leads_list, 5),
        "leads_p95": pct(leads_list, 95),
    }
    wins_summary = {
        "wins_median": pct(wins_list, 50),
        "wins_p05": pct(wins_list, 5),
        "wins_p95": pct(wins_list, 95),
    }
    cpl_summary = {
        "cpl_median": pct(cpls, 50),
        "cpl_p05": pct(cpls, 5),
        "cpl_p95": pct(cpls, 95),
    }
    cac_summary = {
        "cac_customers_median": pct(cacs_customers, 50),
        "cac_customers_p05": pct(cacs_customers, 5),
        "cac_customers_p95": pct(cacs_customers, 95),
    }

    return alloc_summary, revenue_summary, roas_summary, leads_summary, wins_summary, cpl_summary, cac_summary

def explore_allocation_grid(
    total_budget: float,
    n_sims: int = 200,
    seed: int = 42,
    grid_step: float = 0.10,  # 10% increments
    constraints: Constraints = None,
    goal_cfg: GoalConfig = GoalConfig()
) -> Dict[str, Dict]:
    """
    Explore budget allocations on a grid (e.g., 10% steps) and compare performance.
    """
    if constraints is None:
        constraints = Constraints(min_share={})
    names = list(CHANNEL_PRIORS.keys())
    n_channels = len(names)

    # Generate grid allocations summing to 100%
    grid_allocations = []
    step_count = int(1.0 / grid_step)
    for i in range(step_count + 1):
        for j in range(step_count + 1 - i):
            for k in range(step_count + 1 - i - j):
                remaining = step_count - i - j - k
                if remaining >= 0:
                    allocation = [i * grid_step, j * grid_step, k * grid_step, remaining * grid_step]
                    if len(allocation) == n_channels:
                        grid_allocations.append(allocation)

    # Filter valid allocations based on constraints
    valid_allocations = []
    for alloc in grid_allocations:
        valid = True
        for idx, name in enumerate(names):
            if name in constraints.min_share and alloc[idx] < constraints.min_share[name]:
                valid = False; break
            if constraints.max_share and name in constraints.max_share and alloc[idx] > constraints.max_share[name]:
                valid = False; break
        if valid:
            valid_allocations.append(alloc)

    print(f"Exploring {len(valid_allocations)} valid allocation combinations...")

    rng = np.random.default_rng(seed)
    results = {}

    for idx, allocation in enumerate(valid_allocations):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(valid_allocations)}")
        spend_map = {names[i]: allocation[i] * total_budget for i in range(n_channels)}

        revenues, roases, leads_list, wins_list, cpls, cacs = [], [], [], [], [], []
        for _ in range(n_sims):
            draws = {n: sample_channel(CHANNEL_PRIORS[n], rng) for n in names}
            rev = sum(revenue(draws[n], spend_map[n]) for n in names)
            leads = sum(conversions(draws[n], spend_map[n]) for n in names)
            wins = sum(wins_from_revenue(draws[n], revenue(draws[n], spend_map[n])) for n in names)
            roas = rev / total_budget if total_budget > 0 else 0.0
            cpl  = (total_budget / leads) if leads > 0 else float('inf')
            cac  = (total_budget / wins) if wins  > 0 else float('inf')

            revenues.append(rev); roases.append(roas)
            leads_list.append(leads); wins_list.append(wins)
            cpls.append(cpl); cacs.append(cac)

        def pct(x, p): return float(np.percentile(x, p))
        allocation_key = "_".join([f"{a:.1f}" for a in allocation])
        results[allocation_key] = {
            "allocation": {names[i]: allocation[i] for i in range(n_channels)},
            "spend_map": spend_map,
            "revenue": {
                "median": pct(revenues, 50), "p05": pct(revenues, 5), "p95": pct(revenues, 95),
                "mean": float(np.mean(revenues)), "std": float(np.std(revenues))
            },
            "roas": {
                "median": pct(roases, 50), "p05": pct(roases, 5), "p95": pct(roases, 95),
                "mean": float(np.mean(roases)), "std": float(np.std(roases))
            },
            "leads": {
                "median": pct(leads_list, 50), "p05": pct(leads_list, 5), "p95": pct(leads_list, 95),
                "mean": float(np.mean(leads_list)), "std": float(np.std(leads_list))
            },
            "wins": {
                "median": pct(wins_list, 50), "p05": pct(wins_list, 5), "p95": pct(wins_list, 95),
                "mean": float(np.mean(wins_list)), "std": float(np.std(wins_list))
            },
            "cpl": {
                "median": pct(cpls, 50), "p05": pct(cpls, 5), "p95": pct(cpls, 95),
                "mean": float(np.mean(cpls)), "std": float(np.std(cpls))
            },
            "cac_customers": {
                "median": pct(cacs, 50), "p05": pct(cacs, 5), "p95": pct(cacs, 95),
                "mean": float(np.mean(cacs)), "std": float(np.std(cacs))
            }
        }
    return results

def analyze_grid_results(grid_results: Dict, goal_cfg: GoalConfig) -> Dict:
    if goal_cfg.goal == GoalType.REVENUE:
        metric = "revenue"; key = "median"; higher_better = True
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        metric = "leads"; key = "median"; higher_better = True
    elif goal_cfg.goal == GoalType.PROFIT:
        metric = "revenue"; key = "median"; higher_better = True
    else:
        metric = "revenue"; key = "median"; higher_better = True

    sorted_results = sorted(
        grid_results.items(),
        key=lambda x: x[1][metric][key],
        reverse=higher_better
    )
    top_5 = sorted_results[:5]

    risk_ranked = sorted(
        grid_results.items(),
        key=lambda x: x[1][metric]["std"] / x[1][metric]["mean"] if x[1][metric]["mean"] > 0 else float('inf')
    )
    conservative_top_5 = risk_ranked[:5]

    analysis = {
        "top_performers": top_5,
        "conservative_choices": conservative_top_5,
        "total_combinations": len(grid_results),
        "goal_metric": metric,
        "recommendations": {
            "best_overall": top_5[0] if top_5 else None,
            "best_conservative": conservative_top_5[0] if conservative_top_5 else None,
            "risk_tolerance": "Choose from top_performers for aggressive, conservative_choices for stability"
        }
    }
    return analysis

def print_grid_analysis(analysis: Dict, total_budget: float):
    print(f"\n=== GRID EXPLORATION ANALYSIS ===")
    print(f"Explored {analysis['total_combinations']} allocation combinations")
    print(f"Goal metric: {analysis['goal_metric']}")

    print(f"\n--- TOP 5 PERFORMERS ---")
    for i, (key, result) in enumerate(analysis['top_performers']):
        alloc = result['allocation']
        perf = result[analysis['goal_metric']]
        roas = result['roas']
        leads = result['leads']
        wins = result['wins']
        cpl = result['cpl']
        cac = result['cac_customers']
        
        print(f"{i+1}. {key}: {perf['median']:,.0f} [p05: {perf['p05']:,.0f}, p95: {perf['p95']:,.0f}]")
        print(f"   ROAS: {roas['median']:.2f}x [p05: {roas['p05']:.2f}x, p95: {roas['p95']:.2f}x]")
        print(f"   Leads: {leads['median']:,.1f} [p05: {leads['p05']:,.1f}, p95: {leads['p95']:,.1f}]")
        print(f"   Wins: {wins['median']:,.1f} [p05: {wins['p05']:,.1f}, p95: {wins['p95']:,.1f}]")
        print(f"   CPL: ${cpl['median']:,.2f} [p05: ${cpl['p05']:,.2f}, p95: ${cpl['p95']:,.2f}]")
        print(f"   CAC: ${cac['median']:,.2f} [p05: ${cac['p05']:,.2f}, p95: ${cac['p95']:,.2f}]")
        for name, pct in alloc.items():
            print(f"   {name}: {pct*100:.1f}% (${pct*total_budget:,.0f})")
        print()

    print(f"\n--- TOP 5 CONSERVATIVE (LOW RISK) ---")
    for i, (key, result) in enumerate(analysis['conservative_choices']):
        alloc = result['allocation']
        perf = result[analysis['goal_metric']]
        roas = result['roas']
        leads = result['leads']
        wins = result['wins']
        cpl = result['cpl']
        cac = result['cac_customers']
        risk = perf['std'] / perf['mean'] if perf['mean'] > 0 else float('inf')
        
        print(f"{i+1}. {key}: {perf['median']:,.0f} (Risk: {risk:.3f})")
        print(f"   ROAS: {roas['median']:.2f}x [p05: {roas['p05']:.2f}x, p95: {roas['p95']:.2f}x]")
        print(f"   Leads: {leads['median']:,.1f} [p05: {leads['p05']:,.1f}, p95: {leads['p95']:,.1f}]")
        print(f"   Wins: {wins['median']:,.1f} [p05: {wins['p05']:,.1f}, p95: {wins['p95']:,.1f}]")
        print(f"   CPL: ${cpl['median']:,.2f} [p05: ${cpl['p05']:,.2f}, p95: ${cpl['p95']:,.2f}]")
        print(f"   CAC: ${cac['median']:,.2f} [p05: ${cac['p05']:,.2f}, p95: ${cac['p95']:,.2f}]")
        for name, pct in alloc.items():
            print(f"   {name}: {pct*100:.1f}% (${pct*total_budget:,.0f})")
        print()

def print_grid_summary_stats(grid_results: Dict):
    """Print summary statistics across all grid combinations"""
    print(f"\n--- GRID SUMMARY STATISTICS (Across All {len(grid_results)} Combinations) ---")
    
    # Collect all values for each metric
    all_roas = [result['roas']['median'] for result in grid_results.values()]
    all_leads = [result['leads']['median'] for result in grid_results.values()]
    all_wins = [result['wins']['median'] for result in grid_results.values()]
    all_cpl = [result['cpl']['median'] for result in grid_results.values()]
    all_cac = [result['cac_customers']['median'] for result in grid_results.values()]
    
    # Calculate summary stats
    def summary_stats(values, metric_name, format_func=lambda x: f"{x:.2f}"):
        if not values:
            return
        values = [v for v in values if np.isfinite(v)]  # Filter out inf values
        if not values:
            return
        
        min_val, max_val = min(values), max(values)
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        
        print(f"{metric_name:15s}: {format_func(median_val)} (min: {format_func(min_val)}, max: {format_func(max_val)}, std: {format_func(std_val)})")
    
    summary_stats(all_roas, "ROAS", lambda x: f"{x:.2f}x")
    summary_stats(all_leads, "Leads", lambda x: f"{x:,.1f}")
    summary_stats(all_wins, "Wins", lambda x: f"{x:,.1f}")
    summary_stats(all_cpl, "CPL", lambda x: f"${x:,.2f}")
    summary_stats(all_cac, "CAC", lambda x: f"${x:,.2f}")
    
    # Show best and worst performers for each metric
    print(f"\n--- BEST/WORST PERFORMERS BY METRIC ---")
    
    # ROAS
    best_roas = max(grid_results.items(), key=lambda x: x[1]['roas']['median'])
    worst_roas = min(grid_results.items(), key=lambda x: x[1]['roas']['median'])
    print(f"Best ROAS:  {best_roas[0]} → {best_roas[1]['roas']['median']:.2f}x")
    print(f"Worst ROAS: {worst_roas[0]} → {worst_roas[1]['roas']['median']:.2f}x")
    
    # Leads
    best_leads = max(grid_results.items(), key=lambda x: x[1]['leads']['median'])
    worst_leads = min(grid_results.items(), key=lambda x: x[1]['leads']['median'])
    print(f"Best Leads: {best_leads[0]} → {best_leads[1]['leads']['median']:,.1f}")
    print(f"Worst Leads:{worst_leads[0]} → {worst_leads[1]['leads']['median']:,.1f}")
    
    # Wins
    best_wins = max(grid_results.items(), key=lambda x: x[1]['wins']['median'])
    worst_wins = min(grid_results.items(), key=lambda x: x[1]['wins']['median'])
    print(f"Best Wins:  {best_wins[0]} → {best_wins[1]['wins']['median']:,.1f}")
    print(f"Worst Wins: {worst_wins[0]} → {worst_wins[1]['wins']['median']:,.1f}")
    
    # CPL (lower is better)
    best_cpl = min(grid_results.items(), key=lambda x: x[1]['cpl']['median'])
    worst_cpl = max(grid_results.items(), key=lambda x: x[1]['cpl']['median'])
    print(f"Best CPL:   {best_cpl[0]} → ${best_cpl[1]['cpl']['median']:,.2f}")
    print(f"Worst CPL:  {worst_cpl[0]} → ${worst_cpl[1]['cpl']['median']:,.2f}")
    
    # CAC (lower is better)
    best_cac = min(grid_results.items(), key=lambda x: x[1]['cac_customers']['median'])
    worst_cac = max(grid_results.items(), key=lambda x: x[1]['cac_customers']['median'])
    print(f"Best CAC:   {best_cac[0]} → ${best_cac[1]['cac_customers']['median']:,.2f}")
    print(f"Worst CAC:  {worst_cac[0]} → ${worst_cac[1]['cac_customers']['median']:,.2f}")

def export_grid_results_summary(grid_results: Dict, filename: str = None) -> Dict:
    """Export grid results in a structured format for analysis"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grid_results_summary_{timestamp}.json"
    
    summary_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_combinations": len(grid_results),
            "export_tool": "modelling_adv_goals2"
        },
        "summary_stats": {},
        "top_performers": {},
        "all_combinations": {}
    }
    
    # Calculate summary statistics
    all_roas = [result['roas']['median'] for result in grid_results.values()]
    all_leads = [result['leads']['median'] for result in grid_results.values()]
    all_wins = [result['wins']['median'] for result in grid_results.values()]
    all_cpl = [result['cpl']['median'] for result in grid_results.values()]
    all_cac = [result['cac_customers']['median'] for result in grid_results.values()]
    
    def calc_summary_stats(values, metric_name):
        values = [v for v in values if np.isfinite(v)]
        if not values:
            return None
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values))
        }
    
    summary_data["summary_stats"] = {
        "roas": calc_summary_stats(all_roas, "ROAS"),
        "leads": calc_summary_stats(all_leads, "Leads"),
        "wins": calc_summary_stats(all_wins, "Wins"),
        "cpl": calc_summary_stats(all_cpl, "CPL"),
        "cac": calc_summary_stats(all_cac, "CAC")
    }
    
    # Find top performers by each metric
    def find_top_by_metric(metric, key='median', reverse=True):
        valid_results = [(k, v) for k, v in grid_results.items() 
                        if np.isfinite(v[metric][key])]
        if not valid_results:
            return []
        sorted_results = sorted(valid_results, 
                              key=lambda x: x[1][metric][key], 
                              reverse=reverse)
        return sorted_results[:5]
    
    summary_data["top_performers"] = {
        "by_roas": [{"allocation": k, "value": v['roas']['median']} 
                    for k, v in find_top_by_metric('roas')],
        "by_leads": [{"allocation": k, "value": v['leads']['median']} 
                     for k, v in find_top_by_metric('leads')],
        "by_wins": [{"allocation": k, "value": v['wins']['median']} 
                    for k, v in find_top_by_metric('wins')],
        "by_cpl": [{"allocation": k, "value": v['cpl']['median']} 
                   for k, v in find_top_by_metric('cpl', reverse=False)],  # Lower is better
        "by_cac": [{"allocation": k, "value": v['cac_customers']['median']} 
                   for k, v in find_top_by_metric('cac_customers', reverse=False)]  # Lower is better
    }
    
    # Export all combinations with key metrics
    for key, result in grid_results.items():
        summary_data["all_combinations"][key] = {
            "allocation": result['allocation'],
            "metrics": {
                "revenue": result['revenue']['median'],
                "roas": result['roas']['median'],
                "leads": result['leads']['median'],
                "wins": result['wins']['median'],
                "cpl": result['cpl']['median'],
                "cac": result['cac_customers']['median']
            }
        }
    
    # Save to JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Grid results summary exported to: {filename}")
    except Exception as e:
        print(f"❌ Failed to export grid results: {e}")
    
    return summary_data

# -----------------------------
# Demo helpers
# -----------------------------
def print_alloc(title: str, total_budget: float, alloc_summary: Dict[str, Dict[str, float]]):
    print(f"\n=== {title} — Allocation (Median with 5–95% bands) ===")
    for n, s in alloc_summary.items():
        print(
            f"{n:9s}: "
            f"${s['median_$']:,.0f} "
            f"(p05 ${s['p05_$']:,.0f}, p95 ${s['p95_$']:,.0f})  "
            f"{s['median_%']:.1f}% "
            f"(p05 {s['p05_%']:.1f}%, p95 {s['p95_%']:.1f}%)"
        )

def print_outcomes(rev_sum, roas_sum, leads_sum, wins_sum, cpl_sum, cac_sum):
    print("\n--- Outcomes ---")
    print(f"Revenue (median):   ${rev_sum['revenue_median']:,.0f} "
          f"[{rev_sum['revenue_p05']:,.0f} … {rev_sum['revenue_p95']:,.0f}]")
    print(f"ROAS (median):      {roas_sum['roas_median']:.2f} "
          f"[{roas_sum['roas_p05']:.2f} … {roas_sum['roas_p95']:.2f}]")
    print(f"Leads (median):     {leads_sum['leads_median']:,.1f} "
          f"[{leads_sum['leads_p05']:,.1f} … {leads_sum['leads_p95']:,.1f}]")
    print(f"Wins (median):      {wins_sum['wins_median']:,.1f} "
          f"[{wins_sum['wins_p05']:,.1f} … {wins_sum['wins_p95']:,.1f}]")
    print(f"CPL (median):       ${cpl_sum['cpl_median']:,.2f} "
          f"[{cpl_sum['cpl_p05']:,.2f} … {cpl_sum['cpl_p95']:,.2f}]")
    print(f"CAC (customers):    ${cac_sum['cac_customers_median']:,.2f} "
          f"[{cac_sum['cac_customers_p05']:,.2f} … {cac_sum['cac_customers_p95']:,.2f}]")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    total_budget = 100_000.0

    # 1) Maximize REVENUE (baseline)
    alloc, rev_sum, roas_sum, leads_sum, wins_sum, cpl_sum, cac_sum = run_simulation(
        total_budget=total_budget,
        n_sims=300,
        seed=123,
        linked_in_min_share=0.20,
        step=500.0,
        goal_cfg=GoalConfig(goal=GoalType.REVENUE)
    )
    print_alloc("Goal: REVENUE", total_budget, alloc)
    print_outcomes(rev_sum, roas_sum, leads_sum, wins_sum, cpl_sum, cac_sum)

    # # 2) Maximize CONVERSIONS (≈ maximize leads)
    # alloc2, rev2, roas2, leads2, wins2, cpl2, cac2 = run_simulation(
    #     total_budget=total_budget,
    #     n_sims=300,
    #     seed=123,
    #     linked_in_min_share=0.20,
    #     step=500.0,
    #     goal_cfg=GoalConfig(goal=GoalType.CONVERSIONS)
    # )
    # print_alloc("Goal: CONVERSIONS", total_budget, alloc2)
    # print_outcomes(rev2, roas2, leads2, wins2, cpl2, cac2)

    # # 3) Maximize PROFIT with 60% gross margin and ROAS floor 2.0, plus a TikTok cap
    # alloc3, rev3, roas3, leads3, wins3, cpl3, cac3 = run_simulation(
    #     total_budget=total_budget,
    #     n_sims=300,
    #     seed=123,
    #     linked_in_min_share=0.20,
    #     step=500.0,
    #     goal_cfg=GoalConfig(goal=GoalType.PROFIT, gross_margin=0.6, roas_floor=2.0),
    #     max_share={"TikTok": 0.5}
    # )
    # print_alloc("Goal: PROFIT (GM=60%) with ROAS floor 2.0", total_budget, alloc3)
    # print_outcomes(rev3, roas3, leads3, wins3, cpl3, cac3)

    # 4) GRID EXPLORATION — PROFIT optimization (example)
    print("\n" + "="*60)
    print("GRID EXPLORATION: PROFIT optimization")
    print("="*60)
    grid_results_profit = explore_allocation_grid(
        total_budget=total_budget,
        n_sims=200,
        seed=123,
        grid_step=0.10,
        constraints=Constraints(min_share={"LinkedIn": 0.20}),
        goal_cfg=GoalConfig(goal=GoalType.PROFIT, gross_margin=0.6)
    )
    analysis_profit = analyze_grid_results(grid_results_profit, GoalConfig(goal=GoalType.PROFIT))
    print_grid_analysis(analysis_profit, total_budget)
    print_grid_summary_stats(grid_results_profit)
    export_grid_results_summary(grid_results_profit)
