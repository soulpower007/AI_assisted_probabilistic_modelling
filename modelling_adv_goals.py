#!/usr/bin/env python3
"""
Marketing Budget Allocator (Monte Carlo + Saturation + Goal Switches)
- Priors: CPM (lognormal), CTR/CVR (beta), AOV (lognormal), Quality multiplier
- Response curve: conversions_i(S) = theta_i * (1 - exp(-beta_i * S))
  where early slope at S=0 equals base_i = (1000/CPM) * CTR * CVR * quality_adj
  => K_BETA * base / theta with K_BETA = 5.0
- Revenue_i(S) = conversions_i(S) * AOV_i
- Greedy allocation by a marginal "score" per goal:
    REVENUE:   dR/dS = base_i * exp(-beta_i * S) * AOV_i
    CONVERSIONS: dC/dS = base_i * exp(-beta_i * S)
    PROFIT: per-$ profit increment = gross_margin * (dR/dS) - 1
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
# Visualization functions moved to modelling_visualizations.py

# Visualization functions moved to modelling_visualizations.py

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
        # Default to config/channel_priors.json in the same directory as this script
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
    
    # Convert ChannelPriors objects to dictionaries
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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save to JSON
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
    
    # # Apply quality: cheaper CPM, higher CTR/CVR (mildly)
    # cpm_eff = cpm / max(q, 1e-6)
    # ctr_eff = min(ctr * (q ** 0.5), 0.999999)
    # cvr_eff = min(cvr * (q ** 0.5), 0.999999)

    cpm_eff = cpm * (q ** 0.5)
    ctr_eff = min(ctr * (q ** 0.2), 0.5)      # clamp hard; CTRs shouldn’t explode
    cvr_eff = min(cvr * (q ** 0.2), 0.2)      # clamp hard; B2B total CVR to sale is tiny

    # AOV (lognormal)
    aov = float(rng.lognormal(mean=priors.aov_log_mu, sigma=priors.aov_log_sigma))
    
        # === NEW (lead -> SQL -> win) deep-funnel ===
    # If your CVR is click->lead, convert ACV (sale) into "revenue per lead"
    p_lead_to_sql = 0.25
    p_sql_to_win  = 0.15
    effective_aov_per_lead = aov * p_lead_to_sql * p_sql_to_win
    aov_effective = effective_aov_per_lead
    # ============================================


    # ============================================

    # Theta cap (lognormal)
    theta = float(rng.lognormal(mean=priors.theta_log_mu, sigma=priors.theta_log_sigma))

    # base = (1000 / CPM) * CTR * CVR  (this is leads per $ if CVR is click->lead)
    base = (1000.0 / max(cpm_eff, 1e-6)) * ctr_eff * cvr_eff

    # === UPDATED beta (see section 2) ===
    K_BETA = 5.0
    beta = K_BETA * base / max(theta, 1e-9)

    return ChannelDraw(
        name=priors.name, cpm=cpm_eff, ctr=ctr_eff, cvr=cvr_eff,
        aov=aov_effective, theta=theta, base=base, beta=beta
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

def marginal_conversions(draw: ChannelDraw, spend: float) -> float:
    # dC/dS = base * exp(-beta S)
    return draw.base * np.exp(-draw.beta * spend)

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
    # per-$ profit increment = gross_margin * dR/dS - 1 (you pay $1 for each $ spent)
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

    steps = int(np.floor(remaining / step + 1e-9))

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

                # Optional ROAS floor guard: simulate Δrev from this step
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

        # Apply step to best, update totals (rev uses revenue marginal)
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
    return sum(conversions(draws[n], spend_map[n]) for n in spend_map)

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
    revenues, roases, convs, cacs = [], [], [], []

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
        conv = total_conversions(draws, s)
        roas = rev / total_budget if total_budget > 0 else 0.0
        cac = (total_budget / conv) if conv > 0 else float('inf')

        for n in names:
            spends[n].append(s[n])
        revenues.append(rev)
        roases.append(roas)
        convs.append(conv)
        cacs.append(cac)

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
    conv_summary = {
        "conv_median": pct(convs, 50),
        "conv_p05": pct(convs, 5),
        "conv_p95": pct(convs, 95),
    }
    cac_summary = {
        "cac_median": pct(cacs, 50),
        "cac_p05": pct(cacs, 5),
        "cac_p95": pct(cacs, 95),
    }

    return alloc_summary, revenue_summary, roas_summary, conv_summary, cac_summary

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
    
    Args:
        grid_step: Allocation increment (0.10 = 10% steps)
        constraints: Min/max share constraints
        goal_cfg: Optimization goal configuration
    
    Returns:
        Dictionary with grid results and confidence intervals
    """
    if constraints is None:
        constraints = Constraints(min_share={})
    
    names = list(CHANNEL_PRIORS.keys())
    n_channels = len(names)
    
    # Generate grid allocations
    grid_allocations = []
    step_count = int(1.0 / grid_step)
    
    # Generate all possible combinations that sum to 100%
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
        for i, name in enumerate(names):
            if name in constraints.min_share and alloc[i] < constraints.min_share[name]:
                valid = False
                break
            if constraints.max_share and name in constraints.max_share and alloc[i] > constraints.max_share[name]:
                valid = False
                break
        if valid:
            valid_allocations.append(alloc)
    
    print(f"Exploring {len(valid_allocations)} valid allocation combinations...")
    
    # Run Monte Carlo simulation for each allocation
    rng = np.random.default_rng(seed)
    results = {}
    
    for idx, allocation in enumerate(valid_allocations):
        if idx % 10 == 0:  # Progress indicator
            print(f"  Progress: {idx}/{len(valid_allocations)}")
        
        # Convert percentages to dollar amounts
        spend_map = {names[i]: allocation[i] * total_budget for i in range(n_channels)}
        
        # Run Monte Carlo for this allocation
        revenues, roases, convs, cacs = [], [], [], []
        
        for _ in range(n_sims):
            # Sample channel parameters
            draws = {n: sample_channel(CHANNEL_PRIORS[n], rng) for n in names}
            
            # Compute outcomes for this allocation
            rev = sum(revenue(draws[n], spend_map[n]) for n in names)
            conv = sum(conversions(draws[n], spend_map[n]) for n in names)
            roas = rev / total_budget if total_budget > 0 else 0.0
            cac = (total_budget / conv) if conv > 0 else float('inf')
            
            revenues.append(rev)
            roases.append(roas)
            convs.append(conv)
            cacs.append(cac)
        
        # Store results with confidence intervals
        def pct(x, p): return float(np.percentile(x, p))
        
        allocation_key = "_".join([f"{a:.1f}" for a in allocation])
        results[allocation_key] = {
            "allocation": {names[i]: allocation[i] for i in range(n_channels)},
            "spend_map": spend_map,
            "revenue": {
                "median": pct(revenues, 50),
                "p05": pct(revenues, 5),
                "p95": pct(revenues, 95),
                "mean": float(np.mean(revenues)),
                "std": float(np.std(revenues))
            },
            "roas": {
                "median": pct(roases, 50),
                "p05": pct(roases, 5),
                "p95": pct(roases, 95),
                "mean": float(np.mean(roases)),
                "std": float(np.std(roases))
            },
            "conversions": {
                "median": pct(convs, 50),
                "p05": pct(convs, 5),
                "p95": pct(convs, 95),
                "mean": float(np.mean(convs)),
                "std": float(np.std(convs))
            },
            "cac": {
                "median": pct(cacs, 50),
                "p05": pct(cacs, 5),
                "p95": pct(cacs, 95),
                "mean": float(np.mean(cacs)),
                "std": float(np.std(cacs))
            }
        }
    
    return results

def analyze_grid_results(grid_results: Dict, goal_cfg: GoalConfig) -> Dict:
    """
    Analyze grid exploration results to find optimal allocations.
    
    Args:
        grid_results: Output from explore_allocation_grid
        goal_cfg: Goal configuration to determine ranking
    
    Returns:
        Analysis with rankings and recommendations
    """
    # Rank allocations by goal
    if goal_cfg.goal == GoalType.REVENUE:
        metric = "revenue"
        key = "median"
        higher_better = True
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        metric = "conversions"
        key = "median"
        higher_better = True
    elif goal_cfg.goal == GoalType.PROFIT:
        metric = "revenue"  # Use revenue as proxy for profit
        key = "median"
        higher_better = True
    else:
        metric = "revenue"
        key = "median"
        higher_better = True
    
    # Sort by performance
    sorted_results = sorted(
        grid_results.items(),
        key=lambda x: x[1][metric][key],
        reverse=higher_better
    )
    
    # Top 5 allocations
    top_5 = sorted_results[:5]
    
    # Risk analysis (volatility)
    risk_ranked = sorted(
        grid_results.items(),
        key=lambda x: x[1][metric]["std"] / x[1][metric]["mean"] if x[1][metric]["mean"] > 0 else float('inf')
    )
    
    # Conservative (low risk) top 5
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
    """Print grid exploration analysis results."""
    print(f"\n=== GRID EXPLORATION ANALYSIS ===")
    print(f"Explored {analysis['total_combinations']} allocation combinations")
    print(f"Goal metric: {analysis['goal_metric']}")
    
    print(f"\n--- TOP 5 PERFORMERS ---")
    for i, (key, result) in enumerate(analysis['top_performers']):
        alloc = result['allocation']
        perf = result[analysis['goal_metric']]
        roas = result['roas']
        print(f"{i+1}. {key}: {perf['median']:,.0f} [p05: {perf['p05']:,.0f}, p95: {perf['p95']:,.0f}]")
        print(f"   ROAS: {roas['median']:.2f}x [p05: {roas['p05']:.2f}x, p95: {roas['p95']:.2f}x]")
        for name, pct in alloc.items():
            print(f"   {name}: {pct*100:.1f}% (${pct*total_budget:,.0f})")
        print()
    
    print(f"\n--- TOP 5 CONSERVATIVE (LOW RISK) ---")
    for i, (key, result) in enumerate(analysis['conservative_choices']):
        alloc = result['allocation']
        perf = result[analysis['goal_metric']]
        roas = result['roas']
        risk = perf['std'] / perf['mean'] if perf['mean'] > 0 else float('inf')
        print(f"{i+1}. {key}: {perf['median']:,.0f} (Risk: {risk:.3f})")
        print(f"   ROAS: {roas['median']:.2f}x [p05: {roas['p05']:.2f}x, p95: {roas['p95']:.2f}x]")
        for name, pct in alloc.items():
            print(f"   {name}: {pct*100:.1f}% (${pct*total_budget:,.0f})")
        print()

# Visualization functions moved to modelling_visualizations.py

# -----------------------------
# Demo
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

def print_outcomes(rev_sum, roas_sum, conv_sum, cac_sum):
    print("\n--- Outcomes ---")
    print(f"Revenue (median):   ${rev_sum['revenue_median']:,.0f} "
          f"[{rev_sum['revenue_p05']:,.0f} … {rev_sum['revenue_p95']:,.0f}]")
    print(f"ROAS (median):      {roas_sum['roas_median']:.2f} "
          f"[{roas_sum['roas_p05']:.2f} … {roas_sum['roas_p95']:.2f}]")
    print(f"Conversions (med):  {conv_sum['conv_median']:,.1f} "
          f"[{conv_sum['conv_p05']:,.1f} … {conv_sum['conv_p95']:,.1f}]")
    print(f"CAC (median):       ${cac_sum['cac_median']:,.2f} "
          f"[{cac_sum['cac_p05']:,.2f} … {cac_sum['cac_p95']:,.2f}]")

if __name__ == "__main__":
    total_budget = 100_000.0

    # 1) Maximize REVENUE (baseline)
    alloc, rev_sum, roas_sum, conv_sum, cac_sum = run_simulation(
        total_budget=total_budget,
        n_sims=300,
        seed=123,
        linked_in_min_share=0.20,
        step=500.0,
        goal_cfg=GoalConfig(goal=GoalType.REVENUE)
    )
    print_alloc("Goal: REVENUE", total_budget, alloc)
    print_outcomes(rev_sum, roas_sum, conv_sum, cac_sum)

    # 2) Maximize CONVERSIONS (≈ minimize CAC for fixed budget)
    # alloc2, rev2, roas2, conv2, cac2 = run_simulation( 
    #     total_budget=total_budget,
    #     n_sims=300,
    #     seed=123,
    #     linked_in_min_share=0.20,
    #     step=500.0,
    #     goal_cfg=GoalConfig(goal=GoalType.CONVERSIONS)
    # )
    # print_alloc("Goal: CONVERSIONS", total_budget, alloc2)
    # print_outcomes(rev2, roas2, conv2, cac2)

    # 3) Maximize PROFIT with 60% gross margin and ROAS floor 2.0, plus a TikTok cap
    # alloc3, rev3, roas3, conv3, cac3 = run_simulation(
    #     total_budget=total_budget,
    #     n_sims=300,
    #     seed=123,
    #     linked_in_min_share=0.20,
    #     step=500.0,
    #     goal_cfg=GoalConfig(goal=GoalType.PROFIT, gross_margin=0.6, roas_floor=2.0),
    #     max_share={"TikTok": 0.5}  # example cap
    # )
    # print_alloc("Goal: PROFIT (GM=60%) with ROAS floor 2.0", total_budget, alloc3)
    # print_outcomes(rev3, roas3, conv3, cac3)

    # Save channel priors to config file
    # save_channel_priors(CHANNEL_PRIORS, config_path="config/channel_priors.json")
    
    # 4) GRID EXPLORATION - Compare different allocation splits systematically
    # print("\n" + "="*60)
    # print("GRID EXPLORATION: Comparing 10% allocation increments")
    # print("="*60)
    
    # grid_results = explore_allocation_grid(
    #     total_budget=total_budget,
    #     n_sims=200,
    #     seed=123,
    #     grid_step=0.10,  # 10% increments
    #     constraints=Constraints(min_share={"LinkedIn": 0.20}),
    #     goal_cfg=GoalConfig(goal=GoalType.REVENUE)
    # )
    
    # analysis = analyze_grid_results(grid_results, GoalConfig(goal=GoalType.REVENUE))
    # print_grid_analysis(analysis, total_budget)
    
    # 5) GRID EXPLORATION with different goals
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
    
    # # Plot results for grid exploration using visualization module
    # from modelling_visualizations import generate_all_visualizations
    # generate_all_visualizations(grid_results, grid_results_profit, total_budget)