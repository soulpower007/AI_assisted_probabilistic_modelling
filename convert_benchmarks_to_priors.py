#!/usr/bin/env python3
"""
Convert marketing benchmarks (p10/p50/p90) to channel priors (mu/sigma) for modelling_adv_goals.py

This script reads the marketing_benchmarks_config.json file (or latest marketing_benchmarks_*.json)
and converts the percentile ranges to lognormal (mu, sigma) and beta (mean, std) parameters
for use in the Monte Carlo simulation.
"""

import json
import math
import os
import glob
from typing import Dict, Any, Tuple
import numpy as np

# Constants for quantile conversion
Z_90 = 1.2815515655446004
Z_10 = -1.2815515655446004
INTERDECILE_Z = (Z_90 - Z_10)  # ~2.563103131089201

def _to_ratio(x):
    """
    Accepts float (already ratio), or strings like '0.90%', '1.5 %', '0.009'
    Returns a float ratio in [0,1].
    """
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)

def fit_lognormal_from_quantiles(p10: float, p50: float, p90: float) -> Tuple[float, float]:
    """
    For X~LogNormal(mu, sigma): Q(p)=exp(mu + sigma*z_p)
    => sigma = [ln(Q90)-ln(Q10)] / (z90-z10)
       mu    = ln(Q50)  (since median = exp(mu))
    """
    p10 = max(p10, 1e-12)
    p50 = max(p50, 1e-12)
    p90 = max(p90, 1e-12)
    ln10, ln50, ln90 = math.log(p10), math.log(p50), math.log(p90)
    sigma = (ln90 - ln10) / INTERDECILE_Z
    mu = ln50  # median anchor
    # clamp sigma to a reasonable positive range
    sigma = max(sigma, 1e-6)
    return mu, sigma

def approx_mean_std_from_interdecile(p10: float, p50: float, p90: float) -> Tuple[float, float]:
    """
    Use median as mean approx, and interdecile span to back out std assuming ~symmetric beta in [0,1].
    std ≈ (p90 - p10) / (z_0.9 - z_0.1) = (p90 - p10)/2.563...
    """
    mean = float(p50)
    std = max((p90 - p10) / INTERDECILE_Z, 1e-6)
    # clip to [0,1] support
    mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
    # ensure std not bigger than max feasible for a beta with that mean
    max_var = mean * (1 - mean)
    std = float(min(std, 0.99 * math.sqrt(max_var) if max_var > 0 else 1e-6))
    return mean, std

def find_latest_benchmarks_file() -> str:
    """Find the latest marketing benchmarks file."""
    # First try the config file
    config_file = "marketing_benchmarks_config.json"
    if os.path.exists(config_file):
        return config_file
    
    # Otherwise find the latest timestamped file
    pattern = "marketing_benchmarks_*.json"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No marketing benchmarks file found")
    
    # Sort by filename (timestamp) and take the latest
    files.sort(reverse=True)
    return files[0]

def load_benchmarks_data(filename: str) -> Dict[str, Any]:
    """Load benchmarks data from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def convert_benchmarks_to_priors(benchmarks_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert benchmarks data to channel priors format.
    
    Args:
        benchmarks_data: The loaded JSON data from marketing benchmarks file
        
    Returns:
        Dictionary in channel_priors.json format
    """
    channels_data = benchmarks_data["benchmarks"]["channels"]
    priors = {}
    
    for channel_data in channels_data:
        channel_name = channel_data["channel"]
        
        # Extract percentile data
        cpm_data = channel_data["cpm_usd_per_1000"]
        ctr_data = channel_data["ctr_ratio"]
        cvr_data = channel_data["cvr_ratio"]
        aov_data = channel_data["aov_usd"]
        
        # Convert CPM (USD per 1000) → LogNormal(mu, sigma)
        cpm_p10 = float(cpm_data["p10"])
        cpm_p50 = float(cpm_data["p50"])
        cpm_p90 = float(cpm_data["p90"])
        cpm_mu, cpm_sigma = fit_lognormal_from_quantiles(cpm_p10, cpm_p50, cpm_p90)
        
        # Convert CTR (ratio) → Beta(mean, std)
        ctr_p10 = _to_ratio(ctr_data["p10"])
        ctr_p50 = _to_ratio(ctr_data["p50"])
        ctr_p90 = _to_ratio(ctr_data["p90"])
        ctr_mean, ctr_std = approx_mean_std_from_interdecile(ctr_p10, ctr_p50, ctr_p90)
        
        # Convert CVR (ratio) → Beta(mean, std)
        cvr_p10 = _to_ratio(cvr_data["p10"])
        cvr_p50 = _to_ratio(cvr_data["p50"])
        cvr_p90 = _to_ratio(cvr_data["p90"])
        cvr_mean, cvr_std = approx_mean_std_from_interdecile(cvr_p10, cvr_p50, cvr_p90)
        
        # Convert AOV (USD) → LogNormal(mu, sigma)
        aov_p10 = float(aov_data["p10"])
        aov_p50 = float(aov_data["p50"])
        aov_p90 = float(aov_data["p90"])
        aov_mu, aov_sigma = fit_lognormal_from_quantiles(aov_p10, aov_p50, aov_p90)
        
        # Create channel priors entry (keeping default values for quality and theta)
        priors[channel_name] = {
            "name": channel_name,
            "cpm_log_mu": round(cpm_mu, 6),
            "cpm_log_sigma": round(cpm_sigma, 6),
            "ctr_mean": round(ctr_mean, 6),
            "ctr_std": round(ctr_std, 6),
            "cvr_mean": round(cvr_mean, 6),
            "cvr_std": round(cvr_std, 6),
            "quality_log_mu": 0.0,      # Default: ln(1.0)
            "quality_log_sigma": 0.1,   # Default: ~10% sigma
            "theta_log_mu": math.log(5000.0),  # Default: max conversions
            "theta_log_sigma": 0.3,     # Default
            "aov_log_mu": round(aov_mu, 6),
            "aov_log_sigma": round(aov_sigma, 6)
        }
        
        print(f"Converted {channel_name}:")
        print(f"  CPM: p10={cpm_p10}, p50={cpm_p50}, p90={cpm_p90} → mu={cpm_mu:.3f}, sigma={cpm_sigma:.3f}")
        print(f"  CTR: p10={ctr_p10:.4f}, p50={ctr_p50:.4f}, p90={ctr_p90:.4f} → mean={ctr_mean:.4f}, std={ctr_std:.4f}")
        print(f"  CVR: p10={cvr_p10:.4f}, p50={cvr_p50:.4f}, p90={cvr_p90:.4f} → mean={cvr_mean:.4f}, std={cvr_std:.4f}")
        print(f"  AOV: p10={aov_p10}, p50={aov_p50}, p90={aov_p90} → mu={aov_mu:.3f}, sigma={aov_sigma:.3f}")
        print()
    
    return priors

def save_channel_priors(priors: Dict[str, Dict[str, Any]], output_path: str = None):
    """Save the converted priors to channel_priors.json."""
    if output_path is None:
        output_path = os.path.join("config", "channel_priors.json")
    
    # Ensure config directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(priors, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Channel priors saved to: {output_path}")

def main():
    """Main conversion function."""
    try:
        # Find and load the benchmarks file
        benchmarks_file = find_latest_benchmarks_file()
        print(f"📊 Loading benchmarks from: {benchmarks_file}")
        
        benchmarks_data = load_benchmarks_data(benchmarks_file)
        print(f"🏢 Company: {benchmarks_data['benchmarks']['company']}")
        print(f"🌍 Geography: {benchmarks_data['benchmarks']['geography']}")
        print(f"🏭 Industry: {benchmarks_data['benchmarks']['industry']}")
        print()
        
        # Convert to priors format
        print("🔄 Converting percentiles to distribution parameters...")
        priors = convert_benchmarks_to_priors(benchmarks_data)
        
        # Save to config file
        save_channel_priors(priors)
        
        print("🎉 Conversion completed successfully!")
        print(f"📁 Updated {len(priors)} channels in config/channel_priors.json")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise

if __name__ == "__main__":
    main()