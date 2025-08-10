#!/usr/bin/env python3
"""
Example usage of the channel priors config system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling_adv_goals import load_channel_priors, save_channel_priors, ChannelPriors
import numpy as np

def example_read_config():
    """Example: Read channel priors from config."""
    print("=== Reading Channel Priors from Config ===")
    priors = load_channel_priors()
    
    for channel_name, channel_prior in priors.items():
        print(f"\n{channel_name}:")
        print(f"  CPM log(μ): {channel_prior.cpm_log_mu:.3f}")
        print(f"  CTR mean: {channel_prior.ctr_mean:.3f}")
        print(f"  CVR mean: {channel_prior.cvr_mean:.3f}")

def example_update_config():
    """Example: Update and save channel priors."""
    print("\n=== Updating Channel Priors ===")
    
    # Load existing config
    priors = load_channel_priors()
    
    # Update Google's CTR mean
    if "Google" in priors:
        old_ctr = priors["Google"].ctr_mean
        priors["Google"].ctr_mean = 0.035
        print(f"Updated Google CTR mean: {old_ctr:.3f} -> {priors['Google'].ctr_mean:.3f}")
    
    # Add a new channel
    priors["YouTube"] = ChannelPriors(
        name="YouTube",
        cpm_log_mu=np.log(7.0) - 0.5 * (0.3**2),
        cpm_log_sigma=0.3,
        ctr_mean=0.018,
        ctr_std=0.006,
        cvr_mean=0.028,
        cvr_std=0.008,
        theta_log_mu=np.log(5500.0),
        theta_log_sigma=0.3
    )
    print("Added new channel: YouTube")
    
    # Save back to config
    save_channel_priors(priors)
    print("Configuration updated and saved!")

def example_programmatic_update():
    """Example: Programmatically update specific values."""
    print("\n=== Programmatic Updates ===")
    
    # Load config
    priors = load_channel_priors()
    
    # Update multiple channels at once
    updates = {
        "Google": {"ctr_mean": 0.032, "cvr_mean": 0.042},
        "Meta": {"ctr_mean": 0.016, "cvr_mean": 0.032},
    }
    
    for channel_name, updates_dict in updates.items():
        if channel_name in priors:
            for param, value in updates_dict.items():
                setattr(priors[channel_name], param, value)
                print(f"Updated {channel_name}.{param} = {value}")
    
    # Save changes
    save_channel_priors(priors)
    print("Batch updates saved!")

if __name__ == "__main__":
    example_read_config()
    example_update_config()
    example_programmatic_update()