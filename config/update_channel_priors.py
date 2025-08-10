#!/usr/bin/env python3
"""
Utility script to update channel priors configuration.
Usage examples:
    # Update a specific channel's CTR mean
    python update_channel_priors.py --channel Google --ctr_mean 0.035
    
    # Update multiple parameters for a channel
    python update_channel_priors.py --channel Meta --ctr_mean 0.020 --cvr_mean 0.035
    
    # Add a new channel
    python update_channel_priors.py --channel "YouTube" --cpm_log_mu 2.0 --cpm_log_sigma 0.3 --ctr_mean 0.015 --ctr_std 0.005 --cvr_mean 0.025 --cvr_std 0.008
"""

import json
import argparse
import os
import sys

def load_config(config_path: str) -> dict:
    """Load existing config or create empty dict."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: dict, config_path: str):
    """Save config to JSON file with nice formatting."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def update_channel_priors():
    parser = argparse.ArgumentParser(description='Update channel priors configuration')
    parser.add_argument('--channel', required=True, help='Channel name (e.g., Google, Meta, TikTok, LinkedIn)')
    parser.add_argument('--config-path', default='channel_priors.json', help='Path to config file')
    
    # Channel parameters
    parser.add_argument('--cpm_log_mu', type=float, help='CPM log mean')
    parser.add_argument('--cpm_log_sigma', type=float, help='CPM log sigma')
    parser.add_argument('--ctr_mean', type=float, help='CTR mean')
    parser.add_argument('--ctr_std', type=float, help='CTR standard deviation')
    parser.add_argument('--cvr_mean', type=float, help='CVR mean')
    parser.add_argument('--cvr_std', type=float, help='CVR standard deviation')
    parser.add_argument('--quality_log_mu', type=float, help='Quality log mean (default: 0.0)')
    parser.add_argument('--quality_log_sigma', type=float, help='Quality log sigma (default: 0.1)')
    parser.add_argument('--theta_log_mu', type=float, help='Theta log mean')
    parser.add_argument('--theta_log_sigma', type=float, help='Theta log sigma')
    parser.add_argument('--aov_log_mu', type=float, help='AOV log mean (default: 4.875)')
    parser.add_argument('--aov_log_sigma', type=float, help='AOV log sigma (default: 0.3)')
    
    args = parser.parse_args()
    
    # Load existing config
    config = load_config(args.config_path)
    
    # Initialize channel if it doesn't exist
    if args.channel not in config:
        config[args.channel] = {
            "name": args.channel,
            "quality_log_mu": 0.0,
            "quality_log_sigma": 0.1,
            "aov_log_mu": 4.875,
            "aov_log_sigma": 0.3
        }
    
    # Update parameters
    channel_config = config[args.channel]
    
    for param in ['cpm_log_mu', 'cpm_log_sigma', 'ctr_mean', 'ctr_std', 'cvr_mean', 'cvr_std',
                  'quality_log_mu', 'quality_log_sigma', 'theta_log_mu', 'theta_log_sigma',
                  'aov_log_mu', 'aov_log_sigma']:
        value = getattr(args, param)
        if value is not None:
            channel_config[param] = value
            print(f"Updated {args.channel}.{param} = {value}")
    
    # Save config
    save_config(config, args.config_path)
    print(f"Configuration saved to {args.config_path}")

if __name__ == "__main__":
    update_channel_priors()