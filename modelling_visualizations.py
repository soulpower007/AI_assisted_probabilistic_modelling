#!/usr/bin/env python3
"""
Visualization functions for marketing budget allocation modeling.
Contains all plotting and chart generation functions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking terminal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
from typing import Dict
from modelling_adv_goals import GoalType, GoalConfig

# Set style for better-looking plots and ensure non-interactive mode
plt.style.use('default')
sns.set_palette("husl")
plt.ioff()  # Turn off interactive mode

def plot_allocation_heatmap(grid_results: Dict, goal_cfg: GoalConfig, total_budget: float, save_path: str = None):
    """
    Create a heatmap showing performance across different allocation combinations.
    
    Args:
        grid_results: Output from explore_allocation_grid
        goal_cfg: Goal configuration
        total_budget: Total budget amount
        save_path: Optional path to save the plot
    """
    # Determine metric to plot
    if goal_cfg.goal == GoalType.REVENUE:
        metric = "revenue"
        title = "Revenue Performance Heatmap"
        unit = "$"
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        metric = "conversions"
        title = "Conversions Performance Heatmap"
        unit = ""
    elif goal_cfg.goal == GoalType.PROFIT:
        metric = "revenue"  # Use revenue as proxy
        title = "Profit Performance Heatmap (Revenue Proxy)"
        unit = "$"
    
    # Extract data for heatmap
    allocations = []
    performances = []
    risks = []
    
    for key, result in grid_results.items():
        alloc = result['allocation']
        perf = result[metric]
        
        # Create allocation vector for heatmap
        alloc_vector = [alloc.get(name, 0) for name in ['Google', 'Meta', 'LinkedIn', 'TikTok']]
        allocations.append(alloc_vector)
        performances.append(perf['median'])
        risks.append(perf['std'] / perf['mean'] if perf['mean'] > 0 else 0)
    
    allocations = np.array(allocations)
    performances = np.array(performances)
    risks = np.array(risks)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance heatmap
    im1 = ax1.imshow(allocations.T, cmap='YlOrRd', aspect='auto')
    ax1.set_title(f'{title}\n(Median Performance)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Allocation Index', fontsize=12)
    ax1.set_ylabel('Channel', fontsize=12)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(['Google', 'Meta', 'LinkedIn', 'TikTok'])
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Allocation %', fontsize=12)
    
    # Risk heatmap
    im2 = ax2.imshow(allocations.T, cmap='RdYlBu_r', aspect='auto')
    ax2.set_title('Allocation Risk Heatmap\n(Std/Mean)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Allocation Index', fontsize=12)
    ax2.set_ylabel('Channel', fontsize=12)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(['Google', 'Meta', 'LinkedIn', 'TikTok'])
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Allocation %', fontsize=12)
    
    # Add performance annotations
    for i, (perf, risk) in enumerate(zip(performances, risks)):
        if i % 5 == 0:  # Show every 5th allocation
            ax1.text(i, 4.5, f'{perf:,.0f}', ha='center', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax2.text(i, 4.5, f'{risk:.3f}', ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close the plot to free memory

def plot_confidence_intervals(grid_results: Dict, goal_cfg: GoalConfig, top_n: int = 10, save_path: str = None):
    """
    Create confidence interval plots for top performing allocations.
    
    Args:
        grid_results: Output from explore_allocation_grid
        goal_cfg: Goal configuration
        top_n: Number of top allocations to show
        save_path: Optional path to save the plot
    """
    # Determine metric to plot
    if goal_cfg.goal == GoalType.REVENUE:
        metric = "revenue"
        title = "Revenue Confidence Intervals"
        unit = "$"
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        metric = "conversions"
        title = "Conversions Confidence Intervals"
        unit = ""
    elif goal_cfg.goal == GoalType.PROFIT:
        metric = "revenue"
        title = "Profit Confidence Intervals (Revenue Proxy)"
        unit = "$"
    
    # Sort by median performance
    sorted_results = sorted(
        grid_results.items(),
        key=lambda x: x[1][metric]['median'],
        reverse=True
    )[:top_n]
    
    # Prepare data for plotting
    labels = []
    medians = []
    p05s = []
    p95s = []
    means = []
    
    for key, result in sorted_results:
        perf = result[metric]
        alloc_str = "_".join([f"{v*100:.0f}%" for v in result['allocation'].values()])
        labels.append(alloc_str)
        medians.append(perf['median'])
        p05s.append(perf['p05'])
        p95s.append(perf['p95'])
        means.append(perf['mean'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot confidence intervals
    x_pos = np.arange(len(labels))
    ax.errorbar(x_pos, medians, yerr=[np.array(medians) - np.array(p05s), 
                                     np.array(p95s) - np.array(medians)],
                fmt='o', capsize=5, capthick=2, markersize=8, 
                label='Median (5th-95th percentile)', color='#2E86AB')
    
    # Plot means
    ax.scatter(x_pos, means, s=100, marker='s', color='#A23B72', 
               label='Mean', zorder=5)
    
    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Allocation Strategy', fontsize=12)
    ax.set_ylabel(f'{metric.title()} {unit}', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (median, mean) in enumerate(zip(medians, means)):
        ax.annotate(f'{median:,.0f}', (i, median), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        ax.annotate(f'{mean:,.0f}', (i, mean), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontsize=8, color='#A23B72')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close the plot to free memory

def plot_allocation_comparison(grid_results: Dict, goal_cfg: GoalConfig, save_path: str = None):
    """
    Create a stacked bar chart comparing top allocations.
    
    Args:
        grid_results: Output from explore_allocation_grid
        goal_cfg: Goal configuration
        save_path: Optional path to save the plot
    """
    # Determine metric to plot
    if goal_cfg.goal == GoalType.REVENUE:
        metric = "revenue"
        title = "Top Revenue Allocations Comparison"
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        metric = "conversions"
        title = "Top Conversions Allocations Comparison"
    elif goal_cfg.goal == GoalType.PROFIT:
        metric = "revenue"
        title = "Top Profit Allocations Comparison (Revenue Proxy)"
    
    # Get top 8 allocations
    sorted_results = sorted(
        grid_results.items(),
        key=lambda x: x[1][metric]['median'],
        reverse=True
    )[:8]
    
    # Prepare data
    labels = []
    allocations = []
    performances = []
    
    for key, result in sorted_results:
        alloc = result['allocation']
        perf = result[metric]
        
        # Create label
        alloc_str = "_".join([f"{v*100:.0f}%" for v in alloc.values()])
        labels.append(alloc_str)
        
        # Store allocation percentages
        alloc_vector = [alloc.get(name, 0) for name in ['Google', 'Meta', 'LinkedIn', 'TikTok']]
        allocations.append(alloc_vector)
        performances.append(perf['median'])
    
    allocations = np.array(allocations)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Stacked bar chart of allocations
    x_pos = np.arange(len(labels))
    width = 0.7
    
    bottom = np.zeros(len(labels))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    channel_names = ['Google', 'Meta', 'LinkedIn', 'TikTok']
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        ax1.bar(x_pos, allocations[:, i], width, bottom=bottom, 
                label=name, color=color, alpha=0.8)
        bottom += allocations[:, i]
    
    ax1.set_title(f'{title}\nAllocation Breakdown', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Allocation %', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend(title='Channels', loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Bottom plot: Performance comparison
    bars = ax2.bar(x_pos, performances, width, color='#2E86AB', alpha=0.7)
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'{metric.title()} {"" if metric == "conversions" else "$"}', fontsize=12)
    ax2.set_xlabel('Allocation Strategy', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, perf in zip(bars, performances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close the plot to free memory

def plot_risk_vs_performance(grid_results: Dict, goal_cfg: GoalConfig, save_path: str = None):
    """
    Create a scatter plot showing risk vs. performance trade-offs.
    
    Args:
        grid_results: Output from explore_allocation_grid
        goal_cfg: Goal configuration
        save_path: Optional path to save the plot
    """
    # Determine metric to plot
    if goal_cfg.goal == GoalType.REVENUE:
        metric = "revenue"
        title = "Risk vs. Revenue Performance"
        unit = "$"
    elif goal_cfg.goal == GoalType.CONVERSIONS:
        metric = "conversions"
        title = "Risk vs. Conversions Performance"
        unit = ""
    elif goal_cfg.goal == GoalType.PROFIT:
        metric = "revenue"
        title = "Risk vs. Profit Performance (Revenue Proxy)"
        unit = "$"
    
    # Extract data
    performances = []
    risks = []
    allocations = []
    
    for key, result in grid_results.items():
        perf = result[metric]
        performance = perf['median']
        risk = perf['std'] / perf['mean'] if perf['mean'] > 0 else 0
        
        performances.append(performance)
        risks.append(risk)
        allocations.append(result['allocation'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with size based on total allocation
    scatter = ax.scatter(risks, performances, s=100, alpha=0.7, c=performances, 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{metric.title()} {unit}', fontsize=12)
    
    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Risk (Coefficient of Variation)', fontsize=12)
    ax.set_ylabel(f'{metric.title()} {unit}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(risks, performances, 1)
    p = np.poly1d(z)
    ax.plot(risks, p(risks), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Highlight top performers
    top_indices = np.argsort(performances)[-5:]  # Top 5
    for idx in top_indices:
        ax.annotate(f"Top {len(top_indices) - list(top_indices).index(idx)}", 
                    (risks[idx], performances[idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=9, fontweight='bold')
    
    # Highlight low risk options
    low_risk_indices = np.argsort(risks)[:5]  # Lowest 5 risk
    for idx in low_risk_indices:
        if idx not in top_indices:  # Don't double-annotate
            ax.annotate(f"Low Risk", 
                        (risks[idx], performances[idx]),
                        xytext=(10, -15), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9, fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close the plot to free memory

def save_all_plots(grid_results: Dict, grid_results_profit: Dict, total_budget: float, 
                   save_dir: str = "plots", goal_cfg: GoalConfig = None):
    """
    Save all plots to files for easy sharing and analysis.
    
    Args:
        grid_results: Revenue optimization results
        grid_results_profit: Profit optimization results  
        total_budget: Total budget amount
        save_dir: Directory to save plots
        goal_cfg: Goal configuration for naming
    """
    if goal_cfg is None:
        goal_cfg = GoalConfig(goal=GoalType.REVENUE)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving all plots to '{save_dir}/' directory...")
    
    # Revenue optimization plots
    plot_allocation_heatmap(grid_results, GoalConfig(goal=GoalType.REVENUE), 
                           total_budget, f"{save_dir}/revenue_heatmap.png")
    plot_confidence_intervals(grid_results, GoalConfig(goal=GoalType.REVENUE), 
                             10, f"{save_dir}/revenue_confidence_intervals.png")
    plot_allocation_comparison(grid_results, GoalConfig(goal=GoalType.REVENUE),
                              f"{save_dir}/revenue_allocation_comparison.png")
    plot_risk_vs_performance(grid_results, GoalConfig(goal=GoalType.REVENUE),
                             f"{save_dir}/revenue_risk_vs_performance.png")
    
    # Profit optimization plots
    plot_allocation_heatmap(grid_results_profit, GoalConfig(goal=GoalType.PROFIT), 
                           total_budget, f"{save_dir}/profit_heatmap.png")
    plot_confidence_intervals(grid_results_profit, GoalConfig(goal=GoalType.PROFIT), 
                             10, f"{save_dir}/profit_confidence_intervals.png")
    plot_allocation_comparison(grid_results_profit, GoalConfig(goal=GoalType.PROFIT),
                              f"{save_dir}/profit_allocation_comparison.png")
    plot_risk_vs_performance(grid_results_profit, GoalConfig(goal=GoalType.PROFIT),
                             f"{save_dir}/profit_risk_vs_performance.png")
    
    print(f"✅ All plots saved to '{save_dir}/' directory!")

def generate_all_visualizations(grid_results: Dict, grid_results_profit: Dict, total_budget: float):
    """
    Generate all visualizations for the grid exploration results.
    
    Args:
        grid_results: Revenue optimization results
        grid_results_profit: Profit optimization results
        total_budget: Total budget amount
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS FOR GRID EXPLORATION")
    print("="*60)
    
    # Revenue optimization plots
    print("\n--- Revenue Optimization Plots ---")
    plot_allocation_heatmap(grid_results, GoalConfig(goal=GoalType.REVENUE), total_budget)
    plot_confidence_intervals(grid_results, GoalConfig(goal=GoalType.REVENUE), 10)
    plot_allocation_comparison(grid_results, GoalConfig(goal=GoalType.REVENUE))
    plot_risk_vs_performance(grid_results, GoalConfig(goal=GoalType.REVENUE))
    
    # Profit optimization plots
    print("\n--- Profit Optimization Plots ---")
    plot_allocation_heatmap(grid_results_profit, GoalConfig(goal=GoalType.PROFIT), total_budget)
    plot_confidence_intervals(grid_results_profit, GoalConfig(goal=GoalType.PROFIT), 10)
    plot_allocation_comparison(grid_results_profit, GoalConfig(goal=GoalType.PROFIT))
    plot_risk_vs_performance(grid_results_profit, GoalConfig(goal=GoalType.PROFIT))
    
    # Save all plots
    save_all_plots(grid_results, grid_results_profit, total_budget, save_dir="plots") 