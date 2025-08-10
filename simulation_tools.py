import sys
import os
from typing import Dict, List, Tuple
from simulation_models import SimulationParams, SimulationResult, ToolResult, GoalConfig

# Import the simulation function from the existing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modelling_adv_goals2 import GoalType as ModelGoalType, GoalConfig as ModelGoalConfig

class SimulationTool:
    """Tool for running marketing budget allocation simulations"""
    
    def __init__(self):
        self.name = "run_simulation"
    
    def execute(self, params: SimulationParams) -> ToolResult:
        """Execute a simulation with the given parameters"""
        
        try:
            # Convert our Pydantic models to the format expected by the simulation function
            model_goal_config = ModelGoalConfig(
                goal=ModelGoalType(params.goal_config.goal.value),
                gross_margin=params.goal_config.gross_margin,
                roas_floor=params.goal_config.roas_floor
            )
            
            # Import the explore_allocation_grid function
            from modelling_adv_goals2 import explore_allocation_grid, Constraints, analyze_grid_results, print_grid_analysis, print_grid_summary_stats, export_grid_results_summary
            
            # Convert parameters to Constraints format
            constraints = Constraints(
                min_share={"LinkedIn": params.linked_in_min_share} if params.linked_in_min_share > 0 else {},
                max_share=params.max_share
            )
            
            # Run the grid exploration instead of simulation
            grid_results = explore_allocation_grid(
                total_budget=params.total_budget,
                n_sims=params.n_sims,
                seed=params.seed,
                grid_step=0.10,  # 10% increments
                constraints=constraints,
                goal_cfg=model_goal_config
            )
            
            # Enhanced grid analysis and visualization
            if grid_results:
                print(f"\n🔍 Analyzing {len(grid_results)} allocation combinations...")
                
                # Run enhanced analysis
                analysis = analyze_grid_results(grid_results, model_goal_config)
                
                # Display enhanced grid analysis
                print_grid_analysis(analysis, params.total_budget)
                
                # Display summary statistics
                print_grid_summary_stats(grid_results)
                
                # Export results for further analysis
                export_filename = f"grid_results_{params.goal_config.goal.value}_{params.total_budget:,.0f}_{params.n_sims}sims.json"
                export_grid_results_summary(grid_results, export_filename)
                
                # Generate visualizations
                try:
                    from modelling_visualizations import (
                        plot_allocation_heatmap, plot_confidence_intervals, 
                        plot_allocation_comparison, plot_risk_vs_performance,
                        save_all_plots
                    )
                    
                    print(f"\n📊 Generating visualizations...")
                    print("💾 Saving plots to 'plots/' directory (no interactive display)...")
                    
                    # Create plots directory if it doesn't exist
                    plots_dir = "plots"
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Generate individual plots
                    plot_allocation_heatmap(grid_results, model_goal_config, params.total_budget, 
                                         save_path=f"{plots_dir}/allocation_heatmap_{params.goal_config.goal.value}.png")
                    
                    plot_confidence_intervals(grid_results, model_goal_config, top_n=10, 
                                           save_path=f"{plots_dir}/confidence_intervals_{params.goal_config.goal.value}.png")
                    
                    plot_allocation_comparison(grid_results, model_goal_config, 
                                            save_path=f"{plots_dir}/allocation_comparison_{params.goal_config.goal.value}.png")
                    
                    plot_risk_vs_performance(grid_results, model_goal_config, 
                                           save_path=f"{plots_dir}/risk_vs_performance_{params.goal_config.goal.value}.png")
                    
                    print(f"✅ Visualizations saved to {plots_dir}/ directory")
                    print("🎯 You can now continue with your analysis or ask another question!")
                    
                except ImportError as e:
                    print(f"⚠️  Visualization module not available: {e}")
                    print("   Install required packages: pip install matplotlib seaborn")
                except Exception as e:
                    print(f"⚠️  Error generating visualizations: {e}")
                
                # Get the best allocation based on the goal
                best_key = max(grid_results.keys(), 
                             key=lambda k: grid_results[k][model_goal_config.goal.value]["median"])
                best_result = grid_results[best_key]
                
                # Extract summaries from the best allocation
                alloc_summary = best_result["allocation"]
                revenue_summary = best_result["revenue"]
                roas_summary = best_result["roas"]
                conv_summary = best_result["leads"]  # Use 'leads' instead of 'conversions'
                cac_summary = best_result["cac_customers"]  # Use 'cac_customers' instead of 'cac'
                
                # Also extract wins and CPL for enhanced reporting
                wins_summary = best_result["wins"]
                cpl_summary = best_result["cpl"]
            else:
                # Fallback if no results
                alloc_summary = {}
                revenue_summary = {"median": 0, "p05": 0, "p95": 0}
                roas_summary = {"median": 0, "p05": 0, "p95": 0}
                conv_summary = {"median": 0, "p05": 0, "p95": 0}
                cac_summary = {"median": 0, "p05": 0, "p95": 0}
                wins_summary = {"median": 0, "p05": 0, "p95": 0}
                cpl_summary = {"median": 0, "p05": 0, "p95": 0}
            
            # Convert summaries to our Pydantic models
            from simulation_models import ChannelSummary, MetricSummary
            
            # Convert allocation summaries (from explore_allocation_grid format)
            allocations = {}
            for channel, allocation_pct in alloc_summary.items():
                # Convert percentage to dollar amounts
                allocation_dollars = allocation_pct * params.total_budget
                
                # Create ChannelSummary with the expected format
                allocations[channel] = ChannelSummary(
                    median_dollar=allocation_dollars,
                    p05_dollar=allocation_dollars,  # Single allocation, so same as median
                    p95_dollar=allocation_dollars,  # Single allocation, so same as median
                    median_percent=allocation_pct * 100,  # Convert to percentage
                    p05_percent=allocation_pct * 100,    # Single allocation, so same as median
                    p95_percent=allocation_pct * 100     # Single allocation, so same as median
                )
            
            # Convert metric summaries (from explore_allocation_grid format)
            revenue_metric = MetricSummary(
                median=revenue_summary["median"],
                p05=revenue_summary["p05"],
                p95=revenue_summary["p95"]
            )
            
            roas_metric = MetricSummary(
                median=roas_summary["median"],
                p05=roas_summary["p05"],
                p95=roas_summary["p95"]
            )
            
            conv_metric = MetricSummary(
                median=conv_summary["median"],
                p05=conv_summary["p05"],
                p95=conv_summary["p95"]
            )
            
            cac_metric = MetricSummary(
                median=cac_summary["median"],
                p05=cac_summary["p05"],
                p95=cac_summary["p95"]
            )
            
            # Create the result object
            result = SimulationResult(
                allocations=allocations,
                revenue_summary=revenue_metric,
                roas_summary=roas_metric,
                conversions_summary=conv_metric,
                cac_summary=cac_metric
            )
            
            return ToolResult(success=True, result=result.dict())
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Simulation error: {str(e)}"
            )

class SimulationToolManager:
    """Manager for simulation tools"""
    
    def __init__(self):
        self.simulation_tool = SimulationTool()
    
    def run_simulation(self, params: SimulationParams) -> ToolResult:
        """Run a marketing budget allocation simulation"""
        print(f"Running simulation with budget: ${params.total_budget:,.2f}")
        print(f"Goal: {params.goal_config.goal.value}")
        print(f"Number of simulations: {params.n_sims}")
        
        return self.simulation_tool.execute(params)
    
    def format_simulation_results(self, result_data: dict) -> str:
        """Format simulation results for human-readable output"""
        
        try:
            allocations = result_data['allocations']
            revenue_summary = result_data['revenue_summary']
            roas_summary = result_data['roas_summary']
            conversions_summary = result_data['conversions_summary']
            cac_summary = result_data['cac_summary']
            
            # Format the results
            result_text = f"""
## Simulation Results Summary

### Performance Metrics (Median with 5th-95th percentile range)
- **Revenue**: ${revenue_summary['median']:,.2f} [${revenue_summary['p05']:,.2f} - ${revenue_summary['p95']:,.2f}]
- **ROAS**: {roas_summary['median']:.2f}x [{roas_summary['p05']:.2f}x - {roas_summary['p95']:.2f}x]
- **Conversions**: {conversions_summary['median']:,.0f} [{conversions_summary['p05']:,.0f} - {conversions_summary['p95']:,.0f}]
- **CAC**: ${cac_summary['median']:.2f} [${cac_summary['p05']:.2f} - ${cac_summary['p95']:.2f}]

### Budget Allocation (Median allocation with ranges)
"""
            
            # Sort channels by median allocation (handle both alias and field name formats)
            sorted_channels = sorted(allocations.items(), key=lambda x: x[1].get('median_dollar', x[1].get('median_$', 0)), reverse=True)
            
            total_median = sum(channel_data.get('median_dollar', channel_data.get('median_$', 0)) for _, channel_data in sorted_channels)
            
            for channel, channel_data in sorted_channels:
                # Handle both alias format (median_$, median_%) and field name format (median_dollar, median_percent)
                median_dollars = channel_data.get('median_dollar', channel_data.get('median_$', 0))
                median_percent = channel_data.get('median_percent', channel_data.get('median_%', 0))
                p05_dollars = channel_data.get('p05_dollar', channel_data.get('p05_$', 0))
                p95_dollars = channel_data.get('p95_dollar', channel_data.get('p95_$', 0))
                p05_percent = channel_data.get('p05_percent', channel_data.get('p05_%', 0))
                p95_percent = channel_data.get('p95_percent', channel_data.get('p95_%', 0))
                
                result_text += f"- **{channel}**: ${median_dollars:,.2f} ({median_percent:.1f}%)\n"
                result_text += f"  └ Range: ${p05_dollars:,.2f} - ${p95_dollars:,.2f} ({p05_percent:.1f}% - {p95_percent:.1f}%)\n"
            
            result_text += f"\n**Total Median Allocation**: ${total_median:,.2f}"
            
            return result_text
            
        except Exception as e:
            return f"Error formatting results: {str(e)}"