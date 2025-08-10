#!/usr/bin/env python3
"""
Gradio UI for Enhanced Interactive Marketing Agent
Provides a web interface for the marketing agent with visualization display
"""

import gradio as gr
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import subprocess
import sys

# Import the enhanced agent components
from enhanced_interactive_agent2 import EnhancedInteractiveMarketingAgent, ExtractedParameters
from elict_priors_gemini import elicit_priors_with_gemini, BenchmarksResponse, print_benchmarks
from convert_benchmarks_to_priors import main as convert_benchmarks_to_priors
from marketing_chatbot import MarketingChatbot

class GradioMarketingAgent:
    """Gradio interface for the Enhanced Interactive Marketing Agent"""
    
    def __init__(self):
        self.agent = EnhancedInteractiveMarketingAgent()
        self.current_results = None
        self.current_plots_dir = "plots"
        self.chatbot = None
        self.chat_history = []
        
    def check_api_keys(self):
        """Check if required API keys are available"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        status = {
            "gemini": "✅ Available" if gemini_key else "❌ Missing",
            "openai": "✅ Available" if openai_key else "❌ Missing"
        }
        
        return status
    
    def extract_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extract parameters from user input using the agent's extractor"""
        try:
            params = self.agent.parameter_extractor.extract_parameters_with_gemini(user_input)
            
            # Convert to dictionary for Gradio
            return {
                "company": params.company,
                "industry": params.industry or "Not specified",
                "geography": params.geography or "Not specified",
                "channels": params.channels or ["Google", "Meta", "TikTok", "LinkedIn"],
                "model_name": params.model_name,
                "use_grounding": params.use_grounding
            }
        except Exception as e:
            return {"error": f"Failed to extract parameters: {str(e)}"}
    
    def generate_benchmarks(self, company: str, industry: str, geography: str, 
                           channels: List[str], model_name: str, use_grounding: bool) -> Dict[str, Any]:
        """Generate marketing benchmarks using Gemini"""
        try:
            # Convert empty strings to None
            industry = industry if industry and industry != "Not specified" else None
            geography = geography if geography and geography != "Not specified" else None
            
            # Call Gemini API
            result = elicit_priors_with_gemini(
                company=company,
                industry=industry,
                geography=geography,
                channels=channels,
                model_name=model_name,
                use_grounding=use_grounding
            )
            
            self.current_results = result
            
            # Save results for later use
            self._save_gemini_results("marketing_benchmarks_config.json")
            
            # Convert to displayable format
            return self._format_benchmarks_for_display(result)
            
        except Exception as e:
            return {"error": f"Failed to generate benchmarks: {str(e)}"}
    
    def _format_benchmarks_for_display(self, result: BenchmarksResponse) -> Dict[str, Any]:
        """Format benchmarks for Gradio display"""
        formatted = {
            "company": result.company,
            "industry": result.industry or "Not specified",
            "geography": result.geography or "Not specified",
            "timeframe": result.timeframe,
            "channels": []
        }
        
        for channel in result.channels:
            channel_data = {
                "name": channel.channel,
                "cpm": f"${channel.cpm_usd_per_1000.p50:.2f} (${channel.cpm_usd_per_1000.p10:.2f} - ${channel.cpm_usd_per_1000.p90:.2f})",
                "ctr": f"{channel.ctr_ratio.p50:.4f} ({channel.ctr_ratio.p10:.4f} - {channel.ctr_ratio.p90:.4f})",
                "cvr": f"{channel.cvr_ratio.p50:.4f} ({channel.cvr_ratio.p10:.4f} - {channel.cvr_ratio.p90:.4f})",
                "aov": f"${channel.aov_usd.p50:.2f} (${channel.aov_usd.p10:.2f} - ${channel.aov_usd.p90:.2f})",
                "notes": channel.notes or "No additional notes"
            }
            formatted["channels"].append(channel_data)
        
        return formatted
    
    def _save_gemini_results(self, filename: str):
        """Save Gemini results to a JSON file"""
        if self.current_results:
            data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "tool": "gradio_ui",
                    "model": "gemini-2.5-flash"
                },
                "benchmarks": {
                    "company": self.current_results.company,
                    "industry": self.current_results.industry,
                    "geography": self.current_results.geography,
                    "timeframe": self.current_results.timeframe,
                    "channels": []
                }
            }
            
            for channel in self.current_results.channels:
                channel_data = {
                    "channel": channel.channel,
                    "geography": channel.geography,
                    "industry": channel.industry,
                    "cpm_usd_per_1000": {
                        "unit": channel.cpm_usd_per_1000.unit,
                        "p10": channel.cpm_usd_per_1000.p10,
                        "p50": channel.cpm_usd_per_1000.p50,
                        "p90": channel.cpm_usd_per_1000.p90
                    },
                    "ctr_ratio": {
                        "unit": channel.ctr_ratio.unit,
                        "p10": channel.ctr_ratio.p10,
                        "p50": channel.ctr_ratio.p50,
                        "p90": channel.ctr_ratio.p90
                    },
                    "cvr_ratio": {
                        "unit": channel.cvr_ratio.unit,
                        "p10": channel.cvr_ratio.p10,
                        "p50": channel.cvr_ratio.p50,
                        "p90": channel.cvr_ratio.p90
                    },
                    "aov_usd": {
                        "unit": channel.aov_usd.unit,
                        "p10": channel.aov_usd.p10,
                        "p50": channel.aov_usd.p50,
                        "p90": channel.aov_usd.p90
                    },
                    "notes": channel.notes,
                    "sources": [{"title": s.title, "uri": s.uri} for s in (channel.sources or [])]
                }
                data["benchmarks"]["channels"].append(channel_data)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def start_marketing_optimization(self, budget: str, goal: str) -> Dict[str, Any]:
        """Start marketing optimization session and return initial response"""
        try:
            if not self.current_results:
                return {"error": "Please generate benchmarks first"}
            
            # Convert benchmarks to priors
            convert_benchmarks_to_priors()
            
            # Check OpenAI API key
            if not os.getenv('OPENAI_API_KEY'):
                return {"error": "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."}
            
            # Create chatbot instance
            self.chatbot = MarketingChatbot(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Create optimization query
            query = f"Optimize ${budget} budget for maximum {goal}"
            
            # Process the query
            response = self.chatbot.process_single_query(query)
            
            # Add to chat history
            self.chat_history = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
            
            return {
                "query": query,
                "response": response,
                "message": "Marketing optimization started! Use the chat interface below to continue the conversation.",
                "chat_history": self.chat_history
            }
            
        except Exception as e:
            return {"error": f"Marketing optimization failed: {str(e)}"}
    
    def chat_with_agent(self, message: str) -> Dict[str, Any]:
        """Send a message to the marketing agent and get response"""
        try:
            if not self.chatbot:
                return {"error": "Please start marketing optimization first"}
            
            # Add user message to history
            self.chat_history.append({"role": "user", "content": message})
            
            # Get response from chatbot
            response = self.chatbot.process_single_query(message)
            
            # Add assistant response to history
            self.chat_history.append({"role": "assistant", "content": response})
            
            return {
                "response": response,
                "chat_history": self.chat_history
            }
            
        except Exception as e:
            return {"error": f"Chat failed: {str(e)}"}
    
    def get_available_visualizations(self) -> Dict[str, Any]:
        """Get list of available visualizations"""
        try:
            plots_dir = self.current_plots_dir
            html_files = [f for f in os.listdir(".") if f.endswith(".html")]
            
            # Get PNG plots
            png_plots = []
            if os.path.exists(plots_dir):
                png_plots = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
            
            return {
                "html_files": html_files,
                "png_plots": png_plots,
                "plots_directory": plots_dir
            }
        except Exception as e:
            return {"error": f"Failed to get visualizations: {str(e)}"}

def create_gradio_interface():
    """Create the Gradio interface"""
    agent_ui = GradioMarketingAgent()
    
    with gr.Blocks(title="Marketing Agent UI", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 Enhanced Interactive Marketing Agent")
        gr.Markdown("Generate marketing benchmarks and optimize your marketing strategy using AI")
        
        # API Key Status
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔑 API Key Status")
                status_btn = gr.Button("Check API Keys", variant="secondary")
                status_output = gr.JSON(label="API Key Status")
        
        # Parameter Extraction
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🔍 Parameter Extraction")
                user_input = gr.Textbox(
                    label="Describe your marketing needs",
                    placeholder="e.g., 'I need marketing benchmarks for a B2B SaaS company in the US'",
                    lines=3
                )
                extract_btn = gr.Button("Extract Parameters", variant="primary")
                extracted_params = gr.JSON(label="Extracted Parameters")
            
            with gr.Column(scale=1):
                gr.Markdown("## 📋 Manual Parameter Input")
                company = gr.Textbox(label="Company Name", placeholder="Acme Corp")
                industry = gr.Textbox(label="Industry (optional)", placeholder="B2B SaaS")
                geography = gr.Textbox(label="Geography (optional)", placeholder="United States")
                channels = gr.CheckboxGroup(
                    choices=["Google", "Meta", "TikTok", "LinkedIn", "YouTube", "Pinterest"],
                    label="Marketing Channels",
                    value=["Google", "Meta", "TikTok", "LinkedIn"]
                )
                model_name = gr.Dropdown(
                    choices=["gemini-2.5-flash", "gemini-2.0-flash-exp"],
                    label="Gemini Model",
                    value="gemini-2.5-flash"
                )
                use_grounding = gr.Checkbox(label="Use Google Search Grounding", value=True)
        
        # Benchmark Generation
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🚀 Generate Benchmarks")
                generate_btn = gr.Button("Generate Marketing Benchmarks", variant="primary", size="lg")
                benchmarks_output = gr.JSON(label="Generated Benchmarks")
        
        # Marketing Optimization
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🎯 Marketing Optimization")
                budget = gr.Textbox(label="Budget Amount", placeholder="50000")
                goal = gr.Dropdown(
                    choices=["revenue", "conversions", "profit"],
                    label="Optimization Goal",
                    value="revenue"
                )
                start_optimize_btn = gr.Button("Start Marketing Optimization", variant="primary")
                optimization_output = gr.JSON(label="Optimization Results")
        
        # Chat Interface for Marketing Optimization
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💬 Chat with Marketing Agent")
                gr.Markdown("Continue the conversation with the agent to refine your optimization strategy")
                
                # Chat history display
                chat_history_display = gr.HTML(
                    label="Chat History",
                    value="<p>Start optimization to begin chatting with the agent</p>"
                )
                
                # Chat input
                chat_input = gr.Textbox(
                    label="Type your message",
                    placeholder="e.g., 'yes, proceed with simulation' or 'change budget to $75k'",
                    lines=2
                )
                chat_btn = gr.Button("Send Message", variant="primary")
                
                # Chat response
                chat_response = gr.JSON(label="Agent Response")
        
        # Visualizations
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 Visualizations")
                viz_btn = gr.Button("Refresh Visualizations", variant="secondary")
                viz_output = gr.JSON(label="Available Visualizations")
        
        # Display Area for Visualizations
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🖼️ Generated Plots")
                plots_gallery = gr.Gallery(
                    label="Marketing Analysis Plots",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
        
        # HTML Visualizations
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🌐 Interactive Visualizations")
                html_file = gr.Dropdown(
                    choices=[],
                    label="Select HTML Visualization",
                    interactive=True
                )
                html_display = gr.HTML(label="HTML Visualization")
        
        # Event handlers
        status_btn.click(
            fn=agent_ui.check_api_keys,
            outputs=status_output
        )
        
        extract_btn.click(
            fn=agent_ui.extract_parameters,
            inputs=user_input,
            outputs=extracted_params
        )
        
        generate_btn.click(
            fn=agent_ui.generate_benchmarks,
            inputs=[company, industry, geography, channels, model_name, use_grounding],
            outputs=benchmarks_output
        )
        
        start_optimize_btn.click(
            fn=agent_ui.start_marketing_optimization,
            inputs=[budget, goal],
            outputs=optimization_output
        )
        
        # Chat functionality
        def update_chat_history(optimization_result):
            """Update chat history display when optimization starts"""
            if "chat_history" in optimization_result and not optimization_result.get("error"):
                chat_html = ""
                for msg in optimization_result["chat_history"]:
                    role = "👤 User" if msg["role"] == "user" else "🤖 Agent"
                    chat_html += f'<div style="margin: 10px 0; padding: 10px; border-left: 3px solid {"#007bff" if msg["role"] == "user" else "#28a745"}; background: {"#f8f9fa" if msg["role"] == "user" else "#d4edda"};"><strong>{role}:</strong><br>{msg["content"]}</div>'
                return chat_html
            return "<p>Start optimization to begin chatting with the agent</p>"
        
        start_optimize_btn.click(
            fn=update_chat_history,
            inputs=optimization_output,
            outputs=chat_history_display
        )
        
        chat_btn.click(
            fn=agent_ui.chat_with_agent,
            inputs=chat_input,
            outputs=chat_response
        )
        
        # Update chat history display after each message
        def update_chat_display(chat_result):
            """Update chat history display after each message"""
            if "chat_history" in chat_result and not chat_result.get("error"):
                chat_html = ""
                for msg in chat_result["chat_history"]:
                    role = "👤 User" if msg["role"] == "user" else "🤖 Agent"
                    chat_html += f'<div style="margin: 10px 0; padding: 10px; border-left: 3px solid {"#007bff" if msg["role"] == "user" else "#28a745"}; background: {"#f8f9fa" if msg["role"] == "user" else "#d4edda"};"><strong>{role}:</strong><br>{msg["content"]}</div>'
                return chat_html
            return "<p>Error in chat. Please try again.</p>"
        
        chat_btn.click(
            fn=update_chat_display,
            inputs=chat_response,
            outputs=chat_history_display
        )
        
        # Clear chat input after sending
        def clear_chat_input():
            return ""
        
        chat_btn.click(
            fn=clear_chat_input,
            outputs=chat_input
        )
        
        viz_btn.click(
            fn=agent_ui.get_available_visualizations,
            outputs=viz_output
        )
        
        # Update HTML file choices when visualizations are refreshed
        def update_html_choices(viz_data):
            if "html_files" in viz_data and not viz_data.get("error"):
                return gr.Dropdown(choices=viz_data["html_files"])
            return gr.Dropdown(choices=[])
        
        viz_btn.click(
            fn=update_html_choices,
            inputs=viz_output,
            outputs=html_file
        )
        
        # Display HTML visualization
        def display_html(html_file):
            if html_file and os.path.exists(html_file):
                with open(html_file, 'r') as f:
                    content = f.read()
                return content
            return "<p>Select a visualization file to display</p>"
        
        html_file.change(
            fn=display_html,
            inputs=html_file,
            outputs=html_display
        )
        
        # Update plots gallery
        def update_plots_gallery(viz_data):
            if "png_plots" in viz_data and not viz_data.get("error"):
                plots_dir = viz_data.get("plots_directory", "plots")
                plot_paths = [os.path.join(plots_dir, plot) for plot in viz_data["png_plots"]]
                return plot_paths
            return []
        
        viz_btn.click(
            fn=update_plots_gallery,
            inputs=viz_output,
            outputs=plots_gallery
        )
    
    return interface

if __name__ == "__main__":
    # Check if Gradio is installed
    try:
        import gradio
    except ImportError:
        print("Installing Gradio...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        import gradio
    
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 