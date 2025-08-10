#!/usr/bin/env python3
"""
Marketing Budget Optimization Chatbot

A ReACT agent-powered chatbot that helps users optimize marketing budget allocations
across different channels using Monte Carlo simulations.
"""

import os
import sys
from typing import Optional, List
from simulation_agent import SimulationReACTAgent

class MarketingChatbot:
    """Main chatbot interface for marketing budget optimization"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the marketing chatbot
        
        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model: OpenAI model to use for the agent
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.agent = SimulationReACTAgent(api_key=api_key, model=model)
        self.is_running = False
    
    def display_welcome_message(self):
        """Display welcome message and instructions"""
        print("=" * 80)
        print("🚀 Marketing Budget Optimization Chatbot")
        print("=" * 80)
        print()
        

        
        print("I'm here to help you optimize your marketing budget allocation across channels!")
        print()
        print("Available channels: LinkedIn, Facebook, Google, TikTok, YouTube, Twitter")
        print()
        print("You can ask me things like:")
        print("• 'Optimize $50k budget for maximum revenue'")
        print("• 'Run simulation with $100k focusing on conversions'")
        print("• 'Allocate budget for profit with 60% margin and minimum 2.0 ROAS'")
        print("• 'Optimize $75k with at least 20% on LinkedIn'")
        print("• 'Run simulation with TikTok capped at 40%'")
        print()
        print("📋 How it works:")
        print("1. I'll extract parameters from your request")
        print("2. Show you a confirmation with the details")
        print("3. Wait for your approval before running the simulation")
        print("4. You can modify parameters or confirm to proceed")
        print()
        print("Type 'help' for more information, 'quit' or 'exit' to end the conversation.")
        print("-" * 80)
        print()
    
    def display_help(self):
        """Display help information"""
        print()
        print("📋 Help - Marketing Budget Optimization")
        print("-" * 50)
        print()
        print("🎯 Optimization Goals:")
        print("• Revenue: Maximize total revenue")
        print("• Conversions: Maximize number of conversions/leads")
        print("• Profit: Maximize profit (requires gross margin)")
        print()
        print("⚙️  Parameters you can specify:")
        print("• Budget amount: '$50k', '$100,000', '75 thousand dollars'")
        print("• Optimization goal: 'revenue', 'conversions', 'profit'")
        print("• Gross margin: 'with 60% margin' (for profit optimization)")
        print("• ROAS floor: 'minimum 2.0 ROAS'")
        print("• Channel constraints: 'at least 20% on LinkedIn', 'TikTok max 50%'")
        print("• Number of simulations: '500 simulations' (default: 200, max: 1000)")
        print()
        print("💡 Example queries:")
        print("• 'Optimize $50k budget for maximum revenue'")
        print("• 'Run 300 simulations with $80k for conversions'")
        print("• 'Allocate $120k for profit with 65% margin and 2.5 ROAS floor'")
        print("• 'Optimize $60k with LinkedIn at least 25% and TikTok max 40%'")
        print()
        print("🔄 Confirmation Process:")
        print("• I'll show you extracted parameters before running")
        print("• Respond with 'yes' to confirm or 'no' to cancel")
        print("• You can modify by saying 'change budget to $75k'")
        print("• Or 'set TikTok max to 30%' to adjust constraints")
        print()
        print("📊 The simulation will show you:")
        print("• Optimal budget allocation per channel")
        print("• Expected revenue, ROAS, conversions, and CAC")
        print("• Performance metrics with confidence intervals")
        print()
        print("-" * 50)
        print()
    
    def run_interactive_session(self):
        """Run an interactive chat session"""
        self.is_running = True
        self.display_welcome_message()
        
        while self.is_running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for using the Marketing Budget Optimization Chatbot!")
                    print("Happy optimizing! 🚀")
                    break
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    self.display_welcome_message()
                    continue
                
                # Process the query with the agent
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.agent.process_query(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                print("Please try again or type 'help' for assistance.\n")
    
    def process_single_query(self, query: str) -> str:
        """Process a single query and return the response"""
        return self.agent.process_query(query)
    


def main():
    """Main function to run the chatbot"""
    try:
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: OpenAI API key not found!")
            print("Please set the OPENAI_API_KEY environment variable.")
            print("Example: export OPENAI_API_KEY='your-api-key-here'")
            sys.exit(1)
        
        # Create and run the chatbot
        chatbot = MarketingChatbot(api_key=api_key)
        chatbot.run_interactive_session()
        
    except Exception as e:
        print(f"❌ Failed to start chatbot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()