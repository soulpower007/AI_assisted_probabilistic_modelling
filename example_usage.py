#!/usr/bin/env python3
"""
Example usage of the Marketing Budget Optimization Chatbot
"""

import os
from marketing_chatbot import MarketingChatbot

def demonstrate_chatbot():
    """Demonstrate the chatbot with example queries"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Please set OPENAI_API_KEY environment variable to run this example")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize chatbot
    print("🤖 Initializing Marketing Budget Optimization Chatbot...")
    chatbot = MarketingChatbot(api_key=api_key)
    
    # Example queries
    example_queries = [
        # "Optimize $50k budget for maximum revenue",
        # "Run simulation with $75k focusing on conversions with LinkedIn at least 20%",
        "Allocate $100k for profit with 60% margin and minimum 2.0 ROAS",
    ]
    
    print("\n" + "="*80)
    print("🚀 Marketing Budget Optimization Examples")
    print("="*80)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n📋 Example {i}: {query}")
        print("-" * 60)
        
        try:
            response = chatbot.process_single_query(query)
            print(response)
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
        
        print("\n" + "="*80)

def run_interactive_demo():
    """Run a simple interactive demo with confirmation flow"""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    
    chatbot = MarketingChatbot(api_key=api_key)
    
    print("\n🎯 Interactive Demo with Human-in-the-Loop Confirmation")
    print("=" * 65)
    print("This demo shows the confirmation flow:")
    print("1. Make a budget optimization request")
    print("2. Review the extracted parameters")
    print("3. Confirm, modify, or cancel")
    print("4. See the simulation results")
    print()
    print("Type your budget optimization request (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for trying the demo!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Assistant:")
            response = chatbot.process_single_query(user_input)
            print(response)
            
            # If response contains confirmation request, show helpful hint
            if "Is this correct?" in response:
                print("\n💡 Hint: You can respond with:")
                print("   • 'yes' to proceed")
                print("   • 'no' to cancel")
                print("   • 'change budget to $75k' to modify")
                print("   • 'set TikTok max to 30%' to add constraints")
            
        except KeyboardInterrupt:
            print("\n👋 Demo ended!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_demo()
    else:
        demonstrate_chatbot()
        
        print("\n💡 To try the interactive demo:")
        print("python example_usage.py --interactive")
        
        print("\n💡 To run the full chatbot:")
        print("python marketing_chatbot.py")