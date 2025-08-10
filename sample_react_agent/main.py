import os
from react_agent import ReACTAgent
from database_setup import create_sample_database
from knowledge_base import create_knowledge_base

def setup_environment():
    """Set up the database and knowledge base"""
    print("🔧 Setting up environment...")
    
    # Create sample database
    create_sample_database()
    
    # Create knowledge base
    create_knowledge_base()
    
    print("✅ Environment setup complete!")

def run_demo():
    """Run a demonstration of the ReACT agent"""
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize agent
    agent = ReACTAgent(api_key)
    
    # Sample queries to demonstrate different capabilities
    demo_queries = [
        # RAG search queries (policies and general info)
        "What is your return policy?",
        "How can I contact customer service?",
        "What payment methods do you accept?",
        
        # SQL query examples (specific data)
        "How many users are registered in the system?",
        "What are the top 3 most expensive products?",
        "Show me all completed orders from the last month",
        
        # Mixed/ambiguous queries
        "Tell me about shipping costs",
        "What products do you have in the electronics category?",
    ]
    
    print("\n" + "="*60)
    print("🤖 ReACT Agent Demo")
    print("="*60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Demo Query {i}: {query}")
        print("-" * 50)
        
        try:
            response = agent.process_query(query)
            
            print(f"\n💭 Agent's Reasoning:")
            print(f"   Observation: {response.thought.observation}")
            print(f"   Reasoning: {response.thought.reasoning}")
            print(f"   Action: {response.thought.action_needed}")
            print(f"   Confidence: {response.thought.confidence}")
            
            if response.tool_call:
                print(f"\n🔧 Tool Used: {response.tool_call.tool_name}")
                if hasattr(response.tool_call.parameters, 'query'):
                    print(f"   Query: {response.tool_call.parameters.query}")
                if hasattr(response.tool_call.parameters, 'explanation'):
                    print(f"   Explanation: {response.tool_call.parameters.explanation}")
            
            print(f"\n📋 Final Answer:")
            print(f"   {response.final_answer}")
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
        
        print("\n" + "="*60)

def interactive_mode():
    """Run the agent in interactive mode"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    agent = ReACTAgent(api_key)
    
    print("\n🤖 ReACT Agent - Interactive Mode (Now with Conversation History!)")
    print("Type 'quit' to exit, 'demo' to run demo, 'history' to see conversation, or ask any question!")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'demo':
                run_demo()
                continue
            
            if user_input.lower() == 'history':
                print(f"\n📚 {agent.get_conversation_summary()}")
                continue
            
            if not user_input:
                continue
            
            print(f"\n🤖 Agent: Processing your query...")
            response = agent.process_query(user_input)
            
            print(f"\n💭 Reasoning: {response.thought.reasoning}")
            if response.tool_call:
                print(f"🔧 Used tool: {response.tool_call.tool_name}")
            
            print(f"\n📋 Answer: {response.final_answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Setup environment first
    setup_environment()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        interactive_mode()