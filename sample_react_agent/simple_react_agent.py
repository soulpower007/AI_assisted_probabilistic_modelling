import openai
import json
from typing import Optional
from models import (
    AgentResponse, Thought, ToolCall, ActionType, 
    RAGSearchParams, SQLQueryParams, ToolResult
)
from tools import ToolManager

class ReACTAgent:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", db_path: str = "sample_store.db", max_history: int = 5):
        openai.api_key = api_key
        self.model = model
        self.tool_manager = ToolManager(db_path)
        self.max_history = max_history
        self.conversation_history = []  # Store (question, answer) tuples
        
        # Get database schema for context
        schema_result = self.tool_manager.get_database_schema()
        if schema_result.success:
            self.db_schema = schema_result.result
        else:
            self.db_schema = {}
    
    def add_to_history(self, question: str, answer: str):
        """Add a question-answer pair to conversation history"""
        self.conversation_history.append((question, answer))
        
        # Keep only the most recent exchanges
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_summary(self) -> str:
        """Get a formatted summary of recent conversation"""
        if not self.conversation_history:
            return "No previous conversation."
        
        summary = "Recent conversation:\n"
        for i, (q, a) in enumerate(reversed(self.conversation_history[-3:]), 1):
            summary += f"{i}. You asked: '{q}'\n   I answered: '{a[:150]}{'...' if len(a) > 150 else ''}'\n\n"
        return summary
    
    def _create_reasoning_prompt(self, user_query: str) -> str:
        """Create a prompt for the reasoning phase"""
        
        schema_info = json.dumps(self.db_schema, indent=2) if self.db_schema else "No schema available"
        
        # Format conversation history
        history_context = ""
        if self.conversation_history:
            history_context = "\n\nConversation History (most recent first):\n"
            for i, (q, a) in enumerate(reversed(self.conversation_history[-3:]), 1):  # Show last 3 exchanges
                history_context += f"{i}. Q: {q}\n   A: {a[:100]}{'...' if len(a) > 100 else ''}\n"
        
        return f"""
You are a ReACT agent that helps users by either searching a knowledge base or querying a database.

Available Database Schema:
{schema_info}

Available Actions:
1. rag_search - Search the knowledge base for general information, policies, product info, etc.
2. sql_query - Query the database for specific data about users, orders, products, etc.
3. respond - Respond directly if no tool is needed (especially for questions about conversation history)

{history_context}

Current User Query: "{user_query}"

Think step by step:
1. What is the user asking for?
2. Is this a question about policies, general information, or support? -> Use rag_search
3. Is this a question about specific data, statistics, or records? -> Use sql_query  
4. Is this about conversation history or something I can answer directly? -> Use respond
5. Can I answer this directly without tools? -> Use respond

Provide your reasoning in this JSON format (use EXACTLY these lowercase values):
{{
    "observation": "What you observe from the user query",
    "reasoning": "Your step-by-step reasoning process",
    "action_needed": "rag_search|sql_query|respond",
    "confidence": 0.95
}}
"""
    
    def _create_tool_prompt(self, user_query: str, thought: Thought) -> str:
        """Create a prompt for tool parameter generation"""
        
        if thought.action_needed == ActionType.RAG_SEARCH:
            return f"""
Based on the user query: "{user_query}"
You decided to perform a RAG search.

Generate the search parameters in this JSON format:
{{
    "query": "optimized search query for the knowledge base",
    "top_k": 5
}}
"""
        
        elif thought.action_needed == ActionType.SQL_QUERY:
            schema_info = json.dumps(self.db_schema, indent=2)
            return f"""
Based on the user query: "{user_query}"
You decided to perform a SQL query.

Database Schema:
{schema_info}

Generate a SQL query in this JSON format:
{{
    "query": "SELECT ... (valid SQL query)",
    "explanation": "Brief explanation of what this query does"
}}

Important:
- Use proper SQL syntax
- Reference only existing tables and columns
- Use appropriate WHERE clauses and JOINs
- Limit results if needed with LIMIT clause
"""
        
        return ""
    
    def _get_llm_response(self, prompt: str, response_format: str = "json") -> str:
        """Get response from OpenAI LLM"""
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f'{{"error": "LLM Error: {str(e)}"}}'
    
    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute the appropriate tool based on the tool call"""
        
        if tool_call.tool_name == "rag_search":
            params = tool_call.parameters
            return self.tool_manager.perform_rag_search(params.query, params.top_k)
            
        elif tool_call.tool_name == "sql_query":
            params = tool_call.parameters
            return self.tool_manager.execute_sql_query(params.query, params.explanation)
        
        else:
            return ToolResult(
                success=False,
                result=None,
                error=f"Unknown tool: {tool_call.tool_name}"
            )
    
    def _generate_final_answer(self, user_query: str, tool_result: ToolResult) -> str:
        """Generate a final answer based on tool results"""
        
        prompt = f"""
User Query: "{user_query}"

Tool Result: {json.dumps(tool_result.dict(), indent=2)}

Based on the tool result, provide a helpful and natural response to the user's query.
If the tool was successful, use the result data to answer the question.
If there was an error, explain what went wrong and suggest alternatives.

Respond in plain text (not JSON).
"""
        
        response = self._get_llm_response(prompt, "text")
        
        # Clean up the response if it's wrapped in JSON
        if response.startswith('{"') and response.endswith('"}'):
            try:
                parsed = json.loads(response)
                if "response" in parsed:
                    return parsed["response"]
            except:
                pass
        
        return response
    
    def process_query(self, user_query: str) -> AgentResponse:
        """Process a user query using the ReACT methodology"""
        
        print(f"\n🤔 Processing query: {user_query}")
        
        # Step 1: Reasoning phase
        reasoning_prompt = self._create_reasoning_prompt(user_query)
        reasoning_response = self._get_llm_response(reasoning_prompt)
        
        try:
            thought_data = json.loads(reasoning_response)
            
            # Normalize action_needed to lowercase if needed
            if "action_needed" in thought_data:
                action_value = thought_data["action_needed"].lower()
                # Map any variations to correct values
                if action_value in ["rag", "rag_search", "search"]:
                    thought_data["action_needed"] = "rag_search"
                elif action_value in ["sql", "sql_query", "query", "database"]:
                    thought_data["action_needed"] = "sql_query"
                elif action_value in ["respond", "response", "answer"]:
                    thought_data["action_needed"] = "respond"
                else:
                    thought_data["action_needed"] = action_value
            
            thought = Thought(**thought_data)
        except Exception as e:
            # Fallback thought
            thought = Thought(
                observation="Unable to parse reasoning",
                reasoning=f"Error in reasoning: {str(e)}",
                action_needed=ActionType.RESPOND,
                confidence=0.1
            )
        
        print(f"💭 Thought: {thought.reasoning}")
        print(f"🎯 Action: {thought.action_needed}")
        
        # Step 2: Action phase
        if thought.action_needed == ActionType.RESPOND:
            # Check if this is a conversation history question
            if any(keyword in user_query.lower() for keyword in ['previous', 'last', 'history', 'conversation', 'asked', 'before']):
                # Handle conversation history queries directly
                final_answer = self.get_conversation_summary()
            else:
                # Direct response without tools
                final_answer = self._get_llm_response(f"Answer this query directly: {user_query}", "text")
            
            response = AgentResponse(
                thought=thought,
                tool_call=None,
                final_answer=final_answer
            )
            
            # Add to history
            self.add_to_history(user_query, final_answer)
            return response
        
        else:
            # Generate tool parameters
            tool_prompt = self._create_tool_prompt(user_query, thought)
            tool_response = self._get_llm_response(tool_prompt)
            
            try:
                tool_params_data = json.loads(tool_response)
                
                if thought.action_needed == ActionType.RAG_SEARCH:
                    tool_params = RAGSearchParams(**tool_params_data)
                    tool_call = ToolCall(tool_name="rag_search", parameters=tool_params)
                    
                elif thought.action_needed == ActionType.SQL_QUERY:
                    tool_params = SQLQueryParams(**tool_params_data)
                    tool_call = ToolCall(tool_name="sql_query", parameters=tool_params)
                
                # Execute tool
                print(f"🔧 Executing tool: {tool_call.tool_name}")
                tool_result = self._execute_tool(tool_call)
                
                if tool_result.success:
                    print(f"✅ Tool executed successfully")
                    final_answer = self._generate_final_answer(user_query, tool_result)
                else:
                    print(f"❌ Tool execution failed: {tool_result.error}")
                    final_answer = f"I encountered an error: {tool_result.error}"
                
                response = AgentResponse(
                    thought=thought,
                    tool_call=tool_call,
                    final_answer=final_answer
                )
                
                # Add to history
                self.add_to_history(user_query, final_answer)
                return response
                
            except Exception as e:
                error_response = AgentResponse(
                    thought=thought,
                    tool_call=None,
                    final_answer=f"Error generating tool parameters: {str(e)}"
                )
                
                # Add to history even for errors
                self.add_to_history(user_query, error_response.final_answer)
                return error_response