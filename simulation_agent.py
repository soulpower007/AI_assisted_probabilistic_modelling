from openai import OpenAI
import json
import re
from typing import Optional, Dict, Any
from simulation_models import (
    AgentResponse, Thought, SimulationToolCall, ActionType, 
    SimulationParams, GoalConfig, GoalType, ToolResult
)
from simulation_tools import SimulationToolManager

class SimulationReACTAgent:
    """ReACT Agent specialized for marketing budget simulation requests"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_history: int = 10):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tool_manager = SimulationToolManager()
        self.max_history = max_history
        self.conversation_history = []  # Store (question, answer) tuples
        self.pending_parameters = None  # Store parameters awaiting confirmation
    
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
        for i, (q, a) in enumerate(reversed(self.conversation_history[-2:]), 1):
            summary += f"{i}. You asked: '{q}'\n   I answered: '{a[:200]}{'...' if len(a) > 200 else ''}'\n\n"
        return summary
    
    def format_parameters_for_confirmation(self, params: SimulationParams) -> str:
        """Format simulation parameters in a human-readable way for confirmation"""
        
        # Format budget
        budget_str = f"${params.total_budget:,.2f}"
        
        # Format goal
        goal_str = params.goal_config.goal.value.title()
        if params.goal_config.goal == GoalType.PROFIT:
            goal_str += f" (with {params.goal_config.gross_margin*100:.1f}% gross margin)"
        
        if params.goal_config.roas_floor:
            goal_str += f" and minimum {params.goal_config.roas_floor:.1f}x ROAS"
        
        # Format constraints
        constraints = []
        if params.linked_in_min_share > 0.05:  # Only show if different from default
            constraints.append(f"LinkedIn minimum: {params.linked_in_min_share*100:.1f}%")
        
        if params.max_share:
            for channel, max_val in params.max_share.items():
                constraints.append(f"{channel} maximum: {max_val*100:.1f}%")
        
        constraint_str = " with " + ", ".join(constraints) if constraints else ""
        
        # Format simulation settings
        sim_str = f"{params.n_sims} simulations"
        if params.seed != 42:  # Only show if different from default
            sim_str += f" (seed: {params.seed})"
        
        confirmation_msg = f"""
📊 **Simulation Parameters to Confirm:**

💰 **Budget**: {budget_str}
🎯 **Goal**: {goal_str}
🔧 **Simulations**: {sim_str}
⚙️ **Grid Step**: 10% increments{constraint_str}

Is this correct? Type 'yes' to proceed or 'no' to cancel.
You can also modify by saying something like "change budget to $75k" or "set TikTok max to 40%"."""
        
        return confirmation_msg.strip()
    
    def _create_reasoning_prompt(self, user_query: str) -> str:
        """Create a prompt for the reasoning phase"""
        
        # Format conversation history
        history_context = ""
        if self.conversation_history:
            history_context = f"""
Previous Context:
{self.get_conversation_summary()}
"""
        
        prompt = f"""You are a marketing budget optimization assistant that helps users run simulations to find optimal budget allocations across marketing channels.

{history_context}

Current User Query: "{user_query}"

Available Information:
- You can run marketing budget simulations with the following parameters:
  * total_budget: Total marketing budget (default: $100,000)
  * n_sims: Number of simulations (default: 200, max: 1000)
  * seed: Random seed for reproducibility (default: 42)
  * linked_in_min_share: Minimum LinkedIn budget share (default: 0.05 = 5%)
  * linked_in_min_share: Minimum LinkedIn budget share (default: 5%)
  * goal_config: Optimization goal configuration
    - goal: "revenue", "conversions", or "profit"
    - gross_margin: For profit goal (0.0 to 1.0, default: 1.0)
    - roas_floor: Minimum ROAS requirement (optional)
  * max_share: Maximum budget share per channel (optional, e.g., {{"TikTok": 0.5}})

Available Channels: LinkedIn, Facebook, Google, TikTok, YouTube, Twitter

Your task is to analyze the user's request and determine:
1. What they are observing/asking for
2. Your reasoning about their request
3. Whether you need to confirm parameters, run a simulation, or respond directly
4. Your confidence level

Available actions:
- "confirm_parameters": Extract parameters and ask for user confirmation before running (use this for NEW simulation requests)
- "run_simulation": Run simulation directly (only if user has already confirmed parameters)  
- "respond": Provide direct response without simulation

IMPORTANT: For most budget optimization requests, use "confirm_parameters" first!

Respond with a JSON object following this exact structure:
{{
  "thought": {{
    "observation": "What you observe from the user's query",
    "reasoning": "Your detailed reasoning process",
    "action_needed": "confirm_parameters" or "run_simulation" or "respond",
    "confidence": 0.0 to 1.0
  }},
  "tool_call": {{
    "tool_name": "run_simulation",
    "parameters": {{
      "total_budget": number,
      "n_sims": number,
      "seed": number,
      "linked_in_min_share": number,
      "goal_config": {{
        "goal": "revenue|conversions|profit",
        "gross_margin": number,
        "roas_floor": number or null
      }},
      "max_share": {{"channel": number}} or null
    }}
  }} // Only include if action_needed is "run_simulation",
  "confirmation_call": {{
    "tool_name": "confirm_parameters",
    "parameters": {{
      "total_budget": number,
      "n_sims": number,
      "seed": number,
      "linked_in_min_share": number,
      "goal_config": {{
        "goal": "revenue|conversions|profit",
        "gross_margin": number,
        "roas_floor": number or null
      }},
      "max_share": {{"channel": number}} or null
    }}
  }} // Only include if action_needed is "confirm_parameters",
  "final_answer": "Your direct response" // Only include if action_needed is "respond"
}}

Guidelines for parameter extraction:
- Extract budget amounts from phrases like "$50k", "$50,000", "fifty thousand dollars"
- Recognize optimization goals: "maximize revenue", "get more conversions", "optimize for profit"
- Identify constraints: "at least 20% on LinkedIn", "no more than 50% on TikTok"
- Default to reasonable values if not specified
- If the user asks general questions about marketing or simulations without wanting to run one, use "respond" action

Example responses:

For "Optimize $50k budget for revenue":
{{
  "thought": {{
    "observation": "User wants to optimize a $50k budget for revenue",
    "reasoning": "This is a new simulation request, I should extract parameters and confirm before running",
    "action_needed": "confirm_parameters",
    "confidence": 0.9
  }},
  "confirmation_call": {{
    "tool_name": "confirm_parameters",
    "parameters": {{
      "total_budget": 50000.0,
      "goal_config": {{"goal": "revenue"}},
      "n_sims": 200,
      "linked_in_min_share": 0.05
    }}
  }}
}}

For general questions:
{{
  "thought": {{
    "observation": "User is asking about marketing in general",
    "reasoning": "No simulation needed, just provide information",
    "action_needed": "respond",
    "confidence": 0.8
  }},
  "final_answer": "I can help you optimize marketing budgets..."
}}"""

        return prompt
    
    def _extract_parameters_from_query(self, user_query: str) -> SimulationParams:
        """Extract simulation parameters from user query with intelligent defaults"""
        
        params = SimulationParams()  # Start with defaults
        
        # Extract budget
        budget_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $100,000 or $100,000.00
            r'(\d+)k',  # 50k
            r'(\d+)\s*thousand',  # 50 thousand
            r'budget.*?(\d+)',  # budget of 50000
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, user_query.lower())
            if match:
                if 'k' in pattern:
                    params.total_budget = float(match.group(1)) * 1000
                elif 'thousand' in pattern:
                    params.total_budget = float(match.group(1)) * 1000
                else:
                    # Remove commas and convert
                    budget_str = match.group(1).replace(',', '')
                    params.total_budget = float(budget_str)
                break
        
        # Extract goal type
        if any(word in user_query.lower() for word in ['profit', 'profitability', 'margin']):
            params.goal_config.goal = GoalType.PROFIT
            # Look for margin information
            margin_match = re.search(r'margin.*?(\d+(?:\.\d+)?)%?', user_query.lower())
            if margin_match:
                margin_val = float(margin_match.group(1))
                if margin_val > 1:  # Assume percentage
                    params.goal_config.gross_margin = margin_val / 100
                else:
                    params.goal_config.gross_margin = margin_val
        elif any(word in user_query.lower() for word in ['conversion', 'convert', 'leads']):
            params.goal_config.goal = GoalType.CONVERSIONS
        else:
            params.goal_config.goal = GoalType.REVENUE
        
        # Extract ROAS floor
        roas_match = re.search(r'roas.*?(\d+(?:\.\d+)?)', user_query.lower())
        if roas_match:
            params.goal_config.roas_floor = float(roas_match.group(1))
        
        # Extract LinkedIn minimum share
        linkedin_match = re.search(r'linkedin.*?(\d+(?:\.\d+)?)%?', user_query.lower())
        if linkedin_match:
            linkedin_val = float(linkedin_match.group(1))
            if linkedin_val > 1:  # Assume percentage
                params.linked_in_min_share = linkedin_val / 100
            else:
                params.linked_in_min_share = linkedin_val
        
        # Extract maximum shares
        max_share = {}
        channels = ['tiktok', 'facebook', 'google', 'youtube', 'twitter', 'linkedin']
        for channel in channels:
            pattern = f'{channel}.*?(?:max|maximum|cap|limit).*?(\\d+(?:\\.\\d+)?)%?'
            match = re.search(pattern, user_query.lower())
            if match:
                max_val = float(match.group(1))
                if max_val > 1:  # Assume percentage
                    max_share[channel.capitalize()] = max_val / 100
                else:
                    max_share[channel.capitalize()] = max_val
        
        if max_share:
            params.max_share = max_share
        
        # Extract number of simulations
        sims_match = re.search(r'(\d+)\s*simulations?', user_query.lower())
        if sims_match:
            params.n_sims = min(int(sims_match.group(1)), 1000)  # Cap at 1000
        
        return params
    
    def handle_confirmation_response(self, user_response: str) -> tuple[bool, str]:
        """
        Handle user's confirmation response
        Returns (should_proceed, response_message)
        """
        response_lower = user_response.lower().strip()
        
        # Check for confirmation
        if any(word in response_lower for word in ['yes', 'y', 'confirm', 'proceed', 'ok', 'correct', 'right']):
            return True, "Great! Running the simulation now..."
        
        # Check for cancellation
        if any(word in response_lower for word in ['no', 'n', 'cancel', 'stop', 'wrong']):
            self.pending_parameters = None
            return False, "Simulation cancelled. Feel free to ask for a new optimization with different parameters!"
        
        # Check for modifications
        if any(word in response_lower for word in ['change', 'modify', 'update', 'set', 'make']):
            # Try to extract new parameters from the modification request
            try:
                modified_params = self._extract_parameters_from_query(user_response)
                
                # Update parameters based on what was extracted
                # Check if budget was modified (look for significant difference from current)
                if abs(modified_params.total_budget - self.pending_parameters.total_budget) > 1000:
                    self.pending_parameters.total_budget = modified_params.total_budget
                
                # Check if goal was modified
                if modified_params.goal_config.goal != self.pending_parameters.goal_config.goal:
                    self.pending_parameters.goal_config.goal = modified_params.goal_config.goal
                
                if abs(modified_params.goal_config.gross_margin - self.pending_parameters.goal_config.gross_margin) > 0.01:
                    self.pending_parameters.goal_config.gross_margin = modified_params.goal_config.gross_margin
                
                if modified_params.goal_config.roas_floor != self.pending_parameters.goal_config.roas_floor:
                    self.pending_parameters.goal_config.roas_floor = modified_params.goal_config.roas_floor
                
                # Check if simulations count was modified
                if abs(modified_params.n_sims - self.pending_parameters.n_sims) > 10:
                    self.pending_parameters.n_sims = modified_params.n_sims
                
                # Check if LinkedIn minimum was modified
                if abs(modified_params.linked_in_min_share - self.pending_parameters.linked_in_min_share) > 0.01:
                    self.pending_parameters.linked_in_min_share = modified_params.linked_in_min_share
                
                # Update max share constraints
                if modified_params.max_share:
                    if self.pending_parameters.max_share is None:
                        self.pending_parameters.max_share = {}
                    self.pending_parameters.max_share.update(modified_params.max_share)
                
                # Return updated confirmation
                confirmation_msg = self.format_parameters_for_confirmation(self.pending_parameters)
                return False, f"Parameters updated! Here are the new settings:\n\n{confirmation_msg}"
                
            except Exception as e:
                return False, f"I couldn't understand the modification. Please try again or type 'yes' to proceed with current parameters, or 'no' to cancel."
        
        # Unclear response
        return False, "Please respond with 'yes' to confirm, 'no' to cancel, or describe what you'd like to change."
    
    def process_query(self, user_query: str) -> str:
        """Process a user query and return a response"""
        
        try:
            # Check if we have pending parameters awaiting confirmation
            if self.pending_parameters is not None:
                should_proceed, response_msg = self.handle_confirmation_response(user_query)
                
                if should_proceed:
                    # Run the simulation with confirmed parameters
                    result = self.tool_manager.run_simulation(self.pending_parameters)
                    self.pending_parameters = None  # Clear pending parameters
                    
                    if result.success:
                        formatted_results = self.tool_manager.format_simulation_results(result.result)
                        final_response = f"{response_msg}\n\n{formatted_results}"
                    else:
                        final_response = f"{response_msg}\n\nHowever, I encountered an error: {result.error}"
                    
                    self.add_to_history(user_query, final_response)
                    return final_response
                else:
                    # Return response message (could be cancellation or request for modification)
                    self.add_to_history(user_query, response_msg)
                    return response_msg
            
            # No pending parameters, process normally
            # Create reasoning prompt
            prompt = self._create_reasoning_prompt(user_query)
            
            # Get reasoning from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up the response text to ensure valid JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                agent_response_data = json.loads(response_text)
                agent_response = AgentResponse(**agent_response_data)
            except json.JSONDecodeError as e:
                return f"Error parsing agent response: {str(e)}. Raw response: {response_text}"
            
            # Execute the action
            if agent_response.thought.action_needed == ActionType.CONFIRM_PARAMETERS:
                # Extract parameters and ask for confirmation
                if agent_response.confirmation_call is None:
                    params = self._extract_parameters_from_query(user_query)
                else:
                    params = agent_response.confirmation_call.parameters
                
                # Store parameters for confirmation
                self.pending_parameters = params
                
                # Format confirmation message
                confirmation_msg = self.format_parameters_for_confirmation(params)
                self.add_to_history(user_query, confirmation_msg)
                return confirmation_msg
                
            elif agent_response.thought.action_needed == ActionType.RUN_SIMULATION:
                # Direct simulation (should only happen if already confirmed)
                if agent_response.tool_call is None:
                    params = self._extract_parameters_from_query(user_query)
                else:
                    params = agent_response.tool_call.parameters
                
                # Run the simulation
                result = self.tool_manager.run_simulation(params)
                
                if result.success:
                    formatted_results = self.tool_manager.format_simulation_results(result.result)
                    final_response = f"I've run the simulation based on your request. Here are the results:\n\n{formatted_results}"
                else:
                    final_response = f"I encountered an error while running the simulation: {result.error}"
                
                self.add_to_history(user_query, final_response)
                return final_response
                
            else:
                # Direct response
                final_response = agent_response.final_answer or "I can help you run marketing budget simulations. Please describe what you'd like to optimize for."
                self.add_to_history(user_query, final_response)
                return final_response
                
        except Exception as e:
            error_message = f"I encountered an error processing your request: {str(e)}"
            self.add_to_history(user_query, error_message)
            return error_message