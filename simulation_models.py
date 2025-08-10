from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any, Union, Dict
from enum import Enum

class GoalType(str, Enum):
    """Goal types for simulation optimization"""
    REVENUE = "revenue"
    CONVERSIONS = "conversions"
    PROFIT = "profit"

class GoalConfig(BaseModel):
    """Configuration for simulation goals"""
    goal: GoalType = Field(default=GoalType.REVENUE, description="The optimization goal for the simulation")
    gross_margin: float = Field(default=1.0, ge=0.0, le=1.0, description="Gross margin used only for PROFIT goal (1.0 = revenue, no COGS)")
    roas_floor: Optional[float] = Field(default=None, ge=0.0, description="Minimum ROAS requirement during allocation (e.g., 2.0)")

class SimulationParams(BaseModel):
    """Parameters for running marketing budget simulation"""
    total_budget: float = Field(default=100000.0, gt=0, description="Total marketing budget to allocate")
    n_sims: int = Field(default=200, ge=1, le=1000, description="Number of Monte Carlo simulations to run")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    linked_in_min_share: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum share of budget for LinkedIn")
    goal_config: GoalConfig = Field(default_factory=GoalConfig, description="Goal configuration for optimization")
    max_share: Optional[Dict[str, float]] = Field(default=None, description="Maximum budget share per channel (e.g., {'TikTok': 0.5})")

class ChannelSummary(BaseModel):
    """Summary statistics for a channel's allocation"""
    median_dollar: float = Field(alias="median_$", description="Median allocation in dollars")
    p05_dollar: float = Field(alias="p05_$", description="5th percentile allocation in dollars")
    p95_dollar: float = Field(alias="p95_$", description="95th percentile allocation in dollars")
    median_percent: float = Field(alias="median_%", description="Median allocation percentage")
    p05_percent: float = Field(alias="p05_%", description="5th percentile allocation percentage")
    p95_percent: float = Field(alias="p95_%", description="95th percentile allocation percentage")
    
    class Config:
        populate_by_name = True  # Allow both field names and aliases

class MetricSummary(BaseModel):
    """Summary statistics for a performance metric"""
    median: float = Field(description="Median value")
    p05: float = Field(description="5th percentile value")
    p95: float = Field(description="95th percentile value")

class SimulationResult(BaseModel):
    """Result from simulation execution"""
    allocations: Dict[str, ChannelSummary] = Field(description="Budget allocation summaries per channel")
    revenue_summary: MetricSummary = Field(description="Revenue summary statistics")
    roas_summary: MetricSummary = Field(description="ROAS summary statistics")
    conversions_summary: MetricSummary = Field(description="Conversions summary statistics")
    cac_summary: MetricSummary = Field(description="CAC summary statistics")

class ActionType(str, Enum):
    """Available actions for the simulation agent"""
    RUN_SIMULATION = "run_simulation"
    CONFIRM_PARAMETERS = "confirm_parameters"
    RESPOND = "respond"

class Thought(BaseModel):
    """Model for the agent's reasoning process"""
    observation: str = Field(description="What the agent observes from the user query")
    reasoning: str = Field(description="The agent's reasoning process")
    action_needed: ActionType = Field(description="What action the agent decides to take")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level in the decision")

class SimulationToolCall(BaseModel):
    """Model for simulation tool calls"""
    tool_name: Literal["run_simulation"] = Field(description="Name of the simulation tool")
    parameters: SimulationParams = Field(description="Parameters for the simulation")

class ConfirmationToolCall(BaseModel):
    """Model for parameter confirmation calls"""
    tool_name: Literal["confirm_parameters"] = Field(description="Name of the confirmation tool")
    parameters: SimulationParams = Field(description="Parameters to confirm")

class ParameterConfirmation(BaseModel):
    """Model for parameter confirmation before execution"""
    proposed_parameters: SimulationParams = Field(description="The parameters extracted from user query")
    confirmation_message: str = Field(description="Human-readable summary of what will be executed")

class AgentResponse(BaseModel):
    """Final response from the simulation agent"""
    thought: Thought = Field(description="The agent's reasoning process")
    tool_call: Optional[SimulationToolCall] = Field(default=None, description="Tool call if simulation needed")
    confirmation_call: Optional[ConfirmationToolCall] = Field(default=None, description="Confirmation call if parameters need approval")
    final_answer: Optional[str] = Field(default=None, description="Final answer if no tool call needed")

class ToolResult(BaseModel):
    """Result from tool execution"""
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(description="The result from the tool")
    error: Optional[str] = Field(default=None, description="Error message if any")