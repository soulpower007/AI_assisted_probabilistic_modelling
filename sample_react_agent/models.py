from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any, Union
from enum import Enum

class ActionType(str, Enum):
    RAG_SEARCH = "rag_search"
    SQL_QUERY = "sql_query"
    RESPOND = "respond"

class Thought(BaseModel):
    """Model for the agent's reasoning process"""
    observation: str = Field(description="What the agent observes from the user query")
    reasoning: str = Field(description="The agent's reasoning process")
    action_needed: ActionType = Field(description="What action the agent decides to take")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level in the decision")

class RAGSearchParams(BaseModel):
    """Parameters for RAG search"""
    query: str = Field(description="The search query for the knowledge base")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to return")

class SQLQueryParams(BaseModel):
    """Parameters for SQL query"""
    query: str = Field(description="The SQL query to execute")
    explanation: str = Field(description="Explanation of what the query does")

class ToolCall(BaseModel):
    """Model for tool calls"""
    tool_name: str = Field(description="Name of the tool to call")
    parameters: Union[RAGSearchParams, SQLQueryParams] = Field(description="Parameters for the tool")

class AgentResponse(BaseModel):
    """Final response from the agent"""
    thought: Thought = Field(description="The agent's reasoning process")
    tool_call: Optional[ToolCall] = Field(default=None, description="Tool call if needed")
    final_answer: Optional[str] = Field(default=None, description="Final answer if no tool call needed")
    
class ToolResult(BaseModel):
    """Result from tool execution"""
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(description="The result from the tool")
    error: Optional[str] = Field(default=None, description="Error message if any")

class RAGResult(BaseModel):
    """Result from RAG search"""
    documents: List[str] = Field(description="Retrieved documents")
    scores: List[float] = Field(description="Relevance scores")
    answer: str = Field(description="Generated answer based on retrieved documents")

class SQLResult(BaseModel):
    """Result from SQL query"""
    columns: List[str] = Field(description="Column names")
    rows: List[List[Any]] = Field(description="Query result rows")
    row_count: int = Field(description="Number of rows returned")