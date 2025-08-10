import sqlite3
from typing import List, Any
from models import SQLResult, RAGResult, ToolResult
from knowledge_base import KnowledgeBase

class SQLTool:
    def __init__(self, db_path: str = "sample_store.db"):
        self.db_path = db_path
    
    def execute_query(self, query: str) -> ToolResult:
        """Execute a SQL query and return the result"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(query)
            
            # Handle different types of queries
            if query.strip().upper().startswith('SELECT'):
                # For SELECT queries, fetch results
                columns = [description[0] for description in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                
                result = SQLResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows)
                )
                
                conn.close()
                return ToolResult(success=True, result=result.dict())
            
            else:
                # For INSERT, UPDATE, DELETE queries
                affected_rows = cursor.rowcount
                conn.commit()
                conn.close()
                
                result = SQLResult(
                    columns=["affected_rows"],
                    rows=[[affected_rows]],
                    row_count=1
                )
                
                return ToolResult(success=True, result=result.dict())
                
        except sqlite3.Error as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"SQL Error: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Unexpected error: {str(e)}"
            )
    
    def get_schema_info(self) -> ToolResult:
        """Get information about the database schema"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = {}
            
            for table in tables:
                table_name = table[0]
                
                # Get column information for each table
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                schema_info[table_name] = {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "primary_key": bool(col[5])
                        }
                        for col in columns
                    ]
                }
            
            conn.close()
            
            return ToolResult(success=True, result=schema_info)
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Error getting schema: {str(e)}"
            )

class RAGTool:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
    
    def search(self, query: str, top_k: int = 5) -> ToolResult:
        """Perform RAG search on the knowledge base"""
        
        try:
            result = self.knowledge_base.search(query, top_k)
            return ToolResult(success=True, result=result.dict())
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"RAG search error: {str(e)}"
            )

class ToolManager:
    def __init__(self, db_path: str = "sample_store.db"):
        self.sql_tool = SQLTool(db_path)
        self.rag_tool = RAGTool()
    
    def execute_sql_query(self, query: str, explanation: str = "") -> ToolResult:
        """Execute a SQL query with explanation"""
        print(f"Executing SQL Query: {explanation}")
        print(f"Query: {query}")
        return self.sql_tool.execute_query(query)
    
    def perform_rag_search(self, query: str, top_k: int = 5) -> ToolResult:
        """Perform RAG search"""
        print(f"Performing RAG Search: {query}")
        return self.rag_tool.search(query, top_k)
    
    def get_database_schema(self) -> ToolResult:
        """Get database schema information"""
        return self.sql_tool.get_schema_info()