from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    output_parameter: Optional[Dict[str, Any]] = None

class ToolCall(BaseModel):
    name: str
    label: str
    arguments: Dict[str, Any]

class NestfulRow(BaseModel):
    sample_id: str
    input: str
    output: List[ToolCall]
    tools: List[Tool]
    gold_answer: Any
