import operator
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    max_iterations: int
    iteration_count: int
    thread_id: Optional[str]
    uploaded_files: Annotated[list[dict[str, Any]], operator.add]
    current_task: str
    final_response: str
    carry_on: bool
    messages: Annotated[list[BaseMessage], operator.add]
    tasks: list[str]
    ragger_context: list


class NodeType(str, Enum):
    TASK = "Task"
    AGENT = "Agent"
    TOOL = "Tool"
    RESULT = "Result"


class RelationType(str, Enum):
    AGENT_USED = "AGENT_USED"
    TOOL_USED = "TOOL_USED"
    RESULTED_IN = "RESULTED_IN"


class ExtractedEntity(BaseModel):
    id: str = Field(
        description="A short, stable local identifier for this entity, e.g., tool name, task name, agent name"
    )
    task_description: dict[str, Any] = Field(
        description="a brief summary of the task or question given by user to sup agent or by sup agent to sub agent, in case of tool, it's tool arguments"
    )
    type: NodeType


class ExtractedRelation(BaseModel):
    source_id: str = Field(description="id of the source entity")
    target_id: str = Field(description="id of the target entity")
    type: RelationType
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes, if any",
    )


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
