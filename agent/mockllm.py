from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolCallingChatModel(BaseChatModel):
    """
    A fake ChatModel that behaves like ChatGoogleGenerativeAI.
    Feed it a list of AIMessages at construction time.
    Each .invoke() call returns the next message in the sequence.
    """

    # The scripted response sequence
    responses: Iterator[AIMessage] | Iterator[dict[str, Any]]

    # Tools registered via bind_tools() — stored but not used
    bound_tools: List[Any] = []

    class Config:
        # Pydantic needs this because Iterator is not a standard JSON type
        arbitrary_types_allowed = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        The ONE required method. BaseChatModel.invoke() calls this internally.
        We ignore the incoming messages and return the next scripted response.
        """
        response = next(self.responses)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingChatModel":
        """
        ChatGoogleGenerativeAI has bind_tools().
        We must too — otherwise the line:
            mock_llm_with_tools = fake_llm.bind_tools(tools_schema)
        will crash with AttributeError.
        We store the tools but don't forward them anywhere.
        """
        self.bound_tools = list(tools)
        return self  # MUST return self so chaining works

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-chat-model"
