from unittest.mock import Mock

import instructor
import pytest
from pydantic import Field

from atomic_agents import (
    AgentConfig,
    AtomicAgent,
    BaseIOSchema,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from atomic_agents.context import BaseChatHistory, ChatHistory, SystemPromptGenerator


class InputSchema(BaseIOSchema):
    """Test Input Schema"""

    test_field: str = Field(..., description="A test field")


def test_base_chat_history_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseChatHistory()


def test_chat_history_is_subclass_and_instance_of_base():
    assert issubclass(ChatHistory, BaseChatHistory)
    assert isinstance(ChatHistory(), BaseChatHistory)


def test_subclass_missing_abstractmethod_cannot_be_instantiated():
    class IncompleteHistory(BaseChatHistory):
        """A subclass that forgets to implement `copy`."""

        def initialize_turn(self) -> None:
            pass

        def add_message(self, role, content):
            pass

        def get_history(self):
            return []

        def get_current_turn_id(self):
            return None

        def delete_turn_id(self, turn_id):
            pass

        def get_message_count(self):
            return 0

        def dump(self):
            return "{}"

        def load(self, serialized_data):
            pass

        # `copy` intentionally omitted

    with pytest.raises(TypeError):
        IncompleteHistory()


class RecordingChatHistory(ChatHistory):
    """A custom persistent-memory-style backend used to exercise the extension seam."""

    def __init__(self, max_messages=None):
        super().__init__(max_messages=max_messages)
        self.recorded_calls = 0

    def add_message(self, role, content):
        self.recorded_calls += 1
        super().add_message(role, content)


@pytest.fixture
def mock_instructor():
    mock = Mock(spec=instructor.Instructor)
    mock.chat = Mock()
    mock.chat.completions = Mock()
    mock.chat.completions.create = Mock(return_value=BasicChatOutputSchema(chat_message="Test output"))
    return mock


def test_custom_chat_history_subclass_used_by_agent(mock_instructor):
    custom_history = RecordingChatHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=custom_history,
        system_prompt_generator=SystemPromptGenerator(),
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    assert isinstance(agent.history, BaseChatHistory)

    result = agent.run(BasicChatInputSchema(chat_message="Hello"))

    assert result.chat_message == "Test output"
    # One call for the user message, one for the assistant response.
    assert custom_history.recorded_calls == 2
    assert custom_history.get_message_count() == 2
