"""Unit tests for Eden AI provider integration with Atomic Agents."""

import os
import pytest
from unittest.mock import Mock, patch
import instructor
from atomic_agents import (
    AtomicAgent,
    AgentConfig,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from atomic_agents.context import SystemPromptGenerator


def _create_edenai_client(api_key="test-key"):
    """Create an Eden AI client via OpenAI-compatible interface."""
    from openai import OpenAI

    return instructor.from_openai(OpenAI(base_url="https://api.edenai.run/v3", api_key=api_key))


class TestEdenAIClientSetup:
    """Tests for Eden AI client initialization."""

    def test_edenai_client_creation(self):
        """Test that Eden AI client can be created with correct base_url."""
        from openai import OpenAI

        raw_client = OpenAI(base_url="https://api.edenai.run/v3", api_key="test-key")
        assert raw_client.base_url == "https://api.edenai.run/v3/"

    def test_edenai_instructor_wrapping(self):
        """Test that Eden AI client can be wrapped with instructor."""
        client = _create_edenai_client()
        assert isinstance(client, instructor.Instructor)

    def test_edenai_agent_config(self):
        """Test that AgentConfig accepts Eden AI client and model."""
        client = _create_edenai_client()
        config = AgentConfig(
            client=client,
            model="openai/gpt-4o-mini",
        )
        assert config.model == "openai/gpt-4o-mini"
        assert config.assistant_role == "assistant"

    def test_edenai_agent_initialization(self):
        """Test that AtomicAgent can be initialized with Eden AI config."""
        client = _create_edenai_client()
        config = AgentConfig(
            client=client,
            model="openai/gpt-4o-mini",
        )
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
        assert agent.model == "openai/gpt-4o-mini"
        assert agent.assistant_role == "assistant"

    def test_edenai_vendor_prefixed_model(self):
        """Test that a different vendor-prefixed Eden AI model id works."""
        client = _create_edenai_client()
        config = AgentConfig(
            client=client,
            model="anthropic/claude-haiku-4-5",
        )
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
        assert agent.model == "anthropic/claude-haiku-4-5"


class TestEdenAIAgentBehavior:
    """Tests for agent behavior with Eden AI provider."""

    @pytest.fixture
    def mock_edenai_instructor(self):
        mock = Mock(spec=instructor.Instructor)
        mock.chat = Mock()
        mock.chat.completions = Mock()
        mock.chat.completions.create = Mock(return_value=BasicChatOutputSchema(chat_message="Eden AI response"))
        mock_response = BasicChatOutputSchema(chat_message="Eden AI response")
        mock_iter = Mock()
        mock_iter.__iter__ = Mock(return_value=iter([mock_response]))
        mock.chat.completions.create_partial.return_value = mock_iter
        return mock

    @pytest.fixture
    def edenai_agent(self, mock_edenai_instructor):
        config = AgentConfig(
            client=mock_edenai_instructor,
            model="openai/gpt-4o-mini",
        )
        return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    def test_run_with_edenai(self, edenai_agent, mock_edenai_instructor):
        """Test that agent.run works with Eden AI mock client."""
        user_input = BasicChatInputSchema(chat_message="Hello from Eden AI test")
        response = edenai_agent.run(user_input)
        assert response.chat_message == "Eden AI response"
        mock_edenai_instructor.chat.completions.create.assert_called_once()

    def test_run_passes_correct_model(self, edenai_agent, mock_edenai_instructor):
        """Test that the correct model name is passed to the API."""
        user_input = BasicChatInputSchema(chat_message="Test")
        edenai_agent.run(user_input)
        call_kwargs = mock_edenai_instructor.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o-mini"

    def test_run_stream_with_edenai(self, edenai_agent):
        """Test that streaming works with Eden AI mock client."""
        user_input = BasicChatInputSchema(chat_message="Stream test")
        responses = list(edenai_agent.run_stream(user_input))
        assert len(responses) == 1
        assert responses[0].chat_message == "Eden AI response"

    def test_history_tracking_with_edenai(self, edenai_agent):
        """Test that chat history is properly tracked."""
        user_input = BasicChatInputSchema(chat_message="First message")
        edenai_agent.run(user_input)
        history = edenai_agent.history.get_history()
        assert len(history) == 2  # user + assistant

    def test_system_prompt_with_edenai(self, mock_edenai_instructor):
        """Test that system prompt works correctly with Eden AI."""
        spg = SystemPromptGenerator(
            background=["You are a helpful Eden AI-powered assistant."],
        )
        config = AgentConfig(
            client=mock_edenai_instructor,
            model="openai/gpt-4o-mini",
            system_prompt_generator=spg,
        )
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
        prompt = agent.system_prompt_generator.generate_prompt()
        assert "Eden AI" in prompt

    def test_model_api_parameters_with_edenai(self, mock_edenai_instructor):
        """Test that custom API parameters are passed through."""
        config = AgentConfig(
            client=mock_edenai_instructor,
            model="openai/gpt-4o-mini",
            model_api_parameters={"temperature": 0.7, "max_tokens": 1024},
        )
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
        assert agent.model_api_parameters["temperature"] == 0.7
        assert agent.model_api_parameters["max_tokens"] == 1024

    def test_edenai_reset_history(self, edenai_agent):
        """Test that history reset works with Eden AI agent."""
        user_input = BasicChatInputSchema(chat_message="Test")
        edenai_agent.run(user_input)
        edenai_agent.reset_history()
        history = edenai_agent.history.get_history()
        assert len(history) == 0


class TestEdenAIProviderSetup:
    """Tests for the provider setup function from the quickstart example."""

    def test_setup_client_edenai_pattern(self):
        """Test the Eden AI client setup pattern used by the quickstart example."""
        from openai import OpenAI

        api_key = "test-edenai-key"
        raw_client = OpenAI(base_url="https://api.edenai.run/v3", api_key=api_key)
        client = instructor.from_openai(raw_client)
        assert isinstance(client, instructor.Instructor)

    def test_edenai_env_var_detection(self):
        """Test that EDENAI_API_KEY env var can be used."""
        with patch.dict(os.environ, {"EDENAI_API_KEY": "test-env-key"}):
            api_key = os.getenv("EDENAI_API_KEY")
            assert api_key == "test-env-key"
