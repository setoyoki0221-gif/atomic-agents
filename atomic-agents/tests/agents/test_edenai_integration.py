"""Integration tests for Eden AI provider with Atomic Agents.

These tests require a valid EDENAI_API_KEY environment variable.
Run with: pytest -m integration tests/agents/test_edenai_integration.py
"""

import os
import pytest
import instructor
from pydantic import Field
from atomic_agents import (
    AtomicAgent,
    AgentConfig,
    BasicChatInputSchema,
    BasicChatOutputSchema,
    BaseIOSchema,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("EDENAI_API_KEY"),
    reason="EDENAI_API_KEY not set",
)


def _make_edenai_agent(model="openai/gpt-4o-mini", **agent_kwargs):
    """Helper to create an Eden AI-backed agent."""
    from openai import OpenAI

    raw = OpenAI(
        base_url="https://api.edenai.run/v3",
        api_key=os.environ["EDENAI_API_KEY"],
    )
    client = instructor.from_openai(raw)
    config = AgentConfig(client=client, model=model, **agent_kwargs)
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)


@pytest.mark.integration
class TestEdenAILiveChat:
    """Live integration tests against Eden AI API."""

    def test_basic_chat(self):
        """Test a basic chat completion with Eden AI."""
        agent = _make_edenai_agent()
        response = agent.run(BasicChatInputSchema(chat_message="Say hello in one word."))
        assert response.chat_message
        assert len(response.chat_message) > 0

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with Eden AI."""
        agent = _make_edenai_agent()
        r1 = agent.run(BasicChatInputSchema(chat_message="Remember the number 42."))
        assert r1.chat_message  # first response should be non-empty
        r2 = agent.run(BasicChatInputSchema(chat_message="What number did I ask you to remember?"))
        assert r2.chat_message
        # The model should recall "42" from the conversation history
        assert "42" in r2.chat_message

    def test_custom_output_schema(self):
        """Test structured output with a custom schema via Eden AI."""
        from openai import OpenAI

        class AnalysisOutput(BaseIOSchema):
            """Analysis output schema."""

            sentiment: str = Field(..., description="One of: positive, negative, neutral")
            confidence: float = Field(..., description="Confidence score between 0 and 1")

        raw = OpenAI(
            base_url="https://api.edenai.run/v3",
            api_key=os.environ["EDENAI_API_KEY"],
        )
        client = instructor.from_openai(raw)
        config = AgentConfig(client=client, model="openai/gpt-4o-mini")
        agent = AtomicAgent[BasicChatInputSchema, AnalysisOutput](config)

        response = agent.run(BasicChatInputSchema(chat_message="I love this product, it's amazing!"))
        assert response.sentiment in ("positive", "negative", "neutral")
        assert 0 <= response.confidence <= 1
