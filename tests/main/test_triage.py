import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from eaia.main.triage import triage_input
from eaia.schemas import RespondTo, State


@pytest.fixture
def mock_state():
    return State(
        email={
            "page_content": "Test email content",
            "from_email": "sender@example.com",
            "to_email": "recipient@example.com",
            "subject": "Test Subject",
        },
        messages=[],
    )


@pytest.fixture
def mock_config():
    return {
        "configurable": {
            "model": "gpt-4o",
        }
    }


@pytest.mark.asyncio
async def test_triage_input(mock_state, mock_config):
    # Mock dependencies
    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=None)
    
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    mock_llm.bind = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=RespondTo(response="email"))
    
    # Mock the get_few_shot_examples function
    mock_examples = "Mock examples"
    
    # Mock the get_config function
    mock_config_data = {
        "name": "Test",
        "full_name": "Test User",
        "background": "Test background",
        "triage_no": "Test triage no",
        "triage_email": "Test triage email",
        "triage_notify": "Test triage notify",
    }
    
    with patch("eaia.main.triage.ChatOpenAI", return_value=mock_llm), \
         patch("eaia.main.triage.get_few_shot_examples", AsyncMock(return_value=mock_examples)), \
         patch("eaia.main.triage.get_config", AsyncMock(return_value=mock_config_data)):
        
        result = await triage_input(mock_state, mock_config, mock_store)
    
    # Verify the result
    assert "triage" in result
    assert result["triage"].response == "email"
    assert "messages" not in result


@pytest.mark.asyncio
async def test_triage_input_with_messages(mock_state, mock_config):
    # Add messages to the state
    message = MagicMock()
    message.id = "test_id"
    mock_state["messages"] = [message]
    
    # Mock dependencies
    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=None)
    
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    mock_llm.bind = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=RespondTo(response="email"))
    
    # Mock the get_few_shot_examples function
    mock_examples = "Mock examples"
    
    # Mock the get_config function
    mock_config_data = {
        "name": "Test",
        "full_name": "Test User",
        "background": "Test background",
        "triage_no": "Test triage no",
        "triage_email": "Test triage email",
        "triage_notify": "Test triage notify",
    }
    
    with patch("eaia.main.triage.ChatOpenAI", return_value=mock_llm), \
         patch("eaia.main.triage.get_few_shot_examples", AsyncMock(return_value=mock_examples)), \
         patch("eaia.main.triage.get_config", AsyncMock(return_value=mock_config_data)):
        
        result = await triage_input(mock_state, mock_config, mock_store)
    
    # Verify the result
    assert "triage" in result
    assert result["triage"].response == "email"
    assert "messages" in result
    assert len(result["messages"]) == 1
