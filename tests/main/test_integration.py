import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from eaia.main.triage import triage_input
from eaia.main.config import get_config
from eaia.schemas import RespondTo, State
from tests.main.test_config import AsyncMockOpen


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


@pytest.fixture
def mock_config_data():
    return {
        "email": "test@example.com",
        "full_name": "Test User",
        "name": "Test",
        "background": "Test background",
        "triage_no": "Test triage no",
        "triage_email": "Test triage email",
        "triage_notify": "Test triage notify",
    }


@pytest.mark.asyncio
async def test_triage_with_config_integration(mock_state, mock_config, mock_config_data):
    """
    Integration test that verifies the triage_input function works correctly with the get_config function.
    This test mocks the file I/O but tests the actual integration between the two functions.
    """
    # Mock dependencies
    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=None)
    
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    mock_llm.bind = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=RespondTo(response="email"))
    
    # Mock the get_few_shot_examples function
    mock_examples = "Mock examples"
    
    # Mock the aiofiles.open function for config.yaml
    mock_file_content = yaml.dump(mock_config_data)
    async_mock = AsyncMockOpen(read_data=mock_file_content)
    
    with patch("eaia.main.triage.ChatOpenAI", return_value=mock_llm), \
         patch("eaia.main.triage.get_few_shot_examples", AsyncMock(return_value=mock_examples)), \
         patch("aiofiles.open", return_value=async_mock):
        
        # Call the triage_input function which will call the real get_config function
        result = await triage_input(mock_state, mock_config, mock_store)
    
    # Verify the result
    assert "triage" in result
    assert result["triage"].response == "email"
    assert "messages" not in result


@pytest.mark.asyncio
async def test_triage_with_configurable_integration(mock_state, mock_config_data):
    """
    Integration test that verifies the triage_input function works correctly with the get_config function
    when the config is provided in the configurable parameter.
    """
    # Set up the config with the configurable parameter
    config = {
        "configurable": mock_config_data.copy()
    }
    config["configurable"]["model"] = "gpt-4o"
    
    # Mock dependencies
    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=None)
    
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    mock_llm.bind = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=RespondTo(response="notify"))
    
    # Mock the get_few_shot_examples function
    mock_examples = "Mock examples"
    
    with patch("eaia.main.triage.ChatOpenAI", return_value=mock_llm), \
         patch("eaia.main.triage.get_few_shot_examples", AsyncMock(return_value=mock_examples)):
        
        # Call the triage_input function which will call the real get_config function
        result = await triage_input(mock_state, config, mock_store)
    
    # Verify the result
    assert "triage" in result
    assert result["triage"].response == "notify"
    assert "messages" not in result
