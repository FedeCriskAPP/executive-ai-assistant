import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from eaia.main.config import get_config


class AsyncMockOpen:
    """A mock for aiofiles.open that supports the async context manager protocol."""
    def __init__(self, read_data=""):
        self.read_data = read_data
        self.file_mock = MagicMock()
        self.file_mock.read = AsyncMock(return_value=self.read_data)
        
    async def __aenter__(self):
        return self.file_mock
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


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
async def test_get_config_from_configurable():
    # Test when config is provided in configurable
    config = {
        "configurable": {
            "email": "test@example.com",
            "full_name": "Test User",
            "name": "Test",
        }
    }
    
    result = await get_config(config)
    
    assert result == config["configurable"]
    assert result["email"] == "test@example.com"
    assert result["full_name"] == "Test User"
    assert result["name"] == "Test"


@pytest.mark.asyncio
async def test_get_config_from_file(mock_config_data):
    # Test when config is loaded from file
    config = {"configurable": {}}
    
    # Mock the aiofiles.open function
    mock_file_content = yaml.dump(mock_config_data)
    async_mock = AsyncMockOpen(read_data=mock_file_content)
    
    with patch("aiofiles.open", return_value=async_mock):
        result = await get_config(config)
    
    assert result == mock_config_data
    assert result["email"] == "test@example.com"
    assert result["full_name"] == "Test User"
    assert result["name"] == "Test"
