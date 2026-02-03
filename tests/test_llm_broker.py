import pytest
import os
from src.services.llm import LlmBroker
from unittest.mock import MagicMock, patch

class TestLlmBroker:
    @pytest.fixture
    def broker(self):
        """Create a broker instance with dev mode (no API key by default)."""
        # Ensure strict dev mode by clearing env vars if needed
        with patch.dict(os.environ, {}, clear=True):
             # Also patch GroqClient to be None so it forces dev mode
            with patch('src.services.llm.GroqClient', None):
                broker = LlmBroker()
                return broker

    def test_crisis_detection_active(self, broker):
        """Test that crisis keywords trigger the safety response."""
        crisis_inputs = [
            "I want to kill myself",
            "I am thinking of suicide",
            "hurt myself",
        ]
        for text in crisis_inputs:
            resp = broker._check_crisis(text)
            assert resp is not None
            assert "988" in resp
            assert "crisis" in resp.lower()

    def test_crisis_detection_safe(self, broker):
        """Test that normal text does not trigger crisis response."""
        safe_inputs = [
            "I am feeling sad",
            "I want to loose weight",
            "kill the process",  # "kill" without "myself"
        ]
        for text in safe_inputs:
            assert broker._check_crisis(text) is None

    def test_reply_sync_dev_mode(self, broker):
        """Test synchronous reply in dev mode (no external LLM)."""
        # In dev mode, it should just echo with some wrapper
        # We need to mock session history calls since they touch DB (or use in-memory if refactored, 
        # but LlmBroker currently tries to use DB models).
        # Since we didn't refactor LlmBroker to take a storage backend, we must mock the methods that use DB.
        
        with patch.object(broker, 'add_user') as mock_add_user, \
             patch.object(broker, 'add_assistant') as mock_add_assistant, \
             patch.object(broker, 'get_history', return_value=[]), \
             patch.object(broker, 'maybe_condense_history'):
            
            reply = broker.reply_sync("session_1", "Hello world")
            
            assert "You said: Hello world" in reply
            mock_add_user.assert_called_once()
            mock_add_assistant.assert_called_once()
            
    def test_reply_sync_with_citations_forced(self, broker):
        """Test that citations are attached when forced (or always_cite is True)."""
        # Mock citations
        with patch('src.services.llm.citations_v2_gather', return_value=[
            {"source": "test_src", "title": "Test Title", "url": "http://test.com", "year": 2023, "evidence_level": "Level_1"}
        ]):
            with patch.object(broker, 'add_user'), \
                 patch.object(broker, 'add_assistant'), \
                 patch.object(broker, 'get_history', return_value=[]), \
                 patch.object(broker, 'maybe_condense_history'):
                
                broker.always_cite = True # Force it
                reply = broker.reply_sync("session_1", "explain keto")
                
                assert "References:" in reply
                assert "Test Title" in reply

    def test_message_assembly(self, broker):
        """Test that messages are assembled correctly from history."""
        # Mock history
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        with patch.object(broker, 'get_history', return_value=history):
            msgs = broker._messages("session_1")
            
            assert len(msgs) == 3 # System + User + Assistant
            assert msgs[0]["role"] == "system"
            assert msgs[1]["content"] == "Hi"

    def test_fallback_logic_logs_error(self, broker):
        """Test that if provider fails, we log it. (Simulating failure in non-dev mode requires more setup, skipping for simple dev unit test)."""
        pass
