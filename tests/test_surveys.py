import pytest
import json
import time
from src.services.surveys import SurveyManager, phq9_interpretation, gad7_interpretation
from unittest.mock import patch, mock_open

# Sample survey data for testing
SAMPLE_SURVEYS = {
    "surveys": [
        {
            "id": "test_survey",
            "title": "Test Survey",
            "questions": [
                {"question_id": "q1", "base_text": "Q1 text", "expected_type": "string"},
                {"question_id": "q2", "base_text": "Q2 text", "expected_type": "int"}
            ],
            "completion_criteria": 2
        },
        {
             "id": "phq9",
             "title": "PHQ-9",
             "questions": []
        }
    ]
}

class TestSurveyManager:
    
    @pytest.fixture
    def manager(self):
        """Create a manager with mocked file loading."""
        with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_SURVEYS))), \
             patch("os.path.exists", return_value=True):  
            mgr = SurveyManager()
            return mgr

    def test_load_surveys(self, manager):
        """Test that surveys are loaded from JSON."""
        assert "test_survey" in manager.surveys
        assert manager.surveys["test_survey"]["title"] == "Test Survey"

    def test_start_survey(self, manager):
        """Test starting a survey initializes session state."""
        sid = "user1"
        ok = manager.start_survey(sid, "test_survey")
        assert ok is True
        
        st = manager.get_state(sid)
        assert st is not None
        assert st["survey_id"] == "test_survey"
        assert st["status"] == "active"
        assert len(st["pending"]) == 2
        assert len(st["responses"]) == 0

    def test_record_response_flow(self, manager):
        """Test recording responses and completion logic."""
        sid = "user1"
        manager.start_survey(sid, "test_survey")
        
        # 1. Get next question
        q1 = manager.get_next_question(sid)
        assert q1["question_id"] == "q1"
        
        # 2. Record answer
        ok = manager.record_response(sid, "q1", "answer1")
        assert ok is True
        
        st = manager.get_state(sid)
        assert "q1" in st["responses"]
        assert st["responses"]["q1"] == "answer1"
        assert len(st["pending"]) == 1 # q2 left
        assert st["status"] == "active"

        # 3. Record second answer -> should complete
        q2 = manager.get_next_question(sid)
        assert q2["question_id"] == "q2"
        
        ok = manager.record_response(sid, "q2", 10)
        assert ok is True
        
        assert st["status"] == "complete"
        assert len(st["pending"]) == 0
        assert manager.is_complete(sid)

    def test_start_invalid_survey(self, manager):
        """Test starting a non-existent survey fails."""
        ok = manager.start_survey("u2", "does_not_exist")
        assert ok is False

    @patch("src.services.surveys.SurveyManager._persist_session")
    def test_persist_called_on_complete(self, mock_persist, manager):
        """Test that _persist_session is called when survey completes."""
        sid = "u3"
        manager.start_survey(sid, "test_survey")
        manager.record_response(sid, "q1", "a")
        manager.record_response(sid, "q2", "b")
        
        mock_persist.assert_called_with(sid)

    def test_scoring_helpers(self):
        """Test hardcoded scoring interpretation helpers."""
        assert phq9_interpretation(0) == "Minimal or no depression"
        assert phq9_interpretation(27) == "Severe depression"
        
        assert gad7_interpretation(6) == "Mild anxiety"
        assert gad7_interpretation(20) == "Severe anxiety"
