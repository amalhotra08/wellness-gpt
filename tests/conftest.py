import pytest
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, User

@pytest.fixture
def test_app():
    """Create a test app instance."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        # Create a test user
        u = User(username='testuser', password_hash='pbkdf2:sha256:1234')
        db.session.add(u)
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(test_app):
    """Create a test client."""
    return test_app.test_client()

@pytest.fixture
def runner(test_app):
    """Create a test runner."""
    return test_app.test_cli_runner()

@pytest.fixture
def mock_llm_broker(mocker):
    """Mock the LlmBroker to avoid real LLM calls."""
    mock = mocker.patch('app.BROKER')
    # Default behaviors
    mock.reply_sync.return_value = "This is a mock reply."
    mock.stream_reply.return_value = ["This", " is", " a", " mock", " stream."]
    mock.get_history.return_value = []
    mock.history_size.return_value = 0
    return mock
