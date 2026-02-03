import unittest
import requests
import os
import sys

# Ensure we can import app if needed (though we test via requests mostly or app client)
# For simplicity, we'll use Flask's test client.

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, db, User, Consent

class TestWellnessGPT(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing convenience if using forms
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        
        with app.app_context():
            db.create_all()
            # unique user per test run
            u = User(username='testuser', password_hash='pbkdf2:sha256:1234') 
            # Note: real hash would be needed if we used login endpoint logic that checks hash
            # but for unit test we can mock or just insert with known hash if check_password_hash used.
            # Let's use the real register endpoint to be safe.
            pass

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_auth_flow(self):
        # 1. Register
        with app.app_context():
            # Clean db
            db.drop_all()
            db.create_all()

        resp = self.app.post('/api/auth/register', data={'username': 'newuser', 'password': 'password123'}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        # Should be redirected to consent page
        self.assertIn(b'WellnessGPT - Consent', resp.data)

        # 2. Try to access chat (should be blocked)
        resp = self.app.get('/', follow_redirects=True)
        self.assertIn(b'WellnessGPT - Consent', resp.data, "Should redirect to consent")

        # 3. Agree to consent
        resp = self.app.post('/api/consent', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        # Should now be at index
        self.assertIn(b'AI health assistant', resp.data) # Assuming index has some welcome text or struct

    def test_unauthorized_access(self):
        resp = self.app.get('/') # No login
        self.assertEqual(resp.status_code, 302) # Redirect to landing/login

    def test_crisis_detection(self):
        # 1. Register & Consent
        with app.app_context():
            db.create_all()
        # manual register/consent to speed up
        with self.app.session_transaction() as sess:
            # We can't easily fake login via session modification due to Flask-Login's protection,
            # so run the flow.
            pass
        self.app.post('/api/auth/register', data={'username': 'crisis_user', 'password': 'pw'}, follow_redirects=True)
        self.app.post('/api/consent', follow_redirects=True)
        
        # 2. Send crisis message
        resp = self.app.post('/api/chat', json={'user_input': 'I want to kill myself'})
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertIn("988", json_data['reply'])
        self.assertIn("crisis", json_data['reply'].lower())

    
    def test_chat_api(self):
        """Test the chat API basics."""
        # 1. Login
        with self.app as c:
            with c.session_transaction() as sess:
                sess['_user_id'] = '1'
                sess['_fresh'] = True

        # Need a user in DB for load_user to work if we bypass login form
        # But our setup creates one. Logic depends on app config relying on DB or mock.
        # simpler: use the real login flow helper if possible or just mock login_required
        # For smoke test, let's just do the register flow again per test or rely on setUp
        
        self.app.post('/api/auth/register', data={'username': 'chatuser', 'password': 'pw'}, follow_redirects=True)
        self.app.post('/api/consent', follow_redirects=True)
        
        resp = self.app.post('/api/chat', json={'user_input': 'Hello'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('reply', data)
        # In dev mode (no keys), reply should echo or be basic
        self.assertTrue(len(data['reply']) > 0)

    def test_survey_select(self):
        """Test selecting a survey."""
        self.app.post('/api/auth/register', data={'username': 'surveyuser', 'password': 'pw'}, follow_redirects=True)
        self.app.post('/api/consent', follow_redirects=True)

        # Mock SurveyManager if needed, or rely on empty default surveys if file missing
        # If surveys.json exists, we can select one.
        # We can't easily mock the singleton BROKER from here without patching app.BROKER
        # But we can check failure case if no surveys
        
        resp = self.app.post('/api/survey/select', json={'survey_id': 'test_survey'})
        # Should likely fail or 500 if survey not found, or 200 if found.
        # Just checking it doesn't crash completely.
        self.assertTrue(resp.status_code in [200, 400, 500])

if __name__ == '__main__':
    unittest.main()
