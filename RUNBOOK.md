# WellnessGPT Runbook

## Setup
1.  **Prerequisites**: Python 3.10+ (Anaconda or Homebrew).
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```bash
    FLASK_APP=app.py
    SECRET_KEY=your-secure-secret-key-here
    DATABASE_URL=sqlite:///hms_gami.sqlite
    # GROQ_API_KEY=... (optional)
    ```

## Database
Initialize the database:
```bash
# Initialize DB tables (creates hms_gami.sqlite)
python -m flask db init   # Only first time
python -m flask db migrate -m "Initial migration"
python -m flask db upgrade
```
*Note: The app also attempts to auto-create tables on startup via `db.create_all()`.*

## Running the Server
```bash
python app.py
```
Access at `http://localhost:5000`.

## Verification
Run the smoke tests:
```bash
python tests/test_smoke.py
```

## Security Features
- **Consent Gating**: Users must agree to terms before accessing chat.
- **Secure Sessions**: Cookies are `HttpOnly` and `SameSite=Lax`.
- **Environment Config**: Secrets are loaded from `.env`.
