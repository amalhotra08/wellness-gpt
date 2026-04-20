# WellnessGPT

An advanced AI wellness companion powered by an agentic, multi-expert framework.

## Getting Started

### Prerequisites
- Python 3.9+
- Pip
- (Optional) Groq API Key for LLM support

### Installation
1. Clone the repository.
2. Set up the virtual environment.
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your browser to `http://localhost:5000`.

### Groq Mode
Set `GROQ_API_KEY` in your environment or `.env` file to enable the Groq LLM integration by following the instruction below.

**Windows (Command Prompt)**

set GROQ_API_KEY=your_api_key_here

**macOS/Linux (Terminal)**

export GROQ_API_KEY="your_api_key_here"

*Dev Mode: (Not Recommended) If no `GROQ_API_KEY` is set, the app runs in "Dev Mode", echoing responses without hitting an external LLM.*

### Testing
Run the test suite using `pytest`:
```bash
pytest tests/
```

### Deployment
This application relies on an in-memory session store (`LlmBroker`). 
**Critical**: Do not use multi-worker deployments (e.g., gunicorn with multiple workers) or the session state will be fragmented. Deploy with a single worker.

## Project Structure
- `src/services/llm.py`: Core logic for the LLM broker and session management.
- `src/services/surveys.py`: Survey logic and state management.
- `app.py`: Flask entry point and API routes.
- `tests/`: Unit and integration tests.
