# IntervAI

IntervAI is your AI-powered personal interviewer. This Streamlit app uses
Google Gemini to generate role-specific interview questions, evaluate answers,
adapt the next question to demonstrated performance, and prepare a report from
the complete interview.

## Features

- Select from software, data, AI/ML, and DevOps roles.
- Choose an experience level and a 3–10 question interview.
- Receive per-answer scores, strengths, improvements, and an ideal answer.
- Adapt question difficulty based on the previous answer's score and feedback.
- Generate an end-of-interview report covering all submitted answers.
- Preserve interview state and display categorized, user-friendly Gemini errors.
- Avoid repeating questions within an interview or between sessions for the
  same role, experience level, and interview type.

## Completed interview history

Completed interviews are stored separately in
`data\interview_history.json`. Each record contains the interview setup,
questions, submitted answers, saved evaluations, and final report; it does not
contain API credentials. The history page shows score statistics and saved
reports without making Gemini requests. Individual records and the full
history can be deleted after confirmation. This local history file is excluded
by `.gitignore`.

## Interview modes

- **Technical:** role-related concepts, architecture, and practical knowledge.
- **HR:** behavioral and professional questions.
- **Coding:** programming and role-relevant problem-solving questions.
- **Full Interview:** a mix of HR, Technical, and Coding questions.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

## Setup

From the project directory, activate your virtual environment and install the
dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.streamlit\secrets.toml` using
`.streamlit\secrets.toml.example` as a template, then add your own key:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

The application reads the key from Streamlit secrets. Never commit
`.streamlit\secrets.toml` or put the key in source code.

## Gemini models and quotas

The app uses `gemini-3.5-flash-lite`. Transient service failures receive a
small, bounded retry with backoff; quota exhaustion is not repeatedly retried.
If Gemini remains unavailable or the quota is exhausted, the app explains the
failure and preserves the current interview so you can retry later. Gemini
free-tier quotas and reset times are controlled by Google.

## Local question history

Successfully used questions are stored locally in
`data\question_history.json`, grouped by role, experience level, and interview
type. Only normalized question text is saved; candidate answers are not stored
in this file. The history file is excluded by `.gitignore` and remains on the
machine when a new interview starts. Use **Clear Question History** in the
sidebar to remove it after confirmation.

## Run

```powershell
streamlit run app.py
```
