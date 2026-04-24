# 🤖 AI Career Assistant for AI/ML Roles

An intelligent Streamlit application that analyzes resume compatibility with AI/ML job descriptions using OpenAI's GPT API.

## Features

- **PDF Resume Parsing** – Upload any PDF resume and extract text automatically
- **AI-Powered Analysis** – Deep compatibility analysis using GPT-4o-mini
- **Structured Scoring** – Category-wise scoring with justification (not random)
- **Actionable Insights** – Strengths, gaps, suggestions, and improved resume bullets
- **AI/ML Focus** – Evaluation tuned for machine learning, deep learning, NLP, MLOps roles

## Setup & Run Locally

### 1. Prerequisites

- Python 3.9 or higher
- An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 2. Clone / Navigate to the project

```bash
cd d:\NEW
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your API key

Create a `.env` file in the project root:

```bash
copy .env.example .env
```

Then edit `.env` and replace `sk-your-api-key-here` with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 6. Run the application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## How to Use

1. **Upload** your PDF resume using the sidebar file uploader
2. **Paste** the target AI/ML job description in the text area
3. **Click** "🚀 Analyze Resume"
4. **Review** the detailed analysis: match score, category breakdowns, strengths, gaps, and suggestions

## Scoring System

| Category            | Max Score | What It Measures                                  |
|---------------------|-----------|---------------------------------------------------|
| Skills Match        | 40        | Technical skills alignment (frameworks, languages) |
| Experience Match    | 30        | Years & depth of relevant experience              |
| Project Relevance   | 20        | How well projects map to job requirements         |
| Tools & Technologies| 10        | Specific tools, platforms, and infrastructure     |
| **Total**           | **100**   | Sum of all categories                             |

## Project Structure

```
d:\NEW\
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your API key (create from .env.example)
└── README.md           # This file
```

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **PDF Parsing**: PyPDF2
- **Environment**: python-dotenv
