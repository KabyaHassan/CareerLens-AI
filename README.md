# 🤖 AI Career Assistant for AI/ML Roles

An intelligent Streamlit web application that analyzes resume compatibility with AI/ML job descriptions using OpenAI’s GPT model.

## 🚀 Live Demo

The application is deployed as a live web app:  
👉 [Launch App](https://careerlensai0.streamlit.app/)

---

## ✨ Features

- 📄 **PDF Resume Parsing** – Upload and extract resume text automatically  
- 🧠 **AI-Powered Analysis** – GPT-based resume vs job description evaluation  
- 📊 **Structured Scoring System** – Category-wise compatibility scoring  
- 🔍 **Actionable Insights** – Strengths, gaps, and improvement suggestions  
- 🎯 **AI/ML Focused** – Optimized for ML, DL, NLP, and MLOps roles  

---

## ⚙️ Tech Stack

- Streamlit  
- OpenAI GPT-4o-mini  
- PyPDF2  
- Python-dotenv  

---

## 📊 Scoring System

| Category             | Max Score | Description |
|---------------------|-----------|-------------|
| Skills Match         | 40        | Technical skill alignment |
| Experience Match     | 30        | Relevant experience depth |
| Project Relevance    | 20        | Project-to-role mapping |
| Tools & Technologies | 10        | Tools and platforms used |
| **Total**            | **100**   | Overall compatibility score |

---

## 🖥️ Run Locally

```bash
cd d:\NEW
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
