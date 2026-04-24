# Example LLM Prompt

This document shows the exact prompt template sent to the OpenAI API.

---

## System Message

```
You are a precise career analysis assistant. Always respond with valid JSON only.
```

## User Prompt

```
You are an expert AI/ML career advisor and technical recruiter.

TASK:
Analyze the candidate's resume against the provided job description.
Focus specifically on AI/ML relevance: machine learning frameworks, deep learning,
NLP, computer vision, MLOps, data pipelines, research publications, and related skills.

RESUME (CV):
"""
[Extracted PDF text is inserted here]
"""

JOB DESCRIPTION:
"""
[User-provided job description is inserted here]
"""

INSTRUCTIONS:
1. Evaluate the resume against the job description across four categories.
2. For EACH category, first write a 1-2 sentence justification, then assign a score.
3. Compute the final Match Score as the sum of all four category scores.
4. Identify strengths, missing skills, weak areas, and actionable suggestions.
5. Rewrite ONE project bullet from the resume to be more impactful and ATS-friendly.

Return your response as valid JSON with EXACTLY this structure (no markdown fences):
{
  "match_score": <int 0-100>,
  "category_scores": {
    "skills_match": {
      "score": <int 0-40>,
      "reasoning": "<1-2 sentence justification>"
    },
    "experience_match": {
      "score": <int 0-30>,
      "reasoning": "<1-2 sentence justification>"
    },
    "project_relevance": {
      "score": <int 0-20>,
      "reasoning": "<1-2 sentence justification>"
    },
    "tools_technologies": {
      "score": <int 0-10>,
      "reasoning": "<1-2 sentence justification>"
    }
  },
  "strengths": ["<bullet 1>", "<bullet 2>", "..."],
  "missing_skills": ["<bullet 1>", "<bullet 2>", "..."],
  "weak_areas": ["<bullet 1>", "<bullet 2>", "..."],
  "suggestions": ["<bullet 1>", "<bullet 2>", "..."],
  "improved_bullet": "<rewritten project bullet>"
}

IMPORTANT:
- match_score MUST equal the sum of the four category scores.
- Do NOT randomize scores. Base every score strictly on evidence in the resume.
- Be professional and concise.
- Return ONLY the JSON object, no extra text.
```

## Example Response

```json
{
  "match_score": 62,
  "category_scores": {
    "skills_match": {
      "score": 28,
      "reasoning": "Candidate demonstrates proficiency in Python, TensorFlow, and scikit-learn, but lacks experience with PyTorch and Hugging Face Transformers which are explicitly required."
    },
    "experience_match": {
      "score": 18,
      "reasoning": "2 years of ML experience vs. the 3-5 years required. Internship experience is relevant but does not fully meet the seniority expectations."
    },
    "project_relevance": {
      "score": 12,
      "reasoning": "Sentiment analysis and image classification projects are relevant, but no end-to-end ML pipeline or production deployment experience demonstrated."
    },
    "tools_technologies": {
      "score": 4,
      "reasoning": "Familiar with AWS S3 and Docker, but no mention of Kubernetes, MLflow, or CI/CD for ML systems."
    }
  },
  "strengths": [
    "Strong Python programming foundation with data science libraries",
    "Hands-on deep learning project experience (CNNs, RNNs)",
    "Published research in NLP domain"
  ],
  "missing_skills": [
    "PyTorch and Hugging Face Transformers",
    "MLOps tools (MLflow, Kubeflow, Airflow)",
    "Kubernetes and container orchestration",
    "A/B testing and experiment tracking"
  ],
  "weak_areas": [
    "No production ML deployment experience mentioned",
    "Limited to academic/personal projects — no industry-scale work",
    "Resume lacks quantifiable impact metrics"
  ],
  "suggestions": [
    "Add a personal project deploying a model with FastAPI + Docker to demonstrate production skills",
    "Complete a PyTorch-based project and add it to the resume",
    "Quantify project outcomes (e.g., 'Improved accuracy from 78% to 92%')",
    "Add MLflow or Weights & Biases experiment tracking to an existing project",
    "Consider AWS/GCP ML certification to strengthen cloud ML credentials"
  ],
  "improved_bullet": "Developed and deployed a real-time sentiment analysis pipeline using TensorFlow and Flask, processing 10K+ tweets daily with 91% classification accuracy, reducing manual review time by 60%."
}
```
