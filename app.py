"""
AI Career Assistant for CSE Roles
A Streamlit application that analyzes resume compatibility with ANY Computer Science
& Engineering job description using domain-aware evaluation via LLM APIs.
"""

import os
import json
import re
import time
import streamlit as st
from io import BytesIO

import PyPDF2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

PROVIDERS = ["Groq (Free)", "Google Gemini", "OpenAI"]

CSE_DOMAINS = [
    "Auto-Detect from Job Description",
    "Software Engineering (Backend)",
    "Software Engineering (Frontend)",
    "Software Engineering (Full Stack)",
    "Data Science / Data Analytics",
    "Machine Learning / AI",
    "Cybersecurity / Ethical Hacking",
    "DevOps / Cloud Engineering",
    "Mobile App Development (Android/iOS)",
    "Competitive Programming / SDE Roles",
    "Systems / Embedded Engineering",
    "Database / Data Engineering",
    "UI/UX Engineering",
    "Game Development",
    "Blockchain / Web3",
    "QA / Test Engineering",
]


def get_api_key(provider: str) -> str:
    """Resolve API key from sidebar input (session state) or environment variable."""
    key_map = {
        "Groq (Free)": ("groq_key_input", "GROQ_API_KEY"),
        "Google Gemini": ("gemini_key_input", "GOOGLE_API_KEY"),
        "OpenAI": ("openai_key_input", "OPENAI_API_KEY"),
    }
    session_key, env_key = key_map.get(provider, ("", ""))
    sidebar_val = st.session_state.get(session_key, "").strip()
    return sidebar_val if sidebar_val else os.getenv(env_key, "")


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        if not pages:
            raise ValueError("The PDF contains no extractable text (may be scanned/image-based).")
        return "\n".join(pages)
    except PyPDF2.errors.PdfReadError:
        raise ValueError("Unable to read the PDF. It may be corrupted or password-protected.")


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_prompt(cv_text: str, job_description: str, selected_domain: str) -> str:
    """Build the domain-aware structured prompt for the LLM."""

    domain_instruction = ""
    if selected_domain == "Auto-Detect from Job Description":
        domain_instruction = (
            "STEP 1: First, analyze the job description and identify the specific CSE domain "
            "(e.g., Backend Engineering, Data Science, Cybersecurity, DevOps, ML/AI, Mobile Dev, etc.). "
            "Set 'detected_domain' to your identified domain."
        )
    else:
        domain_instruction = (
            f"The job domain is: {selected_domain}. Set 'detected_domain' to \"{selected_domain}\". "
            "Evaluate the resume specifically through the lens of this domain."
        )

    return f"""You are an expert career advisor and technical recruiter specializing in Computer Science & Engineering roles.

{domain_instruction}

TASK:
Analyze the candidate's resume against the provided job description with DOMAIN-AWARE evaluation.
Your scoring must adapt based on the detected/selected domain:
- Backend Engineering → APIs, databases, system design, scalability
- Frontend Engineering → UI frameworks, responsive design, JavaScript ecosystem
- Full Stack → both frontend & backend skills, deployment
- Data Science → statistics, Python, visualization, SQL, ML basics
- ML/AI → deep learning, frameworks (PyTorch/TF), MLOps, research
- Cybersecurity → security tools, networking, OS internals, ethical hacking, compliance
- DevOps/Cloud → CI/CD, Docker, Kubernetes, cloud platforms, IaC
- Mobile Dev → Android/iOS SDKs, React Native/Flutter, mobile UI patterns
- SDE/CP → DSA, problem-solving, competitive programming, system design
- Systems/Embedded → C/C++, RTOS, hardware interfaces, low-level programming
- Other domains → evaluate based on relevant domain-specific skills

RESUME (CV):
\"\"\"
{cv_text}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{job_description}
\"\"\"

INSTRUCTIONS:
1. Identify the job domain (or use the provided one).
2. Evaluate the resume against the job description using domain-specific criteria.
3. For EACH category, write a 1-2 sentence justification, then assign a score.
4. match_score MUST equal the sum of the four category scores.
5. Predict the top 3 best-fit roles for this candidate based on their resume.
6. Create a skill gap roadmap with 4-6 specific skills to learn, each with a resource suggestion.

Return ONLY valid JSON with this EXACT structure:
{{
  "detected_domain": "<identified CSE domain>",
  "match_score": <int 0-100>,
  "best_fit_roles": [
    {{"role": "<role name>", "confidence": "<High/Medium/Low>", "reason": "<why>"}},
    {{"role": "<role name>", "confidence": "<High/Medium/Low>", "reason": "<why>"}},
    {{"role": "<role name>", "confidence": "<High/Medium/Low>", "reason": "<why>"}}
  ],
  "category_scores": {{
    "skills_match": {{
      "score": <int 0-40>,
      "reasoning": "<domain-specific justification>"
    }},
    "experience_match": {{
      "score": <int 0-30>,
      "reasoning": "<domain-specific justification>"
    }},
    "project_relevance": {{
      "score": <int 0-20>,
      "reasoning": "<domain-specific justification>"
    }},
    "tools_technologies": {{
      "score": <int 0-10>,
      "reasoning": "<domain-specific justification>"
    }}
  }},
  "strengths": ["<bullet>", "..."],
  "missing_skills": ["<bullet>", "..."],
  "weak_areas": ["<bullet>", "..."],
  "suggestions": ["<bullet>", "..."],
  "skill_gap_roadmap": [
    {{"skill": "<skill name>", "priority": "<High/Medium/Low>", "resource": "<course/book/platform>"}},
    ...
  ],
  "recommended_career_path": "<2-3 sentence career progression recommendation>",
  "improved_bullet": "<rewritten project bullet from resume>"
}}

IMPORTANT:
- match_score MUST equal the sum of the four category scores.
- Do NOT randomize. Base every score on evidence in the resume.
- Scoring criteria MUST be domain-specific, not generic.
- Be professional and concise.
- Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# LLM Response Parser
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str) -> dict:
    """Parse and clean the raw LLM response into a dict."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# LLM Providers
# ---------------------------------------------------------------------------

def analyze_with_groq(cv_text: str, job_description: str, domain: str) -> dict:
    """Groq API (free) using Llama 3.3 70B."""
    from groq import Groq

    api_key = get_api_key("Groq (Free)")
    if not api_key:
        raise RuntimeError(
            "Groq API key not found. Enter it in the sidebar.\n\n"
            "Get a FREE key at: https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)
    prompt = build_prompt(cv_text, job_description, domain)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise career analysis assistant. Always respond with valid JSON only. No markdown fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        return _parse_llm_response(response.choices[0].message.content)
    except json.JSONDecodeError:
        raise RuntimeError("The AI returned an invalid response. Please try again.")
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")


def analyze_with_gemini(cv_text: str, job_description: str, domain: str) -> dict:
    """Google Gemini API with model fallback."""
    from google import genai

    api_key = get_api_key("Google Gemini")
    if not api_key:
        raise RuntimeError("Google API key not found. Enter it in the sidebar.")

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(cv_text, job_description, domain)
    last_error = None

    for model_name in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt,
                    config={"temperature": 0.2, "max_output_tokens": 3000},
                )
                return _parse_llm_response(response.text)
            except json.JSONDecodeError:
                raise RuntimeError("The AI returned an invalid response. Please try again.")
            except Exception as e:
                last_error = e
                es = str(e)
                if "404" in es or "NOT_FOUND" in es:
                    break
                if "429" in es or "RESOURCE_EXHAUSTED" in es:
                    if attempt == 0:
                        time.sleep(10)
                        continue
                    break
                raise RuntimeError(f"Google Gemini API error: {e}")

    raise RuntimeError(f"Gemini models unavailable. Try Groq (Free). Error: {last_error}")


def analyze_with_openai(cv_text: str, job_description: str, domain: str) -> dict:
    """OpenAI API using GPT-4o-mini."""
    from openai import OpenAI

    api_key = get_api_key("OpenAI")
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Enter it in the sidebar.")

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(cv_text, job_description, domain)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise career analysis assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        return _parse_llm_response(response.choices[0].message.content)
    except json.JSONDecodeError:
        raise RuntimeError("The AI returned an invalid response. Please try again.")
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")


def analyze_with_llm(cv_text: str, job_description: str, provider: str, domain: str) -> dict:
    """Route to the correct LLM provider."""
    dispatch = {
        "Groq (Free)": analyze_with_groq,
        "Google Gemini": analyze_with_gemini,
        "OpenAI": analyze_with_openai,
    }
    return dispatch[provider](cv_text, job_description, domain)


# ---------------------------------------------------------------------------
# Output Renderer
# ---------------------------------------------------------------------------

def format_output(result: dict) -> None:
    """Render the analysis results in the Streamlit UI."""

    # ── Detected Domain ────────────────────────────────────────────────
    domain = result.get("detected_domain", "Unknown")
    st.markdown(
        f"""<div style="text-align:center; padding:12px; border-radius:12px;
            background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #6c63ff33;
            margin-bottom:16px;">
            <span style="color:#8892b0; font-size:13px; text-transform:uppercase;
            letter-spacing:2px;">Detected Domain</span><br>
            <span style="color:#6c63ff; font-size:24px; font-weight:700;">{domain}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Match Score ────────────────────────────────────────────────────
    score = result.get("match_score", 0)
    sc = "#e74c3c" if score < 40 else "#f39c12" if score < 70 else "#2ecc71"
    st.markdown(
        f"""<div style="text-align:center; padding:30px 20px; border-radius:16px;
            background:linear-gradient(135deg,#0f0f1a,#1a1a2e); border:1px solid {sc}33;
            margin-bottom:24px;">
            <p style="font-size:14px; text-transform:uppercase; letter-spacing:3px;
            color:#8892b0; margin-bottom:8px;">Overall Match Score</p>
            <p style="font-size:72px; font-weight:800; color:{sc}; margin:0;
            line-height:1;">{score}<span style="font-size:28px; color:#8892b0;">/100</span></p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Best Fit Roles ─────────────────────────────────────────────────
    best_fit = result.get("best_fit_roles", [])
    if best_fit:
        st.markdown("### 🎯 Best Fit Roles (Top 3)")
        cols = st.columns(3)
        conf_colors = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
        for i, role in enumerate(best_fit[:3]):
            cc = conf_colors.get(role.get("confidence", "Medium"), "#f39c12")
            with cols[i]:
                st.markdown(
                    f"""<div style="padding:16px; border-radius:12px; background:#0f0f1a;
                        border:1px solid #1e1e3a; height:100%; min-height:140px;">
                        <p style="font-weight:700; color:#ccd6f6; font-size:15px;
                        margin-bottom:6px;">{role.get('role','')}</p>
                        <p style="font-size:12px; color:{cc}; font-weight:600;
                        margin-bottom:6px;">● {role.get('confidence','')} Confidence</p>
                        <p style="font-size:12px; color:#8892b0;">{role.get('reason','')}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Category Scores ────────────────────────────────────────────────
    st.markdown("### 📊 Category Scores")
    categories = result.get("category_scores", {})
    labels = {
        "skills_match": ("Skills Match", 40),
        "experience_match": ("Experience Match", 30),
        "project_relevance": ("Project Relevance", 20),
        "tools_technologies": ("Tools & Technologies", 10),
    }
    cols = st.columns(4)
    for idx, (key, (label, mx)) in enumerate(labels.items()):
        cat = categories.get(key, {})
        cs = cat.get("score", 0)
        pct = int((cs / mx) * 100) if mx else 0
        bc = "#e74c3c" if pct < 40 else "#f39c12" if pct < 70 else "#2ecc71"
        with cols[idx]:
            st.markdown(
                f"""<div style="text-align:center; padding:16px 8px; border-radius:12px;
                    background:#0f0f1a; border:1px solid #1e1e3a; height:100%;">
                    <p style="font-size:12px; color:#8892b0; margin-bottom:4px;">{label}</p>
                    <p style="font-size:28px; font-weight:700; color:{bc}; margin:4px 0;">
                        {cs}<span style="font-size:14px; color:#555;">/{mx}</span></p>
                    <div style="background:#1e1e3a; border-radius:6px; height:6px;
                        margin-top:8px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:{bc};
                        border-radius:6px;"></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Score Reasoning ────────────────────────────────────────────────
    st.markdown("### 🧠 Score Reasoning")
    for key, (label, _) in labels.items():
        cat = categories.get(key, {})
        st.markdown(f"**{label}:** {cat.get('reasoning', 'No reasoning provided.')}")

    # ── Strengths / Missing / Weak / Suggestions ───────────────────────
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### ✅ Strengths")
        for item in result.get("strengths", []):
            st.markdown(f"- {item}")
        st.markdown("### 🔻 Weak Areas")
        for item in result.get("weak_areas", []):
            st.markdown(f"- {item}")
    with col_r:
        st.markdown("### ⚠️ Missing Skills")
        for item in result.get("missing_skills", []):
            st.markdown(f"- {item}")
        st.markdown("### 💡 Suggestions")
        for item in result.get("suggestions", []):
            st.markdown(f"- {item}")

    # ── Skill Gap Roadmap ──────────────────────────────────────────────
    roadmap = result.get("skill_gap_roadmap", [])
    if roadmap:
        st.markdown("---")
        st.markdown("### 🗺️ Skill Gap Roadmap")
        for item in roadmap:
            pri = item.get("priority", "Medium")
            pc = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(pri, "🟡")
            st.markdown(
                f"- {pc} **{item.get('skill', '')}** ({pri}) — {item.get('resource', '')}"
            )

    # ── Recommended Career Path ────────────────────────────────────────
    career_path = result.get("recommended_career_path", "")
    if career_path:
        st.markdown("---")
        st.markdown("### 🚀 Recommended Career Path")
        st.success(career_path)

    # ── Improved Resume Bullet ─────────────────────────────────────────
    improved = result.get("improved_bullet", "")
    if improved:
        st.markdown("---")
        st.markdown("### ✍️ Improved Resume Bullet")
        st.info(improved)


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AI Career Assistant", page_icon="🤖", layout="wide")

    # ── CSS ────────────────────────────────────────────────────────────
    st.markdown("""<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        .stApp { font-family: 'Inter', sans-serif; }
        .app-header { text-align: center; padding: 40px 20px 20px; }
        .app-header h1 { font-size: 2.5rem; font-weight: 800;
            background: linear-gradient(135deg, #6c63ff, #3b82f6, #06b6d4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
        .app-header p { color: #8892b0; font-size: 1rem; margin-top: 0; }
        section[data-testid="stSidebar"] { background: #0a0a1a; }
        #MainMenu, footer, header { visibility: hidden; }
        .stFileUploader > div { border-radius: 12px; }
        .stButton > button { width: 100%;
            background: linear-gradient(135deg, #6c63ff, #3b82f6);
            color: white; border: none; padding: 14px 24px; border-radius: 12px;
            font-size: 16px; font-weight: 600; letter-spacing: 0.5px;
            transition: all 0.3s ease; cursor: pointer; }
        .stButton > button:hover { transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(108, 99, 255, 0.35); }
        .stTextArea textarea { border-radius: 12px; border: 1px solid #1e1e3a;
            font-family: 'Inter', sans-serif; }
    </style>""", unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────
    st.markdown("""<div class="app-header">
        <h1>🤖 AI Career Assistant</h1>
        <p>Domain-Aware Resume Analysis for All CSE Roles</p>
    </div>""", unsafe_allow_html=True)

    # ── Main Layout — All Controls in Center ─────────────────────────
    # Row 1: Provider + API Key + Domain
    st.markdown("---")
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        st.markdown("##### 🤖 LLM Provider")
        provider = st.selectbox("Choose provider", PROVIDERS, index=0,
            help="Groq is free and fast (recommended).", label_visibility="collapsed")

    with r1c2:
        if provider == "Groq (Free)":
            st.markdown("##### 🔑 Groq API Key  ·  [Get FREE key](https://console.groq.com/keys)")
            st.text_input("API Key", type="password", key="groq_key_input",
                placeholder="gsk_...", label_visibility="collapsed")
        elif provider == "Google Gemini":
            st.markdown("##### 🔑 Gemini API Key")
            st.text_input("API Key", type="password", key="gemini_key_input",
                placeholder="AIza...", label_visibility="collapsed")
        else:
            st.markdown("##### 🔑 OpenAI API Key")
            st.text_input("API Key", type="password", key="openai_key_input",
                placeholder="sk-proj-...", label_visibility="collapsed")

    with r1c3:
        st.markdown("##### 🎯 Job Domain")
        selected_domain = st.selectbox("Domain", CSE_DOMAINS, index=0,
            label_visibility="collapsed",
            help="Choose a CSE domain or let AI auto-detect.")

    # Row 2: Resume Upload + Job Description
    st.markdown("---")
    r2c1, r2c2 = st.columns([1, 2])

    with r2c1:
        st.markdown("##### 📄 Upload Resume (PDF)")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"],
            label_visibility="collapsed")

    with r2c2:
        st.markdown("##### 📝 Job Description")
        job_description = st.text_area("Paste the job description here", height=180,
            placeholder="e.g. We are looking for a Backend Engineer with experience in Node.js, PostgreSQL, Docker...",
            label_visibility="collapsed")

    # Analyze Button
    st.markdown("")
    bc1, bc2, bc3 = st.columns([1, 2, 1])
    with bc2:
        analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

    # ── Landing Content (before analysis) ──────────────────────────────
    if not analyze_btn:
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; margin:10px 0 20px;">
            <h3 style="color:#ccd6f6;">✨ What You'll Get</h3>
        </div>
        """, unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        features = [
            ("📊", "Domain-Aware Scoring", "Category scores adapted to Backend, AI/ML, Cybersecurity, DevOps & more."),
            ("🎯", "Best Fit Roles", "Top 3 career role predictions based on your resume profile."),
            ("🗺️", "Skill Gap Roadmap", "Prioritized skills to learn with recommended resources."),
        ]
        for col, (icon, title, desc) in zip([f1, f2, f3], features):
            with col:
                st.markdown(f"""
                <div style="padding:20px; border-radius:14px; background:#0f0f1a;
                    border:1px solid #1e1e3a; text-align:center; min-height:150px;
                    display:flex; flex-direction:column; justify-content:center; align-items:center;">
                    <p style="font-size:28px; margin-bottom:6px;">{icon}</p>
                    <p style="font-weight:600; color:#ccd6f6; font-size:14px; margin-bottom:4px;">{title}</p>
                    <p style="font-size:12px; color:#8892b0; line-height:1.4;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-top:24px;">
            <p style="color:#555; font-size:12px;">Powered by Groq / Gemini / OpenAI &bull; Built with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Validation ─────────────────────────────────────────────────────
    if not uploaded_file:
        st.error("📄 Please upload a PDF resume above.")
        return
    if not job_description.strip():
        st.error("📝 Please enter a job description above.")
        return

    # ── Analysis ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.spinner("🔍 Analyzing your resume with domain-aware evaluation..."):
        try:
            cv_text = extract_text_from_pdf(uploaded_file)
        except ValueError as e:
            st.error(f"PDF Error: {e}")
            return

        if len(cv_text.strip()) < 50:
            st.warning("⚠️ Very little text extracted from PDF. Results may be unreliable.")

        try:
            result = analyze_with_llm(cv_text, job_description, provider, selected_domain)
        except RuntimeError as e:
            st.error(f"Analysis Error: {e}")
            return

    format_output(result)


if __name__ == "__main__":
    main()

