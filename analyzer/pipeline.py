"""
pipeline.py
Lightweight version for cloud deployment:
- KeyBERT for keyword extraction (no torch needed separately)
- HuggingFace Inference API for feedback generation
- All heavy local model inference removed
"""

import re
import os
import requests
from keybert import KeyBERT
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN", "")

GENERIC_IGNORE = {
    "skills", "experience", "work", "team", "client", "role", "job", "people",
    "day", "high", "strong", "good", "new", "best", "key", "level", "impact",
    "driving", "lasting", "reward", "culture", "ability", "proven",
    "demonstrated", "outstanding", "willingness", "fluency", "required",
    "plus", "huge", "moving", "fast", "able", "figure", "things", "using",
    "looking", "someone", "will", "not", "this", "that", "with", "from",
    "attention", "comfortable", "operating", "environment", "startup",
    "execution", "speed", "studio", "similar", "independently",
}

TECH_VOCAB = [
    "claude", "codex", "chatgpt", "cursor", "copilot", "gpt-4", "gpt",
    "openai", "langchain", "llm", "rag", "prompt engineering", "fine tuning",
    "agentic ai", "ai agents", "generative ai", "hugging face",
    "python", "javascript", "typescript", "java", "kotlin", "swift",
    "scala", "golang", "rust", "c++", "c#", "bash", "sql",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "transfer learning",
    "neural networks", "transformers", "bert", "scikit-learn", "pytorch",
    "tensorflow", "keras", "xgboost", "lightgbm", "stable diffusion",
    "feature engineering", "model training", "model evaluation", "tensorboard",
    "next.js", "react", "vue", "angular", "node.js", "fastapi", "flask",
    "django", "rest api", "graphql", "microservices", "vercel", "netlify",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "looker", "metabase", "looker studio", "google sheets",
    "excel", "etl", "data pipeline", "data engineering", "data analysis",
    "data visualization", "business intelligence", "spark", "pyspark",
    "airflow", "dbt", "bigquery", "snowflake",
    "mysql", "postgresql", "sqlite", "mongodb", "redis", "elasticsearch",
    "firebase", "supabase", "nosql",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ci/cd", "mlops", "devops", "git", "github", "linux",
    "zapier", "make", "n8n", "automation", "workflow automation", "api integration",
    "google analytics", "segment", "mixpanel", "gtm", "google tag manager",
    "facebook ads", "google ads", "paid ads", "funnels", "landing pages",
    "supply chain", "demand forecasting", "inventory management",
    "s&op", "production planning", "logistics", "erp", "sap",
    "statistics", "time series", "forecasting", "regression", "classification",
    "clustering", "bayesian", "optimization", "linear programming",
    "operations research", "stochastic modelling", "gurobi", "cplex",
    "master's degree", "bachelor's degree", "phd",
    "agile", "scrum", "data engineering",
]

STRONG_VERBS = [
    "built", "developed", "designed", "implemented", "architected", "led",
    "optimized", "automated", "deployed", "launched", "created", "engineered",
    "reduced", "improved", "increased", "delivered", "shipped", "scaled",
    "integrated", "migrated", "researched", "published", "trained",
    "fine-tuned", "benchmarked", "mentored", "managed",
]

WEAK_VERBS = [
    "helped", "assisted", "worked on", "responsible for", "participated",
    "supported", "involved in", "contributed to", "was part of",
]

INDUSTRY_KEYWORDS = [
    "healthcare", "finance", "banking", "fintech", "edtech", "e-commerce",
    "retail", "manufacturing", "supply chain", "logistics", "automotive",
    "telecom", "media", "entertainment", "gaming", "cybersecurity",
    "cloud computing", "saas", "consulting", "research", "academia",
    "computer vision", "nlp", "robotics", "iot", "medical imaging",
    "real estate", "proptech",
]

_kw_model = None


def get_kw_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT()
    return _kw_model


def word_boundary_match(term: str, text_lower: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"
    return bool(re.search(pattern, text_lower))


def vocab_scan(jd_text: str) -> list[str]:
    found = []
    lower = jd_text.lower()
    for term in TECH_VOCAB:
        if word_boundary_match(term, lower):
            found.append(term)
    return found


def extract_qualifications_section(jd_text: str) -> str:
    patterns = [
        r"(qualifications?.*?)(?:\n{2,}|your impact|about the|we offer|compensation|$)",
        r"(requirements?.*?)(?:\n{2,}|benefits|about the|we offer|compensation|$)",
        r"(what we.re looking for.*?)(?:\n{2,}|benefits|compensation|$)",
        r"(what you.ll need.*?)(?:\n{2,}|benefits|compensation|$)",
        r"(required.*?)(?:\n{2,}|benefits|compensation|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, jd_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return jd_text


def detect_role_type(jd_text: str) -> str:
    lower = jd_text.lower()
    role_signals = {
        "ML/AI Engineer": ["machine learning", "deep learning", "model training", "pytorch", "tensorflow", "mlops"],
        "Data Analyst": ["sql", "dashboard", "tableau", "power bi", "data analysis", "reporting", "excel"],
        "Software Engineer": ["backend", "frontend", "api", "system design", "next.js", "react", "node"],
        "AI/Automation Builder": ["ai agent", "codex", "claude", "chatgpt", "automation", "workflow", "n8n", "zapier"],
        "Data Scientist": ["statistics", "forecasting", "regression", "classification", "research", "jupyter"],
        "DevOps/Cloud": ["docker", "kubernetes", "ci/cd", "terraform", "aws", "azure", "gcp"],
    }
    scores = {role: sum(1 for s in signals if s in lower) for role, signals in role_signals.items()}
    return max(scores, key=scores.get)


def extract_keywords(jd_text: str) -> list[str]:
    vocab_matches = vocab_scan(jd_text)
    qual_section = extract_qualifications_section(jd_text)
    kw_model = get_kw_model()
    raw_keybert = kw_model.extract_keywords(
        qual_section,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=15,
        use_mmr=True,
        diversity=0.7
    )
    keybert_filtered = []
    for kw, score in raw_keybert:
        words = kw.lower().split()
        if all(w in GENERIC_IGNORE for w in words):
            continue
        if len(words) == 1 and len(kw) <= 3 and kw.lower() not in {"sql", "aws", "gcp", "ml", "ai"}:
            continue
        if len(words) > 3:
            continue
        if len(kw) < 3:
            continue
        keybert_filtered.append(kw)

    all_keywords = vocab_matches.copy()
    existing_lower = {k.lower() for k in all_keywords}
    for kw in keybert_filtered:
        if kw.lower() not in existing_lower:
            all_keywords.append(kw)
            existing_lower.add(kw.lower())

    return all_keywords[:25]


def analyze_experience_quality(resume_text: str) -> dict:
    lines = resume_text.split("\n")
    bullet_lines = [
        l.strip() for l in lines
        if l.strip().startswith(("•", "-", "*")) or (len(l.strip()) > 20 and l.strip()[0].isupper())
    ]
    strong_verb_lines = []
    weak_verb_lines = []
    quantified_lines = []

    for line in bullet_lines:
        lower = line.lower()
        if re.search(r"\d+\s*(%|x|k|ms|sec|users|samples|images|fps|gb|tb|mb)", lower) or re.search(r"\d{2,}", lower):
            quantified_lines.append(line)
        if any(lower.startswith(v) or f" {v}" in lower for v in STRONG_VERBS):
            strong_verb_lines.append(line)
        elif any(v in lower for v in WEAK_VERBS):
            weak_verb_lines.append(line)

    return {
        "strong_verb_lines": strong_verb_lines[:4],
        "weak_verb_lines": weak_verb_lines[:4],
        "quantified_lines": quantified_lines[:4],
        "has_metrics": len(quantified_lines) > 0,
        "total_bullets": len(bullet_lines),
        "strong_verb_count": len(strong_verb_lines),
        "weak_verb_count": len(weak_verb_lines),
        "quantified_count": len(quantified_lines),
    }


def analyze_experience(resume_text: str) -> dict:
    lines = resume_text.split("\n")
    internship_lines = []
    role_lines = []
    found_industries = []
    date_pattern = re.compile(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\.\-,]*\d{4}|"
        r"\d{4}\s*[–\-]\s*(?:\d{4}|present)",
        re.IGNORECASE
    )
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(w in lower for w in ["intern", "internship", "trainee"]):
            internship_lines.append(stripped)
        if date_pattern.search(stripped) and len(stripped) > 10:
            role_lines.append(stripped)
        for industry in INDUSTRY_KEYWORDS:
            if industry in lower and industry not in found_industries:
                found_industries.append(industry)

    return {
        "internships": list(set(internship_lines))[:5],
        "role_lines": list(set(role_lines))[:6],
        "industries": found_industries[:8],
        "has_internship": len(internship_lines) > 0,
        "experience_count": len(role_lines),
    }


def match_experience_to_jd(resume_text: str, jd_text: str) -> dict:
    exp = analyze_experience(resume_text)
    jd_lower = jd_text.lower()
    years_match = re.search(r"(\d)\+?\s*years?", jd_lower)
    required_years = int(years_match.group(1)) if years_match else 0
    resume_industries = exp["industries"]
    jd_industries = [ind for ind in INDUSTRY_KEYWORDS if ind in jd_lower]
    industry_overlap = [ind for ind in resume_industries if ind in jd_industries]
    industry_gaps = [ind for ind in jd_industries if ind not in resume_industries]
    return {
        "candidate_roles": exp["role_lines"],
        "candidate_internships": exp["internships"],
        "candidate_industries": resume_industries,
        "jd_required_years": required_years,
        "has_relevant_internship": exp["has_internship"],
        "industry_overlap": industry_overlap,
        "industry_gaps": industry_gaps[:5],
        "experience_count": exp["experience_count"],
    }


def generate_feedback(resume_text: str, jd_text: str, score: int, missing_keywords: list[str]) -> str:
    """
    Generate feedback using HuggingFace Inference API.
    Falls back to a rule-based response if API is unavailable.
    """
    missing_str = ", ".join(missing_keywords[:5]) if missing_keywords else "none"

    if HF_TOKEN:
        prompt = f"""Resume score: {score}%. Missing skills: {missing_str}.
Resume: {resume_text[:300]}
Job: {jd_text[:300]}
Give 3 specific bullet point improvements to add to this resume:"""

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 200, "repetition_penalty": 3.0}},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
        except Exception:
            pass

    # Rule-based fallback
    lines = []
    if missing_keywords:
        lines.append(f"1. Add these missing skills to your Skills section: {', '.join(missing_keywords[:4])}.")
    if score < 50:
        lines.append("2. Tailor your summary to directly mention the role title and core technologies from the JD.")
    lines.append("3. Add metrics to your bullet points — include numbers like accuracy %, dataset size, or time saved.")
    return "\n".join(lines)


def get_missing_keywords(resume_text: str, jd_keywords: list[str]) -> list[str]:
    resume_lower = resume_text.lower()
    missing = []
    for kw in jd_keywords:
        if not word_boundary_match(kw, resume_lower):
            missing.append(kw)
    return missing