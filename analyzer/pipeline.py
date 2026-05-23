"""
pipeline.py
Production-optimized lightweight ATS resume analyzer

Features:
- KeyBERT keyword extraction
- Semantic skill matching
- HuggingFace Inference API feedback
- Lightweight cloud deployment
- No heavy local inference
- Backward-compatible API structure
"""

import os
import re
import requests

from functools import lru_cache
from collections import Counter

from dotenv import load_dotenv
from keybert import KeyBERT

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN", "")

# =========================================================
# GENERIC FILTERS
# =========================================================

GENERIC_IGNORE = {
    # generic
    "skills", "experience", "work", "team", "client", "role", "job",
    "people", "company", "candidate", "position", "opportunity",
    "knowledge", "ability", "understanding", "background",

    # fluff
    "strong", "good", "best", "high", "excellent", "outstanding",
    "motivated", "passionate", "dynamic", "innovative", "creative",
    "professional", "dedicated", "reliable", "responsible",

    # hiring junk
    "required", "preferred", "minimum", "plus", "desired",
    "qualifications", "requirements", "responsibilities",
    "overview", "description", "applicant",

    # common verbs
    "developed", "implemented", "managed", "worked", "working",
    "building", "creating", "supporting", "improving", "designed",

    # structural
    "environment", "startup", "execution", "attention",
    "comfortable", "independently", "operating",

    # noise
    "using", "with", "from", "this", "that", "will",
    "able", "someone", "things", "similar",

    # useless abstractions
    "technology", "technologies", "solution", "solutions",
    "system", "systems", "platform", "platforms",
    "process", "processes", "services", "operations",

    # business junk
    "business", "stakeholders", "users", "customers",
    "projects", "deliverables", "goals", "objectives",
}

STOP_PHRASES = {
    "work experience",
    "job description",
    "preferred qualifications",
    "required qualifications",
    "key responsibilities",
    "about the role",
    "about us",
    "what you'll do",
    "what we offer",
    "join our team",
}

NORMALIZATION_MAP = {
    "nodejs": "node.js",
    "reactjs": "react",
    "nextjs": "next.js",
    "js": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "ai": "artificial intelligence",
}

RELATED_SKILLS = {
    "deep learning": ["pytorch", "tensorflow", "keras"],
    "machine learning": ["scikit-learn", "xgboost", "lightgbm"],
    "frontend": ["react", "vue", "angular"],
    "backend": ["fastapi", "flask", "django", "node.js"],
    "cloud": ["aws", "azure", "gcp"],
    "llm": ["langchain", "rag", "openai", "hugging face"],
}

# =========================================================
# TECH VOCAB
# =========================================================

TECH_VOCAB = [
    "claude", "codex", "chatgpt", "cursor", "copilot", "gpt-4", "gpt",
    "openai", "langchain", "llm", "rag", "prompt engineering",
    "fine tuning", "agentic ai", "ai agents", "generative ai",
    "hugging face",

    "python", "javascript", "typescript", "java", "kotlin",
    "swift", "scala", "golang", "rust", "c++", "c#", "bash", "sql",

    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "reinforcement learning", "transfer learning",
    "neural networks", "transformers", "bert",
    "scikit-learn", "pytorch", "tensorflow",
    "keras", "xgboost", "lightgbm", "stable diffusion",

    "feature engineering", "model training",
    "model evaluation", "tensorboard",

    "next.js", "react", "vue", "angular",
    "node.js", "fastapi", "flask", "django",
    "rest api", "graphql", "microservices",

    "vercel", "netlify",

    "pandas", "numpy", "matplotlib", "plotly",
    "tableau", "power bi", "looker", "metabase",
    "looker studio", "excel",

    "etl", "data pipeline", "data engineering",
    "data analysis", "data visualization",
    "business intelligence",

    "spark", "pyspark", "airflow", "dbt",
    "bigquery", "snowflake",

    "mysql", "postgresql", "sqlite",
    "mongodb", "redis", "elasticsearch",
    "firebase", "supabase", "nosql",

    "aws", "azure", "gcp", "google cloud",
    "docker", "kubernetes", "terraform",
    "ci/cd", "mlops", "devops",
    "git", "github", "linux",

    "zapier", "make.com", "n8n",
    "automation", "workflow automation",
    "api integration",

    "statistics", "forecasting",
    "regression", "classification",
    "clustering", "optimization",

    "agile", "scrum",
]

# =========================================================
# EXPERIENCE ANALYSIS
# =========================================================

STRONG_VERBS = [
    "built", "developed", "designed", "implemented",
    "architected", "led", "optimized", "automated",
    "deployed", "launched", "created", "engineered",
    "reduced", "improved", "increased", "delivered",
    "shipped", "scaled", "integrated", "migrated",
    "researched", "published", "trained",
    "fine-tuned", "benchmarked", "mentored", "managed",
]

WEAK_VERBS = [
    "helped", "assisted", "worked on",
    "responsible for", "participated",
    "supported", "involved in",
    "contributed to", "was part of",
]

INDUSTRY_KEYWORDS = [
    "healthcare", "finance", "banking",
    "fintech", "edtech", "e-commerce",
    "retail", "manufacturing",
    "supply chain", "logistics",
    "automotive", "telecom",
    "media", "gaming",
    "cybersecurity", "cloud computing",
    "saas", "consulting",
    "research", "academia",
    "computer vision", "nlp",
    "robotics", "iot",
    "medical imaging",
]

# =========================================================
# MODEL LOADING
# =========================================================

@lru_cache(maxsize=1)
def get_kw_model() -> KeyBERT:
    return KeyBERT()

# =========================================================
# HELPERS
# =========================================================

def word_boundary_match(term: str, text_lower: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"
    return bool(re.search(pattern, text_lower))


def normalize_keyword(kw: str) -> str:
    kw = kw.lower().strip()
    return NORMALIZATION_MAP.get(kw, kw)


def clean_keyword(kw: str) -> str:
    kw = kw.strip().lower()

    kw = re.sub(r"[^a-zA-Z0-9\+\#\.\-\s]", "", kw)
    kw = re.sub(r"\s+", " ", kw)

    return kw.strip()


def semantic_match(keyword: str, resume_lower: str) -> bool:
    keyword = normalize_keyword(keyword)

    if word_boundary_match(keyword, resume_lower):
        return True

    related = RELATED_SKILLS.get(keyword, [])

    return any(
        word_boundary_match(r, resume_lower)
        for r in related
    )

# =========================================================
# VOCAB SCAN
# =========================================================

def vocab_scan(jd_text: str) -> list[str]:
    found = []

    lower = jd_text.lower()

    for term in TECH_VOCAB:
        if word_boundary_match(term, lower):
            found.append(term)

    return found

# =========================================================
# JD SECTION EXTRACTION
# =========================================================

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

# =========================================================
# ROLE DETECTION
# =========================================================

def detect_role_type(jd_text: str) -> str:
    lower = jd_text.lower()

    role_signals = {
        "ML/AI Engineer": [
            "machine learning", "deep learning",
            "model training", "pytorch",
            "tensorflow", "mlops"
        ],

        "Data Analyst": [
            "sql", "dashboard", "tableau",
            "power bi", "data analysis"
        ],

        "Software Engineer": [
            "backend", "frontend",
            "api", "react", "node"
        ],

        "AI/Automation Builder": [
            "ai agent", "automation",
            "workflow", "n8n", "zapier"
        ],

        "Data Scientist": [
            "statistics", "forecasting",
            "regression", "classification"
        ],

        "DevOps/Cloud": [
            "docker", "kubernetes",
            "terraform", "aws"
        ],
    }

    scores = {
        role: sum(1 for s in signals if s in lower)
        for role, signals in role_signals.items()
    }

    return max(scores, key=scores.get)

# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(jd_text: str) -> list[str]:
    vocab_matches = vocab_scan(jd_text)

    qual_section = extract_qualifications_section(jd_text)

    kw_model = get_kw_model()

    raw_keybert = kw_model.extract_keywords(
        qual_section,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=40,
        use_mmr=True,
        diversity=0.7
    )

    keywords = []

    for kw, score in raw_keybert:

        if score < 0.35:
            continue

        kw = clean_keyword(kw)

        if not kw:
            continue

        if kw in STOP_PHRASES:
            continue

        words = kw.split()

        if all(w in GENERIC_IGNORE for w in words):
            continue

        if len(words) > 3:
            continue

        if len(kw) < 3:
            continue

        if len(words) == 1 and len(kw) <= 3:
            if kw not in {"sql", "aws", "gcp", "ml", "ai"}:
                continue

        keywords.append(normalize_keyword(kw))

    all_keywords = []

    seen = set()

    for kw in vocab_matches + keywords:
        kw = normalize_keyword(kw)

        if kw not in seen:
            seen.add(kw)
            all_keywords.append(kw)

    return all_keywords[:30]

# =========================================================
# EXPERIENCE QUALITY ANALYSIS
# =========================================================

def analyze_experience_quality(resume_text: str) -> dict:
    lines = resume_text.split("\n")

    bullet_lines = [
        l.strip()
        for l in lines
        if l.strip().startswith(("•", "-", "*"))
        or (len(l.strip()) > 20 and l.strip()[0].isupper())
    ]

    strong_verb_lines = []
    weak_verb_lines = []
    quantified_lines = []

    for line in bullet_lines:
        lower = line.lower()

        if (
            re.search(
                r"\d+\s*(%|x|k|ms|sec|users|samples|images|fps|gb|tb|mb)",
                lower
            )
            or re.search(r"\d{2,}", lower)
        ):
            quantified_lines.append(line)

        if any(
            lower.startswith(v) or f" {v}" in lower
            for v in STRONG_VERBS
        ):
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

# =========================================================
# EXPERIENCE EXTRACTION
# =========================================================

def analyze_experience(resume_text: str) -> dict:
    lines = resume_text.split("\n")

    internship_lines = []
    role_lines = []
    found_industries = []

    date_pattern = re.compile(
        r"("
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[a-z]*[\s\.\-,]*\d{4}"
        r"|"
        r"\d{1,2}[\/\-]\d{4}"
        r"|"
        r"\d{4}\s*[–\-]\s*(?:\d{4}|present|current)"
        r")",
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

# =========================================================
# EXPERIENCE MATCHING
# =========================================================

def match_experience_to_jd(
    resume_text: str,
    jd_text: str
) -> dict:

    exp = analyze_experience(resume_text)

    jd_lower = jd_text.lower()

    years_match = re.search(
        r"(\d)\+?\s*years?",
        jd_lower
    )

    required_years = (
        int(years_match.group(1))
        if years_match
        else 0
    )

    resume_industries = exp["industries"]

    jd_industries = [
        ind
        for ind in INDUSTRY_KEYWORDS
        if ind in jd_lower
    ]

    industry_overlap = [
        ind
        for ind in resume_industries
        if ind in jd_industries
    ]

    industry_gaps = [
        ind
        for ind in jd_industries
        if ind not in resume_industries
    ]

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

# =========================================================
# FEEDBACK GENERATION
# =========================================================

def generate_feedback(
    resume_text: str,
    jd_text: str,
    score: int,
    missing_keywords: list[str]
) -> str:

    missing_str = (
        ", ".join(missing_keywords[:5])
        if missing_keywords
        else "none"
    )

    if HF_TOKEN:

        prompt = f"""
You are an ATS resume reviewer.

Analyze the resume against the job description.

Return:
1. Missing technical skills
2. Weak resume areas
3. Missing measurable achievements
4. Experience gaps
5. Exact resume improvements

Keep the response concise, practical, and professional.

Resume score: {score}%

Missing skills:
{missing_str}

Resume:
{resume_text[:1000]}

Job Description:
{jd_text[:1000]}
"""

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}"
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 220,
                        "repetition_penalty": 3.0
                    }
                },
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                if (
                    isinstance(result, dict)
                    and result.get("error")
                ):
                    raise ValueError(result["error"])

                if isinstance(result, list) and result:
                    return (
                        result[0]
                        .get("generated_text", "")
                        .strip()
                    )

        except Exception:
            pass

    # fallback response

    lines = []

    if missing_keywords:
        lines.append(
            f"1. Add missing skills: "
            f"{', '.join(missing_keywords[:5])}."
        )

    if score < 50:
        lines.append(
            "2. Tailor your summary and projects "
            "to match the job description more closely."
        )

    lines.append(
        "3. Add measurable metrics such as "
        "accuracy %, users served, "
        "latency reduction, or automation impact."
    )

    return "\n".join(lines)

# =========================================================
# MISSING KEYWORDS
# =========================================================

def get_missing_keywords(
    resume_text: str,
    jd_keywords: list[str]
) -> list[str]:

    resume_lower = resume_text.lower()

    missing = []

    for kw in jd_keywords:

        if not semantic_match(kw, resume_lower):
            missing.append(kw)

    return missing