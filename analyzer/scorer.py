"""
scorer.py
Combines semantic similarity score and keyword match rate
into a final ATS-style score with a breakdown.
"""

from analyzer.embeddings import compute_similarity, similarity_to_percentage
from analyzer.pipeline import extract_keywords, get_missing_keywords


def compute_keyword_match_score(resume_text: str, jd_keywords: list[str]) -> float:
    """
    Calculate what percentage of JD keywords appear in the resume.
    Returns a float between 0.0 and 1.0.
    """
    if not jd_keywords:
        return 0.0

    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw.lower() in resume_lower]
    return round(len(matched) / len(jd_keywords), 4)


def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    """
    Main scoring function. Returns a full breakdown dict with:
    - final_score: weighted ATS score out of 100
    - similarity_score: semantic similarity percentage
    - keyword_score: keyword match percentage
    - matched_keywords: list of JD keywords found in resume
    - missing_keywords: list of JD keywords not found in resume
    - all_keywords: full list of extracted JD keywords
    - rating: label like "Excellent", "Good", "Fair", "Poor"
    """
    # Semantic similarity (60% weight)
    similarity = compute_similarity(resume_text, jd_text)
    similarity_pct = similarity_to_percentage(similarity)

    # Keyword extraction and matching (40% weight)
    jd_keywords = extract_keywords(jd_text)
    keyword_match = compute_keyword_match_score(resume_text, jd_keywords)
    keyword_pct = int(keyword_match * 100)

    # Weighted final score
    final_score = int((similarity_pct * 0.6) + (keyword_pct * 0.4))

    # Matched vs missing keywords
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw.lower() in resume_lower]
    missing = [kw for kw in jd_keywords if kw.lower() not in resume_lower]

    # Rating label
    if final_score >= 80:
        rating = "Excellent"
    elif final_score >= 60:
        rating = "Good"
    elif final_score >= 40:
        rating = "Fair"
    else:
        rating = "Poor"

    return {
        "final_score": final_score,
        "similarity_score": similarity_pct,
        "keyword_score": keyword_pct,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "all_keywords": jd_keywords,
        "rating": rating,
    }