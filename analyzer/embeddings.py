"""
embeddings.py
Loads a HuggingFace sentence-transformer model, encodes resume and
job description text into vectors, and computes semantic similarity.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Using a lightweight but strong model — good balance of speed and accuracy
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def load_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model once and cache it globally.
    Avoids reloading on every function call.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embedding(text: str) -> np.ndarray:
    """
    Encode a single piece of text into a vector embedding.
    Returns a 1D numpy array.
    """
    model = load_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between resume and job description embeddings.
    Returns a float between 0.0 and 1.0.
    1.0 = perfect match, 0.0 = completely unrelated.
    """
    resume_vec = get_embedding(resume_text).reshape(1, -1)
    jd_vec = get_embedding(jd_text).reshape(1, -1)

    similarity = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(float(similarity), 4)


def compute_section_similarities(resume_sections: dict, jd_text: str) -> dict:
    """
    Compute similarity scores for individual resume sections against the JD.
    resume_sections: dict like {"Skills": "Python, ML...", "Experience": "..."}
    Returns: dict like {"Skills": 0.82, "Experience": 0.74}
    """
    results = {}
    jd_vec = get_embedding(jd_text).reshape(1, -1)

    for section_name, section_text in resume_sections.items():
        if not section_text.strip():
            continue
        sec_vec = get_embedding(section_text).reshape(1, -1)
        sim = cosine_similarity(sec_vec, jd_vec)[0][0]
        results[section_name] = round(float(sim), 4)

    return results


def similarity_to_percentage(score: float) -> int:
    """
    Convert raw cosine similarity (0 to 1) to a readable percentage.
    """
    return int(score * 100)