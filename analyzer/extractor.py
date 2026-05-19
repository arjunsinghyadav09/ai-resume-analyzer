"""
extractor.py
Handles extracting clean text from uploaded resume files.
Supports PDF and plain text input.
"""

import pdfplumber
import re


def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file object (e.g. from Streamlit file_uploader).
    Returns cleaned plain text string.
    """
    full_text = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    raw = "\n".join(full_text)
    return clean_text(raw)


def extract_text_from_string(text: str) -> str:
    """
    Clean and return text that was pasted directly (plain text input).
    """
    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Remove excessive whitespace, special characters, and normalize newlines.
    """
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def get_word_count(text: str) -> int:
    return len(text.split())


def get_section_hints(text: str) -> list[str]:
    """
    Detect likely resume section headings present in the text.
    Useful for quick validation that the uploaded file is a resume.
    """
    common_sections = [
        "experience", "education", "skills", "projects",
        "summary", "objective", "certifications", "achievements",
        "internship", "publications", "languages"
    ]
    found = []
    lower = text.lower()
    for section in common_sections:
        if section in lower:
            found.append(section.capitalize())
    return found