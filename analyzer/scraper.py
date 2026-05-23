"""
scraper.py
Fetches full job description text from any public job posting URL.
Works with Naukri, Indeed, Internshala, Glassdoor, company career pages, etc.
LinkedIn will fail with a clear message since it requires login.
"""

import re
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
}

NOISE_TAGS = [
    "script", "style", "nav", "header",
    "footer", "aside", "noscript", "iframe",
]

LOGIN_SIGNALS = [
    "sign in to view", "please log in",
    "join to see", "create an account to",
    "authwall", "login required",
]


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_login_wall(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in LOGIN_SIGNALS)


def fetch_jd_from_url(url: str) -> dict:
    """
    Fetch and return all meaningful text content from a job posting URL.

    Returns a dict with:
        success  (bool)   — whether extraction succeeded
        text     (str)    — extracted job description text
        error    (str)    — error message if success is False, else None
    """

    url = url.strip()

    if not url.startswith("http"):
        return {
            "success": False,
            "text": "",
            "error": "Invalid URL. Make sure it starts with https://",
        }

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "text": "",
            "error": (
                "Request timed out. "
                "Please try again or paste the JD manually."
            ),
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "text": "",
            "error": (
                "Could not connect to the URL. "
                "Check the link and try again."
            ),
        }

    except requests.exceptions.HTTPError:
        status = response.status_code

        if status == 403:
            return {
                "success": False,
                "text": "",
                "error": (
                    "This site blocked the request — it likely requires login "
                    "(common with LinkedIn). "
                    "Please paste the job description manually."
                ),
            }

        if status == 404:
            return {
                "success": False,
                "text": "",
                "error": "Job posting not found (404). The link may have expired.",
            }

        return {
            "success": False,
            "text": "",
            "error": (
                f"HTTP {status} error. "
                "Please paste the job description manually."
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"Unexpected error: {str(e)}",
        }

    # Parse and clean HTML
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = clean_extracted_text(text)

    # Check for login wall
    if detect_login_wall(text):
        return {
            "success": False,
            "text": "",
            "error": (
                "This page requires login to view the full job description. "
                "Please paste the JD manually."
            ),
        }

    # Check minimum content
    if len(text) < 150:
        return {
            "success": False,
            "text": "",
            "error": (
                "Not enough content extracted from this page. "
                "Please paste the job description manually."
            ),
        }

    return {
        "success": True,
        "text": text[:8000],
        "error": None,
    }