"""
scraper.py
Fetches job description text from any job posting URL.

Strategy:
  1. Jina AI Reader (r.jina.ai) — handles JS-heavy sites like Oracle HCM,
     Workday, Greenhouse, Lever, LinkedIn, Indeed, Naukri, etc.
  2. Direct BeautifulSoup — fallback for simple static pages if Jina fails.

No API key required. No headless browser needed.
"""

import re
import requests
from bs4 import BeautifulSoup


# ── Request Headers ────────────────────────────────────────────────────────────

BROWSER_HEADERS = {
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

JINA_HEADERS = {
    "Accept": "text/plain",
    "X-Return-Format": "text",
    "X-Timeout": "15",
}

# ── Noise to strip from BeautifulSoup ─────────────────────────────────────────

NOISE_TAGS = [
    "script", "style", "nav", "header",
    "footer", "aside", "noscript", "iframe",
]

# ── Login wall detection ───────────────────────────────────────────────────────

LOGIN_SIGNALS = [
    "sign in to view",
    "please log in",
    "join to see",
    "create an account to",
    "authwall",
    "login required",
    "sign up to apply",
    "you must be signed in",
    "register to apply",
]

# ── Known platforms ────────────────────────────────────────────────────────────

KNOWN_PLATFORMS = {
    "linkedin.com": "LinkedIn",
    "naukri.com": "Naukri",
    "indeed.com": "Indeed",
    "glassdoor.com": "Glassdoor",
    "internshala.com": "Internshala",
    "wellfound.com": "Wellfound",
    "angel.co": "AngelList",
    "oraclecloud.com": "Oracle HCM",
    "myworkdayjobs.com": "Workday",
    "workday.com": "Workday",
    "greenhouse.io": "Greenhouse",
    "lever.co": "Lever",
    "taleo.net": "Taleo",
    "successfactors.com": "SAP SuccessFactors",
    "icims.com": "iCIMS",
    "smartrecruiters.com": "SmartRecruiters",
    "jobvite.com": "Jobvite",
    "breezy.hr": "Breezy HR",
    "bamboohr.com": "BambooHR",
    "recruitee.com": "Recruitee",
    "zoho.com": "Zoho Recruit",
    "freshteam.com": "Freshteam",
    "keka.com": "Keka",
    "darwinbox.com": "Darwinbox",
    "shine.com": "Shine",
    "monster.com": "Monster",
    "foundit.in": "Foundit",
    "timesjobs.com": "TimesJobs",
    "hirist.com": "Hirist",
    "instahyre.com": "Instahyre",
    "iimjobs.com": "IIMJobs",
    "cutshort.io": "Cutshort",
    "unstop.com": "Unstop",
    "hackerearth.com": "HackerEarth",
    "hackerrank.com": "HackerRank",
    "ambitionbox.com": "AmbitionBox",
    "apna.co": "Apna",
    "fresherworld.com": "Fresherworld",
    "freshersworld.com": "Freshersworld",
    "placementindia.com": "PlacementIndia",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_platform(url: str) -> str:
    url_lower = url.lower()
    for domain, name in KNOWN_PLATFORMS.items():
        if domain in url_lower:
            return name
    return "Job Posting"


def has_login_wall(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in LOGIN_SIGNALS)


def is_sufficient(text: str, min_words: int = 80) -> bool:
    return len(text.split()) >= min_words


# ── Method 1: Jina AI Reader ───────────────────────────────────────────────────

def fetch_via_jina(url: str) -> dict:
    jina_url = f"https://r.jina.ai/{url}"
    try:
        response = requests.get(
            jina_url,
            headers=JINA_HEADERS,
            timeout=25,
        )
        if response.status_code == 200:
            text = clean_text(response.text)
            if is_sufficient(text):
                return {"success": True, "text": text[:9000]}
    except Exception:
        pass
    return {"success": False, "text": ""}


# ── Method 2: Direct BeautifulSoup ────────────────────────────────────────────

def fetch_via_bs4(url: str) -> dict:
    try:
        response = requests.get(
            url,
            headers=BROWSER_HEADERS,
            timeout=15,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        return {"success": False, "text": "", "error_code": "timeout"}

    except requests.exceptions.HTTPError:
        code = response.status_code
        return {"success": False, "text": "", "error_code": f"http_{code}"}

    except requests.exceptions.ConnectionError:
        return {"success": False, "text": "", "error_code": "connection"}

    except Exception:
        return {"success": False, "text": "", "error_code": "unknown"}

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = clean_text(text)

    if not is_sufficient(text):
        return {"success": False, "text": "", "error_code": "insufficient"}

    return {"success": True, "text": text[:9000], "error_code": None}


# ── Main Entry Point ───────────────────────────────────────────────────────────

def fetch_jd_from_url(url: str) -> dict:
    url = url.strip()
    platform = detect_platform(url)

    if not url.startswith("http"):
        return {
            "success": False,
            "text": "",
            "platform": platform,
            "error": "Invalid URL. Make sure it starts with https://",
        }

    # Try Jina AI Reader first
    jina = fetch_via_jina(url)
    if jina["success"]:
        text = jina["text"]
        if has_login_wall(text):
            return {
                "success": False,
                "text": "",
                "platform": platform,
                "error": (
                    f"{platform} requires login to view this job. "
                    "Please paste the job description manually."
                ),
            }
        return {"success": True, "text": text, "platform": platform, "error": None}

    # Fallback: direct BeautifulSoup
    bs4 = fetch_via_bs4(url)
    if bs4["success"]:
        text = bs4["text"]
        if has_login_wall(text):
            return {
                "success": False,
                "text": "",
                "platform": platform,
                "error": (
                    f"{platform} requires login to view this job. "
                    "Please paste the job description manually."
                ),
            }
        return {"success": True, "text": text, "platform": platform, "error": None}

    error_code = bs4.get("error_code", "unknown")
    error_messages = {
        "timeout": (
            "The page took too long to respond. "
            "Try again or paste the JD manually."
        ),
        "http_403": (
            f"{platform} blocked the request. "
            "Please paste the job description manually."
        ),
        "http_404": "Job posting not found — the link may have expired.",
        "http_429": (
            f"{platform} is rate-limiting right now. "
            "Try again in a minute or paste the JD manually."
        ),
        "connection": "Could not connect. Check the link and try again.",
        "insufficient": (
            "Not enough content extracted. "
            "Please paste the job description manually."
        ),
    }

    return {
        "success": False,
        "text": "",
        "platform": platform,
        "error": error_messages.get(
            error_code,
            "Could not fetch this URL. Please paste the JD manually."
        ),
    }