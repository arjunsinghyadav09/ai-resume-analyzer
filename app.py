"""
app.py - AI Resume Analyzer Streamlit Frontend
"""

import re
import streamlit as st
from analyzer.extractor import extract_text_from_pdf, get_section_hints
from analyzer.scorer import compute_ats_score
from analyzer.pipeline import (
    generate_feedback, analyze_experience_quality,
    match_experience_to_jd, detect_role_type,
    extract_keywords, get_missing_keywords, STRONG_VERBS
)

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown(
    "Upload your resume and paste a job description for a full ATS analysis, "
    "experience evaluation, and copy-ready rewrite suggestions."
)
st.divider()

# ── Input Columns ──────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    upload_option = st.radio(
        "Input method", ["Upload PDF", "Paste Text"], horizontal=True
    )
    resume_text = ""

    if upload_option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extracting text..."):
                resume_text = extract_text_from_pdf(uploaded_file)
            st.success(f"Extracted {len(resume_text.split())} words.")
            sections = get_section_hints(resume_text)
            if sections:
                st.caption("Detected sections: " + ", ".join(sections))
    else:
        resume_text = st.text_area("Paste your resume text here", height=300)

with col2:
    st.subheader("Job Description")

    jd_url = st.text_input(
        "Paste a job URL",
        placeholder="https://naukri.com/... or Indeed, Internshala, Oracle, Workday..."
    )

    jd_text = ""

    if jd_url:
        with st.spinner("Fetching job description..."):
            from analyzer.scraper import fetch_jd_from_url
            result_scrape = fetch_jd_from_url(jd_url)
        if result_scrape["success"]:
            word_count = len(result_scrape["text"].split())
            st.success(f"Fetched from {result_scrape['platform']} — {word_count} words extracted.")
            jd_text = result_scrape["text"]
            with st.expander("Preview fetched JD"):
                preview = result_scrape["text"]
                st.text(preview[:1000] + "..." if len(preview) > 1000 else preview)
        else:
            st.error(result_scrape["error"])

    st.markdown("<div style='text-align:center; color:gray; margin: 8px 0'>— OR —</div>", unsafe_allow_html=True)

    manual_jd = st.text_area(
        "Paste the job description manually",
        height=200,
        placeholder="Copy and paste the full job description here..."
    )

    if manual_jd.strip():
        jd_text = manual_jd

# ── Analyze Button ─────────────────────────────────────────────────────────────

st.divider()
analyze_clicked = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze_clicked:
    if not resume_text.strip():
        st.error("Please provide your resume.")
    elif not jd_text.strip():
        st.error("Please paste or fetch a job description.")
    else:
        with st.spinner("Running full analysis..."):
            result = compute_ats_score(resume_text, jd_text)
            role_type = detect_role_type(jd_text)
            exp_quality = analyze_experience_quality(resume_text)
            exp_match = match_experience_to_jd(resume_text, jd_text)

        # ── Section 1: ATS Score ───────────────────────────────────────
        st.divider()
        st.subheader("ATS Score")
        st.caption(f"Role detected: **{role_type}**")
        s1, s2, s3 = st.columns(3)
        rating_colors = {
            "Excellent": "🟢", "Good": "🔵", "Fair": "🟡", "Poor": "🔴"
        }
        with s1:
            st.metric("Overall ATS Score", f"{result['final_score']}%")
            st.markdown(f"**Rating:** {rating_colors[result['rating']]} {result['rating']}")
        with s2:
            st.metric("Semantic Similarity", f"{result['similarity_score']}%")
        with s3:
            st.metric("Keyword Match", f"{result['keyword_score']}%")

        # ── Section 2: Keyword Analysis ────────────────────────────────
        st.divider()
        st.subheader("Keyword Analysis")
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Matched Keywords**")
            if result["matched_keywords"]:
                for kw in result["matched_keywords"]:
                    st.markdown(f"✅ `{kw}`")
            else:
                st.markdown("_None found_")
        with k2:
            st.markdown("**Missing Keywords**")
            if result["missing_keywords"]:
                for kw in result["missing_keywords"]:
                    st.markdown(f"❌ `{kw}`")
            else:
                st.markdown("_You cover all detected keywords_")

        # ── Section 3: Experience Analysis ────────────────────────────
        st.divider()
        st.subheader("Experience Analysis")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Your Roles and Internships**")
            if exp_match["candidate_internships"]:
                for line in exp_match["candidate_internships"]:
                    st.markdown(f"🎓 {line}")
            else:
                st.warning("No internships detected in your resume.")
            if exp_match["candidate_roles"]:
                for line in exp_match["candidate_roles"][:4]:
                    st.markdown(f"💼 {line}")
            if exp_match["candidate_industries"]:
                st.markdown("_Industry exposure:_")
                for ind in exp_match["candidate_industries"]:
                    st.markdown(f"🏭 {ind.title()}")
        with e2:
            st.markdown("**Experience vs JD Requirements**")
            req = exp_match["jd_required_years"]
            if req > 0:
                if exp_match["experience_count"] >= req:
                    st.success(f"JD needs {req}+ years — relevant experience found.")
                else:
                    st.warning(f"JD needs {req}+ years — limited experience detected.")
            else:
                st.info("No specific years of experience required.")
            if exp_match["industry_overlap"]:
                for ind in exp_match["industry_overlap"]:
                    st.markdown(f"✅ {ind.title()}")
            if exp_match["industry_gaps"]:
                for ind in exp_match["industry_gaps"]:
                    st.markdown(f"❌ {ind.title()} — not reflected in your resume")

        # ── Section 4: Resume Writing Quality ─────────────────────────
        st.divider()
        st.subheader("Resume Writing Quality")
        q1, q2, q3 = st.columns(3)
        with q1:
            st.markdown("**Strong Action Verb Bullets**")
            if exp_quality["strong_verb_lines"]:
                for line in exp_quality["strong_verb_lines"]:
                    st.markdown(
                        f"✅ _{line[:90]}..._" if len(line) > 90 else f"✅ _{line}_"
                    )
            else:
                st.warning("No strong action verbs found. Use: Built, Developed, Optimized, Deployed...")
        with q2:
            st.markdown("**Weak Bullets to Rewrite**")
            if exp_quality["weak_verb_lines"]:
                for line in exp_quality["weak_verb_lines"]:
                    st.markdown(
                        f"⚠️ _{line[:90]}..._" if len(line) > 90 else f"⚠️ _{line}_"
                    )
                st.caption("Replace 'helped', 'assisted', 'responsible for' with impact verbs.")
            else:
                st.success("No weak verb patterns found.")
        with q3:
            st.markdown("**Quantified Achievements**")
            if exp_quality["quantified_lines"]:
                for line in exp_quality["quantified_lines"]:
                    st.markdown(
                        f"📊 _{line[:90]}..._" if len(line) > 90 else f"📊 _{line}_"
                    )
            else:
                st.warning("No metrics found. Add numbers like: improved accuracy by 15%, processed 10k samples.")
        st.caption(
            f"Bullets analysed: {exp_quality['total_bullets']} | "
            f"Strong verbs: {exp_quality['strong_verb_count']} | "
            f"Quantified: {exp_quality['quantified_count']}"
        )

        # ── Section 5: Gap Analysis + Rewrite Suggestions ─────────────
        st.divider()
        st.subheader("Resume vs Job Description — Gap Analysis")
        st.caption(
            "What your resume currently has vs what this role actually needs, "
            "with copy-ready rewrites."
        )

        matched = result["matched_keywords"]
        missing = result["missing_keywords"]

        if matched:
            st.markdown("**What your resume already covers:**")
            cols = st.columns(4)
            for i, kw in enumerate(matched[:8]):
                cols[i % 4].success(f"✅ {kw}")

        if missing:
            st.markdown("**What this role needs that your resume is missing:**")
            cols = st.columns(4)
            for i, kw in enumerate(missing[:12]):
                cols[i % 4].error(f"❌ {kw}")

        st.divider()
        st.subheader("Copy-Ready Rewrite Suggestions")
        st.caption("Each suggestion below has a copy button on the right. Click it, paste directly into your resume.")

        if exp_quality["weak_verb_lines"]:
            st.markdown("#### Weak Bullets — Suggested Rewrites")
            rewrite_map = {
                "contributed": "Built",
                "helped": "Developed",
                "assisted": "Implemented",
                "responsible for": "Engineered",
                "worked on": "Delivered",
                "participated": "Led",
                "supported": "Optimized",
                "was part of": "Contributed directly to",
                "involved in": "Executed",
            }
            for line in exp_quality["weak_verb_lines"]:
                st.markdown(f"**Original:** _{line}_")
                rewritten = line
                for weak, strong in rewrite_map.items():
                    if weak in line.lower():
                        rewritten = re.sub(
                            re.escape(weak), strong, line, flags=re.IGNORECASE
                        )
                        break
                st.markdown("**Suggested rewrite:**")
                st.code(rewritten, language="")
                st.caption("Add a metric to this line for even more impact, e.g. 'improving X by Y%'")
                st.markdown("---")

        if missing:
            st.markdown("#### Missing Keywords — Suggested Bullets to Add")
            bullet_templates = {
                "claude": "Used Claude as a daily AI coding assistant to accelerate development, debug code, and automate repetitive tasks in production workflows.",
                "codex": "Leveraged OpenAI Codex for AI-assisted code generation, reducing development time by automating boilerplate and repetitive coding tasks.",
                "chatgpt": "Integrated ChatGPT into daily workflows for rapid prototyping, code review assistance, and automating research and documentation tasks.",
                "cursor": "Used Cursor AI editor for AI-pair-programming, enabling faster code iteration and intelligent refactoring across multiple projects.",
                "agentic ai": "Built agentic AI workflows that chain LLM calls, tool use, and conditional logic to automate multi-step operational tasks.",
                "ai agents": "Designed and deployed AI agents using LangChain that autonomously execute tasks, retrieve information, and update systems.",
                "next.js": "Built and deployed a full-stack web application using Next.js and Vercel, with server-side rendering and API routes.",
                "vercel": "Deployed production web applications to Vercel with CI/CD pipeline integration and custom domain configuration.",
                "metabase": "Created operational dashboards in Metabase to track KPIs, visualize trends, and support data-driven decision making.",
                "looker studio": "Built automated reporting dashboards in Looker Studio connected to live data sources for real-time business insights.",
                "google sheets": "Automated data workflows in Google Sheets using formulas, pivot tables, and Apps Script to replace manual reporting processes.",
                "automation": "Built end-to-end automation workflows that eliminated manual data entry, reducing processing time by over 60%.",
                "sql": "Wrote complex SQL queries with joins, window functions, and aggregations to extract and analyze large operational datasets.",
                "rest api": "Designed and consumed REST APIs for system integration, enabling seamless data flow between internal tools and third-party services.",
                "flask": "Built and deployed a Flask REST API backend to serve ML model predictions with JSON responses and authentication.",
                "docker": "Containerized ML application using Docker for consistent deployment across development and production environments.",
                "paid ads": "Supported paid ads campaign tracking by building conversion dashboards and integrating analytics pixel data.",
                "funnels": "Designed and monitored conversion funnels using analytics tools to identify drop-off points and improve user flow.",
                "landing pages": "Built responsive landing pages with tracking pixels, A/B test variants, and form integrations for lead generation.",
            }
            for kw in missing[:8]:
                template = bullet_templates.get(kw.lower())
                if template:
                    st.markdown(f"**Add this bullet for `{kw}`:**")
                    st.code(template, language="")
                else:
                    generic = (
                        f"Applied {kw} in a project context to "
                        f"[describe task], achieving [describe outcome or metric]."
                    )
                    st.markdown(f"**Add this bullet for `{kw}`:**")
                    st.code(generic, language="")
                st.markdown("")

        # ── Section 6: AI Feedback ─────────────────────────────────────
        st.divider()
        st.subheader("AI Feedback")
        with st.spinner("Generating AI feedback..."):
            feedback = generate_feedback(
                resume_text, jd_text,
                result["final_score"],
                result["missing_keywords"]
            )
        st.code(feedback, language="")

        st.divider()
        st.caption("Powered by Sentence Transformers, KeyBERT, Google Flan-T5, and Streamlit.")