"""
Step 4: ATS Score Calculator
==============================
Calculates an Applicant Tracking System (ATS) compatibility score.

ATS systems automatically screen resumes. This module simulates
what an ATS looks for and scores accordingly.

Scoring Breakdown (Total: 100 points):
  - Skills Match:        30 points
  - Section Presence:    25 points
  - Keyword Density:     20 points
  - Resume Length:       15 points
  - Formatting Quality:  10 points
"""

import re
from typing import Dict, List


# ─── ATS Keywords that most systems look for ─────────────────────────────────
ATS_POWER_KEYWORDS = [
    # Action verbs
    "developed", "implemented", "designed", "built", "created", "managed",
    "led", "optimized", "improved", "increased", "reduced", "achieved",
    "delivered", "launched", "maintained", "collaborated", "architected",
    "automated", "deployed", "integrated", "analyzed", "engineered",
    
    # Quantifiers (ATS loves metrics)
    "%", "percent", "million", "thousand", "users", "revenue", "performance",
    
    # Professional terms
    "cross-functional", "stakeholder", "agile", "scrum", "roadmap",
    "scalable", "production", "enterprise", "end-to-end"
]


def score_skills(skills_list: List[str]) -> Dict:
    """
    Score based on number and quality of skills. (30 points max)
    
    Scoring:
    - 0 skills:     0 pts
    - 1-5 skills:   10 pts
    - 6-10 skills:  18 pts
    - 11-15 skills: 24 pts
    - 16+ skills:   30 pts
    """
    count = len(skills_list)
    
    if count == 0:
        score = 0
    elif count <= 5:
        score = 10
    elif count <= 10:
        score = 18
    elif count <= 15:
        score = 24
    else:
        score = 30
    
    return {
        'score': score,
        'max': 30,
        'skills_found': count,
        'feedback': f"Found {count} skills. {'Great!' if count >= 15 else 'Add more technical skills.' if count < 10 else 'Good coverage.'}"
    }


def score_sections(sections_present: Dict) -> Dict:
    """
    Score based on presence of standard resume sections. (25 points max)
    
    Each section is weighted by ATS importance:
    - Contact Info: 5 pts (critical)
    - Experience:   7 pts (most important)
    - Education:    5 pts (required)
    - Skills:       4 pts (key for ATS)
    - Summary:      2 pts (recommended)
    - Projects:     1 pt
    - Certs:        1 pt
    """
    weights = {
        'contact_info': 5,
        'experience_work': 7,
        'education': 5,
        'skills_section': 4,
        'summary_objective': 2,
        'projects': 1,
        'certifications': 1,
        'achievements': 0  # bonus
    }
    
    total_score = 0
    section_details = {}
    
    for section, present in sections_present.items():
        points = weights.get(section, 0) if present else 0
        total_score += points
        section_details[section] = {
            'present': present,
            'points': points,
            'max': weights.get(section, 0)
        }
    
    # Cap at 25
    total_score = min(total_score, 25)
    
    missing_critical = [
        s for s, p in sections_present.items()
        if not p and weights.get(s, 0) >= 4
    ]
    
    return {
        'score': total_score,
        'max': 25,
        'section_details': section_details,
        'missing_critical': missing_critical,
        'feedback': f"Found {sum(1 for p in sections_present.values() if p)}/{len(sections_present)} sections."
    }


def score_keyword_density(resume_text: str) -> Dict:
    """
    Score ATS power keyword density. (20 points max)
    
    More action verbs and measurable results = better ATS score.
    """
    text_lower = resume_text.lower()
    words = text_lower.split()
    total_words = len(words)
    
    if total_words == 0:
        return {'score': 0, 'max': 20, 'keywords_found': 0, 'feedback': 'No text found'}
    
    found_keywords = [kw for kw in ATS_POWER_KEYWORDS if kw in text_lower]
    keyword_count = len(found_keywords)
    
    # Check for quantified achievements (numbers in context)
    has_metrics = bool(re.search(r'\d+[\s]*(%|percent|users|customers|million|k\b)', text_lower))
    
    # Score calculation
    base_score = min(keyword_count * 1.2, 16)  # up to 16 pts
    metrics_bonus = 4 if has_metrics else 0
    total = min(int(base_score + metrics_bonus), 20)
    
    return {
        'score': total,
        'max': 20,
        'keywords_found': keyword_count,
        'has_metrics': has_metrics,
        'sample_keywords': found_keywords[:8],
        'feedback': (
            f"Found {keyword_count} power keywords. "
            f"{'✓ Quantified achievements detected!' if has_metrics else 'Add metrics (%, numbers, users) to boost score.'}"
        )
    }


def score_resume_length(resume_text: str) -> Dict:
    """
    Score based on resume length. (15 points max)
    
    ATS ideal: 400-800 words (1-2 pages).
    Too short = lacking detail. Too long = likely filtered.
    """
    words = len(resume_text.split())
    
    if words < 100:
        score, feedback = 3, f"Too short ({words} words). Aim for 400-800 words."
    elif words < 300:
        score, feedback = 7, f"Somewhat short ({words} words). Expand your experience descriptions."
    elif words <= 800:
        score, feedback = 15, f"Excellent length ({words} words). ATS-friendly!"
    elif words <= 1200:
        score, feedback = 10, f"Slightly long ({words} words). Consider condensing to 2 pages."
    else:
        score, feedback = 5, f"Too long ({words} words). ATS may truncate. Aim for max 2 pages."
    
    return {
        'score': score,
        'max': 15,
        'word_count': words,
        'feedback': feedback
    }


def score_formatting(resume_text: str, sections_present: Dict) -> Dict:
    """
    Score formatting quality. (10 points max)
    
    Checks for:
    - Consistent date formats
    - Email address present
    - Phone number present
    - LinkedIn/GitHub URLs
    - Bullet point usage
    """
    score = 0
    checks = {}
    
    # Email check (2 pts)
    has_email = bool(re.search(r'\b[\w._%+-]+@[\w.-]+\.[a-z]{2,}\b', resume_text, re.I))
    checks['email'] = has_email
    score += 2 if has_email else 0
    
    # Phone check (2 pts)
    has_phone = bool(re.search(r'(\+?\d[\d\s\-()]{8,}\d)', resume_text))
    checks['phone'] = has_phone
    score += 2 if has_phone else 0
    
    # LinkedIn/GitHub (2 pts)
    has_profile = bool(re.search(r'(linkedin\.com|github\.com|gitlab\.com)', resume_text, re.I))
    checks['professional_profile'] = has_profile
    score += 2 if has_profile else 0
    
    # Dates present (2 pts)
    has_dates = bool(re.search(r'\b(20\d{2}|19\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', resume_text, re.I))
    checks['dates'] = has_dates
    score += 2 if has_dates else 0
    
    # Multiple sections (2 pts)
    section_count = sum(1 for v in sections_present.values() if v)
    checks['multiple_sections'] = section_count >= 4
    score += 2 if section_count >= 4 else 0
    
    return {
        'score': min(score, 10),
        'max': 10,
        'checks': checks,
        'feedback': f"Passed {sum(1 for v in checks.values() if v)}/{len(checks)} formatting checks."
    }


def calculate_ats_score(resume_text: str, skills_data: Dict, sections_data: Dict) -> Dict:
    """
    Main ATS scoring function. Combines all sub-scores.
    
    Args:
        resume_text: Full resume text
        skills_data: Output from skill_analyzer.detect_skills()
        sections_data: Output from skill_analyzer.detect_resume_sections()
    
    Returns:
        dict: Complete ATS analysis with total score and breakdown
    """
    # Calculate each component
    skills_score = score_skills(skills_data.get('all_skills', []))
    sections_score = score_sections(sections_data)
    keyword_score = score_keyword_density(resume_text)
    length_score = score_resume_length(resume_text)
    format_score = score_formatting(resume_text, sections_data)
    
    # Total score
    total = (
        skills_score['score'] +
        sections_score['score'] +
        keyword_score['score'] +
        length_score['score'] +
        format_score['score']
    )
    
    # Grade assignment
    if total >= 85:
        grade, grade_label = 'A', 'Excellent'
    elif total >= 70:
        grade, grade_label = 'B', 'Good'
    elif total >= 55:
        grade, grade_label = 'C', 'Fair'
    elif total >= 40:
        grade, grade_label = 'D', 'Needs Work'
    else:
        grade, grade_label = 'F', 'Poor'
    
    # Overall feedback
    if total >= 80:
        overall_feedback = "Your resume is well-optimized for ATS systems."
    elif total >= 60:
        overall_feedback = "Good resume, some improvements will boost your ATS score significantly."
    else:
        overall_feedback = "Your resume needs optimization to pass ATS screening effectively."
    
    return {
        'total_score': total,
        'max_score': 100,
        'grade': grade,
        'grade_label': grade_label,
        'overall_feedback': overall_feedback,
        'breakdown': {
            'skills': skills_score,
            'sections': sections_score,
            'keywords': keyword_score,
            'length': length_score,
            'formatting': format_score,
        }
    }
