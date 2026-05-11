"""
Step 3: Skill Analysis Module
================================
Detects technical and soft skills from resume text.
Uses keyword matching with a comprehensive skills database.
"""

import re
from typing import List, Dict


# ─── Skills Database ─────────────────────────────────────────────────────────
# Organized by category for better analysis and reporting

SKILLS_DATABASE = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "dart",
        "perl", "shell", "bash", "powershell", "vba", "cobol", "fortran"
    ],
    "Web Frontend": [
        "html", "css", "html5", "css3", "react", "reactjs", "vue", "vuejs", "angular",
        "angularjs", "svelte", "nextjs", "nuxtjs", "gatsby", "bootstrap", "tailwind",
        "sass", "scss", "less", "jquery", "webpack", "babel", "vite"
    ],
    "Web Backend": [
        "django", "flask", "fastapi", "node", "nodejs", "express", "expressjs",
        "spring", "spring boot", "laravel", "rails", "ruby on rails", "asp.net",
        "fastify", "nestjs", "graphql", "rest", "restful", "api"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra",
        "dynamodb", "oracle", "mssql", "sql server", "elasticsearch", "neo4j",
        "firebase", "supabase", "nosql", "mariadb"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "ci/cd", "github actions", "gitlab ci",
        "heroku", "vercel", "netlify", "linux", "nginx", "apache"
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "neural network", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
        "numpy", "matplotlib", "seaborn", "hugging face", "transformers",
        "llm", "rag", "langchain", "openai", "generative ai"
    ],
    "Data & Analytics": [
        "data analysis", "data science", "data engineering", "etl", "tableau",
        "power bi", "excel", "hadoop", "spark", "kafka", "airflow", "dbt",
        "snowflake", "bigquery", "looker", "jupyter"
    ],
    "Version Control & Collaboration": [
        "git", "github", "gitlab", "bitbucket", "svn", "jira", "confluence",
        "trello", "slack", "agile", "scrum", "kanban", "devops"
    ],
    "Mobile Development": [
        "android", "ios", "react native", "flutter", "xamarin", "ionic",
        "swift", "kotlin", "mobile development"
    ],
    "Security": [
        "cybersecurity", "penetration testing", "ethical hacking", "owasp",
        "ssl", "encryption", "oauth", "jwt", "security", "firewall"
    ],
    "Soft Skills": [
        "leadership", "communication", "problem solving", "teamwork", "collaboration",
        "project management", "time management", "analytical", "critical thinking",
        "presentation", "mentoring", "adaptability"
    ]
}

# Flatten for quick lookup
ALL_SKILLS_FLAT = {}
for category, skills in SKILLS_DATABASE.items():
    for skill in skills:
        ALL_SKILLS_FLAT[skill.lower()] = category


def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace for matching."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def detect_skills(resume_text: str) -> Dict:
    """
    Scan resume text for known skills.
    
    Algorithm:
    1. Normalize the resume text
    2. Check each skill keyword against the text
    3. Use word boundary matching to avoid false positives
       (e.g., "R" skill shouldn't match "React")
    4. Group found skills by category
    
    Args:
        resume_text: Cleaned resume text
    Returns:
        dict: {
            'all_skills': list of all found skills,
            'by_category': dict of category -> skills list,
            'skill_count': total number of skills found,
            'top_skills': most important skills found
        }
    """
    if not resume_text:
        return {'all_skills': [], 'by_category': {}, 'skill_count': 0, 'top_skills': []}
    
    normalized_text = normalize_text(resume_text)
    found_skills = {}  # skill -> category
    
    for skill, category in ALL_SKILLS_FLAT.items():
        # Build regex pattern
        # For multi-word skills: exact phrase match
        # For single-word short skills (<=2 chars): word boundary
        if len(skill) <= 2:
            pattern = r'\b' + re.escape(skill) + r'\b'
        else:
            pattern = re.escape(skill)
        
        if re.search(pattern, normalized_text):
            found_skills[skill] = category
    
    # Group by category
    by_category = {}
    for skill, category in found_skills.items():
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(skill.title())
    
    # Define high-value/important skills
    high_value_skills = [
        "python", "javascript", "react", "nodejs", "django", "sql", "aws",
        "docker", "kubernetes", "machine learning", "tensorflow", "pytorch",
        "java", "spring boot", "postgresql", "mongodb"
    ]
    
    top_skills = [s.title() for s in found_skills.keys() if s in high_value_skills]
    
    return {
        'all_skills': [s.title() for s in found_skills.keys()],
        'by_category': by_category,
        'skill_count': len(found_skills),
        'top_skills': top_skills[:10]  # limit to top 10
    }


def detect_resume_sections(resume_text: str) -> Dict:
    """
    Detect which standard sections exist in the resume.
    Used for ATS scoring.
    
    Args:
        resume_text: Resume text
    Returns:
        dict: section_name -> bool (present or not)
    """
    text_lower = resume_text.lower()
    
    sections = {
        'contact_info': bool(re.search(r'(email|phone|linkedin|github|address)', text_lower)),
        'summary_objective': bool(re.search(r'(summary|objective|profile|about me)', text_lower)),
        'experience_work': bool(re.search(r'(experience|work history|employment|career)', text_lower)),
        'education': bool(re.search(r'(education|degree|university|college|school|academic)', text_lower)),
        'skills_section': bool(re.search(r'(skills|technologies|competencies|expertise)', text_lower)),
        'projects': bool(re.search(r'(project|portfolio|built|developed|created)', text_lower)),
        'certifications': bool(re.search(r'(certification|certificate|certified|credential)', text_lower)),
        'achievements': bool(re.search(r'(achievement|award|honor|recognition|accomplishment)', text_lower)),
    }
    
    return sections


def get_missing_skills_suggestions(found_skills: List[str], job_role: str = "Software Developer") -> List[str]:
    """
    Suggest skills that are commonly expected but not found.
    
    Args:
        found_skills: List of detected skills (lowercase)
        job_role: Target job role for customized suggestions
    Returns:
        list: Suggested skills to add
    """
    found_lower = [s.lower() for s in found_skills]
    
    # Common must-have skills for software roles
    essential_skills = {
        "Software Developer": ["git", "sql", "python", "javascript", "docker", "agile"],
        "Data Scientist": ["python", "sql", "pandas", "numpy", "machine learning", "tableau"],
        "DevOps Engineer": ["docker", "kubernetes", "aws", "terraform", "jenkins", "linux"],
        "Full Stack Developer": ["react", "nodejs", "sql", "git", "docker", "rest"],
    }
    
    target_skills = essential_skills.get(job_role, essential_skills["Software Developer"])
    
    missing = [skill.title() for skill in target_skills if skill not in found_lower]
    return missing
