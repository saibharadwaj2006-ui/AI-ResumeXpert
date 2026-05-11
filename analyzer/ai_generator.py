"""
Step 6: AI Content Generator
================================
Generates AI-powered suggestions and interview questions using Gemini API.
Uses RAG context for better, more personalized responses.
"""

import os
import json
import re
from typing import Dict, List


def get_gemini_client():
    """Initialize Gemini API client."""
    try:
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except ImportError:
        return None


def call_gemini(prompt: str, model=None) -> str:
    """
    Call Gemini API with a prompt.
    Returns the response text or a fallback message.
    """
    if model is None:
        model = get_gemini_client()
    
    if model is None:
        return None
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return None


def generate_resume_suggestions(resume_text: str, skills: List[str], 
                                  ats_score: float, rag_pipeline=None) -> str:
    """
    Generate personalized resume improvement suggestions.
    
    RAG Flow:
    1. Query RAG for relevant resume sections
    2. Pass context + analysis to Gemini
    3. Get targeted, specific suggestions
    
    Args:
        resume_text: Full resume text
        skills: Detected skills list
        ats_score: Current ATS score
        rag_pipeline: ResumeRAGPipeline instance
    Returns:
        str: AI-generated suggestions
    """
    # RAG: Retrieve relevant context
    if rag_pipeline:
        context = rag_pipeline.get_context_for_query(
            "professional summary experience achievements work history"
        )
    else:
        context = resume_text[:1500]
    
    prompt = f"""You are an expert resume coach and ATS optimization specialist.

RESUME CONTEXT (extracted via RAG):
{context}

ANALYSIS RESULTS:
- ATS Score: {ats_score}/100
- Skills Detected: {', '.join(skills[:20]) if skills else 'None detected'}

Please provide specific, actionable resume improvement suggestions in these areas:
1. **Professional Summary** - How to make it more impactful
2. **Skills Section** - Missing skills to add, how to format
3. **Work Experience** - How to quantify achievements with metrics
4. **ATS Optimization** - Specific changes to improve ATS score
5. **Overall Recommendations** - Top 3 priority improvements

Be specific and reference the actual content from the resume where possible.
Format your response with clear sections and bullet points.
Keep it practical and immediately actionable."""

    result = call_gemini(prompt)
    if result:
        return result
    
    # Fallback suggestions if API unavailable
    return generate_fallback_suggestions(skills, ats_score)


def generate_interview_questions(resume_text: str, skills: List[str], rag_pipeline=None) -> Dict:
    """
    Generate categorized interview questions based on resume.
    
    Categories:
    - Technical (SQL, Python, specific tech)
    - Behavioral (HR questions)
    - Project-based (based on resume projects)
    - Role-specific
    
    Args:
        resume_text: Full resume text
        skills: Detected skills
        rag_pipeline: ResumeRAGPipeline instance
    Returns:
        dict: Questions organized by category
    """
    # RAG: Get project and experience context
    if rag_pipeline:
        project_context = rag_pipeline.get_context_for_query("projects built developed created implemented")
        experience_context = rag_pipeline.get_context_for_query("work experience responsibilities managed led")
    else:
        project_context = resume_text[:800]
        experience_context = resume_text[800:1600]
    
    # Identify top technical skills for question generation
    tech_skills_priority = ['python', 'sql', 'javascript', 'react', 'django', 'java', 
                             'machine learning', 'aws', 'docker', 'nodejs']
    candidate_tech_skills = [s for s in skills if s.lower() in tech_skills_priority][:3]
    
    prompt = f"""You are a senior technical interviewer preparing questions for a candidate.

CANDIDATE'S PROJECT EXPERIENCE (from RAG):
{project_context}

CANDIDATE'S WORK EXPERIENCE (from RAG):
{experience_context}

CANDIDATE'S TOP SKILLS: {', '.join(skills[:15]) if skills else 'Not specified'}

Generate a comprehensive set of interview questions in JSON format.
Return ONLY valid JSON, no other text.

{{
  "technical_questions": [
    {{"question": "...", "category": "Python/SQL/etc", "difficulty": "Easy/Medium/Hard"}},
    ... (5 questions total)
  ],
  "behavioral_questions": [
    {{"question": "...", "tip": "What to look for in answer"}},
    ... (4 questions total)
  ],
  "project_questions": [
    {{"question": "...", "purpose": "What this tests"}},
    ... (4 questions total)
  ],
  "hr_questions": [
    {{"question": "...", "category": "Culture/Salary/Growth"}},
    ... (3 questions total)
  ]
}}

Make questions specific to the candidate's actual experience and skills."""

    result = call_gemini(prompt)
    
    if result:
        # Try to parse JSON from response
        try:
            # Clean up response (remove markdown code blocks if present)
            clean = re.sub(r'```json\s*|\s*```', '', result).strip()
            questions = json.loads(clean)
            return questions
        except json.JSONDecodeError:
            pass
    
    # Fallback questions
    return generate_fallback_questions(skills)


def generate_fallback_suggestions(skills: List[str], ats_score: float) -> str:
    """Fallback suggestions when AI API is unavailable."""
    suggestions = f"""## Resume Improvement Suggestions

**Note:** AI API not configured. Showing template suggestions based on analysis.

### 1. Professional Summary
- Write a 3-4 sentence summary highlighting your years of experience, top skills, and career goal
- Use keywords from job descriptions you're targeting
- Quantify your impact (e.g., "5+ years building scalable web applications")

### 2. Skills Section
- **Found {len(skills)} skills**: {', '.join(skills[:10]) if skills else 'None detected - Add a dedicated skills section!'}
- Organize skills by category (Languages, Frameworks, Tools, Cloud)
- List skills in order of proficiency/relevance

### 3. Work Experience Improvements
- Start each bullet point with a strong action verb (Built, Developed, Led, Optimized)
- Add metrics to every achievement: "Reduced load time by 40%" not just "Improved performance"
- Follow the STAR format: Situation, Task, Action, Result

### 4. ATS Optimization (Current Score: {ats_score:.0f}/100)
- Include exact keywords from job descriptions
- Avoid tables, headers, footers, and text boxes
- Use standard section titles (Experience, Education, Skills)
- Save as a .docx or simple PDF

### 5. Top Priority Actions
1. 🔴 Add quantified achievements with percentages and numbers
2. 🟡 Include LinkedIn profile URL and GitHub portfolio link  
3. 🟢 Tailor your resume for each job application

---
*Configure GEMINI_API_KEY in .env file for personalized AI suggestions.*"""
    
    return suggestions


def generate_fallback_questions(skills: List[str]) -> Dict:
    """Fallback interview questions when AI API is unavailable."""
    
    has_python = any('python' in s.lower() for s in skills)
    has_sql = any('sql' in s.lower() for s in skills)
    has_js = any(s.lower() in ['javascript', 'react', 'nodejs'] for s in skills)
    
    technical = []
    if has_python:
        technical.extend([
            {"question": "Explain the difference between a list and a tuple in Python.", "category": "Python", "difficulty": "Easy"},
            {"question": "How does Python's GIL affect multi-threaded applications?", "category": "Python", "difficulty": "Hard"},
        ])
    if has_sql:
        technical.extend([
            {"question": "Explain the difference between INNER JOIN and LEFT JOIN.", "category": "SQL", "difficulty": "Easy"},
            {"question": "How would you optimize a slow SQL query?", "category": "SQL", "difficulty": "Medium"},
        ])
    if has_js:
        technical.append({"question": "Explain event bubbling vs event capturing in JavaScript.", "category": "JavaScript", "difficulty": "Medium"})
    
    technical.append({"question": "Describe a challenging technical problem you solved recently.", "category": "General", "difficulty": "Medium"})
    
    return {
        "technical_questions": technical or [
            {"question": "What programming languages are you most comfortable with?", "category": "General", "difficulty": "Easy"},
            {"question": "Explain RESTful API design principles.", "category": "Architecture", "difficulty": "Medium"},
        ],
        "behavioral_questions": [
            {"question": "Tell me about a time you worked under tight deadlines.", "tip": "Look for prioritization and delivery"},
            {"question": "Describe a conflict with a teammate and how you resolved it.", "tip": "Look for communication and empathy"},
            {"question": "What's your greatest professional achievement?", "tip": "Look for metrics and impact"},
            {"question": "Why are you looking for a new opportunity?", "tip": "Look for growth motivation"},
        ],
        "project_questions": [
            {"question": "Walk me through the most complex project on your resume.", "purpose": "Tests depth of technical knowledge"},
            {"question": "What would you do differently if you rebuilt this project today?", "purpose": "Tests learning and reflection"},
            {"question": "How did you handle scalability challenges in your projects?", "purpose": "Tests system design thinking"},
            {"question": "What was the biggest technical challenge you faced?", "purpose": "Tests problem-solving ability"},
        ],
        "hr_questions": [
            {"question": "Where do you see yourself in 5 years?", "category": "Growth"},
            {"question": "What are your salary expectations?", "category": "Salary"},
            {"question": "Are you open to relocation or remote work?", "category": "Logistics"},
        ]
    }
