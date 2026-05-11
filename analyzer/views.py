"""
Step 7: Django Views
======================
Handles all HTTP requests for the Resume Analyzer.
Each view function corresponds to a URL endpoint.
"""

import json
import os
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

from .models import Resume
from .pdf_extractor import extract_resume_text
from .skill_analyzer import detect_skills, detect_resume_sections, get_missing_skills_suggestions
from .ats_scorer import calculate_ats_score
from .rag_pipeline import ResumeRAGPipeline
from .ai_generator import generate_resume_suggestions, generate_interview_questions


def home(request):
    """
    Home page: Show upload form and recent resumes.
    GET: Show the upload form
    POST: Handle file upload → redirect to results
    """
    if request.method == 'POST':
        # ── Step 1: Validate uploaded file ────────────────────────────────
        if 'resume_file' not in request.FILES:
            messages.error(request, 'Please select a PDF file to upload.')
            return redirect('home')
        
        uploaded_file = request.FILES['resume_file']
        
        # Validate file type
        if not uploaded_file.name.lower().endswith('.pdf'):
            messages.error(request, 'Only PDF files are accepted.')
            return redirect('home')
        
        # Validate file size (10MB max)
        if uploaded_file.size > 10 * 1024 * 1024:
            messages.error(request, 'File too large. Maximum size is 10MB.')
            return redirect('home')
        
        # ── Step 2: Save to database ───────────────────────────────────────
        resume = Resume.objects.create(
            file=uploaded_file,
            filename=uploaded_file.name,
            status='processing'
        )
        
        try:
            # ── Step 3: Extract text from PDF ─────────────────────────────
            extraction_result = extract_resume_text(resume.file)
            
            if not extraction_result['success']:
                raise ValueError(f"Failed to extract text: {extraction_result.get('error', 'Unknown error')}")
            
            resume_text = extraction_result['text']
            resume.extracted_text = resume_text
            
            # ── Step 4: Analyze skills ─────────────────────────────────────
            skills_data = detect_skills(resume_text)
            sections_data = detect_resume_sections(resume_text)
            
            resume.skills_found = json.dumps(skills_data['all_skills'])
            
            # ── Step 5: Calculate ATS score ────────────────────────────────
            ats_result = calculate_ats_score(resume_text, skills_data, sections_data)
            resume.ats_score = ats_result['total_score']
            resume.ats_breakdown = json.dumps(ats_result)
            
            # ── Step 6: Build RAG Pipeline ─────────────────────────────────
            rag = ResumeRAGPipeline(resume_text)
            
            # ── Step 7: Generate AI suggestions ───────────────────────────
            suggestions = generate_resume_suggestions(
                resume_text,
                skills_data['all_skills'],
                ats_result['total_score'],
                rag_pipeline=rag
            )
            resume.ai_suggestions = suggestions
            
            # ── Step 8: Generate interview questions ───────────────────────
            interview_qs = generate_interview_questions(
                resume_text,
                skills_data['all_skills'],
                rag_pipeline=rag
            )
            resume.interview_questions = json.dumps(interview_qs)
            
            # ── Step 9: Mark as complete ───────────────────────────────────
            resume.status = 'completed'
            resume.save()
            
            return redirect('results', pk=resume.pk)
        
        except Exception as e:
            resume.status = 'failed'
            resume.error_message = str(e)
            resume.save()
            messages.error(request, f'Analysis failed: {str(e)}')
            return redirect('home')
    
    # GET: Show upload form with recent analyses
    recent_resumes = Resume.objects.filter(status='completed').order_by('-uploaded_at')[:5]
    return render(request, 'analyzer/home.html', {'recent_resumes': recent_resumes})


def results(request, pk):
    """
    Results page: Display full analysis for a resume.
    """
    resume = get_object_or_404(Resume, pk=pk)
    
    # Parse stored JSON data
    skills_list = resume.get_skills_list()
    ats_data = resume.get_ats_breakdown()
    interview_qs = resume.get_interview_questions()
    
    # Get missing skills suggestions
    missing_skills = get_missing_skills_suggestions(skills_list)
    
    # Organize skills by category for display
    from .skill_analyzer import detect_skills
    if resume.extracted_text:
        skills_detailed = detect_skills(resume.extracted_text)
    else:
        skills_detailed = {'by_category': {}, 'all_skills': skills_list}
    
    # Check if AI was used (Gemini configured)
    ai_configured = bool(os.getenv('GEMINI_API_KEY', ''))
    
    context = {
        'resume': resume,
        'skills_list': skills_list,
        'skills_by_category': skills_detailed.get('by_category', {}),
        'ats_data': ats_data,
        'interview_questions': interview_qs,
        'missing_skills': missing_skills,
        'ai_configured': ai_configured,
        'ats_score': resume.ats_score,
        'ats_grade': ats_data.get('grade', 'N/A'),
        'ats_label': ats_data.get('grade_label', ''),
    }
    
    return render(request, 'analyzer/results.html', context)


def history(request):
    """Show all previously analyzed resumes."""
    resumes = Resume.objects.all().order_by('-uploaded_at')
    return render(request, 'analyzer/history.html', {'resumes': resumes})


def delete_resume(request, pk):
    """Delete a resume analysis."""
    resume = get_object_or_404(Resume, pk=pk)
    if request.method == 'POST':
        # Delete the file too
        if resume.file and os.path.exists(resume.file.path):
            os.remove(resume.file.path)
        resume.delete()
        messages.success(request, 'Resume deleted successfully.')
    return redirect('history')


def api_rag_query(request, pk):
    """
    API endpoint: Query the RAG pipeline for a resume.
    Used for interactive Q&A about a resume.
    
    POST: { "query": "What are this person's Python skills?" }
    Response: { "context": "...", "chunks_used": 3 }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    resume = get_object_or_404(Resume, pk=pk)
    
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({'error': 'Query required'}, status=400)
        
        if not resume.extracted_text:
            return JsonResponse({'error': 'No text extracted from resume'}, status=400)
        
        # Build RAG pipeline and query
        rag = ResumeRAGPipeline(resume.extracted_text)
        context = rag.get_context_for_query(query)
        pipeline_info = rag.get_pipeline_info()
        
        return JsonResponse({
            'context': context,
            'pipeline_info': pipeline_info,
            'query': query
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
