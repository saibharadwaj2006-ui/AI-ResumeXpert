from django.db import models
from django.utils import timezone


class Resume(models.Model):
    """
    Model to store uploaded resume information.
    
    Step 1: When user uploads a PDF, we create this record
    to track the file and all analysis results.
    """
    # File storage
    file = models.FileField(upload_to='resumes/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(default=timezone.now)
    
    # Extracted content
    extracted_text = models.TextField(blank=True)
    
    # Analysis results (stored as JSON strings)
    skills_found = models.TextField(blank=True, default='[]')   # JSON list
    ats_score = models.FloatField(default=0.0)
    ats_breakdown = models.TextField(blank=True, default='{}')  # JSON dict
    
    # AI-generated content
    ai_suggestions = models.TextField(blank=True)
    interview_questions = models.TextField(blank=True, default='{}')
    
    # Processing status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Resume'
        verbose_name_plural = 'Resumes'

    def __str__(self):
        return f"{self.filename} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

    def get_skills_list(self):
        """Parse JSON skills field into Python list."""
        import json
        try:
            return json.loads(self.skills_found)
        except (json.JSONDecodeError, TypeError):
            return []

    def get_ats_breakdown(self):
        """Parse JSON ATS breakdown into Python dict."""
        import json
        try:
            return json.loads(self.ats_breakdown)
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_interview_questions(self):
        """Parse JSON interview questions into Python dict."""
        import json
        try:
            return json.loads(self.interview_questions)
        except (json.JSONDecodeError, TypeError):
            return {}
