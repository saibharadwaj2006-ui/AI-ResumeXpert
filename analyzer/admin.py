from django.contrib import admin
from .models import Resume


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ['filename', 'ats_score', 'status', 'uploaded_at']
    list_filter = ['status']
    search_fields = ['filename']
    readonly_fields = ['uploaded_at', 'extracted_text']
