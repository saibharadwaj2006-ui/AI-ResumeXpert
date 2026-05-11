from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('results/<int:pk>/', views.results, name='results'),
    path('history/', views.history, name='history'),
    path('delete/<int:pk>/', views.delete_resume, name='delete_resume'),
    path('api/rag-query/<int:pk>/', views.api_rag_query, name='api_rag_query'),
]
