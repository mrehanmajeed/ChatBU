from django.urls import path
from .views import AskView, HealthView

urlpatterns = [
    path('ask/', AskView.as_view(), name='chatbot-ask'),
    path('health/', HealthView.as_view(), name='chatbot-health'),
]
