from django.conf import settings
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='chatbot/index.html',
                                  extra_context={'chatbot_name': settings.CHATBOT_NAME}), name='home'),
    path('admin/', admin.site.urls),
    path('chatbot/', include('chatbot.urls')),
]
