from django.contrib import admin
from .models import QueryLog

@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ("id", "question", "query_category", "is_answered", "confidence_score", "latency_ms", "created_at")
    list_filter = ("created_at", "query_category", "is_answered")
    search_fields = ("question", "answer")
    readonly_fields = ("created_at",)
    date_hierarchy = "created_at"
