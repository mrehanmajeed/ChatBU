from django.db import models

# QueryLog model for tracking and analyzing user queries
class QueryLog(models.Model):
    # Core query data
    question = models.TextField()
    answer = models.TextField(blank=True)
    latency_ms = models.IntegerField(null=True, blank=True)
    retrieved_doc_ids = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Analytics fields (for Power BI or any BI tool)
    session_id = models.CharField(max_length=100, null=True, blank=True, db_index=True, 
                                   help_text="Unique session identifier for tracking user conversations")
    
    # Classification and quality metrics
    query_category = models.CharField(max_length=50, null=True, blank=True, db_index=True,
                                      help_text="Query category (see chatbot/utils/analytics.py)")
    
    is_answered = models.BooleanField(default=True, db_index=True,
                                      help_text="Whether the query was successfully answered")
    
    # Context and metadata
    confidence_score = models.FloatField(null=True, blank=True,
                                         help_text="Average similarity score of retrieved documents")
    num_docs_retrieved = models.IntegerField(null=True, blank=True,
                                             help_text="Number of documents retrieved for this query")
    
    # User tracking (optional - can be IP hash or actual user ID)
    user_identifier = models.CharField(max_length=100, null=True, blank=True, db_index=True,
                                       help_text="Anonymous user identifier (hashed IP or user ID)")

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['session_id', '-created_at']),
            models.Index(fields=['query_category', '-created_at']),
        ]

    def __str__(self):
        return self.question[:60]
