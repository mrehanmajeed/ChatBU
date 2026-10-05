"""
Query enrichment for analytics (Power BI, Metabase, plain SQL...).
Each question is tagged with a category, an anonymous user id and a session id.
"""

import hashlib
import re
from typing import Dict, Optional

# Edit these to match the topics of your own documents
CATEGORY_KEYWORDS = {
    'Pricing': ['fee', 'fees', 'price', 'pricing', 'cost', 'charges', 'payment', 'refund', 'discount'],
    'Eligibility': ['eligible', 'eligibility', 'requirement', 'requirements', 'criteria', 'qualify'],
    'Process': ['how to', 'apply', 'application', 'procedure', 'process', 'steps', 'register', 'registration'],
    'Policy': ['policy', 'policies', 'rule', 'rules', 'regulation', 'regulations', 'guideline', 'guidelines'],
    'Deadline': ['deadline', 'last date', 'due date', 'closing date', 'schedule', 'timeline'],
    'Contact': ['contact', 'email', 'phone', 'address', 'location', 'where is', 'office'],
}


def _has_keyword(text: str, keyword: str) -> bool:
    """Whole-word match, so 'fee' doesn't match 'feedback'."""
    return re.search(rf'\b{re.escape(keyword)}\b', text) is not None


def classify_query_category(question: str) -> Optional[str]:
    """Return the category with the most keyword hits, or 'General'."""
    question_lower = question.lower()
    scores = {
        category: sum(1 for kw in keywords if _has_keyword(question_lower, kw))
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'General'


def get_client_ip(request) -> str:
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def anonymize_user(request) -> str:
    """Hashed IP, so no raw IPs are stored."""
    return hashlib.sha256(get_client_ip(request).encode()).hexdigest()[:12]


def generate_session_id(request) -> str:
    if hasattr(request, 'session') and request.session.session_key:
        return request.session.session_key
    session_string = f"{get_client_ip(request)}_{request.META.get('HTTP_USER_AGENT', '')}"
    return hashlib.sha256(session_string.encode()).hexdigest()[:16]


def enrich_query_data(question: str, answer: str, request, metadata: Dict) -> Dict:
    return {
        'session_id': generate_session_id(request),
        'query_category': classify_query_category(question),
        'user_identifier': anonymize_user(request),
        'is_answered': not metadata.get('retrieval_failed') and not metadata.get('llm_failed'),
        'confidence_score': metadata.get('avg_similarity'),
        'num_docs_retrieved': metadata.get('num_docs_retrieved'),
    }
