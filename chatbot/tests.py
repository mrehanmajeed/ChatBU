from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from .core.rag import RAGResult, RetrievedChunk
from .models import QueryLog
from .utils.analytics import classify_query_category


class AnalyticsTests(SimpleTestCase):
    def test_category_whole_word_match(self):
        self.assertEqual(classify_query_category("What is the fee?"), "Pricing")
        self.assertEqual(classify_query_category("Where do I leave feedback?"), "General")
        self.assertEqual(classify_query_category("What are the rules for refunds?"), "Policy")


class AskApiTests(TestCase):
    """The RAG pipeline is mocked, so these run without models, an index or an API key."""

    def fake_result(self, question, **kwargs):
        return RAGResult(
            question=question,
            answer="You can borrow 5 books.",
            context="[Source: policy.pdf]\nMembers may borrow up to 5 books.",
            retrieved_chunks=[RetrievedChunk("Members may borrow up to 5 books.", 0.8, {"source": "policy.pdf"})],
            metadata={"num_docs_retrieved": 1, "avg_similarity": 0.8, "model": "test-model", "llm_failed": False},
        )

    def test_ask_returns_answer_and_logs_query(self):
        with patch("chatbot.views.ask_with_details", side_effect=self.fake_result):
            res = self.client.post("/chatbot/ask/", {"question": "How many books?"}, content_type="application/json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["answer"], "You can borrow 5 books.")
        self.assertEqual(res.json()["matches"][0]["source"], "policy.pdf")
        log = QueryLog.objects.get()
        self.assertTrue(log.is_answered)
        self.assertEqual(log.retrieved_doc_ids, ["policy.pdf"])

    def test_empty_question_rejected(self):
        res = self.client.post("/chatbot/ask/", {"question": ""}, content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_pipeline_error_hides_details_in_production(self):
        with patch("chatbot.views.ask_with_details", side_effect=RuntimeError("secret internals")):
            res = self.client.post("/chatbot/ask/", {"question": "hi"}, content_type="application/json")
        self.assertEqual(res.status_code, 500)
        self.assertNotIn("secret internals", res.content.decode())
