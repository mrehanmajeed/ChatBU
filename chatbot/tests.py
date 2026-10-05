from django.test import SimpleTestCase

from .utils.analytics import classify_query_category


class AnalyticsTests(SimpleTestCase):
    def test_category_whole_word_match(self):
        self.assertEqual(classify_query_category("What is the fee?"), "Pricing")
        self.assertEqual(classify_query_category("Where do I leave feedback?"), "General")
        self.assertEqual(classify_query_category("What are the rules for refunds?"), "Policy")
