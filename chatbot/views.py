"""REST endpoints for the RAG chatbot."""

import logging
from time import perf_counter

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from chatbot.core.rag import ask_with_details, system_status
from .models import QueryLog
from .serializers import AskRequestSerializer, AskResponseSerializer
from .utils.analytics import enrich_query_data

logger = logging.getLogger(__name__)


class AskView(APIView):
    """POST {"question": "..."} -> answer generated from the indexed documents."""

    def post(self, request, *args, **kwargs):
        serializer = AskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data['question']

        start = perf_counter()

        try:
            result = ask_with_details(question)
        except Exception as e:
            logger.error(f"Error in AskView: {e}", exc_info=True)
            payload = {
                'question': question,
                'answer': 'Sorry, there was an error processing your question. Please try again.',
                'contexts': [],
                'matches': [],
                'latency_ms': int((perf_counter() - start) * 1000),
            }
            if settings.DEBUG:
                payload['error'] = str(e)
            return Response(payload, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        latency = int((perf_counter() - start) * 1000)

        matches = [
            {
                'source': chunk.metadata.get('source', f'Source {i+1}'),
                'chunk_id': i,
                'snippet': chunk.content[:300],
                'confidence': chunk.score,
            }
            for i, chunk in enumerate(result.retrieved_chunks)
        ]

        # Logging must never break the answer
        try:
            enriched = enrich_query_data(question, result.answer, request, result.metadata)
            QueryLog.objects.create(
                question=question,
                answer=result.answer,
                latency_ms=latency,
                retrieved_doc_ids=[m['source'] for m in matches],
                **enriched,
            )
        except Exception as log_err:
            logger.warning(f"Failed to log query: {log_err}")

        payload = {
            'question': question,
            'answer': result.answer,
            'contexts': [result.context],
            'matches': matches,
            'latency_ms': latency,
            'metadata': {
                'model': result.metadata.get('model'),
                'num_docs_retrieved': result.metadata['num_docs_retrieved'],
                'avg_similarity': result.metadata['avg_similarity'],
            }
        }
        return Response(AskResponseSerializer(payload).data, status=status.HTTP_200_OK)


class HealthView(APIView):
    """GET -> whether the index and models are loaded."""

    def get(self, request, *args, **kwargs):
        try:
            info = system_status()
            return Response({
                'status': 'ok' if info.get('pipeline_ready') else 'initializing',
                'pipeline_ready': info.get('pipeline_ready', False),
                'vector_store_size': info['retriever'].get('vector_store_size', 0),
                'model': info['llm'].get('model'),
            })
        except Exception as e:
            logger.error(f"Health check error: {e}", exc_info=True)
            # Usually "vector store not found" -> tell the operator to build the index
            return Response({'status': 'error', 'detail': str(e), 'pipeline_ready': False},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
