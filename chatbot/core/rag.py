"""Core LangChain RAG pipeline.

This module exposes a minimal, production-ready retrieval augmented generation
pipeline built entirely on LangChain primitives. The pipeline takes care of
loading the existing FAISS vector store, running semantic retrieval, assembling
context, and delegating answer generation to a hosted LLM such as Gemini via
Google's Generative AI API. No local fine-tuning or custom model loading is
required.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


_DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
_VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))
_DEFAULT_TOP_K = int(os.getenv("TOP_K_DOCS", "10"))
_DEFAULT_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.28"))
_LLM_MODEL = os.getenv("GENAI_MODEL_NAME", "models/gemini-2.5-flash")
_LLM_TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", "0.3"))
_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("GENAI_MAX_OUTPUT_TOKENS", "2048"))
_MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_LENGTH", "100000"))

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context provided below. "
    "The context is made of excerpts from the documents this assistant was set up with.\n\n"
    "INSTRUCTIONS:\n"
    "1. Answer from the provided context. Synthesize information from multiple excerpts when needed.\n"
    "2. When the context contains lists, criteria or step-by-step information, present them fully.\n"
    "3. Use bullet points, numbered lists and short sections to keep answers readable.\n"
    "4. If the context only covers a related topic, say what IS available and how it relates.\n"
    "5. If the context has no relevant information, say you don't have information about that "
    "in the documents. Never make up facts that are not in the context.\n"
    "6. Answer in the same language the question was asked in."
)


@dataclass
class RetrievedChunk:
    """Representation of a retrieved document chunk."""

    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResult:
    """Structured output from the RAG pipeline."""

    question: str
    answer: str
    context: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict[str, Any]


class RAGPipeline:
    """High-level LangChain RAG pipeline.

    The pipeline is intentionally simple: it loads the FAISS vector store, runs a
    semantic search, formats the retrieved chunks, and sends them through a
    Gemini chat model using a lightweight prompt template.
    """

    def __init__(
        self,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        vector_store_path: Path = _VECTOR_STORE_PATH,
        top_k: int = _DEFAULT_TOP_K,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
        llm_model: str = _LLM_MODEL,
        llm_temperature: float = _LLM_TEMPERATURE,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.vector_store_path = vector_store_path
        self.top_k = max(1, top_k)
        self.similarity_threshold = similarity_threshold
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

        self.embeddings = self._load_embeddings(embedding_model)
        self.vectorstore = self._load_vector_store(vector_store_path)
        self.llm = self._load_llm(llm_model, llm_temperature)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT + "\n\nContext:\n{context}"),
                ("human", "Question: {question}"),
            ]
        )

    def _load_embeddings(self, model_name: str) -> HuggingFaceEmbeddings:
        logger.info("Loading embeddings model %s", model_name)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_vector_store(self, path: Path) -> FAISS:
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {index_path}. Run 'python manage.py build_langchain_index' first."
            )
        logger.info("Loading FAISS vector store from %s", path)
        try:
            return FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)
        except KeyError as exc:  # Common when upgrading from old langchain/pydantic pickles
            if "__fields_set__" in str(exc):
                raise RuntimeError(
                    "Vector store was created with an older LangChain version. "
                    "Please rebuild it via 'python manage.py build_langchain_index'."
                ) from exc
            raise

    def _load_llm(self, model: str, temperature: float) -> ChatGoogleGenerativeAI:
        api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set; downstream calls will fail until configured")
        else:
            logger.info("Using Gemini API key ending with %s", api_key[-4:])
        logger.info("Initialising ChatGoogleGenerativeAI model %s (temperature=%s)", model, temperature)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=_LLM_MAX_OUTPUT_TOKENS,
            google_api_key=api_key,
        )

    def answer_question(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        return_result: bool = False,
    ) -> RAGResult | str:
        """Generate an answer for the given question."""

        top_k = max(1, top_k or self.top_k)
        threshold = threshold if threshold is not None else self.similarity_threshold

        retrieved = self._retrieve_with_scores(question, top_k=top_k, threshold=threshold)
        if not retrieved:
            logger.warning(
                f"No documents met similarity threshold {threshold} for query: '{question[:100]}...'. "
                f"Consider lowering threshold or rephrasing query."
            )
            fallback = (
                "I couldn't find relevant information in the documents for your question. "
                "Please try rephrasing it."
            )
            result = RAGResult(
                question=question,
                answer=fallback,
                context="",
                retrieved_chunks=[],
                metadata={
                    "top_k": top_k,
                    "threshold": threshold,
                    "num_docs_retrieved": 0,
                    "avg_similarity": 0.0,
                    "model": self.llm_model,
                    "retrieval_failed": True,
                },
            )
            return result if return_result else result.answer

        context = self._build_context([doc for doc, _ in retrieved])
        
        # Log retrieval quality for debugging
        logger.info(
            f"Retrieved {len(retrieved)} chunks for query: '{question[:80]}...' | "
            f"Avg similarity: {sum(s for _, s in retrieved) / len(retrieved):.3f} | "
            f"Top score: {retrieved[0][1]:.3f}"
        )
        
        messages = self.prompt.format_messages(context=context, question=question)
        
        llm_failed = False
        try:
            response = self.llm.invoke(messages)
            answer = response.text.strip()  # .text joins content blocks (newer Gemini models return a list)
            
            # Detect if model refused despite having context
            if len(answer) < 50 and any(phrase in answer.lower() for phrase in 
                ["i don't", "i do not", "not found", "no information", "cannot find"]):
                logger.warning(
                    f"Model gave short/negative response despite retrieving {len(retrieved)} chunks. "
                    f"Response: '{answer}'. Context length: {len(context)} chars."
                )
        except Exception as e:
            logger.error("LLM invocation failed: %s | Query: '%s'", e, question[:100], exc_info=True)
            llm_failed = True
            answer = (
                "I apologize, but there was a technical error generating a response. "
                "The system retrieved relevant information but couldn't process it. Please try again."
            )

        retrieved_chunks = [
            RetrievedChunk(
                content=doc.page_content[9:].strip() if doc.page_content.startswith("passage: ") else doc.page_content.strip(),
                score=score,
                metadata=doc.metadata
            )
            for doc, score in retrieved
        ]

        metadata = {
            "top_k": top_k,
            "threshold": threshold,
            "num_docs_retrieved": len(retrieved_chunks),
            "avg_similarity": sum(score for _, score in retrieved) / len(retrieved),
            "model": self.llm_model,
            "llm_failed": llm_failed,
        }

        result = RAGResult(
            question=question,
            answer=answer,
            context=context,
            retrieved_chunks=retrieved_chunks,
            metadata=metadata,
        )

        return result if return_result else result.answer

    def _retrieve_with_scores(
        self,
        query: str,
        *,
        top_k: int,
        threshold: float,
    ) -> List[Tuple[Document, float]]:
        # Add E5 query prefix for optimal retrieval
        prefixed_query = f"query: {query}"
        
        # Fetch 2x candidates for better filtering
        search_k = max(top_k * 2, 20)
        raw_results = self.vectorstore.similarity_search_with_score(prefixed_query, k=search_k)
        processed: List[Tuple[Document, float]] = []

        for doc, distance in raw_results:
            similarity = 1.0 / (1.0 + distance)
            if similarity >= threshold:
                processed.append((doc, similarity))

        # Fallback: if threshold too strict, take top results anyway
        if not processed and raw_results:
            logger.info(
                f"No results above threshold {threshold}. Taking top {top_k} results as fallback. "
                f"Best score: {1.0 / (1.0 + raw_results[0][1]):.3f}"
            )
            processed = [
                (doc, 1.0 / (1.0 + distance))
                for doc, distance in sorted(raw_results, key=lambda item: item[1])[:top_k]
            ]

        processed.sort(key=lambda item: item[1], reverse=True)
        return processed[:top_k]

    def _build_context(self, documents: Sequence[Document]) -> str:
        chunks: List[str] = []
        remaining = _MAX_CONTEXT_CHARS

        for doc in documents:
            source = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"
            # Strip E5 passage prefix before building context
            content = doc.page_content
            if content.startswith("passage: "):
                content = content[9:]  # Remove exactly "passage: " (9 characters)
            content = content.strip()
            formatted = f"[Source: {source}]\n{content}"
            if len(formatted) > remaining:
                formatted = formatted[: max(remaining, 0)]
            if not formatted:
                continue
            chunks.append(formatted)
            remaining -= len(formatted)
            if remaining <= 0:
                break

        return "\n\n".join(chunks)

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "retriever": {
                "vector_store_loaded": self.vectorstore is not None,
                "embeddings_loaded": self.embeddings is not None,
                "embedding_model": self.embedding_model_name,
                "vector_store_path": str(self.vector_store_path),
                "vector_store_size": getattr(self.vectorstore.index, "ntotal", 0),
                "top_k": self.top_k,
                "threshold": self.similarity_threshold,
            },
            "llm": {
                "model": self.llm_model,
                "temperature": self.llm_temperature,
            },
            "pipeline_ready": self.vectorstore is not None,
        }


_pipeline: Optional[RAGPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    """Load the pipeline once per process (first request is slow while models load)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipeline()
    return _pipeline


def ask(question: str, **kwargs: Any) -> str:
    return get_pipeline().answer_question(question, **kwargs)


def ask_with_details(question: str, **kwargs: Any) -> RAGResult:
    result = get_pipeline().answer_question(question, return_result=True, **kwargs)
    if isinstance(result, RAGResult):
        return result
    raise RuntimeError("Unexpected pipeline response type")


def system_status() -> Dict[str, Any]:
    return get_pipeline().get_system_status()
