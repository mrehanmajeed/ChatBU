"""
Document ingestion: PDF / TXT / MD files and web pages -> chunks -> FAISS vector store.
PDFs are read with PyMuPDF, with optional Tesseract OCR for scanned pages.
"""

import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Provide a default user agent if environment not set
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "ChatBU-Ingestion/1.0"

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'intfloat/e5-base-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '450'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '75'))
VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'data/vector_store')

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}


class LangChainIngestion:
    """Loads documents, splits them into token-sized chunks and stores embeddings in FAISS."""

    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading tokenizer for {EMBEDDING_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        def token_length_function(text: str) -> int:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=token_length_function,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(f"Ingestion initialized with chunk_size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens")

    def ingest_documents(self, documents_directory: str) -> Dict[str, Any]:
        """Ingest every supported file (recursively) in the given directory."""
        start_time = time.time()
        docs_dir = Path(documents_directory)

        if not docs_dir.exists():
            return {'error': f"Directory not found: {documents_directory}"}

        files = sorted(p for p in docs_dir.rglob('*')
                       if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            return {'error': f"No {', '.join(sorted(SUPPORTED_EXTENSIONS))} files found in {documents_directory}"}

        logger.info(f"Found {len(files)} files to process")

        documents = []
        processed_files = []
        failed_files = []

        for file in files:
            try:
                logger.info(f"Processing: {file.name}")
                if file.suffix.lower() == '.pdf':
                    file_docs = self._extract_pdf(str(file))
                else:
                    text = file.read_text(encoding='utf-8', errors='ignore').strip()
                    file_docs = [Document(page_content=text, metadata={})] if text else []
                for d in file_docs:
                    d.metadata.update({
                        'source': file.name,
                        'file_path': str(file.relative_to(docs_dir)),
                        'file_type': file.suffix.lower().lstrip('.'),
                        'ingestion_time': time.time()
                    })
                documents.extend(file_docs)
                processed_files.append(file.name)
                logger.info(f"OK {file.name}: {len(file_docs)} sections extracted")
            except Exception as e:
                logger.error(f"FAILED Failed to process {file.name}: {e}")
                failed_files.append(file.name)

        if not documents:
            return {'error': 'No documents successfully processed', 'failed_files': failed_files}

        chunks = self._split(documents)
        result = self._create_vector_store(chunks)

        return {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_sections': len(documents),
            'total_chunks': len(chunks),
            'processing_time_seconds': round(time.time() - start_time, 2),
            'vector_store_updated': result.get('success', False),
            'total_vectors': result.get('total_vectors', 0)
        }

    def ingest_web_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Ingest web pages using WebBaseLoader, falling back to requests + BeautifulSoup."""
        start_time = time.time()

        if not urls:
            return {'error': 'No URLs provided'}

        logger.info(f"Processing {len(urls)} URLs")

        documents = []
        processed_urls = []
        failed_urls = []

        headers = {"User-Agent": os.getenv("USER_AGENT")}

        for url in urls:
            try:
                logger.info(f"Loading: {url}")
                try:
                    web_documents = WebBaseLoader(web_paths=[url]).load()
                except Exception as primary_err:
                    logger.warning(f"Primary loader failed for {url}: {primary_err}; falling back to requests")
                    resp = requests.get(url, headers=headers, timeout=20)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()
                    web_documents = [Document(page_content=soup.get_text(separator=' '), metadata={})]

                cleaned_docs = []
                for doc in web_documents:
                    content = self._clean_web_content(doc.page_content)
                    if len(content) < 100:
                        continue
                    doc.page_content = content
                    doc.metadata.update({
                        'source': self._get_url_source_name(url),
                        'url': url,
                        'file_type': 'web',
                        'ingestion_time': time.time()
                    })
                    cleaned_docs.append(doc)
                if not cleaned_docs:
                    failed_urls.append(url)
                    continue
                documents.extend(cleaned_docs)
                processed_urls.append(url)
                logger.info(f"OK {url}: {len(cleaned_docs)} sections loaded")
            except Exception as e:
                logger.error(f"FAILED Failed to process {url}: {e}")
                failed_urls.append(url)

        if not documents:
            return {'error': 'No web documents successfully processed', 'failed_urls': failed_urls}

        chunks = self._split(documents)
        result = self._create_vector_store(chunks)
        return {
            'processed_urls': processed_urls,
            'failed_urls': failed_urls,
            'total_sections': len(documents),
            'total_chunks': len(chunks),
            'processing_time_seconds': round(time.time() - start_time, 2),
            'vector_store_updated': result.get('success', False),
            'total_vectors': result.get('total_vectors', 0)
        }

    def _split(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        # E5 embedding models expect a "passage: " prefix on indexed text
        for chunk in chunks:
            chunk.page_content = f"passage: {chunk.page_content}"
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} sections")
        return chunks

    def _create_vector_store(self, chunks: List[Document]) -> Dict[str, Any]:
        """Create the FAISS store, or add to it if one already exists."""
        try:
            if not chunks:
                return {'success': False, 'error': 'No chunks to process'}

            logger.info(f"Creating embeddings for {len(chunks)} chunks...")

            if (self.vector_store_path / "index.faiss").exists():
                vectorstore = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True  # safe: we only load an index we built ourselves
                )
                vectorstore.add_documents(chunks)
            else:
                vectorstore = FAISS.from_documents(chunks, self.embeddings)

            vectorstore.save_local(str(self.vector_store_path))
            total_vectors = vectorstore.index.ntotal
            logger.info(f"Vector store saved with {total_vectors} total vectors")
            return {'success': True, 'total_vectors': total_vectors, 'new_chunks': len(chunks)}

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            return {'success': False, 'error': str(e)}

    def _get_url_source_name(self, url: str) -> str:
        """Generate a readable source name from a URL."""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        name = path_parts[-1] if path_parts else parsed.netloc
        return f"web_{name}".replace('-', '_').replace('.', '_')[:50]

    def _clean_web_content(self, content: str) -> str:
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'Skip to.*?content', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Cookie.*?policy', '', content, flags=re.IGNORECASE)
        return content.strip()

    def _extract_pdf(self, path: str) -> List[Document]:
        """Page-level extraction with PyMuPDF, using OCR for scanned pages when Tesseract is installed."""
        docs: List[Document] = []

        def needs_ocr(t: str) -> bool:
            if not t or len(t) < 60:
                return True
            # Almost no lowercase letters or punctuation suggests layout/garbage text
            lowercase_ratio = sum(c.islower() for c in t) / max(len(t), 1)
            punctuation_ratio = sum(c in '.;:,' for c in t) / max(len(t), 1)
            return lowercase_ratio < 0.08 and punctuation_ratio < 0.005

        with fitz.open(path) as pdf:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                text = page.get_text().strip()
                ocr_used = False

                if pytesseract and needs_ocr(text) and page.get_images(full=True):
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        image = Image.open(io.BytesIO(pix.tobytes('png')))
                        ocr_text = pytesseract.image_to_string(image)
                        if len(ocr_text.strip()) > len(text) * 1.2:
                            text = (text + '\n' + ocr_text).strip()
                            ocr_used = True
                    except Exception as ocr_err:  # e.g. Tesseract binary not installed
                        logger.debug(f"OCR failed on page {page_index+1} of {path}: {ocr_err}")

                text = self._clean_pdf_page_text(text)

                # Skip decorative cover/title pages that carry no real content
                words = text.split()
                upper_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
                alnum_or_space = sum(ch.isalnum() or ch.isspace() for ch in text)
                non_alnum_ratio = (len(text) - alnum_or_space) / max(len(text), 1)
                low_signal = (len(text) < 80
                              or (upper_ratio > 0.85 and '.' not in text and ';' not in text)
                              or non_alnum_ratio > 0.35)
                if page_index < 2 and low_signal:
                    logger.debug(f"Skipping low-signal cover page {page_index+1} in {os.path.basename(path)}")
                    continue
                if not text:
                    continue

                docs.append(Document(page_content=text, metadata={'page': page_index + 1, 'ocr_used': ocr_used}))
        return docs

    def _clean_pdf_page_text(self, text: str) -> str:
        """Normalize PDF/OCR text: odd punctuation, glued letters/digits, hyphenation, whitespace."""
        if not text:
            return text
        replacements = {
            '–': '-', '—': '-', '―': '-', '­': '',
            '‘': "'", '’': "'", '“': '"', '”': '"',
            '™': '', '®': '', '©': ''
        }
        for a, b in replacements.items():
            text = text.replace(a, b)
        text = re.sub(r'[#*_~]{3,}', ' ', text)
        text = re.sub(r'[•●▪]+', ' ', text)  # bullet chars
        text = re.sub(r'([A-Za-z])-[\r\n]+([A-Za-z])', r'\1\2', text)  # "pro-\ngram" -> "program"
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'(?m)^[A-Za-z]$', ' ', text)  # isolated single letters (OCR noise)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s+\n', '\n', text)
        text = re.sub(r'\n\s+', '\n', text)
        return text.strip()
