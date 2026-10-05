# ChatBU — Chat with your documents

[![tests](https://github.com/mrehanmajeed/ChatBU/actions/workflows/tests.yml/badge.svg)](https://github.com/mrehanmajeed/ChatBU/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ChatBU is a self-hosted **RAG (Retrieval-Augmented Generation) chatbot**. Drop your PDFs, text or Markdown files into a folder, build the index with one command, and get a chat website + REST API that answers questions **only from your documents**, with sources.

Use it for a university, company handbook, product docs, policies, FAQs — anything.

## Features

- 📄 **Your documents** — `.pdf`, `.txt`, `.md` (sub-folders included), plus optional web pages
- 🔍 **OCR** for scanned PDFs (when Tesseract is installed)
- 💬 **Chat web UI** at `/` and a **REST API** at `/chatbot/ask/`
- 📚 **Source citations** with every answer
- 🌐 Answers in the language of the question (English, Urdu, Roman Urdu, ...)
- 🛡️ **Rate limiting** per IP (protects your LLM API quota)
- 📊 **Query analytics** stored in the database (category, latency, confidence) — connect Power BI, Metabase, or plain SQL
- 🗄️ SQLite by default (zero setup), MySQL or PostgreSQL optional
- 🐳 **Docker** ready

## Tech stack

| Layer | Technology |
|---|---|
| Backend / API | Django 5, Django REST Framework |
| RAG orchestration | LangChain |
| Embeddings | HuggingFace `intfloat/e5-base-v2` (runs locally on CPU) |
| Vector store | FAISS |
| LLM | Google Gemini (`gemini-2.5-flash`, free tier available) |
| Document parsing | PyMuPDF, Tesseract OCR, BeautifulSoup (web pages) |
| Database | SQLite / MySQL / PostgreSQL |
| Serving | Gunicorn + WhiteNoise, Docker |

## How it works

```
             build_langchain_index (one time)
data/documents/*.pdf|txt|md ──► extract text (+OCR) ──► split into chunks ──► embeddings ──► FAISS index
data/urls.txt (optional)   ──┘

             every question
question ──► embedding ──► FAISS: top-K similar chunks ──► Gemini (prompt + chunks) ──► answer + sources
                                                                                     └► logged to DB
```

## Quick start (local)

**Requirements:** Python 3.11+ (tested on 3.13), Git, a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone
git clone https://github.com/mrehanmajeed/ChatBU.git
cd ChatBU

# 2. Virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env        # Windows (cmd): copy .env.example .env
# Open .env and set GOOGLE_API_KEY and SECRET_KEY

# 5. Add your documents
#    Put your .pdf / .txt / .md files in data/documents/
#    (optional) add web page URLs to data/urls.txt

# 6. Build the index (first run downloads the embedding model, ~440 MB)
python manage.py build_langchain_index

# 7. Create the database and run
python manage.py migrate
python manage.py runserver
```

Open **http://127.0.0.1:8000** and start chatting.

> Generate a `SECRET_KEY`: `python -c "import secrets; print(secrets.token_urlsafe(50))"`

### Updating documents

Added or changed files? Rebuild and restart the server:

```bash
python manage.py build_langchain_index            # full rebuild (safe: old index is kept if the build fails)
python manage.py build_langchain_index --append   # only add new files to the existing index
```

Other options: `--docs-dir PATH`, `--urls-file PATH`, `--skip-docs`, `--skip-web`.

### OCR for scanned PDFs (optional)

Install the Tesseract binary — Windows: [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) (add it to PATH), Ubuntu: `sudo apt install tesseract-ocr`, macOS: `brew install tesseract`. Without it, text PDFs still work; only scanned pages are skipped.

## Run with Docker

```bash
cp .env.example .env     # set GOOGLE_API_KEY and SECRET_KEY
# put your files in data/documents/

docker compose build
docker compose run --rm app python manage.py build_langchain_index
docker compose up -d
```

Open **http://localhost:8000**. Everything that must persist (documents, index, SQLite DB, model cache) lives in `./data`. After changing documents, run the build command again and `docker compose restart`.

## Configuration

All settings are in `.env` (see [`.env.example`](.env.example) for the full list).

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Gemini API key |
| `SECRET_KEY` | — | **Required.** Django secret key |
| `DEBUG` | `False` | `True` only for local development |
| `ALLOWED_HOSTS` | — | Your domain(s), comma-separated. Required when `DEBUG=False` |
| `CSRF_TRUSTED_ORIGINS` | — | e.g. `https://chat.example.com` (needed for admin login over HTTPS) |
| `SECURE_HTTPS` | `False` | `True` once served over HTTPS: redirects HTTP, secure cookies, HSTS |
| `CORS_ALLOWED_ORIGINS` | — | Only if a frontend on another domain calls the API |
| `CHATBOT_NAME` | `ChatBU` | Title shown in the chat UI |
| `RATE_LIMIT` | `20/min` | Max questions per IP |
| `DB_ENGINE` | `sqlite` | `sqlite`, `mysql` (`pip install mysqlclient`) or `postgresql` (`pip install "psycopg[binary]"`) + `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` |
| `EMBEDDING_MODEL` | `intfloat/e5-base-v2` | Embedding model. **Rebuild the index after changing it.** |
| `GENAI_MODEL_NAME` | `models/gemini-2.5-flash` | Gemini model |
| `TOP_K_DOCS` | `10` | Chunks sent to the LLM per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `450` / `75` | Chunk size in tokens |

## API

**Ask a question**

```bash
curl -X POST http://127.0.0.1:8000/chatbot/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

```json
{
  "question": "What is the refund policy?",
  "answer": "...",
  "matches": [{"source": "policy.pdf", "chunk_id": 0, "snippet": "...", "confidence": 0.82}],
  "latency_ms": 1840,
  "metadata": {"model": "models/gemini-2.5-flash", "num_docs_retrieved": 10, "avg_similarity": 0.71}
}
```

**Health check:** `GET /chatbot/health/` returns `200` when the index is loaded, `503` if it has not been built yet.

## Analytics

Every question is stored in the `chatbot_querylog` table: question, answer, latency, sources, category, confidence score, whether it was answered, and an anonymous (hashed) user/session id. Browse it in the Django admin (`python manage.py createsuperuser`, then `/admin/`) or connect any BI tool (Power BI, Metabase, Grafana) to the database.

Categories are keyword-based; edit `CATEGORY_KEYWORDS` in [`chatbot/utils/analytics.py`](chatbot/utils/analytics.py) to match your domain.

## Project structure

```
├── Unibot/                     # Django project (settings, urls, wsgi)
├── chatbot/
│   ├── core/rag.py             # Retrieval + Gemini answer generation
│   ├── ingestion/              # PDF/TXT/MD/web loading, chunking, FAISS index
│   ├── management/commands/    # build_langchain_index command
│   ├── templates/chatbot/      # Chat web UI
│   ├── utils/analytics.py      # Query categories, anonymous ids
│   ├── models.py               # QueryLog table
│   └── views.py                # /chatbot/ask/ and /chatbot/health/
├── data/
│   ├── documents/              # ← put your files here (git-ignored)
│   ├── urls.txt                # optional web pages to index
│   └── vector_store/           # built index (git-ignored)
├── .github/workflows/          # CI: runs tests on every push
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── .env.example
```

## Deploying to production

1. Set `DEBUG=False`, a strong `SECRET_KEY`, `ALLOWED_HOSTS=your-domain.com`, `CSRF_TRUSTED_ORIGINS=https://your-domain.com` and `SECURE_HTTPS=True`.
2. Run with Docker (`docker compose up -d`) or `gunicorn Unibot.wsgi:application --workers 1 --threads 4 --timeout 180` (Linux).
3. Put it behind a reverse proxy with HTTPS (Nginx, Caddy, or a platform like Render/Railway).
   Verify with `python manage.py check --deploy`.
4. Keep **one worker** per container: each worker loads the embedding model (~1 GB RAM) and keeps its own rate-limit counter. Scale with more containers + a shared cache if needed.

## Running tests

```bash
python manage.py test
```

Tests mock the LLM pipeline, so they need no API key, model or index. They also run automatically on every push via GitHub Actions.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Vector store not found` / health returns 503 | Run `python manage.py build_langchain_index` |
| `API key not valid` in logs | Check `GOOGLE_API_KEY` in `.env` |
| Answers ignore new documents | Rebuild the index and restart the server |
| `Bad Request (400)` with `DEBUG=False` | Add your domain to `ALLOWED_HOSTS` |
| First answer takes ~20s after start | Models load in the background at startup; wait until the log stops, or check `/chatbot/health/` |

## License

[MIT](LICENSE) — free to use, modify and distribute.
