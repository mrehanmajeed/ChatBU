"""WSGI entry point (used by gunicorn and runserver)."""

import logging
import os
import threading

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Unibot.settings')

application = get_wsgi_application()


def _warm_up():
    # Load the models + index at startup so the first question isn't slow
    from chatbot.core.rag import get_pipeline
    try:
        get_pipeline()
    except Exception as e:  # e.g. index not built yet; requests will report it
        logging.getLogger(__name__).warning(f"RAG warm-up skipped: {e}")


threading.Thread(target=_warm_up, daemon=True).start()
