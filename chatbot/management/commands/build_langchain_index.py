"""python manage.py build_langchain_index

Builds the FAISS index from the documents folder (+ optional web URLs file).
Default is a full rebuild; use --append to add to the existing index.
"""

import json
import os
import shutil
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from chatbot.ingestion.document_processor import LangChainIngestion, VECTOR_STORE_PATH


class Command(BaseCommand):
    help = "Build (or append to) the FAISS index from your documents folder and web URLs"

    def add_arguments(self, parser):
        parser.add_argument('--docs-dir', default=os.getenv('DOCUMENTS_PATH', 'data/documents'),
                            help='Folder with .pdf/.txt/.md files (default: data/documents)')
        parser.add_argument('--urls-file', default=os.getenv('WEB_URLS_FILE', 'data/urls.txt'),
                            help='Text file with one URL per line (default: data/urls.txt)')
        parser.add_argument('--append', action='store_true', help='Append to existing index instead of rebuilding')
        parser.add_argument('--skip-docs', action='store_true', help='Skip documents folder')
        parser.add_argument('--skip-web', action='store_true', help='Skip web URLs')

    def handle(self, *args, **options):
        final_path = Path(VECTOR_STORE_PATH)
        # Full rebuild goes to a temp folder first, so a failed build never destroys the working index
        build_path = final_path if options['append'] else final_path.with_name(final_path.name + '_tmp')
        if build_path != final_path and build_path.exists():
            shutil.rmtree(build_path)

        self.stdout.write(self.style.MIGRATE_HEADING('Building index...'))
        ingestion = LangChainIngestion(vector_store_path=str(build_path))
        summary = {'documents': {'skipped': True}, 'web': {'skipped': True}}

        if not options['skip_docs']:
            self.stdout.write(f"Processing documents from {options['docs_dir']} ...")
            summary['documents'] = ingestion.ingest_documents(options['docs_dir'])

        urls_file = Path(options['urls_file'])
        if not options['skip_web'] and urls_file.exists():
            urls = [ln.strip() for ln in urls_file.read_text(encoding='utf-8').splitlines()
                    if ln.strip() and not ln.strip().startswith('#')]
            if urls:
                self.stdout.write(f"Processing {len(urls)} URLs from {urls_file} ...")
                summary['web'] = ingestion.ingest_web_urls(urls)

        doc_chunks = summary['documents'].get('total_chunks', 0)
        web_chunks = summary['web'].get('total_chunks', 0)
        errors = [f"{k}: {v['error']}" for k, v in summary.items() if v.get('error')]

        if not (build_path / 'index.faiss').exists():
            if build_path != final_path:
                shutil.rmtree(build_path, ignore_errors=True)
            raise CommandError("No index was built. " + " | ".join(errors or ['Nothing to ingest.'])
                               + f"\nAdd .pdf/.txt/.md files to {options['docs_dir']} and/or URLs to {urls_file}.")

        if build_path != final_path:
            shutil.rmtree(final_path, ignore_errors=True)
            build_path.rename(final_path)

        with open(final_path / 'ingestion_stats.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        for err in errors:
            self.stdout.write(self.style.WARNING(f'Warning - {err}'))
        self.stdout.write(self.style.SUCCESS(
            f'Index ready at {final_path}  |  document chunks: {doc_chunks}  |  web chunks: {web_chunks}'))
        self.stdout.write('Restart the server so it loads the new index.')
