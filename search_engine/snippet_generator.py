import re
from pathlib import Path
from typing import Optional
from search_engine.config import Config


class SnippetGenerator:
    def __init__(self, corpus_dir: Path = Config.CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self.window = Config.SNIPPET_WINDOW

    def generate_snippet(self, document_id: str, query_terms: list) -> Optional[str]:
        """
        Extracts a snippet containing the first query term match from the document.
        """
        file_path = self.corpus_dir / document_id
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[Error] Cannot read file for snippet: {file_path} — {e}")
            return None

        text_lower = text.lower()

        for term in query_terms:
            match = re.search(re.escape(term.lower()), text_lower)
            if match:
                start_idx = max(0, match.start() - self.window)
                end_idx = min(len(text), match.end() + self.window)
                snippet = text[start_idx:end_idx].strip().replace("\n", " ")
                return f"...{snippet}..."
        return None
