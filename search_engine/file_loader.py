import os
from pathlib import Path
from typing import Dict
from search_engine.config import Config


class FileLoader:
    def __init__(self, corpus_dir: Path = Config.CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS

    def load_documents(self) -> Dict[str, str]:
        """
        Recursively loads and reads all supported text/markdown documents from the corpus directory.
        Returns a dictionary mapping filename (relative path) to content.
        """
        documents = {}

        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory '{self.corpus_dir}' not found.")

        for root, _, files in os.walk(self.corpus_dir):
            for file in files:
                filepath = Path(root) / file
                if filepath.suffix.lower() in self.supported_extensions:
                    try:
                        content = filepath.read_text(encoding="utf-8")
                        rel_path = str(filepath.relative_to(self.corpus_dir))
                        documents[rel_path] = content
                    except UnicodeDecodeError:
                        print(f"[Warning] Skipped unreadable file: {filepath}")
                    except Exception as e:
                        print(f"[Error] Failed to read {filepath}: {e}")

        return documents
