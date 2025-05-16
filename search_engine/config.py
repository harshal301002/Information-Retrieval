from pathlib import Path

class Config:
    CORPUS_DIR = Path("data")  # Should point to the folder with category subfolders
    SUPPORTED_EXTENSIONS = {".txt", ".md"}
    STOPWORDS_LANGUAGE = "english"
    ENABLE_STEMMING = False
    MAX_FEATURES = None
    TOP_K_RESULTS = 5
    SNIPPET_WINDOW = 30
