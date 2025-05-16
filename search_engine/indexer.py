from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from search_engine.text_processor import TextProcessor


class Indexer:
    def __init__(self, max_features=None):
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._custom_tokenizer,
            preprocessor=None,
            lowercase=False,
            stop_words=None,
            max_features=max_features,
        )
        self.text_processor = TextProcessor()
        self.document_ids: List[str] = []
        self.tfidf_matrix = None

    def _custom_tokenizer(self, text: str) -> List[str]:
        return self.text_processor.preprocess(text)

    def build_index(self, documents: Dict[str, str]):
        """
        Builds the TF-IDF index for the given documents.
        Stores the matrix and mapping of document IDs.
        """
        self.document_ids = list(documents.keys())
        raw_corpus = list(documents.values())
        self.tfidf_matrix = self.vectorizer.fit_transform(raw_corpus)

    def get_tfidf_matrix(self):
        return self.tfidf_matrix

    def get_vectorizer(self):
        return self.vectorizer

    def get_document_ids(self):
        return self.document_ids
