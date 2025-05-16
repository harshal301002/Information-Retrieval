from typing import List, Tuple, Dict
from search_engine.query_expander import expand_query
from search_engine.text_processor import TextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np


class RetrievalEngine:
    def __init__(self, documents: Dict[str, str]):
        self.docs = documents
        self.doc_ids = list(documents.keys())
        self.text_processor = TextProcessor()

        # Preprocessing
        self.preprocessed_docs = [
            self.text_processor.preprocess(text) for text in documents.values()
        ]
        self.raw_docs = list(documents.values())

        # === Build TF-IDF model ===
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda text: text,
            lowercase=False,
            preprocessor=None,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_docs)

        # === Build BM25 ===
        self.bm25 = BM25Okapi(self.preprocessed_docs)

        # === Load sentence embedding model ===
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.doc_embeddings = self.embedder.encode(self.raw_docs, convert_to_tensor=True)

    def search(self, query: str, top_k=5, method="tfidf") -> List[Tuple[str, float]]:
        query_tokens = self.text_processor.preprocess(query)
        query_str = " ".join(query_tokens)

        if method == "tfidf":
            return self._search_tfidf(query_tokens, top_k)
        elif method == "bm25":
            return self._search_bm25(query_tokens, top_k)
        elif method == "dense":
            return self._search_dense(query_str, top_k)
        else:
            raise ValueError("Unknown retrieval method: choose tfidf, bm25, or dense")

    def _search_tfidf(self, tokens, top_k):
        query_vector = self.tfidf_vectorizer.transform([tokens])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[i], round(similarities[i], 4)) for i in top_indices if similarities[i] > 0]

    def _search_bm25(self, tokens, top_k):
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], round(scores[i], 4)) for i in top_indices if scores[i] > 0]

    def _search_dense(self, query, top_k):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        return [(self.doc_ids[i], round(float(scores[i]), 4)) for i in top_indices if float(scores[i]) > 0]

    def run_search_with_qe(engine, query: str, method: str, top_k: int = 5, use_qe: bool = False):
        expanded_query = expand_query(query) if use_qe else query
        return engine.search(expanded_query, top_k=top_k, method=method)