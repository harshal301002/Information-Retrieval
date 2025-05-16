import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from search_engine.config import Config

# Ensure necessary NLTK resources are downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words(Config.STOPWORDS_LANGUAGE))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = Config.ENABLE_STEMMING

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess a given text: tokenize, normalize, remove stopwords,
        and apply lemmatization or stemming.
        Returns a list of processed tokens.
        """
        # Remove punctuation, newlines, and lower the text
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = nltk.word_tokenize(text)
        tokens = [
            self._normalize_token(token)
            for token in tokens
            if token not in self.stop_words and token.isalpha()
        ]

        return tokens

    def _normalize_token(self, token: str) -> str:
        if self.use_stemming:
            return self.stemmer.stem(token)
        else:
            return self.lemmatizer.lemmatize(token)
