import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def expand_query(query: str, max_terms: int = 20) -> str:
    terms = query.lower().split()
    expanded = set(terms)
    for term in terms:
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                if len(expanded) >= max_terms:
                    break
                expanded.add(lemma.name().replace("_", " "))
    return " ".join(expanded)
