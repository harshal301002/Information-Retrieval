import sys
from search_engine.file_loader import FileLoader
from search_engine.retrieval_engine import RetrievalEngine
from search_engine.snippet_generator import SnippetGenerator
from search_engine.text_processor import TextProcessor
from search_engine.config import Config
from search_engine.query_expander import expand_query


def run_search_with_qe(engine, query: str, method: str, top_k: int = 5, use_qe: bool = False):
    if use_qe:
        expanded = expand_query(query)
        print(f"\n🔁 Expanded Query: {expanded}")
        query = expanded
    return engine.search(query, top_k=top_k, method=method)


def main():
    print("🔍 Personal Notes Search Engine")
    print("Indexing documents...")

    # Load and index documents
    loader = FileLoader()
    documents = loader.load_documents()
    if not documents:
        print("No documents found in data directory.")
        sys.exit(1)

    engine = RetrievalEngine(documents)
    snippet_gen = SnippetGenerator()

    print(f"Indexed {len(documents)} documents.")
    print("Available methods: tfidf, bm25, dense")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🔎 Enter query: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        method = input("⚙️ Method (tfidf/bm25/dense): ").strip().lower()
        if method not in {"tfidf", "bm25", "dense"}:
            print("❌ Invalid method. Defaulting to tfidf.")
            method = "tfidf"

        use_qe = input("➕ Use Query Expansion? (y/n): ").strip().lower() == "y"

        results = run_search_with_qe(engine, query, method, Config.TOP_K_RESULTS, use_qe=use_qe)
        query_terms = TextProcessor().preprocess(query)

        if not results:
            print("⚠️ No relevant documents found.\n")
            continue

        print(f"\n📄 Top {len(results)} results using {method.upper()} {'with QE' if use_qe else ''}:\n")
        for rank, (doc_id, score) in enumerate(results, start=1):
            snippet = snippet_gen.generate_snippet(doc_id, query_terms) or ""
            print(f"{rank}. {doc_id} (Score: {score})")
            print(f"   {snippet}\n")


if __name__ == "__main__":
    main()
