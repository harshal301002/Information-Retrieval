import os
from search_engine.file_loader import FileLoader
from search_engine.retrieval_engine import RetrievalEngine
from search_engine.eval_ground_truth import GROUND_TRUTH
from search_engine.evaluator import precision_at_k, mean_reciprocal_rank, ndcg_at_k
from search_engine.query_expander import expand_query

# Normalize file paths
def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/")

# Run evaluation for a method (with or without QE)
def evaluate(engine, method: str, use_qe: bool = False, top_k: int = 5):
    all_precisions = []
    all_ndcgs = []
    all_retrieved_ids = []

    for query, relevant_docs in GROUND_TRUTH.items():
        query_expanded = expand_query(query) if use_qe else query
        results = engine.search(query_expanded, method=method, top_k=top_k)

        retrieved_ids = [normalize_path(doc_id) for doc_id, _ in results]
        relevant_ids = [normalize_path(doc_id) for doc_id in relevant_docs]

        all_precisions.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        all_ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, top_k))
        all_retrieved_ids.append(retrieved_ids)

    mrr = mean_reciprocal_rank(all_retrieved_ids, list(GROUND_TRUTH.values()))
    return sum(all_precisions) / len(all_precisions), sum(all_ndcgs) / len(all_ndcgs), mrr


def print_results(method, label, precision, ndcg, mrr):
    print(f"\n📌 Method: {method.upper()} ({label})")
    print(f"Precision@5: {precision:.4f}")
    print(f"nDCG@5:     {ndcg:.4f}")
    print(f"MRR:        {mrr:.4f}")


def main():
    print("\n🔍 Evaluation on", len(GROUND_TRUTH), "queries")
    print("Comparing: TF-IDF, BM25, Dense with/without Query Expansion")
    print("-" * 60)

    methods = ["tfidf", "bm25", "dense"]

    documents = FileLoader().load_documents()
    engine = RetrievalEngine(documents)

    for method in methods:
        p, n, m = evaluate(engine, method, use_qe=False)
        print_results(method, "No QE", p, n, m)

    print("\n" + "-" * 60)

    for method in methods:
        p, n, m = evaluate(engine, method, use_qe=True)
        print_results(method, "With QE", p, n, m)


if __name__ == "__main__":
    main()
