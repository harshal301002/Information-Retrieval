from typing import List, Dict
import numpy as np

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return sum(doc in relevant_set for doc in retrieved_k) / k

def mean_reciprocal_rank(results: List[List[str]], ground_truths: List[List[str]]) -> float:
    reciprocal_ranks = []
    for retrieved, relevant in zip(results, ground_truths):
        rank = next((i + 1 for i, doc in enumerate(retrieved) if doc in relevant), 0)
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)
    return np.mean(reciprocal_ranks)

def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    dcg = 0
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0
