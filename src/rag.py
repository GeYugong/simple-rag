import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "data" / "index" / "tfidf_matrix.npz"
VOCAB_PATH = ROOT / "data" / "index" / "tfidf_vocab.json"
IDF_PATH = ROOT / "data" / "index" / "tfidf_idf.npy"
META_PATH = ROOT / "data" / "index" / "meta.json"
PARAMS_PATH = ROOT / "data" / "index" / "index_params.json"
VECTORIZER_PATH = ROOT / "data" / "index" / "tfidf_vectorizer.pkl"


def load_index_and_meta():
    if (
        not MATRIX_PATH.exists()
        or not VOCAB_PATH.exists()
        or not IDF_PATH.exists()
        or not META_PATH.exists()
        or not PARAMS_PATH.exists()
        or not VECTORIZER_PATH.exists()
    ):
        raise RuntimeError("Index not found or incomplete. Run: python src/ingest.py")

    matrix = load_npz(str(MATRIX_PATH)).tocsr()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return matrix, vectorizer, meta


def retrieve(query: str, k: int = 4, min_score: float = 1e-9) -> List[Dict]:
    matrix, vectorizer, meta = load_index_and_meta()

    q = vectorizer.transform([query])
    scores = (matrix @ q.T).toarray().ravel()

    positive_idx = np.where(scores > min_score)[0]
    if positive_idx.size == 0:
        return []

    k = max(1, min(k, positive_idx.size))
    candidate_scores = scores[positive_idx]
    top_local = np.argpartition(-candidate_scores, k - 1)[:k]
    top_idx = positive_idx[top_local]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for idx in top_idx:
        m = meta[int(idx)]
        results.append({"score": float(scores[int(idx)]), **m})
    return results


def naive_generate_answer(query: str, contexts: List[Dict]) -> str:
    ctx_text = "\n\n".join(
        [
            f"[{i+1}] {c['source']}#chunk{c['chunk_id']} (score={c['score']:.3f})\n{c['text']}"
            for i, c in enumerate(contexts)
        ]
    )

    answer = (
        "【这是一个不调用 LLM 的最小 RAG 演示】\n"
        f"问题：{query}\n\n"
        "检索到的相关片段如下（先确认检索是否靠谱）：\n"
        f"{ctx_text}\n\n"
        "下一步我们会把这些片段作为 context 交给 LLM，生成更自然的答案。"
    )
    return answer


def main():
    query = input("请输入问题：").strip()
    if not query:
        print("空问题，退出。")
        return

    contexts = retrieve(query, k=4)
    if not contexts:
        print("没有检索到内容。")
        return

    print("\n" + "=" * 80)
    print(naive_generate_answer(query, contexts))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
