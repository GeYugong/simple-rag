import json
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = OUT_DIR / "tfidf_matrix.npz"
VOCAB_PATH = OUT_DIR / "tfidf_vocab.json"
IDF_PATH = OUT_DIR / "tfidf_idf.npy"
META_PATH = OUT_DIR / "meta.json"
PARAMS_PATH = OUT_DIR / "index_params.json"


def read_all_docs(doc_dir: Path) -> List[Dict]:
    docs = []
    for p in sorted(doc_dir.glob("**/*")):
        if p.is_file() and p.suffix.lower() in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append({"path": str(p.relative_to(ROOT)), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main():
    docs = read_all_docs(DOC_DIR)
    if not docs:
        raise RuntimeError(f"No docs found in {DOC_DIR}. Put some .txt/.md files there.")

    chunks = []
    meta = []
    for d in docs:
        cs = chunk_text(d["text"])
        for i, c in enumerate(cs):
            chunks.append(c)
            meta.append({"source": d["path"], "chunk_id": i, "text": c})

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000)
    matrix = vectorizer.fit_transform(chunks)

    from scipy.sparse import save_npz
    import numpy as np

    save_npz(str(MATRIX_PATH), matrix)
    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    np.save(str(IDF_PATH), vectorizer.idf_.astype("float32"))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    PARAMS_PATH.write_text(
        json.dumps({"kind": "tfidf", "rows": int(matrix.shape[0]), "cols": int(matrix.shape[1])}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Ingest done.")
    print(f"Matrix: {MATRIX_PATH}")
    print(f"Meta: {META_PATH}")
    print(f"Chunks: {len(meta)}")


if __name__ == "__main__":
    main()
