import os
import re
import json
import pickle
from typing import List, Dict, Tuple

import numpy as np
import faiss
import pdfplumber
import gradio as gr
from sentence_transformers import SentenceTransformer


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CHUNKS_PATH = os.path.join(DATA_DIR, "syllabus_chunks.jsonl")
FAISS_PATH = os.path.join(DATA_DIR, "syllabus_index.faiss")
META_PATH = os.path.join(DATA_DIR, "syllabus_metadata.pkl")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_lines_with_fontsize(pdf_path: str) -> List[Dict]:
    all_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size"])
            line_map = {}
            for w in words:
                key = round(w["top"], 1)
                line_map.setdefault(key, []).append(w)

            for _, line_words in sorted(line_map.items()):
                text = " ".join([w["text"] for w in line_words]).strip()
                if not text:
                    continue
                avg_size = sum([w["size"] for w in line_words]) / len(line_words)
                all_lines.append({"text": text, "size": avg_size})
    return all_lines


def build_chunks(all_lines: List[Dict]) -> List[Dict]:
    documents = []
    doc_id = 1

    course_code_indices = [
        i for i, ln in enumerate(all_lines)
        if re.search(r"Course Code\s+\d{2}[A-Z]{2,3}\d{2,3}", ln["text"])
    ]

    if not course_code_indices:
        # fallback: treat entire doc as one block if pattern not found
        block_text = "\n".join([ln["text"] for ln in all_lines])
        documents.append({
            "id": doc_id,
            "course_code": "UNKNOWN",
            "course_title": "Syllabus",
            "section": "Full Document",
            "content": block_text.strip()
        })
        return documents

    for idx, start_i in enumerate(course_code_indices):
        end_i = course_code_indices[idx + 1] if idx + 1 < len(course_code_indices) else len(all_lines)
        block_lines = all_lines[start_i:end_i]

        code_match = re.search(r"Course Code\s+(\d{2}[A-Z]{2,3}\d{2,3})", block_lines[0]["text"])
        course_code = code_match.group(1) if code_match else "UNKNOWN"

        # Largest font above course code within last ~8 lines
        title = "Unknown Course"
        search_start = max(0, start_i - 8)
        candidate_lines = all_lines[search_start:start_i]
        if candidate_lines:
            best = max(candidate_lines, key=lambda x: x["size"])
            title = best["text"].strip()

        block_text = "\n".join([ln["text"] for ln in block_lines])

        def add_section(section: str, content: str):
            nonlocal doc_id
            content = content.strip()
            if not content:
                return
            documents.append({
                "id": doc_id,
                "course_code": course_code,
                "course_title": title,
                "section": section,
                "content": content
            })
            doc_id += 1

        # Objectives
        obj_match = re.search(
            r"Course Objectives?:\s*(.*?)(?=Module\s*[–-]|Course Outcomes|Textbooks|References|$)",
            block_text,
            re.DOTALL
        )
        if obj_match:
            add_section("Course Objectives", obj_match.group(1))

        # Modules
        module_matches = re.finditer(
            r"(Module\s*[–-]\s*\d+)\s*(.*?)(?=Module\s*[–-]\s*\d+|Course Outcomes|Textbooks|References|$)",
            block_text,
            re.DOTALL
        )
        for m in module_matches:
            add_section(m.group(1).strip(), m.group(2))

        # Outcomes
        out_match = re.search(
            r"Course Outcomes?:\s*(.*?)(?=Textbooks|References|$)",
            block_text,
            re.DOTALL
        )
        if out_match:
            add_section("Course Outcomes", out_match.group(1))

        # Textbooks
        tb_match = re.search(
            r"Textbooks?:\s*(.*?)(?=References|$)",
            block_text,
            re.DOTALL
        )
        if tb_match:
            add_section("Textbooks", tb_match.group(1))

        # References
        ref_match = re.search(r"References?:\s*(.*?)(?=$)", block_text, re.DOTALL)
        if ref_match:
            add_section("References", ref_match.group(1))

    return documents


def save_jsonl(docs: List[Dict], jsonl_path: str) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def build_index(docs: List[Dict]) -> Tuple[faiss.IndexFlatL2, SentenceTransformer, np.ndarray]:
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [d["content"] for d in docs]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embedding_model, embeddings


def persist_index(index: faiss.Index, docs: List[Dict]) -> None:
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)


def load_index_and_docs():
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        docs = pickle.load(f)
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, docs, embedding_model


def retrieve(query: str, index, docs, embedding_model, k: int = 3) -> List[Dict]:
    q = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q, k)
    results = [docs[i] for i in indices[0] if 0 <= i < len(docs)]
    return results


def format_answer(retrieved: List[Dict]) -> str:
    if not retrieved:
        return "No relevant chunks found."
    out = ["### 📚 Answer based on your syllabus (retrieved context)\n"]
    for doc in retrieved:
        out.append(f"**[{doc.get('course_title','')} — {doc.get('section','')}]**")
        out.append(doc.get("content", "").strip())
        out.append("\n---\n")
    return "\n\n".join(out)


def build_or_load(pdf_path: str):
    # If index exists, load it; otherwise build from the provided pdf
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        return load_index_and_docs()

    all_lines = extract_lines_with_fontsize(pdf_path)
    docs = build_chunks(all_lines)
    save_jsonl(docs, CHUNKS_PATH)

    index, _, _ = build_index(docs)
    persist_index(index, docs)

    return load_index_and_docs()


def ui_answer(pdf_file, question: str):
    if pdf_file is None:
        return "Please upload a syllabus PDF first."
    if not question or not question.strip():
        return "Please enter a question."

    index, docs, embedding_model = build_or_load(pdf_file)
    retrieved = retrieve(question, index, docs, embedding_model, k=3)
    return format_answer(retrieved)


def main():
    demo = gr.Interface(
        fn=ui_answer,
        inputs=[
            gr.File(label="Upload syllabus PDF", file_types=[".pdf"]),
            gr.Textbox(lines=2, label="Question", placeholder="Ask about modules, objectives, outcomes, etc...")
        ],
        outputs=gr.Markdown(label="Answer"),
        title="Student Syllabus Bot (Local RAG)",
        description="Upload a syllabus PDF and ask questions. The app retrieves the most relevant sections using embeddings + FAISS."
    )
    demo.launch()


if __name__ == "__main__":
    main()
