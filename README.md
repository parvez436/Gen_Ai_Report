# Student Syllabus Bot (Local RAG Chatbot)

A local Retrieval-Augmented Generation (RAG) chatbot for syllabus PDFs.  
It extracts syllabus sections, builds embeddings with SentenceTransformers, indexes them in FAISS, and serves a Gradio UI for Q&A.

## Features
- Local PDF parsing (no Google Drive required)
- Chunking into sections (Objectives, Modules, Outcomes, Textbooks, References)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector search: FAISS
- Web UI: Gradio
- Optional “LLM answer mode” (you can add later)

## Screenshots

first screen shot shows the retrieval of chunks
![Screenshot 1](assets/screenshot1.jpeg)

second and third screenshots show the model generating a valid output from the retrieved chunks

![Screenshot 2](assets/screenshot2.jpeg)
![Screenshot 3](assets/screenshot3.jpeg)

## Quickstart (Local)
### 1) Create environment + install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the app
```bash
python app.py
```

Then open the Gradio URL shown in the terminal.

## How it works
1. You provide a syllabus PDF
2. The PDF is parsed and split into structured chunks
3. Chunks are embedded and stored in a FAISS index
4. A query is embedded → top-k chunks are retrieved → shown as the answer context

## Project Structure
- `app.py`: main Gradio app (local-friendly)
- `notebooks/`: development notebook version
- `data/`: generated index + metadata
- `assets/`: screenshots for README

## License
MIT
