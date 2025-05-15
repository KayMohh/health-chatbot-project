# RAG Chatbot (Vercel-Compatible)

A simple Retrieval-Augmented Generation (RAG) chatbot built with:

- **Backend**: FastAPI + LangChain + FAISS + Hugging Face Transformers
- **Frontend**: Next.js + Vercel

## 🔧 Setup Instructions

### 1. Run Backend Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload