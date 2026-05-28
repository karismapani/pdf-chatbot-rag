# pdf-chatbot-rag
A chatbot that answers questions from PDF documents using LangChain and RAG
# PDF ChatBot using RAG & LangChain

A conversational AI chatbot that answers questions from PDF documents using LangChain and RAG (Retrieval-Augmented Generation) pipeline.

## 🚀 Features
- Upload any PDF document
- Ask questions in natural language
- Get accurate answers retrieved directly from the PDF
- Powered by LangChain, FAISS vector store, and Google Gemini

## 🛠️ Tech Stack
- Python
- LangChain
- FAISS (Vector Store)
- Google Gemini API
- PyPDF

## ⚙️ How It Works
1. PDF is loaded and split into chunks
2. Chunks are converted to vector embeddings using FAISS
3. User query is matched with relevant chunks
4. LLM generates answer based on retrieved context

## 📦 Installation
pip install langchain-google-genai faiss-cpu pypdf langchain-text-splitters langchain-core langchain-community

## 🔑 Setup
Add your Gemini API key:
export GOOGLE_API_KEY="your-api-key"

## 📌 Status
🚧 Under Development
