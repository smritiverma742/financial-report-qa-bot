# Financial Report Q&A Bot

Chat with any annual report, earnings transcript, or 10-K filing 
using RAG (Retrieval-Augmented Generation).

## Live Demo
[Link to your HuggingFace Space]

## What it does
Upload any financial PDF → ask questions in plain English → 
get answers with source citations.

## Tech stack
- LangChain — document chunking and retrieval pipeline
- LiteLLM — universal LLM interface (works with GPT-4o, Claude, Gemini)  
- ChromaDB — vector store for semantic search
- HuggingFace Embeddings — free local embeddings (no API cost)
- Streamlit — interactive web UI
- pdfplumber — PDF text extraction

## Built while working on Broker Vote Intelligence at CLSA
This project extends the same RAG + LiteLLM patterns I use 
at CLSA to process 120+ broker client emails daily.

## How to run locally
pip install -r requirements.txt
streamlit run app.py
