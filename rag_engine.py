
import os
import pdfplumber
import litellm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path: str) -> str:
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text += f"\n\n--- Page {page_num} ---\n{text}"
    return all_text

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks: list, collection_name: str = "financial_report"):
    print("Building embeddings... (1-2 min first time)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def retrieve_relevant_chunks(vectorstore, question: str, k: int = 4) -> list:
    results = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in results]

def ask_llm_with_context(question: str, context_chunks: list, model: str = "gpt-4o-mini") -> str:
    context = "\n\n".join([f"[Chunk {i+1}]:\n{chunk}"
                              for i, chunk in enumerate(context_chunks)])
    prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this information in the document."
Be specific with numbers, dates, and figures.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content

def process_pdf(pdf_path: str):
    print(f"Processing: {pdf_path}")
    text   = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    store  = build_vector_store(chunks)
    return store

def answer_question(vectorstore, question: str, model: str = "gpt-4o-mini") -> dict:
    chunks = retrieve_relevant_chunks(vectorstore, question)
    answer = ask_llm_with_context(question, chunks, model)
    return {
        "question": question,
        "answer": answer,
        "sources": chunks,
        "chunks_used": len(chunks)
    }
