import os, pdfplumber, litellm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text += f"\n\n--- Page {i} ---\n{text}"
    return all_text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks, collection_name="financial_report"):
    print("Building embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    store = Chroma.from_texts(texts=chunks, embedding=embeddings,
                               collection_name=collection_name, persist_directory="./chroma_db")
    print(f"Stored {len(chunks)} chunks")
    return store

def retrieve_relevant_chunks(vectorstore, question, k=4):
    return [doc.page_content for doc in vectorstore.similarity_search(question, k=k)]

def ask_llm_with_context(question, context_chunks, model="gpt-4o-mini"):
    context = "\n\n".join([f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(context_chunks)])
    prompt = f"""You are a financial analyst assistant.
Answer ONLY using the context below. If not found, say so.
Be specific with numbers and dates.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""
    response = litellm.completion(model=model,
                                   messages=[{"role": "user", "content": prompt}],
                                   temperature=0, max_tokens=500)
    return response.choices[0].message.content

def process_pdf(pdf_path):
    text   = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    return build_vector_store(chunks)

def answer_question(vectorstore, question, model="gpt-4o-mini"):
    chunks = retrieve_relevant_chunks(vectorstore, question)
    answer = ask_llm_with_context(question, chunks, model)
    return {"question": question, "answer": answer, "sources": chunks, "chunks_used": len(chunks)}
