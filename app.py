import streamlit as st
import os, sys
sys.path.insert(0, ".")
from rag_engine import process_pdf, answer_question

st.set_page_config(page_title="Financial Report Q&A", page_icon="📊", layout="wide")
st.title("📊 Financial Report Q&A Bot")
st.caption("Upload any annual report or earnings PDF and ask questions about it")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Upload PDF\n2. Split into chunks\n3. Store as vectors\n4. Question finds relevant chunks\n5. LLM answers from chunks only")

if "vectorstore"  not in st.session_state: st.session_state.vectorstore  = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "pdf_name"     not in st.session_state: st.session_state.pdf_name     = None

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file:
        pdf_path = f"./{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing... (1-2 min first time)"):
                try:
                    st.session_state.vectorstore  = process_pdf(pdf_path)
                    st.session_state.pdf_name     = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Ready! Ask anything about {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.success(f"Using: {uploaded_file.name}")

    st.subheader("Sample Questions")
    for q in ["What was the total revenue?","What are the key risk factors?",
               "What was the net profit?","Summarise the CEO letter","What was YoY growth?"]:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending = q

with col2:
    st.subheader("Ask a Question")
    question = st.text_input("Your question", placeholder="e.g. What was revenue in Q3?")
    if "pending" in st.session_state:
        question = st.session_state.pending
        del st.session_state.pending
    if st.button("Ask", type="primary", use_container_width=True):
        if not st.session_state.vectorstore:
            st.warning("Upload a PDF first")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.warning("Enter API key in sidebar")
        elif question:
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(st.session_state.vectorstore, question, model)
                    st.session_state.chat_history.append(result)
                except Exception as e:
                    st.error(f"Error: {e}")

if st.session_state.chat_history:
    st.divider()
    st.subheader("Conversation")
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {item['question']}")
        st.info(f"**Answer:** {item['answer']}")
        with st.expander(f"View {item['chunks_used']} source chunks used"):
            for j, chunk in enumerate(item["sources"], 1):
                st.markdown(f"**Chunk {j}:**")
                st.text(chunk[:400] + "..." if len(chunk) > 400 else chunk)
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()
