import streamlit as st
import requests
import time
from pathlib import Path

# ---------------- CONFIG ----------------
FASTAPI_URL = "http://127.0.0.1:8000/ask"   
UPLOAD_URL = "http://127.0.0.1:8000/uploads"
VECTORSTORE_DIR = Path("D:/langchain_models/rag_app/backend/vectorstores")

st.set_page_config(page_title="💬 RAG Chat Assistant", layout="wide")
st.title("💬 RAG Chat Assistant")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---------------- SIDEBAR ----------------
st.sidebar.title("📄 Documents & Uploads")

# Auto-populate documents from subdirectories
if VECTORSTORE_DIR.exists():
    documents = [sd.name for sd in VECTORSTORE_DIR.iterdir() if sd.is_dir()]
else:
    documents = []

# Document selector (SIDEBAR)
if documents:
    st.sidebar.selectbox("Select document",documents,key="doc_name")
else:
    st.sidebar.info("No documents found in vectorstore.")

# Upload PDFs (with sidebar placeholder)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["PDF","TXT"],
    accept_multiple_files=True
)

if uploaded_files:
    for pdf in uploaded_files:
        placeholder = st.sidebar.empty()
        placeholder.info(f"Uploading {pdf.name}...")
        try:
            response = requests.post(
                UPLOAD_URL,
                files={"file": (pdf.name, pdf.getvalue(), "application/pdf")}
            )
            if response.status_code == 200:
                placeholder.success(f"Uploaded: {pdf.name}")
            else:
                placeholder.error(f"Failed: {pdf.name}")
        except requests.exceptions.RequestException as e:
            placeholder.error(f"Upload error: {e}")
        time.sleep(2)
        placeholder.empty()

# Clear chat button
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    placeholder = st.sidebar.empty()
    placeholder.success("Chat cleared!")
    time.sleep(2)
    placeholder.empty()

# ---------------- CHAT ----------------
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    if not st.session_state.doc_name:
        st.warning("Please select a document first.")
    else:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        FASTAPI_URL,
                        json={
                            "text": question,
                            "doc_name": st.session_state.doc_name
                        },
                        timeout=30
                    )
                    data = response.json()

                    if response.status_code == 200 and "answer" in data:
                        answer = data["answer"]
                    else:
                        answer = "⚠️ Sorry, something went wrong."
                        st.error("Backend error")
                        st.write(data)

                except requests.exceptions.RequestException as e:
                    answer = f"⚠️ Error contacting backend: {e}"
                    st.error(answer)

            st.markdown(answer)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
