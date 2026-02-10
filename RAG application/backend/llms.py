from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
groq_llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.5)

from langchain_huggingface import HuggingFaceEmbeddings
embedding_llm = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=embedding_llm)


