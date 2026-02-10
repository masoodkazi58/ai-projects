from langchain_community.vectorstores import FAISS
from llms import embedder
from config import VECTORSTORE_DIR

def load_retriever(doc_name: str, k: int = 5):
    vectorstore = FAISS.load_local(
        f"{VECTORSTORE_DIR}/{doc_name}",
        embedder,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
