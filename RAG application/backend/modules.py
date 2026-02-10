import asyncio
from embedding import build_embeddings_if_needed
from retreiver import load_retriever
from chain import build_chain
from llms import groq_llm
from config import search_vectorstore



async def model_question(question: str, doc_name: str):
    # Check if vectorstore exists
    exists = await asyncio.to_thread(search_vectorstore, file=doc_name)

    if not exists:
        await asyncio.to_thread(build_embeddings_if_needed)

    retriever = await asyncio.to_thread(load_retriever, doc_name)
    chain = await asyncio.to_thread(build_chain, retriever, groq_llm)

    # ainvoke is async → await is correct
    answer = await chain.ainvoke(question)
    return answer



if __name__ == "__main__":
    # print(model_question("what is large language model")
    pass 