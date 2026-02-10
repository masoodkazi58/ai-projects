from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(retriever, llm):
    prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Use ONLY the information provided in the context below.
You may summarize, paraphrase, or reorganize the content,
but do NOT add any new information.

If the context does not contain enough information
to answer the question, say: "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

    return (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
