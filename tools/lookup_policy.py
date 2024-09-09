from langchain_core.tools import tool
from retrievers.vector_store_retriever import retriever

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    docs = retriever.query(query, k=2)
    print(docs)
    return "\n\n".join([doc["page_content"] for doc in docs])


