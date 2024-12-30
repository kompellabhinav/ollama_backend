from pinecone import Pinecone
from config.__init__ import pinecone_api_key

def query_pinecone(index, query_embedding, top_k=5):
    """
    Queries Pinecone index with the provided embedding and retrieves top-k results.
    """
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results

def get_pinecone_index(index_name):
    """
    Retrieves the specified Pinecone index.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return index