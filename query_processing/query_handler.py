from embeddings.embedding_service import get_embedding
from pinecone_service.pinecone_utils import query_pinecone
from config.__init__ import pinecone_init
from langchain_ollama import OllamaLLM
import logging
import json

SIMILARITY_LOWER_BOUND = 0.3
SIMILARITY_THRESHOLD = 0.6
messages = []
index_name = "rag-project"

# Initialize the Ollama LLM with a system prompt to enforce JSON array output
system_prompt = """
You are a helpful  4assistant. When generating follow-up questions, please provide them in the following JSON array format:

[
  {"question": "First question"},
  {"question": "Second question"},
  {"question": "Third question"}
]

Return only the JSON array and nothing else. Follow this format strictly
"""

# The model with the system prompt
llama_model = OllamaLLM(model="llama3.1", system_prompt=system_prompt)

def process_query(query, top_k=5, similarity_threshold=SIMILARITY_THRESHOLD):
    # Generate query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return {"error": "Failed to generate embedding for the query."}

    # Retrieve documents from Pinecone
    results = retrieve_documents(query_embedding)
    if not results['matches']:
        return {"error": "No documents were retrieved."}

    # Filter out results with a score less than SIMILARITY_LOWER_BOUND
    filtered_matches = [match for match in results['matches'] if match['score'] >= SIMILARITY_LOWER_BOUND]

    # If no results remain after filtering, return NULL status
    if not filtered_matches:
        return {
            "status": "NULL"
        }

    # Check if the top score meets the threshold
    if filtered_matches[0]['score'] < SIMILARITY_LOWER_BOUND:
        return {
            "status": "NULL"
        }

    if is_score_above_threshold({"matches": filtered_matches}, threshold=similarity_threshold):
        response_data = {
            "status": "relevant",
            "documents": []
        }
    else:
        prompt = generate_follow_up_questions({"matches": filtered_matches}, query)
        follow_up_questions = get_follow_up_questions(prompt)
        response_data = {
            "status": "not_relevant",
            "follow_up_questions": follow_up_questions,
            "documents": []
        }

    for match in filtered_matches:
        document = {
            "score": match['score'],
            "topic": match['metadata'].get('Topic', 'N/A'),
            "url": match['metadata'].get('URL', 'N/A'),
            "description": match['metadata'].get('Description', 'N/A')
        }
        response_data["documents"].append(document)

    return response_data

def retrieve_documents(query_embedding):
    pc = pinecone_init()
    index = pc.Index(index_name)
    # Query Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    return results

def is_score_above_threshold(results, threshold=SIMILARITY_THRESHOLD):
    if not results['matches']:
        return False
    top_score = results['matches'][0]['score']
    return top_score >= threshold

def generate_follow_up_questions(documents, original_query, num_questions=3):
    document_texts = []
    for i, match in enumerate(documents['matches'][:3]):
        doc_text = match['metadata'].get('Description', '')
        document_texts.append(f"Document {i+1}: {doc_text}")

    prompt = (
        f"The user's query is: '{original_query}'\n\n"
        "The retrieved documents are not sufficiently relevant.\n"
        "Based on the following documents, generate follow-up questions to help clarify the user's intent.\n\n" +
        "\n".join(document_texts) +
        f"\n\nPlease provide {num_questions} follow-up questions.\n"
        "Remember to return the questions as a JSON array in the specified format."
    )
    return prompt

def get_follow_up_questions(prompt):
    messages.append({"role": "user", "content": prompt})
    logging.info(f"messages: {messages}")
    
    # Invoke the model with the prompt
    response = llama_model.invoke(prompt)
    
    # Log the raw response
    print(response)
    logging.info(f"Raw response from model: {response}")
    questions = response
    
    messages.append({"role": "assistant", "content": response})
    return questions
