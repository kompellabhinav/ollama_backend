from embeddings.embedding_service import get_embedding
from pinecone_service.pinecone_utils import query_pinecone
from config.__init__ import pinecone_init  # Removed openai import
import logging

# For evaluation purposes
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score, f1_score

# Import Llama model (assuming you have a library for it)
from langchain_ollama import OllamaLLM  # Importing the Llama model class

# Initialize the Llama model
llama_model = OllamaLLM(model="llama3.2:latest")  # Specify your desired Llama model

SIMILARITY_THRESHOLD = 0.8
messages = []
index_name = "rag-project"

def process_query(query, top_k=5, similarity_threshold=SIMILARITY_THRESHOLD):
    # Generate query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return {"error": "Failed to generate embedding for the query."}

    # Retrieve documents from Pinecone
    results = retrieve_documents(query_embedding)
    if not results['matches']:
        return {"error": "No documents were retrieved."}

    # Check if the top score meets the threshold
    if is_score_above_threshold(results, threshold=similarity_threshold):
        response_data = {
            "status": "relevant",
            "documents": []
        }       
        for match in results['matches']:
            document = {
                "score": match['score'],
                "topic": match['metadata'].get('Topic', 'N/A'),
                "url": match['metadata'].get('Video URL', 'N/A'),
                "description": match['metadata'].get('Description', 'N/A')
            }
            response_data["documents"].append(document)
        return response_data
    else:
        prompt = generate_follow_up_questions(results, query)
        follow_up_questions = get_follow_up_questions(prompt)

        response_data = {
            "status": "not_relevant",
            "follow_up_questions": follow_up_questions
        }
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

def generate_follow_up_questions(documents, original_query, num_questions=1):
    document_texts = []
    for i, match in enumerate(documents['matches'][:3]):
        doc_text = match['metadata'].get('Description', '')
        document_texts.append(f"Document {i+1}: {doc_text}")

    prompt = (
        f"The user's query is: '{original_query}'\n\n"
        "The retrieved documents are not sufficiently relevant.\n"
        "Based on the following documents, generate follow-up questions to help clarify the user's intent.\n\n" +
        "\n".join(document_texts) +
        f"\n\nPlease provide {num_questions} follow-up questions."
    )
    return prompt

def get_follow_up_questions(prompt):
    messages.append({"role": "user", "content": prompt})
    logging.info(f"messages: {messages}")

    # Use Llama model to generate follow-up questions
    response = llama_model.invoke(prompt)  # Call invoke method on llama_model
    questions = response  # Adjust based on how your model returns data
    
    messages.append({"role": "assistant", "content": questions})
    
    return questions

def evaluate_response(test_queries, expected_responses):
    bleu_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    faithful_scores = []

    for query, expected in zip(test_queries, expected_responses):
        response = process_query(query)

        if response.get("error"):
            print(f"Error processing query '{query}': {response['error']}")
            continue  # Skip this query and continue with the next one

        if response["status"] == "not_relevant":
            print(f"Query didn't retrieve any relevant document.")
            continue

        generated_response = " ".join([doc['description'] for doc in response["documents"]])

        # BLEU score evaluation
        bleu_score = sentence_bleu([expected.split()], generated_response.split())
        bleu_scores.append(bleu_score)

        # Token-based precision, recall, and F1-Score
        expected_tokens = set(expected.split())
        generated_tokens = set(generated_response.split())
        true_positives = len(expected_tokens.intersection(generated_tokens))
        
        precision = true_positives / len(generated_tokens) if len(generated_tokens) > 0 else 0
        recall = true_positives / len(expected_tokens) if len(expected_tokens) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Faithfulness check (arbitrary condition)
        faithful_score = 1 if bleu_score > 0.5 else 0
        faithful_scores.append(faithful_score)

    # Average metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_faithfulness = sum(faithful_scores) / len(faithful_scores) if faithful_scores else 0

    # Print evaluation results
    print(f"Average BLEU score: {avg_bleu}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1-Score: {avg_f1}")
    print(f"Average Faithfulness: {avg_faithfulness}")

# Prepare the dataset
test_queries = [
    "What is the importance of skin-to-skin contact for babies, and what should I observe during this time?"
]
expected_responses = [
    "Skin-to-skin contact is crucial for newborns as it helps regulate their body temperature, supports breastfeeding, and enhances bonding. During this time, observe the baby’s breathing, skin color, and temperature. The baby should appear calm, with normal breathing and a steady heartbeat."
]

# Let's run the evaluation
if __name__ == "__main__":
    evaluate_response(test_queries, expected_responses)
