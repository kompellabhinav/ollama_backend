import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access the API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
def pinecone_init():
    pc = Pinecone(api_key=pinecone_api_key)
    return pc