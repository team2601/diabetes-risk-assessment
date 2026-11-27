import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "models/diabetes_model.joblib"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "diabetes_docs"
MAX_ITERATIONS = 5


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)