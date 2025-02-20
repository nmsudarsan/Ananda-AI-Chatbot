import os
from dotenv import load_dotenv

# Load environment variables
def load_env():
    """Load environment variables for the application."""
    load_dotenv()
    os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
