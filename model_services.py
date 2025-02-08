import requests
import os
import time
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def validate_environment():
    """Validate required environment variables."""
    required_vars = ["OPENAI_API_KEY", "HUGGINGFACE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

validate_environment()

def query_openai(prompt, model="text-davinci-003", retries=3, backoff_factor=0.5):
    """Query OpenAI GPT model with retries."""
    url = "https://api.openai.com/v1/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7
    }
    for attempt in range(retries):
        try:
            logger.info(f"Sending request to OpenAI with prompt: {prompt}")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            logger.info("OpenAI request successful.")
            return response.json().get("choices", [{}])[0].get("text", "No response")
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                logger.error(f"OpenAI API request failed after {retries} attempts: {e}")
                return f"OpenAI API request failed: {e}"
            time.sleep(backoff_factor * (2 ** attempt))
    return "OpenAI API request failed."

def query_huggingface(prompt, model="gpt2"):
    """Query Hugging Face model."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": prompt}
    try:
        logger.info(f"Sending request to Hugging Face with prompt: {prompt}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info("Hugging Face request successful.")
        response_data = response.json()
        if isinstance(response_data, list) and response_data:
            return response_data[0].get("generated_text", "No response")
        return "Unexpected Hugging Face response format."
    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face API request failed: {e}")
        return f"Hugging Face API request failed: {e}"
    except ValueError:
        logger.error("Error parsing Hugging Face response.")
        return "Error parsing Hugging Face response."

# Example usage
if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    print("OpenAI Response:", query_openai(user_prompt))
    print("Hugging Face Response:", query_huggingface(user_prompt))