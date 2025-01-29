import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not OPENAI_API_KEY or not HUGGINGFACE_API_KEY:
    raise ValueError("Missing API keys. Please check your environment variables.")

def query_openai(prompt):
    """Query OpenAI GPT model."""
    url = "https://api.openai.com/v1/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("text", "No response")
    except requests.exceptions.RequestException as e:
        return f"OpenAI API request failed: {e}"
    except ValueError:
        return "Error parsing OpenAI response."

def query_huggingface(prompt):
    """Query Hugging Face model."""
    url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": prompt}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        if isinstance(response_data, list) and response_data:
            return response_data[0].get("generated_text", "No response")
        return "Unexpected Hugging Face response format."
    except requests.exceptions.RequestException as e:
        return f"Hugging Face API request failed: {e}"
    except ValueError:
        return "Error parsing Hugging Face response."

# Example usage
if __name__ == "__main__":
    user_prompt = "Tell me a joke about AI."
    print("OpenAI Response:", query_openai(user_prompt))
    print("Hugging Face Response:", query_huggingface(user_prompt))
