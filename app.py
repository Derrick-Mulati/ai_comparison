import openai
from transformers import pipeline

# Set up OpenAI API key
openai.api_key = "your-openai-api-key"

# Set up Hugging Face model (using a pre-trained model from the transformers library)
huggingface_model = pipeline("text-generation", model="gpt2")

def openai_gpt3(prompt):
    """Generate a response using OpenAI's GPT-3.5."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def huggingface_gpt2(prompt):
    """Generate a response using Hugging Face's GPT-2."""
    response = huggingface_model(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'].strip()

def compare_ai_responses(prompt):
    """Compare responses from OpenAI GPT-3.5 and Hugging Face GPT-2."""
    print(f"Prompt: {prompt}\n")

    # Get response from OpenAI GPT-3.5
    gpt3_response = openai_gpt3(prompt)
    print(f"OpenAI GPT-3.5 Response:\n{gpt3_response}\n")

    # Get response from Hugging Face GPT-2
    gpt2_response = huggingface_gpt2(prompt)
    print(f"Hugging Face GPT-2 Response:\n{gpt2_response}\n")

    # Compare the responses (you can add more sophisticated comparison logic)
    if gpt3_response == gpt2_response:
        print("Both models generated the same response.")
    else:
        print("The models generated different responses.")

# Example usage
if __name__ == "__main__":
    prompt = "Explain the concept of artificial intelligence in simple terms."
    compare_ai_responses(prompt)