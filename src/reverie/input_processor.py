import re
import string
from typing import Dict, List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer

# Default tokenizer, can be configured/changed later
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"


def load_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> PreTrainedTokenizer:
    """Loads a Hugging Face tokenizer.

    Args:
        model_name: The name of the pre-trained model whose tokenizer is needed.

    Returns:
        The loaded tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except OSError as e:
        # Handle cases where the model name is invalid or not found
        print(f"Error loading tokenizer for model '{model_name}': {e}")
        # Consider raising the error or falling back to a default safe tokenizer
        # For now, let's re-raise to make the problem visible
        raise


def clean_text(text: str) -> str:
    """Performs basic cleaning of input text.

    - Converts to lowercase.
    - Removes punctuation (optional, decided against for now as BERT handles it).
    - Normalizes whitespace.

    Args:
        text: The raw input text string.

    Returns:
        The cleaned text string.
    """
    text = text.lower()
    # text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_input(
    text: str, tokenizer: PreTrainedTokenizer
) -> Dict[str, List[int]]:
    """Cleans and tokenizes the input text using the provided tokenizer.

    Args:
        text: The raw input text string.
        tokenizer: The Hugging Face tokenizer instance.

    Returns:
        A dictionary containing the token IDs ('input_ids') and attention mask,
        ready for input into a Transformer model.
    """
    cleaned_text = clean_text(text)
    # Tokenizers handle padding, truncation, and adding special tokens
    # We'll return PyTorch tensors ('pt')
    # Max length can be adjusted based on model/context window needs
    encoded_input = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,  # Standard max length for BERT
    )
    # The tokenizer returns a BatchEncoding object which behaves like a dict
    # For clarity, explicitly return the structure expected by most models
    return {
        "input_ids": encoded_input["input_ids"].squeeze().tolist(), # Convert tensor to list for easier handling initially
        "attention_mask": encoded_input["attention_mask"].squeeze().tolist(),
    }


# Example Usage (optional, for testing)
if __name__ == "__main__":
    example_text = " This is an Example... input text!! With extra spaces. "
    print(f"Original: {example_text}")

    loaded_tokenizer = load_tokenizer()
    print(f"Using tokenizer: {loaded_tokenizer.name_or_path}")

    cleaned = clean_text(example_text)
    print(f"Cleaned: {cleaned}")

    tokenized_output = tokenize_input(example_text, loaded_tokenizer)
    print(f"Tokenized (Input IDs): {tokenized_output['input_ids']}")
    print(f"Tokenized (Attention Mask): {tokenized_output['attention_mask']}")

    # Decode for verification
    decoded_text = loaded_tokenizer.decode(
        tokenized_output["input_ids"],
        skip_special_tokens=False # Show special tokens like [CLS], [SEP]
    )
    print(f"Decoded: {decoded_text}")