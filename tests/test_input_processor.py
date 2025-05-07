import pytest
from typing import TYPE_CHECKING
from transformers import PreTrainedTokenizer, AutoTokenizer

# Import functions from the module we are testing
from src.reverie.input_processor import (
    clean_text,
    tokenize_input,
    load_tokenizer,
    DEFAULT_MODEL_NAME,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

# Fixture to load the tokenizer once for all tests in this module
@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    """Provides a shared tokenizer instance for tests."""
    return load_tokenizer(DEFAULT_MODEL_NAME)


def test_clean_text():
    """Tests the clean_text function."""
    assert clean_text("  Hello World!  ") == "hello world!"
    assert clean_text("Multiple   Spaces") == "multiple spaces"
    assert clean_text("\nNew\tLines\r") == "new lines"
    # assert clean_text("Keep Punctuation.,!?") == "keep punctuation.,!?" # If punctuation removal is disabled
    assert clean_text("") == ""
    assert clean_text("already clean") == "already clean"


def test_tokenize_input(tokenizer: PreTrainedTokenizer):
    """Tests the tokenize_input function."""
    text = "This is a test."
    tokenized = tokenize_input(text, tokenizer)

    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], list)
    assert isinstance(tokenized["attention_mask"], list)
    assert len(tokenized["input_ids"]) == 512 # Check if padding to max_length worked
    assert len(tokenized["attention_mask"]) == 512

    # Check if the start and end tokens are correct for BERT
    assert tokenized["input_ids"][0] == tokenizer.cls_token_id
    # Find the index of the first SEP token (end of sequence)
    sep_token_index = -1
    for i, token_id in enumerate(tokenized["input_ids"]):
        if token_id == tokenizer.sep_token_id:
            sep_token_index = i
            break
    assert sep_token_index > 0 # SEP token should exist
    # Check padding token (assuming bert-base-uncased uses 0 for padding)
    assert tokenized["input_ids"][sep_token_index + 1] == tokenizer.pad_token_id

    # Check attention mask
    assert tokenized["attention_mask"][0] == 1 # CLS token attended
    assert tokenized["attention_mask"][sep_token_index] == 1 # SEP token attended
    assert tokenized["attention_mask"][sep_token_index + 1] == 0 # Padding not attended


def test_tokenize_input_empty(tokenizer: PreTrainedTokenizer):
    """Tests tokenize_input with empty string."""
    text = ""
    tokenized = tokenize_input(text, tokenizer)
    assert isinstance(tokenized["input_ids"], list)
    assert isinstance(tokenized["attention_mask"], list)
    assert len(tokenized["input_ids"]) == 512
    assert tokenized["input_ids"][0] == tokenizer.cls_token_id
    assert tokenized["input_ids"][1] == tokenizer.sep_token_id
    assert tokenized["input_ids"][2] == tokenizer.pad_token_id
    assert tokenized["attention_mask"][0] == 1
    assert tokenized["attention_mask"][1] == 1
    assert tokenized["attention_mask"][2] == 0

def test_load_tokenizer_invalid(monkeypatch: "MonkeyPatch"):
    """Tests that load_tokenizer raises an error for an invalid model name."""
    # Mock AutoTokenizer.from_pretrained to raise OSError
    def mock_from_pretrained(*args, **kwargs):
        raise OSError("Model not found")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", mock_from_pretrained)

    with pytest.raises(OSError, match="Model not found"):
        load_tokenizer("invalid-model-name-that-does-not-exist")