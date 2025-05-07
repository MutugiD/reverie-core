import pytest
import torch
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Assume default model name is consistent
from src.reverie.input_processor import (
    tokenize_input,
    load_tokenizer,
    DEFAULT_MODEL_NAME,
)
from src.reverie.text_analyzer import (
    TextAnalyzer,
    DEFAULT_SENTIMENT_MODEL_NAME,
    DEFAULT_NER_MODEL_NAME,
    DEFAULT_ZERO_SHOT_MODEL_NAME,
    CANDIDATE_TOPICS
)
from transformers import Pipeline

# Fixture to load tokenizer and analyzer once per module
@pytest.fixture(scope="module")
def setup_analyzer() -> Dict[str, Any]:
    """Provides a shared TextAnalyzer instance with mocked pipelines."""
    # Mock the pipeline function to avoid loading heavy models in tests
    with patch('transformers.pipeline') as mock_pipeline_func:
        # Configure mock pipeline return values
        mock_ner_pipeline = MagicMock(spec=Pipeline)
        mock_topic_pipeline = MagicMock(spec=Pipeline)

        # Define side effects for the pipeline calls based on task name
        def pipeline_side_effect(task, model, tokenizer=None, **kwargs):
            if task == "ner":
                return mock_ner_pipeline
            elif task == "zero-shot-classification":
                return mock_topic_pipeline
            else:
                raise ValueError(f"Unexpected pipeline task: {task}")

        mock_pipeline_func.side_effect = pipeline_side_effect

        analyzer = TextAnalyzer(
            base_model_name=DEFAULT_MODEL_NAME,
            sentiment_model_name=DEFAULT_SENTIMENT_MODEL_NAME,
            ner_model_name=DEFAULT_NER_MODEL_NAME,
            zero_shot_model_name=DEFAULT_ZERO_SHOT_MODEL_NAME
        )
        # Store mocks for assertions
        analyzer._test_mocks = {
            "ner_pipeline": mock_ner_pipeline,
            "topic_pipeline": mock_topic_pipeline
        }
        return {"analyzer": analyzer}


def test_analyzer_init(setup_analyzer):
    """Tests if the TextAnalyzer initializes correctly with all models/pipelines."""
    analyzer = setup_analyzer["analyzer"]
    assert analyzer.base_model is not None
    assert analyzer.sentiment_model is not None
    assert isinstance(analyzer.ner_pipeline, MagicMock)
    assert isinstance(analyzer.topic_pipeline, MagicMock)
    assert analyzer.base_tokenizer is not None
    assert analyzer.sentiment_tokenizer is not None
    assert analyzer.base_model_name == DEFAULT_MODEL_NAME
    assert analyzer.sentiment_model_name == DEFAULT_SENTIMENT_MODEL_NAME
    assert analyzer.ner_model_name == DEFAULT_NER_MODEL_NAME
    assert analyzer.zero_shot_model_name == DEFAULT_ZERO_SHOT_MODEL_NAME
    assert analyzer.base_model.config.model_type == "deberta-v2"
    assert analyzer.sentiment_model.config.model_type == "distilbert"
    assert analyzer.base_model.training is False
    assert analyzer.sentiment_model.training is False

def test_analyze_structure_sentiment_ner_topics(setup_analyzer):
    """Tests analysis output structure, sentiment, NER, and topics (mocked pipelines)."""
    analyzer = setup_analyzer["analyzer"]
    mock_ner_pipeline = analyzer._test_mocks["ner_pipeline"]
    mock_topic_pipeline = analyzer._test_mocks["topic_pipeline"]

    # Configure mock pipeline outputs for this test
    mock_ner_pipeline.return_value = [
        {'entity_group': 'PER', 'score': 0.99, 'word': 'Alice', 'start': 11, 'end': 16},
        {'entity_group': 'LOC', 'score': 0.98, 'word': 'Berlin', 'start': 26, 'end': 32}
    ]
    mock_topic_pipeline.return_value = {
        'sequence': '...',
        'labels': ['urban', 'curiosity', 'animals', 'calm'],
        'scores': [0.95, 0.88, 0.75, 0.4]
    }

    text = "My friend Alice visited Berlin and saw a fox."
    analysis = analyzer.analyze(text)

    assert "last_hidden_state" in analysis
    assert "pooler_output" in analysis
    assert "sentiment" in analysis
    assert "sentiment_score" in analysis
    assert "concepts" in analysis
    assert "entities" in analysis
    assert "topics" in analysis

    assert isinstance(analysis["last_hidden_state"], torch.Tensor)
    assert isinstance(analysis["sentiment"], str)
    assert isinstance(analysis["sentiment_score"], float)
    assert isinstance(analysis["concepts"], list)
    assert isinstance(analysis["entities"], list)
    assert isinstance(analysis["topics"], list)

    assert analysis["sentiment"] in ["POSITIVE", "NEUTRAL"]

    expected_concepts = {"alice", "berlin"}
    assert set(analysis["concepts"]) == expected_concepts
    assert analysis["entities"] == mock_ner_pipeline.return_value

    expected_topics = [
        {"topic": "urban", "score": 0.95},
        {"topic": "curiosity", "score": 0.88},
        {"topic": "animals", "score": 0.75}
    ]
    assert analysis["topics"] == expected_topics
    mock_topic_pipeline.assert_called_once_with(text, CANDIDATE_TOPICS, multi_label=True)

def test_analyze_empty_input(setup_analyzer):
    """Tests the analyzer with empty input text."""
    analyzer = setup_analyzer["analyzer"]
    mock_ner_pipeline = analyzer._test_mocks["ner_pipeline"]
    mock_topic_pipeline = analyzer._test_mocks["topic_pipeline"]

    # Configure mocks for empty input
    mock_ner_pipeline.return_value = []
    mock_topic_pipeline.return_value = {'sequence': '', 'labels': [], 'scores': []}

    text = ""
    analysis = analyzer.analyze(text)

    assert "last_hidden_state" in analysis
    assert "sentiment" in analysis
    assert "sentiment_score" in analysis
    assert "concepts" in analysis
    assert "entities" in analysis
    assert "topics" in analysis

    assert analysis["last_hidden_state"].shape == (1, 512, 768)
    assert analysis["concepts"] == []
    assert analysis["entities"] == []
    assert analysis["topics"] == []

    # Verify pipelines were called
    mock_ner_pipeline.assert_called_once_with(text)
    mock_topic_pipeline.assert_called_once_with(text, CANDIDATE_TOPICS, multi_label=True)

@patch('transformers.pipeline', return_value=MagicMock())
def test_analyzer_init_invalid_models(mock_pipeline):
    """Tests that TextAnalyzer initialization raises errors for invalid models."""
    invalid_name = "invalid-model-name-that-does-not-exist"

    # Test invalid base model
    with pytest.raises(OSError):
        TextAnalyzer(base_model_name=invalid_name)
    # Test invalid sentiment model
    with pytest.raises(OSError):
        TextAnalyzer(sentiment_model_name=invalid_name)
    # Test invalid NER model - pipeline creation error
    with patch('transformers.pipeline', side_effect=OSError("NER model failed")):
        with pytest.raises(OSError, match="NER model failed"):
            TextAnalyzer(ner_model_name=invalid_name)
    # Test invalid Zero-Shot model - pipeline creation error
    with patch('transformers.pipeline', side_effect=OSError("ZeroShot model failed")):
        with pytest.raises(OSError, match="ZeroShot model failed"):
            TextAnalyzer(zero_shot_model_name=invalid_name)