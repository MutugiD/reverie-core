import pytest
from unittest.mock import patch, MagicMock
import torch
from typing import Dict, Any

from src.reverie.concept_mapper import (
    ConceptMapper,
    DreamscapeCommand,
    DEFAULT_MAPPER_MODEL_PATH
)
from transformers import PreTrainedTokenizer, PreTrainedModel

@pytest.fixture
def mapper() -> ConceptMapper:
    """Provides a ConceptMapper instance."""
    return ConceptMapper()

def test_map_fear_forest_negative(mapper: ConceptMapper):
    """Test mapping fear and forest concepts with negative sentiment."""
    analysis = {
        "concepts": ["fear", "forest"], # NER concepts are already lowercase
        "sentiment": "NEGATIVE",
        "sentiment_score": 0.8
    }
    commands = mapper.map_analysis_to_commands(analysis)
    assert len(commands) == 2
    # Check fear command (lighting)
    assert commands[0].action == "change_lighting"
    assert commands[0].parameters["color"] == "dark_red"
    assert commands[0].parameters["_sentiment"] == "NEGATIVE"
    assert commands[0].parameters["_triggering_concept"] == "fear"
    assert pytest.approx(commands[0].parameters["brightness"]) == 0.3 * 0.7 * 0.8 # base * sentiment_mod * intensity
    # Check forest command (theme)
    assert commands[1].action == "set_theme"
    assert commands[1].parameters["theme"] == "forest"
    assert commands[1].parameters["_sentiment"] == "NEGATIVE"
    assert commands[1].parameters["_triggering_concept"] == "forest"

def test_map_joy_positive(mapper: ConceptMapper):
    """Test mapping joy with positive sentiment."""
    analysis = {"concepts": ["joy"], "sentiment": "POSITIVE", "sentiment_score": 0.9}
    commands = mapper.map_analysis_to_commands(analysis)
    assert len(commands) == 1
    assert commands[0].action == "change_lighting"
    assert commands[0].parameters["color"] == "yellow"
    assert commands[0].parameters["_sentiment"] == "POSITIVE"
    assert commands[0].parameters["_triggering_concept"] == "joy"
    assert pytest.approx(commands[0].parameters["brightness"]) == 0.9 * 1.2 * 0.9 # base * sentiment_mod * intensity

def test_map_ner_concepts(mapper: ConceptMapper):
    """Test mapping concepts derived from NER."""
    analysis = {
        "concepts": ["berlin", "alice"], # NER result
        "sentiment": "NEUTRAL",
        "sentiment_score": 0.6
    }
    commands = mapper.map_analysis_to_commands(analysis)
    assert len(commands) == 2 # Both Berlin and Alice should map
    actions = {cmd.action for cmd in commands}
    assert "set_theme" in actions
    assert "spawn_entity" in actions
    # Check parameters of one command for detail
    for cmd in commands:
        if cmd.action == "spawn_entity":
            assert cmd.parameters["type"] == "friendly_figure"
            assert cmd.parameters["_sentiment"] == "NEUTRAL"
            assert cmd.parameters["_triggering_concept"] == "alice"

def test_map_unknown_concept(mapper: ConceptMapper):
    """Test mapping an unknown concept with neutral sentiment."""
    analysis = {"concepts": ["unknownconcept"], "sentiment": "NEUTRAL", "sentiment_score": 0.5}
    commands = mapper.map_analysis_to_commands(analysis)
    # No mapping, neutral sentiment -> no fallback command
    assert len(commands) == 0

def test_map_unknown_concept_negative(mapper: ConceptMapper):
    """Test mapping an unknown concept with negative sentiment (fallback)."""
    analysis = {"concepts": ["unknownconcept"], "sentiment": "NEGATIVE", "sentiment_score": 0.7}
    commands = mapper.map_analysis_to_commands(analysis)
    # No mapping, but negative sentiment -> fallback command
    assert len(commands) == 1
    assert commands[0].action == "change_atmosphere"
    assert commands[0].parameters["mood"] == "negative"
    assert commands[0].parameters["intensity"] == 0.7

def test_map_no_concept_positive(mapper: ConceptMapper):
    """Test mapping with no concepts but positive sentiment."""
    analysis = {"concepts": [], "sentiment": "POSITIVE", "sentiment_score": 0.6}
    commands = mapper.map_analysis_to_commands(analysis)
    assert len(commands) == 1
    assert commands[0].action == "change_atmosphere"
    assert commands[0].parameters["mood"] == "positive"
    assert commands[0].parameters["intensity"] == 0.6

def test_map_no_concept_neutral(mapper: ConceptMapper):
    """Test mapping with no concepts and neutral sentiment."""
    analysis = {"concepts": [], "sentiment": "NEUTRAL", "sentiment_score": 0.5}
    commands = mapper.map_analysis_to_commands(analysis)
    assert len(commands) == 0

# --- Test Initialization (Mocked) ---
@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModelForSeq2SeqLM.from_pretrained')
def test_concept_mapper_init_success(mock_model_load, mock_tokenizer_load):
    """Test successful initialization (mocked)."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_tokenizer_load.return_value = mock_tokenizer
    mock_model_load.return_value = mock_model

    mapper = ConceptMapper()
    assert mapper.tokenizer is mock_tokenizer
    assert mapper.model is mock_model
    mock_tokenizer_load.assert_called_once_with(DEFAULT_MAPPER_MODEL_PATH)
    mock_model_load.assert_called_once_with(DEFAULT_MAPPER_MODEL_PATH)
    mock_model.eval.assert_called_once()
    mock_model.to.assert_called_once()

@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModelForSeq2SeqLM.from_pretrained')
def test_concept_mapper_init_failure(mock_model_load, mock_tokenizer_load):
    """Test initialization failure (mocked)."""
    mock_tokenizer_load.side_effect = OSError("Model not found")
    mapper = ConceptMapper()
    assert mapper.tokenizer is None
    assert mapper.model is None
    mock_model_load.assert_not_called()

# --- Test Parsing Logic --- (Directly test the parser method)
@pytest.fixture
def mapper_instance() -> ConceptMapper:
    """Provides a ConceptMapper instance (model loading might fail but parser works)."""
    # Suppress print statements during test init if model load fails
    with patch('builtins.print'):
        mapper = ConceptMapper(model_path="dummy/path/to/avoid/load/attempt/in/fixture")
        # Ensure model is None so inference isn't attempted if called accidentally
        mapper.model = None
        mapper.tokenizer = None
    return mapper

def test_parse_command_string(mapper_instance: ConceptMapper):
    """Test the command string parsing logic directly."""
    parser = mapper_instance._parse_command_string

    # Test case 1: Valid string
    str1 = "action=set_theme theme=beach | action=change_lighting color=yellow brightness=0.8"
    cmd1 = parser(str1)
    assert len(cmd1) == 2
    assert cmd1[0].action == "set_theme"
    assert cmd1[0].parameters == {"theme": "beach"}
    assert cmd1[1].action == "change_lighting"
    assert cmd1[1].parameters == {"color": "yellow", "brightness": 0.8}

    # Test case 2: String with extra spaces and pipe
    str2 = " action=spawn type=orb |  action=sound name=zap volume=1 "
    cmd2 = parser(str2)
    assert len(cmd2) == 2
    assert cmd2[0].action == "spawn"
    assert cmd2[0].parameters == {"type": "orb"}
    assert cmd2[1].action == "sound"
    assert cmd2[1].parameters == {"name": "zap", "volume": 1}

    # Test case 3: Invalid format (missing action=)
    str3 = "theme=forest | action=light color=green"
    cmd3 = parser(str3)
    assert len(cmd3) == 1 # Only the second part is valid
    assert cmd3[0].action == "light"

    # Test case 4: Empty string
    str4 = ""
    cmd4 = parser(str4)
    assert len(cmd4) == 0

    # Test case 5: String with only pipes
    str5 = " | | "
    cmd5 = parser(str5)
    assert len(cmd5) == 0

    # Test case 6: Boolean values
    str6 = "action=modify effect=visible value=true | action=set flag=false"
    cmd6 = parser(str6)
    assert len(cmd6) == 2
    assert cmd6[0].parameters == {"effect": "visible", "value": True}
    assert cmd6[1].parameters == {"flag": False}

# Note: Testing the map_analysis_to_commands method now requires
# actually running inference or a more complex mocking setup covering
# tokenizer calls, model.generate, and decode, which was causing issues.
# These tests primarily cover initialization and parsing logic.