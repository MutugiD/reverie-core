import pytest
import os
import json
from typing import Dict, Any

from src.reverie.dreamscape_manager import DreamscapeManager
from src.reverie.concept_mapper import DreamscapeCommand # Use the command class

TEST_STATE_FILE = "test_dreamscape_state.json"

@pytest.fixture
def manager() -> DreamscapeManager:
    """Provides a DreamscapeManager instance with a default state."""
    # Ensure clean state for each test
    if os.path.exists(TEST_STATE_FILE):
        os.remove(TEST_STATE_FILE)
    m = DreamscapeManager()
    yield m
    # Cleanup after test
    if os.path.exists(TEST_STATE_FILE):
        os.remove(TEST_STATE_FILE)

def test_manager_init_default(manager: DreamscapeManager):
    """Test initialization with default state."""
    state = manager.get_state()
    assert state["theme"] == "neutral_void"
    assert state["lighting"]["color"] == "white"
    assert len(state["entities"]) == 0
    assert state["last_command_info"] is None

def test_manager_init_custom_state():
    """Test initialization with a custom state."""
    custom_state = {"theme": "underwater", "lighting": {"brightness": 0.2}}
    manager = DreamscapeManager(initial_state=custom_state)
    state = manager.get_state()
    assert state["theme"] == "underwater"
    assert state["lighting"]["brightness"] == 0.2
    # Check if defaults are merged (they aren't in current init, maybe desirable?)
    # assert "soundscape" in state # This would fail currently

def test_apply_single_command_lighting(manager: DreamscapeManager):
    """Test applying a single lighting command."""
    command = DreamscapeCommand(action="change_lighting", parameters={"color": "red", "brightness": 0.4})
    manager.apply_commands([command])
    state = manager.get_state()
    assert state["lighting"]["color"] == "red"
    assert state["lighting"]["brightness"] == 0.4
    assert state["last_command_info"]["action"] == "change_lighting"

def test_apply_multiple_commands(manager: DreamscapeManager):
    """Test applying multiple commands."""
    commands = [
        DreamscapeCommand(action="set_theme", parameters={"theme": "desert"}),
        DreamscapeCommand(action="spawn_entity", parameters={"type": "cactus"}),
        DreamscapeCommand(action="spawn_entity", parameters={"type": "mirage"})
    ]
    manager.apply_commands(commands)
    state = manager.get_state()
    assert state["theme"] == "desert"
    assert len(state["entities"]) == 2
    assert state["entities"][0]["type"] == "cactus"
    assert state["entities"][1]["type"] == "mirage"
    assert state["last_command_info"]["action"] == "spawn_entity" # Last command applied

def test_apply_unknown_command(manager: DreamscapeManager, caplog):
    """Test applying an unknown command action."""
    command = DreamscapeCommand(action="fly_to_moon", parameters={})
    manager.apply_commands([command])
    # Check logs for warning (requires caplog fixture from pytest)
    assert "Unknown command action: fly_to_moon" in caplog.text
    assert state["last_command_info"]["action"] == "fly_to_moon" # Info is still stored

def test_save_load_state(manager: DreamscapeManager):
    """Test saving and loading the state."""
    # Apply some changes first
    commands = [DreamscapeCommand(action="set_theme", parameters={"theme": "cyberpunk"})]
    manager.apply_commands(commands)
    original_state = manager.get_state()

    manager.save_state(TEST_STATE_FILE)
    assert os.path.exists(TEST_STATE_FILE)

    loaded_manager = DreamscapeManager.load_state(TEST_STATE_FILE)
    loaded_state = loaded_manager.get_state()

    assert loaded_state == original_state

def test_load_state_file_not_found(caplog):
    """Test loading state when the file does not exist."""
    manager = DreamscapeManager.load_state("non_existent_file.json")
    assert "Failed to load state" in caplog.text
    assert "Returning manager with default state" in caplog.text
    # Should have the default state
    state = manager.get_state()
    assert state["theme"] == "neutral_void"

def test_load_state_invalid_json(tmp_path, caplog):
    """Test loading state from an invalid JSON file."""
    invalid_json_file = tmp_path / "invalid_state.json"
    invalid_json_file.write_text("this is not json{")

    manager = DreamscapeManager.load_state(str(invalid_json_file))
    assert "Failed to load state" in caplog.text
    assert "JSONDecodeError" in caplog.text
    assert "Returning manager with default state" in caplog.text
    # Should have the default state
    state = manager.get_state()
    assert state["theme"] == "neutral_void"