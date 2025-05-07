import json
from typing import Dict, Any, List, Optional
import logging
import torch

# Import the command structure from concept_mapper
from .concept_mapper import DreamscapeCommand

# Setup basic logging
logger = logging.getLogger("DreamscapeManager")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class DreamscapeManager:
    """Manages the state of the Dreamscape and applies changes."""

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initializes the DreamscapeManager.

        Args:
            initial_state: An optional dictionary representing the starting state.
                           If None, a default state is created.
        """
        if initial_state:
            self.state = initial_state
        else:
            self.state = self._get_default_state()
        logger.info(f"DreamscapeManager initialized with state: {self.state}")

    def _get_default_state(self) -> Dict[str, Any]:
        """Returns the default initial state of the Dreamscape."""
        return {
            "theme": "neutral_void",
            "lighting": {"color": "white", "brightness": 0.7},
            "soundscape": {"sound": "ambient_hum", "volume": 0.5},
            "entities": [],
            "modifiers": {},
            "last_command_info": None # Store info about last action
        }

    def apply_commands(self, commands: List[DreamscapeCommand]):
        """Applies a list of commands to modify the Dreamscape state.

        Args:
            commands: A list of DreamscapeCommand objects.
        """
        if not commands:
            logger.debug("No commands to apply.")
            return

        logger.info(f"Applying {len(commands)} commands: {commands}")
        for command in commands:
            self._apply_single_command(command)
            # Store info about the last applied command (simplified)
            self.state["last_command_info"] = {
                "action": command.action,
                "params": command.parameters
            }

        logger.info(f"New Dreamscape state: {self.get_state_summary()}")

    def _apply_single_command(self, command: DreamscapeCommand):
        """Applies a single command to the state. (Basic implementation)"""
        action = command.action
        params = command.parameters

        try:
            if action == "change_lighting":
                self.state["lighting"].update(params)
                logger.debug(f"Applied lighting change: {params}")
            elif action == "change_soundscape":
                self.state["soundscape"].update(params)
                logger.debug(f"Applied soundscape change: {params}")
            elif action == "set_theme":
                self.state["theme"] = params.get("theme", self.state["theme"])
                logger.debug(f"Applied theme change: {params}")
            elif action == "spawn_entity":
                entity = {"type": params.get("type"), "id": f"entity_{len(self.state['entities']) + 1}"}
                self.state["entities"].append(entity)
                logger.debug(f"Spawned entity: {entity}")
            elif action == "modify_environment":
                effect = params.get("effect")
                if effect:
                    self.state["modifiers"][effect] = params
                    logger.debug(f"Applied environment modifier: {effect}={params}")
            elif action == "change_atmosphere": # General mood setter
                mood = params.get("mood", "neutral")
                intensity = params.get("intensity", 0.5)
                # Example: Adjust multiple params based on mood
                if mood == "positive":
                    self.state["lighting"]["brightness"] = min(1.0, 0.7 + 0.3 * intensity)
                    self.state["lighting"]["color"] = "light_yellow"
                elif mood == "negative":
                    self.state["lighting"]["brightness"] = max(0.1, 0.5 - 0.4 * intensity)
                    self.state["lighting"]["color"] = "dark_blue"
                logger.debug(f"Applied atmosphere change: {mood} ({intensity})")
            # Add more command handlers here...
            else:
                logger.warning(f"Unknown command action: {action}")
        except Exception as e:
            logger.error(f"Error applying command {command}: {e}", exc_info=True)

    def get_state(self) -> Dict[str, Any]:
        """Returns the complete current state of the Dreamscape."""
        return self.state

    def get_state_summary(self) -> str:
         """Returns a concise summary of the state for logging."""
         # Avoid serializing potentially large/complex objects directly in logs
         summary = {
             k: v for k, v in self.state.items() if k not in ['entities']
         }
         summary['entity_count'] = len(self.state['entities'])
         try:
            return json.dumps(summary)
         except TypeError:
             return str(summary) # Fallback if not JSON serializable

    def record_user_feedback(self, feedback_type: str, associated_command: Optional[Dict] = None):
        """Records explicit user feedback (Placeholder).

        Args:
            feedback_type: Type of feedback (e.g., 'positive', 'negative', 'clarify').
            associated_command: The command info dict from the state that prompted feedback.
        """
        # In a real implementation:
        # - Store feedback (e.g., input text, analysis, command, feedback_type) in a database/log.
        # - This data could be used later for RLHF or creating more training examples.
        feedback_data = {
            "type": feedback_type,
            "triggering_command": associated_command,
            "timestamp": torch.utils.data.get_worker_info() if torch.utils.data.get_worker_info() else None # Simple timestamp
        }
        logger.info(f"User feedback received: {feedback_data}")
        # For now, just log it.

    def save_state(self, filepath: str):
        """Saves the current state to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.state, f, indent=4)
            logger.info(f"Dreamscape state saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save state to {filepath}: {e}")

    @classmethod
    def load_state(cls, filepath: str) -> 'DreamscapeManager':
        """Loads state from a JSON file and returns a new manager instance."""
        try:
            with open(filepath, 'r') as f:
                loaded_state = json.load(f)
            logger.info(f"Dreamscape state loaded from {filepath}")
            return cls(initial_state=loaded_state)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load state from {filepath}: {e}. Returning manager with default state.")
            return cls() # Return default state manager on load failure

# Example Usage
if __name__ == "__main__":
    manager = DreamscapeManager()
    print("Initial State:", manager.get_state())

    # Simulate commands from ConceptMapper
    commands_to_apply = [
        DreamscapeCommand(action="change_lighting", parameters={"color": "blue", "brightness": 0.5}),
        DreamscapeCommand(action="spawn_entity", parameters={"type": "glowing_orb"})
    ]
    manager.apply_commands(commands_to_apply)
    print("State after commands:", manager.get_state_summary())

    # Test save/load
    state_file = "dreamscape_state.json"
    manager.save_state(state_file)
    loaded_manager = DreamscapeManager.load_state(state_file)
    print("Loaded State Summary:", loaded_manager.get_state_summary())
    assert manager.get_state() == loaded_manager.get_state() # Verify state matches
    print(f"State successfully saved to and loaded from {state_file}")

    # Clean up the test file
    import os
    try:
        os.remove(state_file)
        print(f"Removed test state file: {state_file}")
    except OSError:
        pass