from typing import Dict, Any, List, Optional
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel, pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
import logging

logger = logging.getLogger(__name__)

# Placeholder for Dreamscape command structure - Keep this for parsing the output
class DreamscapeCommand:
    def __init__(self, action: str, parameters: Dict[str, Any]):
        self.action = action
        self.parameters = parameters

    def __repr__(self) -> str:
        return f"Command(action='{self.action}', params={self.parameters})"

# Default model for Concept Mapping - Changed to LongT5
# DEFAULT_MODEL_NAME = "./models/concept_mapper_t5_small" # Previous default
DEFAULT_MODEL_NAME = "google/long-t5-tglobal-base" # New default - assumes fine-tuned version will be saved here or loaded from HF

class ConceptMapper:
    """Maps analyzed text and dream state to actionable commands using a Seq2Seq model."""
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME):
        """Initializes the ConceptMapper with a pre-trained Seq2Seq model.

        Args:
            model_name_or_path: Path or Hugging Face identifier for the model.
        """
        self.model_name = model_name_or_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ConceptMapper initializing with model: {self.model_name} on device: {self.device}")
        try:
            # Use AutoModel for flexibility, handles T5 and potentially others
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure pad_token is set if eos_token is used and pad_token is None
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                 logger.warning(f"Tokenizer for {self.model_name} has eos_token but no pad_token. Setting pad_token to eos_token.")
                 self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode by default
            logger.info(f"ConceptMapper loaded model {self.model_name} successfully.")
        except OSError as e:
            logger.error(f"Error loading model {self.model_name}: {e}. Ensure the model exists locally or on Hugging Face Hub.")
            # Depending on requirements, could raise error or fallback
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during ConceptMapper initialization: {e}")
            raise

    def _parse_command_string(self, command_str: str) -> List[DreamscapeCommand]:
        """Parses the model's output string into DreamscapeCommand objects.

        Handles variations in spacing and basic error checking.
        Format: action=ACTION [key=value] [key=value] | action=ACTION ...
        """
        commands = []
        if not command_str:
            return commands

        # Split commands by '|'
        command_parts = command_str.split('|')
        for part in command_parts:
            part = part.strip() # Remove leading/trailing whitespace
            if not part:
                continue

            elements = part.split() # Split by space
            if not elements or not elements[0].startswith("action="):
                print(f"Warning: Skipping invalid command part format: '{part}'")
                continue # Skip if first element isn't 'action=...'

            try:
                action = elements[0].split("=", 1)[1].strip()
                if not action:
                     print(f"Warning: Skipping command part with empty action: '{part}'")
                     continue

                parameters = {}
                for element in elements[1:]:
                    element = element.strip()
                    if not element or "=" not in element:
                        # Allow valueless keys potentially later? For now, skip if no '='
                        print(f"Warning: Skipping invalid parameter format '{element}' in command '{part}'")
                        continue

                    key, value = element.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                         print(f"Warning: Skipping parameter with empty key '{element}' in command '{part}'")
                         continue

                    # Attempt type conversion
                    if value.lower() == 'true':
                        parsed_value = True
                    elif value.lower() == 'false':
                        parsed_value = False
                    else:
                        try:
                            # Try float first, then int
                            if '.' in value or 'e' in value.lower():
                                parsed_value = float(value)
                            else:
                                parsed_value = int(value)
                        except ValueError:
                            parsed_value = value # Keep as string if conversion fails

                    parameters[key] = parsed_value

                commands.append(DreamscapeCommand(action=action, parameters=parameters))
            except IndexError:
                 print(f"Warning: Skipping malformed command part (IndexError): '{part}'")
            except Exception as e:
                 print(f"Warning: Unexpected error parsing command part '{part}': {e}")

        return commands


    def map_analysis_to_commands(
        self, analysis_results: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None # Add current_state input
    ) -> List[DreamscapeCommand]:
        """Maps analysis results dictionary to commands using the model, considering current state.

        Args:
            analysis_results: A dictionary from TextAnalyzer.analyze().
            current_state: Optional dictionary representing the current Dreamscape state
                           from DreamscapeManager.get_state().

        Returns:
            A list of DreamscapeCommand objects, or an empty list if model failed.
        """
        if not self.model or not self.tokenizer:
            print("ConceptMapper model not loaded. Returning empty command list.")
            return []

        # Extract info from analysis
        raw_text = analysis_results.get("raw_input_text", "")
        # Extract just the topic labels for the input string
        topics = [t["topic"] for t in analysis_results.get("topics", [])]
        sentiment = analysis_results.get("sentiment", "NEUTRAL")

        # Extract info from current_state (if provided)
        current_theme = "None"
        last_action = "None"
        if current_state:
            current_theme = current_state.get("theme", "None")
            last_cmd_info = current_state.get("last_command_info")
            if last_cmd_info and isinstance(last_cmd_info, dict):
                last_action = last_cmd_info.get("action", "None")

        # Construct input string for the T5 model including state
        input_str = (
            f"map concepts to commands: text: {raw_text} | sentiment: {sentiment} | "
            f"topics: {', '.join(topics)} | current_theme: {current_theme} | "
            f"last_action: {last_action}"
        )
        print(f"ConceptMapper input string: {input_str}") # Log the input string

        # Tokenize input
        model_inputs = self.tokenizer(input_str, return_tensors="pt", max_length=128, truncation=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        # Generate output tokens
        try:
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs['attention_mask'],
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode generated tokens
            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            print(f"ConceptMapper generated command string: {generated_text}")

            # Parse the generated string into commands
            commands = self._parse_command_string(generated_text)

        except Exception as e:
            print(f"Error during ConceptMapper model inference or parsing: {e}")
            commands = []

        return commands

# Example Usage updated
if __name__ == "__main__":
    mapper = ConceptMapper()

    if mapper.model:
        # Simulate analysis results
        analysis1 = {
            "raw_input_text": "I felt a deep sense of fear walking through the dark woods alone.",
            "concepts": ["fear", "woods"],
            "sentiment": "NEGATIVE",
            "sentiment_score": 0.95,
            "topics": [{"topic": "fear", "score": 0.9}, {"topic": "nature", "score": 0.8}],
            "entities": []
        }
        # Simulate current state
        state1 = {"theme": "neutral_void", "last_command_info": None}
        commands1 = mapper.map_analysis_to_commands(analysis1, state1)
        print(f"Analysis 1 -> State 1 -> Commands: {commands1}")

        analysis2 = {
            "raw_input_text": "Alice saw a strange glowing fox in Berlin.",
            "concepts": ["alice", "fox", "berlin"],
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.6,
            "topics": [{"topic": "urban", "score": 0.9}, {"topic": "animals", "score": 0.8}],
            # Provide example entity dictionaries, even if incomplete
            "entities": [
                {"word": "Alice", "entity_group": "PER", "score": 0.99},
                {"word": "Berlin", "entity_group": "LOC", "score": 0.98}
            ]
        }
        state2 = {"theme": "beach", "last_command_info": {"action": "change_soundscape", "params": {}}}
        commands2 = mapper.map_analysis_to_commands(analysis2, state2)
        print(f"Analysis 2 -> State 2 -> Commands: {commands2}")
    else:
        print("Skipping ConceptMapper inference example as model failed to load.")