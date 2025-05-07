# Concept Mapping Dataset - Generation Guidelines

This document provides guidelines for creating effective training examples for the Reverie `ConceptMapper` model (currently `t5-small`). The goal is to train the model to map analyzed text input and current Dreamscape state to appropriate command sequences.

## Data Format (JSON Lines - `data/concept_mapping_dataset.jsonl`)

Each line in the file must be a valid JSON object with the following keys:

*   `raw_text` (string): The simulated dreamer input text.
*   `current_theme` (string): The hypothetical theme of the Dreamscape *before* this input (e.g., "forest", "cityscape", "beach", "library", "neutral_void"). Use "None" if it's the very first input.
*   `last_action` (string): The primary action of the *last* command applied by the `DreamscapeManager` (e.g., "set_theme", "spawn_entity", "change_lighting", "None").
*   `topics` (list of strings): Key thematic elements or concepts relevant to the `raw_text` (lowercase). Should align with topics potentially identified by `TextAnalyzer` (see `CANDIDATE_TOPICS` in `text_analyzer.py`).
*   `sentiment` (string): The overall sentiment of the `raw_text` ("POSITIVE", "NEGATIVE", "NEUTRAL").
*   `target_text` (string): The desired output sequence of commands, formatted as specified below.

## Target Text Format

The `target_text` string should represent one or more commands, separated by ` | ` (space, pipe, space).
Each command follows the format: `action=ACTION_NAME param1=value1 param2=value2 ...`

*   `ACTION_NAME` should match actions handled by `DreamscapeManager._apply_single_command` (e.g., `set_theme`, `change_lighting`, `spawn_entity`, `modify_environment`, `change_soundscape`, `change_atmosphere`).
*   Parameters (`key=value`) should be space-separated.
*   Values should generally be strings, integers, or floats. Boolean values should be `true` or `false` (lowercase).

**Example `target_text`:** `action=set_theme theme=forest style=dark | action=change_lighting color=dark_red brightness=0.2`

## Generation Principles

1.  **Diversity:** Cover a wide range of:
    *   Sentiments (Positive, Negative, Neutral).
    *   Topics/Themes (fear, joy, nature, urban, memory, abstract, etc.).
    *   Input Text Styles (declarative, questioning, fragmented memories, observations).
    *   Dreamscape States (`current_theme`, `last_action`). Ensure variety here to test the state feedback.
    *   Command Combinations (single vs. multiple commands, different action types).
2.  **Coherence:** The `target_text` (commands) should be a *plausible and coherent* response to the `raw_text`, considering the `sentiment`, `topics`, and `current_state` fields.
    *   *State Influence:* The `target_text` should ideally reflect the `current_theme` and `last_action`. E.g., adding "mist" (`action=modify_environment effect=mist`) might be relevant if the `current_theme` is already "forest", but less so if it's "beach". A `last_action` of "spawn_entity" might make another `spawn_entity` less likely immediately after, unless the `raw_text` strongly implies it.
    *   *Sentiment Influence:* Negative sentiment might trigger darker colors, lower brightness, unsettling sounds, or negative mood atmospheres. Positive sentiment triggers brighter elements, pleasant sounds, etc.
    *   *Topic Influence:* Specific topics should map to relevant actions (e.g., "ocean" topic -> `action=set_theme theme=ocean` or `action=change_soundscape sound=waves`).
3.  **Complexity:** Include both simple (one command) and complex (multiple commands) `target_text` examples.
4.  **Realism (Conceptual):** While grounded in the defined actions, the generated scenarios should *feel* somewhat dreamlike or subconscious-driven, reflecting the project's goal.
5.  **Consistency:** Maintain consistent formatting for the JSON structure and the `target_text` command strings.

## LLM Assistance (e.g., using Deepseek Coder V2)

When using an LLM to assist generation:

*   **Prompt Engineering:** Design prompts that provide clear instructions and context, including the desired JSON format and the generation principles.
    *   *Example Prompt Idea:* "Generate a JSONL entry for training a text-to-command model for a dream simulation. The input text should reflect [feeling/scenario, e.g., 'a feeling of joyful discovery']. The current dreamscape theme is '[theme, e.g., library]' and the last action was '[action, e.g., spawn_entity]'. Identify relevant topics and sentiment. Generate a plausible target command string reflecting the input and state. Follow JSON format: { 'raw_text': '...', 'current_theme': '...', ... 'target_text': 'action=... | action=...' }"
*   **Batching & Variation:** Generate examples in batches, varying the seed feelings, scenarios, themes, and last actions provided in the prompts.
*   **Review and Refine:** **Critically review all LLM-generated examples.** Ensure they adhere to the guidelines, especially coherence and command format. Edit or discard examples that are nonsensical, malformed, or unhelpful.
*   **Focus on Edge Cases:** Prompt the LLM specifically for challenging or less common scenarios (e.g., contradictory inputs, rapid sentiment shifts, abstract concepts).

## Exploration of Existing Datasets

Consider adapting data from:

*   **Text Adventure Datasets:** May contain text descriptions linked to discrete actions.
*   **Story Generation Datasets:** Could provide narrative text, but actions would need to be manually annotated.
*   **Dialogue Datasets:** May contain emotional expressions and context, but less environmental interaction.
*   *(Requires significant adaptation effort)*