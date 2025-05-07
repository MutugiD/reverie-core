# Reverie Manual Testing Guide

This document outlines the steps for manually testing the core feedback loop of the Reverie system.

## Objective

To verify that dreamer input is processed, analyzed (including sentiment, NER, topics), mapped to commands considering the current Dreamscape state, and that these commands update the state correctly.

## Prerequisites

*   Python environment set up with all dependencies from `requirements.txt` installed.
*   The `ConceptMapper` model has been trained (`python scripts/train_concept_mapper.py`) and saved to `models/concept_mapper_t5_small`.
*   Necessary Hugging Face models (base, sentiment, NER, zero-shot) are downloadable or cached.
*   (Optional but Recommended) spaCy English model downloaded for advanced NLP features (placeholders): `python -m spacy download en_core_web_sm`

## Testing Steps

This testing is conceptual and currently best performed via an interactive Python session or a dedicated test script.

1.  **Initialization:**
    *   Import necessary classes: `TextAnalyzer`, `ConceptMapper`, `DreamscapeManager` from `src.reverie`.
    *   Instantiate each class:
        ```python
        from src.reverie.text_analyzer import TextAnalyzer
        from src.reverie.concept_mapper import ConceptMapper
        from src.reverie.dreamscape_manager import DreamscapeManager

        analyzer = TextAnalyzer() # Loads all analysis models
        mapper = ConceptMapper()   # Loads the fine-tuned T5 model
        manager = DreamscapeManager() # Initializes with default state
        ```
    *   *Verification:* Check for any errors during initialization (e.g., model loading issues printed to console).

2.  **First Input & Analysis:**
    *   Define a sample input string:
        ```python
        input_text_1 = "I feel lost in a dark forest."
        ```
    *   Analyze the text:
        ```python
        analysis_1 = analyzer.analyze(input_text_1)
        print("--- Analysis 1 ---")
        print(analysis_1)
        ```
    *   *Verification:* Examine the `analysis_1` dictionary. Check if `sentiment`, `concepts` (from NER), and `topics` seem reasonable for the input text.

3.  **Get Current State:**
    *   Retrieve the initial state from the manager:
        ```python
        state_1 = manager.get_state()
        print("--- State 1 (Initial) ---")
        print(state_1)
        ```
    *   *Verification:* Confirm it matches the default state (e.g., `theme: neutral_void`).

4.  **Map to Commands (with State):**
    *   Generate commands using the analysis and the current state:
        ```python
        commands_1 = mapper.map_analysis_to_commands(analysis_1, state_1)
        print("--- Commands 1 ---")
        print(f"Generated command string: {getattr(mapper, '_last_generated_text', 'N/A')} ") # Hypothetical attribute to see raw output
        print(commands_1)
        ```
    *   *Verification:* Observe the raw command string printed by the mapper (if logging added) and the final parsed `DreamscapeCommand` objects. Do they seem like a plausible response to the input and initial state?

5.  **Apply Commands:**
    *   Apply the generated commands to the manager:
        ```python
        manager.apply_commands(commands_1)
        ```
    *   *Verification:* Check console logs from `DreamscapeManager` for confirmation of applied actions.

6.  **Check State Update:**
    *   Retrieve the updated state:
        ```python
        state_2 = manager.get_state()
        print("--- State 2 (After Commands 1) ---")
        print(state_2)
        ```
    *   *Verification:* Compare `state_2` to `state_1`. Does it reflect the changes expected from `commands_1` (e.g., updated theme, lighting, new entities)? Is `last_command_info` updated?

7.  **Second Input & Analysis (Testing Feedback):**
    *   Define a second input that might relate to or contrast with the first:
        ```python
        input_text_2 = "A friendly light appears ahead."
        ```
    *   Analyze the second input:
        ```python
        analysis_2 = analyzer.analyze(input_text_2)
        print("--- Analysis 2 ---")
        print(analysis_2)
        ```
    *   *Verification:* Check the analysis results for the second input.

8.  **Map to Commands (with Updated State):**
    *   Generate commands using the second analysis and the *updated* state (`state_2`):
        ```python
        commands_2 = mapper.map_analysis_to_commands(analysis_2, state_2)
        print("--- Commands 2 ---")
        print(f"Generated command string: {getattr(mapper, '_last_generated_text', 'N/A')} ")
        print(commands_2)
        ```
    *   *Verification:* Critically observe the generated commands. Does the T5 model appear to have considered `state_2` (e.g., the `current_theme` being `forest`, the `last_action` potentially being `change_soundscape`) when generating `commands_2`? Or does it seem solely based on `analysis_2`? (Note: With limited training data, the influence of state might be minimal or nonsensical).

9.  **Apply Commands & Check State:**
    *   Apply the second set of commands:
        ```python
        manager.apply_commands(commands_2)
        ```
    *   Check the final state:
        ```python
        state_3 = manager.get_state()
        print("--- State 3 (After Commands 2) ---")
        print(state_3)
        ```
    *   *Verification:* Confirm the state reflects the changes from `commands_2`.

10. **Provide Feedback (Optional/Conceptual):**
    *   If the generated commands (`commands_1` or `commands_2`) were particularly good or bad, simulate providing feedback:
        ```python
        # Example positive feedback on the last command
        last_command_info = state_3.get("last_command_info")
        manager.record_user_feedback('positive', last_command_info)

        # Example negative feedback
        # manager.record_user_feedback('negative', last_command_info)
        ```
    *   *Verification:* Check logs for the feedback message.

11. **Repeat:** Continue with different inputs and scenarios, observing how the system reacts and how state potentially influences command generation.

## Expected Behavior vs. Reality

*   **Expected:** With sufficient training data, the `ConceptMapper` model should learn to generate command sequences that are not only relevant to the immediate input text (`analysis_results`) but also coherent with the `current_state`. For example, inputting "make it warmer" might result in different commands depending on whether the `current_theme` is `beach` or `arctic`.
*   **Reality (with limited data):** The fine-tuned T5 model likely hasn't learned strong correlations between the state inputs and the target commands. Its output might be repetitive or primarily driven by the input text and sentiment, with minimal influence from the `current_theme` or `last_action`. More data and potentially adjustments to the input prompt format for the T5 model are needed for true stateful adaptation.