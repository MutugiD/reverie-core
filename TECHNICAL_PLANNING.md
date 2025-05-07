# Reverie: Technical Planning Document

This document outlines the core technical components, initial training objectives, data handling strategies, and input preprocessing steps for the Reverie AI.

## 1. Core Functional Modules

Based on `REVERIE_DEFINITION.md` and `ARCHITECTURE_RESEARCH.md`, the initial implementation will require these key Python modules/classes:

*   **Input Processor (`input_processor.py`):**
    *   Receives raw text input from the dreamer.
    *   Performs preprocessing steps (cleaning, tokenization).
    *   Feeds processed text into the Text Analyzer.
*   **Text Analyzer (`text_analyzer.py`):**
    *   Utilizes a pre-trained Transformer model (from Hugging Face, e.g., BERT-based for analysis or T5/BART for sequence-to-sequence mapping).
    *   Performs NLP tasks: NER, topic modeling, sentiment analysis, concept extraction.
    *   Outputs a structured representation of the key thoughts, concepts, entities, and emotional tone.
    *   Implements logic for prioritizing concepts based on significance/repetition.
*   **Concept Mapper (`concept_mapper.py`):**
    *   Takes the structured output from the Text Analyzer.
    *   Translates high-priority concepts into actionable commands or parameters for the Dreamscape Manager.
    *   This might involve rule-based logic initially, potentially evolving into a learned mapping (e.g., using a fine-tuned seq2seq model).
    *   Handles the "implicit clarification prompt" logic by generating ambiguous commands if input analysis is uncertain.
*   **Dreamscape Manager (`dreamscape_manager.py`):**
    *   Maintains the current state of the Dreamscape (entities, environment parameters, etc.).
    *   Receives commands from the Concept Mapper.
    *   Executes changes to the Dreamscape state (e.g., modifying environment variables, triggering procedural generation updates, adding/removing symbolic elements).
    *   Interfaces with any rendering/output systems (TBD - initially, might just log state changes).
*   **State Storage (`state_storage.py` / Database):**
    *   Handles persistence of the Dreamscape state between sessions.
    *   Stores dreamer profile information, including learned personalization/mappings.
    *   (Initial Plan: Use simple file storage like JSON or Pickle for the state? Consider a database like SQLite or a vector database for more complex recall later).

## 2. Initial Training Goals

*   **Text Analyzer Fine-tuning:** Fine-tune a pre-trained Transformer model (e.g., BERT) on relevant datasets (sentiment, emotion, topic classification) to improve performance on domain-specific language (dreams, subconscious thoughts).
*   **Concept Extraction:** Develop and evaluate methods (potentially fine-tuning NER models or using topic modeling) to reliably extract key concepts, themes, and entities from dreamer input.
*   **Concept-to-Symbol Mapping (Initial):** Create an initial, likely rule-based or simple dictionary-based, mapping between common extracted concepts (e.g., "fear", "joy", "forest", "water") and corresponding symbolic Dreamscape actions/parameters. This provides a baseline for the `ConceptMapper`.
*   *(Future Goal):* Train a sequence-to-sequence model (e.g., T5) to directly translate processed dreamer input into structured Dreamscape commands, learning the mapping end-to-end.

## 3. Data Storage Plan (Initial)

*   **Dreamscape State:** Represent the current state as a Python dictionary or a custom class structure. Store attributes like environment theme, key parameters (lighting, weather), list of core entities/symbols, etc.
*   **Persistence:** Initially serialize the state object to a JSON file for simplicity between sessions.
*   **Dreamer Profile:** Store basic settings and any learned personalization data (e.g., custom concept mappings) in a separate JSON file per dreamer.
*   *(Future Considerations):* Explore databases (SQLite for relational data, vector databases like ChromaDB or Pinecone for semantic recall/memory) as complexity grows.

## 4. Input Preprocessing Steps

*   **Input:** Raw text string from the dreamer.
*   **Cleaning:**
    *   Convert to lowercase.
    *   Remove punctuation (or handle it strategically based on model needs).
    *   Handle extraneous whitespace.
*   **Tokenization:** Use the tokenizer corresponding to the chosen Hugging Face Transformer model.
*   **Output:** Token IDs suitable for input into the Transformer model.