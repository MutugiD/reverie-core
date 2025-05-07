# Reverie: Advanced Features Design

This document outlines design considerations for more advanced features like the feedback loop and dataset creation/fine-tuning strategy.

## 1. Feedback Loop Design

**Goal:** Allow Reverie to learn and adapt its responses based on the effectiveness and reception of its environmental changes, moving beyond simple text input -> immediate change.

**Mechanism Ideas:**

*   **Implicit Feedback via State Analysis:**
    *   The `DreamscapeManager` state could be fed back into the `TextAnalyzer` or `ConceptMapper`.
    *   *Example:* If the Dreamscape state contains conflicting elements (e.g., theme="calm_beach", last_command="spawn_entity(type=demon)") or rapidly oscillates, the system could trigger a clarification prompt or dampen future responses.
    *   *Example:* Analyze sequences of states. If the dreamer repeatedly inputs text leading to negative sentiment *after* Reverie introduces certain elements, Reverie could learn to avoid or modify those elements in response to similar future inputs.
*   **Explicit Feedback (Future):**
    *   Allow the dreamer to provide direct feedback (e.g., "I like this change", "This feels wrong").
    *   This feedback would directly adjust the weights or mappings in the `ConceptMapper` or potentially fine-tune the underlying models (requires Reinforcement Learning from Human Feedback - RLHF - setup).
*   **Memory Integration (via Mnesis):**
    *   A future Mnesis component could provide historical context.
    *   *Example:* If the dreamer previously reacted positively to "forest" themes, Reverie could favor forest-related commands when similar positive concepts arise later.

**Initial Implementation Focus:**

*   Start with simple state feedback: The `ConceptMapper` could receive the `DreamscapeManager.get_state()` and adjust its command generation based on the current theme or recent commands to promote coherence.
*   Log interaction sequences (Input -> Analysis -> Commands -> State Change) for future analysis and potential offline training/fine-tuning.

## 2. Dataset Strategy

**Goal:** Obtain or create data to train/fine-tune Reverie's components, especially the `TextAnalyzer` and `ConceptMapper`, for the specific domain of subconscious exploration and dream-like text.

**Approach:**

1.  **Leverage Pre-trained Models:** Start with robust base models (e.g., BERT, DistilBERT, T5) pre-trained on large text corpora (like those used for models on Hugging Face Hub). This provides general language understanding.
2.  **Domain Adaptation (General):**
    *   Fine-tune the base NLP models (`TextAnalyzer`) on datasets related to emotions, creative writing, dialogue, or potentially anonymized psychological text (if ethically sourced).
    *   *Datasets:* `emotion`, `commonsense_dialogues`, Project Gutenberg text, potentially abstract/metaphor datasets.
    *   *Goal:* Adapt the models to the *style* and *vocabulary* common in dream descriptions or subconscious explorations.
3.  **Task-Specific Fine-tuning (Core Task):**
    *   **Sentiment:** Fine-tune a sequence classification head on emotion/sentiment datasets (as done with `distilbert-base-uncased-finetuned-sst-2-english`).
    *   **Concept Extraction:** Fine-tune models for NER or use topic modeling techniques on relevant texts to identify key dream elements/concepts.
    *   **Concept-to-Command Mapping:** This is the most custom part. Requires creating a dataset mapping example text inputs (or extracted concepts/sentiments) to desired `DreamscapeCommand` sequences.
        *   *Creation:* Could start manually, defining ideal responses for representative inputs. Could potentially use LLMs (like GPT-4) to help generate initial mappings based on rules, which are then curated.
        *   *Format:* Pairs of `(Input Text/AnalysisResult) -> [DreamscapeCommand, ...] `
        *   *Model:* Could fine-tune a sequence-to-sequence model (like T5 or BART) on this custom dataset to learn the mapping directly.
4.  **Personalization Data:**
    *   Store per-dreamer interaction logs.
    *   Periodically (offline or online, if feasible) fine-tune the individual dreamer's version of the `ConceptMapper` or even parts of the `TextAnalyzer` based on their specific inputs and implicit/explicit feedback, leading to personalized Dreamscapes.

**Initial Implementation Focus:**

*   Utilize the existing pre-trained sentiment model.
*   Focus on implementing a robust *placeholder* for concept extraction in `TextAnalyzer` and the dictionary-based `ConceptMapper`.
*   Set up logging to gather interaction data suitable for *future* creation of the concept-to-command mapping dataset.