from typing import Dict, List, Any, Optional

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import spacy

from .input_processor import DEFAULT_MODEL_NAME, load_tokenizer

# Specific models for tasks
DEFAULT_SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_NER_MODEL_NAME = "microsoft/deberta-v3-base-finetuned-conll2003-english"
DEFAULT_ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

# Predefined candidate topics for zero-shot classification
CANDIDATE_TOPICS = [
    "fear", "anxiety", "joy", "calm", "sadness", "anger", "surprise", "curiosity",
    "nature", "urban", "liminal_space", "abstract",
    "memory", "dream", "escape", "pursuit", "discovery", "secrets", "connection", "isolation",
    "family", "work", "relationships",
    "animals", "objects", "body_sensation"
]

class TextAnalyzer:
    """Analyzes tokenized text input using Transformer models."""

    def __init__(
        self,
        base_model_name: str = DEFAULT_MODEL_NAME,
        sentiment_model_name: str = DEFAULT_SENTIMENT_MODEL_NAME,
        ner_model_name: str = DEFAULT_NER_MODEL_NAME,
        zero_shot_model_name: str = DEFAULT_ZERO_SHOT_MODEL_NAME,
        load_spacy: bool = True
    ):
        """Initializes the TextAnalyzer with base, sentiment, NER, and Zero-Shot models.

        Args:
            base_model_name: Name of the base HF model (embeddings).
            sentiment_model_name: Name of the sentiment classification model.
            ner_model_name: Name of the token classification model for NER.
            zero_shot_model_name: Name of the zero-shot classification model for topics.
            load_spacy: Option to load spacy for advanced NLP tasks.
        """
        self.base_model_name = base_model_name
        self.sentiment_model_name = sentiment_model_name
        self.ner_model_name = ner_model_name
        self.zero_shot_model_name = zero_shot_model_name
        self.candidate_topics = CANDIDATE_TOPICS
        self.nlp = None
        if load_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                # Add experimental coref if available (requires separate install/model)
                # Check if component factory exists before trying to add
                if "coref" in spacy.registry.factories:
                    # Replace with actual coref component name if using spacy-experimental etc.
                    # self.nlp.add_pipe("coref", config={"alias": "coref"})
                    print("Experimental coref component found but not added automatically.")
                elif "neuralcoref" in spacy.registry.factories:
                     # Example: Using the older neuralcoref library if installed
                     # import neuralcoref
                     # neuralcoref.add_to_pipe(self.nlp)
                     print("Neuralcoref component found but not added automatically.")
                else:
                     print("No known coreference resolution component found in spaCy registry.")
                print(f"spaCy model '{self.nlp.meta['name']}' loaded for advanced NLP tasks.")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found.")
                print("Download it using: python -m spacy download en_core_web_sm")
                print("Advanced NLP features (coreference, relations) will be disabled.")
            except ImportError:
                 print("Warning: spaCy library not installed (`pip install spacy`). Advanced NLP features disabled.")
            except Exception as e:
                 print(f"An unexpected error occurred loading spaCy: {e}")

        try:
            # Load base model
            self.base_model: PreTrainedModel = AutoModel.from_pretrained(base_model_name)
            self.base_tokenizer: PreTrainedTokenizer = load_tokenizer(base_model_name)
            self.base_model.eval()

            # Load sentiment model
            self.sentiment_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                sentiment_model_name
            )
            self.sentiment_tokenizer: PreTrainedTokenizer = load_tokenizer(
                sentiment_model_name
            )
            self.sentiment_model.eval()

            # Load NER pipeline
            self.ner_pipeline: Pipeline = pipeline(
                "ner",
                model=ner_model_name,
                tokenizer=ner_model_name,
                aggregation_strategy="simple"
            )

            # Load Zero-Shot Classification pipeline for topics
            self.topic_pipeline: Pipeline = pipeline(
                "zero-shot-classification",
                model=zero_shot_model_name
            )

        except OSError as e:
            print(
                f"Error loading models ('{base_model_name}', '{sentiment_model_name}', '{ner_model_name}', or '{zero_shot_model_name}'): {e}"
            )
            raise
        except Exception as e:
             print(f"An unexpected error occurred during initialization: {e}")
             raise

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """Performs analysis: embeddings, sentiment, NER, topics, coref (placeholder)."""
        analysis_results = {"raw_input_text": raw_text}

        # --- Tokenize for Base Model (Embeddings) ---
        from .input_processor import tokenize_input
        tokenized_base = tokenize_input(raw_text, self.base_tokenizer)
        input_ids_base = torch.tensor([tokenized_base["input_ids"]])
        attention_mask_base = torch.tensor([tokenized_base["attention_mask"]])

        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids_base, attention_mask=attention_mask_base
            )
        analysis_results["last_hidden_state"] = base_outputs.last_hidden_state
        analysis_results["pooler_output"] = getattr(base_outputs, 'pooler_output', None)

        # --- Sentiment Analysis ---
        tokenized_sentiment = tokenize_input(raw_text, self.sentiment_tokenizer)
        input_ids_sentiment = torch.tensor([tokenized_sentiment["input_ids"]])
        attention_mask_sentiment = torch.tensor([tokenized_sentiment["attention_mask"]])

        with torch.no_grad():
            sentiment_outputs = self.sentiment_model(
                input_ids=input_ids_sentiment, attention_mask=attention_mask_sentiment
            )
        logits = sentiment_outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        analysis_results["sentiment"] = self.sentiment_model.config.id2label[predicted_class_id]
        analysis_results["sentiment_score"] = probabilities[0, predicted_class_id].item()

        # --- NER Analysis ---
        try:
            ner_results = self.ner_pipeline(raw_text)
            analysis_results["entities"] = ner_results
            analysis_results["concepts"] = list(set([entity["word"].lower() for entity in ner_results]))
        except Exception as e:
            print(f"Error during NER processing: {e}")
            analysis_results["entities"] = []
            analysis_results["concepts"] = []

        # --- Topic Classification (Zero-Shot) ---
        try:
            topic_threshold = 0.5
            topic_results = self.topic_pipeline(raw_text, self.candidate_topics, multi_label=True)
            relevant_topics = [
                {"topic": label, "score": score}
                for label, score in zip(topic_results["labels"], topic_results["scores"])
                if score >= topic_threshold
            ]
            analysis_results["topics"] = relevant_topics
        except Exception as e:
            print(f"Error during Zero-Shot Topic processing: {e}")
            analysis_results["topics"] = []

        # --- Advanced NLP ---
        doc = None
        if self.nlp and raw_text:
            try:
                doc = self.nlp(raw_text)
            except Exception as e:
                print(f"Error processing text with spaCy: {e}")
                doc = None # Ensure doc is None if spacy fails

        # Resolve Coreferences (Placeholder implementation)
        coref_clusters = self._extract_coreferences(doc)
        analysis_results["coreferences"] = coref_clusters
        # Placeholder for text with resolved references
        analysis_results["resolved_text"] = self._resolve_text_with_coref(raw_text, coref_clusters)

        # Extract Relations (Placeholder implementation)
        analysis_results["relations"] = self._extract_relations(doc)

        return analysis_results

    # --- Placeholder Methods for Advanced NLP ---
    def _extract_coreferences(self, doc: Optional[spacy.tokens.Doc]) -> List[List[str]]:
        """Placeholder: Extracts coreference clusters if available."""
        clusters = []
        # Example check for neuralcoref style extension
        if doc and hasattr(doc._, 'coref_clusters'):
            print("Coreference clusters found via doc._.coref_clusters")
            for cluster in doc._.coref_clusters:
                # Convert spaCy spans to strings
                mentions = [mention.text for mention in cluster.mentions]
                clusters.append(mentions)
        # Add checks for other coref implementations here if needed
        if clusters:
             print(f"Extracted Coref Clusters: {clusters}")
        return clusters

    def _resolve_text_with_coref(self, text: str, clusters: List[List[str]]) -> str:
        """Placeholder: Returns original text. Future: Replace pronouns with main mentions."""
        # TODO: Implement logic to replace pronouns (e.g., he, she, it, they)
        # based on the extracted coreference clusters.
        # Example: "Alice went to the store. She bought milk." -> "Alice went to the store. Alice bought milk."
        # This is non-trivial and requires careful handling of grammar and context.
        return text # Return original text for now

    def _extract_relations(self, doc: Optional[spacy.tokens.Doc]) -> List[Any]:
        """Placeholder for relation extraction."""
        # Requires a specific relation extraction model/component added to the spacy pipeline
        if doc: # and doc has relation extraction component
            # relations = extract_relations_from_doc(doc)
            # Process relations...
            pass
        return [] # Return empty list for now

# Example Usage updated
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    print(f"Analyzer loaded with base model: {analyzer.base_model_name}")
    print(f"Analyzer loaded with sentiment model: {analyzer.sentiment_model_name}")
    print(f"Analyzer loaded with NER model: {analyzer.ner_model_name}")
    print(f"Analyzer loaded with Zero-Shot model: {analyzer.zero_shot_model_name}")

    example_texts = [
        "I dreamt I was walking through a dark forest near Berlin when suddenly I saw a glowing fox.",
        "My friend Alice felt pure joy watching the sunset over the Pacific Ocean.",
        "The meeting with Mr. Smith about the Acme Corp project was surprisingly calm.",
        "Running endlessly down a dark tunnel, feeling chased."
    ]

    for example_text in example_texts:
        print(f"\n--- Analyzing: '{example_text}' ---")
        analysis = analyzer.analyze(example_text)

        print(f"Analysis Results:")
        print(f"  Sentiment: {analysis['sentiment']} (Score: {analysis['sentiment_score']:.4f})")
        print(f"  Concepts (NER): {analysis['concepts']}")
        print(f"  Topics (Zero-Shot): {analysis['topics']}")