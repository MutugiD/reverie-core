import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import logging
import os
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use LongT5 as the base model
BASE_MODEL_NAME = "google/long-t5-tglobal-base"
# Load from both datasets
OLD_DATASET_FILE = "data/concept_mapping_dataset.jsonl"
LLM_GENERATED_DATASET_FILE = "data/llm_generated_concept_mapping.jsonl"
OUTPUT_DIR = "./models/concept_mapper_longt5_base" # New output directory
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3, # Adjust epochs as needed
    # Reduce batch size due to larger model, use accumulation
    "per_device_train_batch_size": 2, # Was 4 for t5-small
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4, # Effective batch size = 2 * 4 = 8
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 50,
    "evaluation_strategy": "epoch", # Evaluate at the end of each epoch
    "save_strategy": "epoch",       # Save checkpoint at the end of each epoch
    "save_total_limit": 2,          # Keep only the last 2 checkpoints
    "load_best_model_at_end": True, # Load the best model found during training
    "metric_for_best_model": "loss", # Use validation loss to determine the best model
    "greater_is_better": False,
    "predict_with_generate": True, # Needed for Seq2Seq evaluation (e.g., BLEU/ROUGE if added later)
    "fp16": torch.cuda.is_available(), # Enable mixed precision if CUDA is available
    # "push_to_hub": False, # Uncomment to push to Hugging Face Hub
}
MAX_SOURCE_LENGTH = 1024 # Max input length (LongT5 can handle more, but keep reasonable)
MAX_TARGET_LENGTH = 128  # Max output command sequence length

# --- Data Loading and Preprocessing ---

def transform_old_format(example: Dict[str, Any]) -> Dict[str, str]:
    """Constructs the 'input_text' field from the old dataset format."""
    prefix = "map concepts to commands: "
    raw_text = example.get("raw_text", "")
    sentiment = example.get("sentiment", "neutral")
    topics = example.get("topics", [])
    current_theme = example.get("current_theme", "unknown")
    last_action = example.get("last_action", "none")
    target_text = example.get("target_text", "") # Keep target text

    # Ensure topics is a list of strings before joining
    topic_str = ", ".join(map(str, topics)) if topics else "none"

    input_text = (
        f"{prefix}text: {raw_text} | sentiment: {sentiment} | "
        f"topics: {topic_str} | "
        f"current_theme: {current_theme} | last_action: {last_action}"
    )
    return {"input_text": input_text, "target_text": target_text}

def preprocess_function(examples: Dict[str, List[str]], tokenizer) -> Dict[str, List[int]]:
    """Tokenizes the input ('input_text') and target ('target_text') text."""
    inputs = examples["input_text"]
    targets = examples["target_text"]

    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    logging.info("Starting Concept Mapper Training Script")

    # --- Load Model and Tokenizer ---
    logging.info(f"Loading base model and tokenizer: {BASE_MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
        # Ensure pad token is set if needed (important for T5 variants)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            logging.warning(f"Tokenizer for {BASE_MODEL_NAME} has eos_token but no pad_token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # If we modify the tokenizer, we might need to resize model embeddings
            # model.resize_token_embeddings(len(tokenizer))
            # NOTE: Resizing might not be needed if pad_token was already in vocab but just not set.
            #       Let's proceed without resizing first. If errors occur, uncomment the line above.

    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        return

    # --- Load and Prepare Datasets ---
    logging.info("Attempting to load and process datasets...")
    processed_datasets: List[Dataset] = []

    # 1. Load and transform the old dataset
    if os.path.exists(OLD_DATASET_FILE):
        try:
            logging.info(f"Loading old format dataset: {OLD_DATASET_FILE}")
            old_ds = load_dataset("json", data_files=OLD_DATASET_FILE, split='train')
            # Check if it needs transformation (has old columns, lacks 'input_text')
            if "raw_text" in old_ds.column_names and "input_text" not in old_ds.column_names:
                logging.info(f"Transforming {OLD_DATASET_FILE} to new format...")
                transformed_old_ds = old_ds.map(
                    transform_old_format,
                    remove_columns=old_ds.column_names # Remove all old columns
                )
                logging.info(f"Transformed {len(transformed_old_ds)} examples from {OLD_DATASET_FILE}")
                processed_datasets.append(transformed_old_ds)
            elif "input_text" in old_ds.column_names and "target_text" in old_ds.column_names:
                logging.info(f"{OLD_DATASET_FILE} seems to be in the new format already. Using as is.")
                processed_datasets.append(old_ds.select_columns(["input_text", "target_text"]))
            else:
                logging.warning(f"Skipping {OLD_DATASET_FILE}: Does not match old or new expected format.")
        except Exception as e:
            logging.error(f"Failed to load or transform {OLD_DATASET_FILE}: {e}")
    else:
        logging.warning(f"Old dataset file not found: {OLD_DATASET_FILE}. Skipping.")

    # 2. Load the LLM generated dataset (expects new format)
    if os.path.exists(LLM_GENERATED_DATASET_FILE):
        try:
            logging.info(f"Loading LLM generated dataset: {LLM_GENERATED_DATASET_FILE}")
            llm_ds = load_dataset("json", data_files=LLM_GENERATED_DATASET_FILE, split='train')
            # Validate required columns for the new format
            if all(col in llm_ds.column_names for col in ["input_text", "target_text"]):
                logging.info(f"Loaded {len(llm_ds)} examples from {LLM_GENERATED_DATASET_FILE}")
                # Keep only the necessary columns
                processed_datasets.append(llm_ds.select_columns(["input_text", "target_text"]))
            else:
                logging.error(f"LLM generated dataset {LLM_GENERATED_DATASET_FILE} is missing required columns ('input_text', 'target_text'). Skipping.")
        except Exception as e:
            logging.error(f"Failed to load or validate LLM generated dataset {LLM_GENERATED_DATASET_FILE}: {e}. Ensure it is valid JSON Lines.")
    else:
        logging.warning(f"LLM generated dataset file not found: {LLM_GENERATED_DATASET_FILE}. Skipping.")

    # Check if any datasets were successfully processed
    if not processed_datasets:
        logging.error("No valid datasets were loaded or processed. Exiting training.")
        return

    # Concatenate datasets
    if len(processed_datasets) > 1:
        final_dataset = concatenate_datasets(processed_datasets)
        logging.info(f"Concatenated datasets. Total examples: {len(final_dataset)}")
    else:
        final_dataset = processed_datasets[0]
        logging.info(f"Using single processed dataset. Total examples: {len(final_dataset)}")

    # Split into train/validation
    if len(final_dataset) < 10:
         logging.warning("Dataset too small for train/eval split. Using all data for training.")
         split_dataset = DatasetDict({'train': final_dataset, 'test': None})
    else:
        split_dataset = final_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset.get("test") # Use .get() for safety if split failed

    logging.info(f"Final dataset split: Train={len(train_dataset)}, Eval={len(eval_dataset) if eval_dataset else 0}")

    # Preprocess (Tokenize) the final dataset
    logging.info("Tokenizing datasets...")
    try:
        tokenized_train_dataset = train_dataset.map(
            lambda ex: preprocess_function(ex, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_eval_dataset = None
        if eval_dataset:
            tokenized_eval_dataset = eval_dataset.map(
                lambda ex: preprocess_function(ex, tokenizer),
                batched=True,
                remove_columns=eval_dataset.column_names
            )
    except Exception as e:
        logging.error(f"Error during tokenization: {e}", exc_info=True)
        return

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # --- Setup Trainer ---
    logging.info("Setting up Seq2Seq Trainer...")
    training_args_obj = Seq2SeqTrainingArguments(**TRAINING_ARGS)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args_obj,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    logging.info("Starting training...")
    try:
        train_result = trainer.train()
        logging.info("Training finished.")

        # --- Save Model and Metrics ---
        logging.info(f"Saving model and tokenizer to {OUTPUT_DIR}")
        trainer.save_model() # Saves the tokenizer too
        tokenizer.save_pretrained(OUTPUT_DIR) # Explicitly save tokenizer again just in case

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logging.info(f"Training metrics: {metrics}")

        # --- Evaluate (if eval dataset exists) ---
        if tokenized_eval_dataset:
            logging.info("Starting evaluation...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            logging.info(f"Evaluation metrics: {eval_metrics}")

    except Exception as e:
        logging.error(f"An error occurred during training or saving: {e}", exc_info=True)

    logging.info("Training script finished.")

if __name__ == "__main__":
    # Ensure output and log dirs exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if "logging_dir" in TRAINING_ARGS:
         os.makedirs(TRAINING_ARGS["logging_dir"], exist_ok=True)
    main()
