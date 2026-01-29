import os
import json
import logging
import numpy as np
from typing import Dict, List, Any
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    EvalPrediction
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

import evaluate
import torch

# Updated paths based on your file structure
MODEL_NAME: str = "bert-base-cased"
HUMAN_DATA_PATH: str = "data/human_text_pre2022.jsonl"
AI_DATA_PATH: str = "data/ai_generated_text.jsonl"
OUTPUT_DIR: str = "results"
MAX_LENGTH: int = 256
BATCH_SIZE: int = 4
EPOCHS: int = 3
LEARNING_RATE: float = 2e-4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data() -> Dataset:
    texts: List[str] = []
    labels: List[int] = []
    
    # Check if files exist
    if not os.path.exists(HUMAN_DATA_PATH):
        logger.warning(f"Human data file not found at: {HUMAN_DATA_PATH}")
    
    if not os.path.exists(AI_DATA_PATH):
        logger.warning(f"AI data file not found at: {AI_DATA_PATH}")

    if os.path.exists(HUMAN_DATA_PATH):
        with open(HUMAN_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    texts.append(record["text"])
                    labels.append(0) # 0 for Human
                except Exception as e:
                    logger.warning(f"Skipping invalid line in human data: {e}")
                
    if os.path.exists(AI_DATA_PATH):
        with open(AI_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    texts.append(record["ai_text"])
                    labels.append(1) # 1 for AI
                except Exception as e:
                    logger.warning(f"Skipping invalid line in AI data: {e}")
    
    if not texts:
        raise ValueError("No data loaded! Please check your data paths.")
                
    # Create dataset
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels) # type: ignore


def main():
    dataset: Dataset = load_data()
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2) # type: ignore
    
    logger.info(f"Training on {len(dataset['train'])} samples, Testing on {len(dataset['test'])} samples")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    tokenized_datasets = dataset.map(tokenize, batched=True)
    
    # Define label mappings for better inference usability
    id2label = {0: "Human", 1: "AI"}
    label2id = {"Human": 0, "AI": 1}

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "value"]
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch", # Renamed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        # use_mps_device=torch.backends.mps.is_available(), # Auto-detected usually, can leave if needed
        dataloader_pin_memory=False,
        logging_steps=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],   # type: ignore
        eval_dataset=tokenized_datasets["test"],     # type: ignore
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    logger.info("starting training")
    trainer.train()
    
    logger.info("Saving model...")
    # This saves both the adapter config and weights
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    
    print(f"Training Complete! Model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")
    
    
if __name__ == "__main__":
    main()