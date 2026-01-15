import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from datasets import Dataset

def train_liquid_gliner():
    # 1. Configuration
    model_name = "tabachain/liquid-mamba2-hybrid-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")

    # Custom Config for GLiNER
    # We need to match the backbone's hidden size (1024 as per my wrapper)
    config = GLiNERConfig(
        model_name=model_name,
        model_type="span_level", # Standard GLiNER span level
        hidden_size=1024,        # Output size of the backbone wrapper
        max_width=12,
        dropout=0.1,
        labels_encoder=None,     # Not used for single encoder
        name="liquid-gliner"
    )
    
    # We can explicitly set the vocab size if known, to avoid resize issues
    config.vocab_size = 64402

    # 2. Initialize Model
    # This will trigger my modified Transformer code in encoder.py
    # which detects "liquid-mamba2-hybrid" and loads the custom backbone.
    print("Initializing GLiNER model with Liquid Mamba 2 backbone...")
    try:
        model = GLiNER(config)
        model.to(device)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise e
    
    # 3. Dummy Dataset
    # GLiNER expects data with "tokenized_text" and "ner" (list of [start, end, label])
    data = [
        {"tokenized_text": ["Liquid", "AI", "is", "cool"], "ner": [[0, 1, "ORG"]]},
        {"tokenized_text": ["Tokyo", "is", "a", "city"], "ner": [[0, 0, "LOC"]]},
        {"tokenized_text": ["I", "love", "Python"], "ner": [[2, 2, "LANG"]]},
    ] * 10
    
    dataset = Dataset.from_list(data)
    
    # 4. Data Collator
    collator = DataCollatorWithPadding(model.config) 
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="liquid_gliner_checkpoints",
        learning_rate=1e-5,
        weight_decay=0.01,
        others_lr=1e-4, 
        others_weight_decay=0.01,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=5,
        use_cpu=not torch.cuda.is_available(),
    )
    
    # 6. Train
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=collator
    )
    
    trainer.train()
    print("Training finished!")

if __name__ == "__main__":
    train_liquid_gliner()
