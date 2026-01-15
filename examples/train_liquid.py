import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import SpanDataCollator
from datasets import load_dataset
import logging

def convert_to_gliner_format(example):
    # If already in GLiNER format, return as is
    if "tokenized_text" in example and "ner" in example:
        return example
    
    # Check for standard HF NER format (tokens + ner_tags)
    if "tokens" in example and "ner_tags" in example:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        # We need to convert tags (ints) to spans (start, end, label)
        # Assuming we have access to the features to get label names, 
        # but inside map function it's harder. 
        # Usually datasets have 'features' metadata. 
        # For now, let's assume if it fails we might need manual mapping if names aren't in features.
        # Ideally, we processed this globaly before map.
        # BUT for this script, let's assume the user has a GLiNER-compatible dataset 
        # OR standard Bio/IOB with features.
        
        return {"tokenized_text": tokens, "ner": []} # Placeholder if simple conversion logic isn't here

    # Fallback/Pass-through
    return example

# 1. Configuration
model_name = "tabachain/liquid-mamba2-hybrid-v1"
dataset_id = "tabachain/synthetic-pii-ja"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Custom Config for GLiNER
config = GLiNERConfig(
    model_name=model_name,
    # model_type argument removed as it is a property
    hidden_size=1024,
    max_width=12,
    dropout=0.1,
    labels_encoder=None,
    name="liquid-gliner"
)
config.vocab_size = 64402

# 2. Initialize Model
print("Initializing GLiNER model with Liquid Mamba 2 backbone...")
try:
    model = GLiNER(config)
    model.to(device)
    print("Model initialized successfully!")
except Exception as e:
    print(f"Failed to initialize model: {e}")
    raise e

# 3. Load Dataset
print(f"Loading dataset: {dataset_id}")
dataset = load_dataset(dataset_id)

# Flatten if it's a DatasetDict with 'train'
if "train" in dataset:
    train_ds = dataset["train"]
    if "test" in dataset:
        eval_ds = dataset["test"]
    else:
        # Split train if no test
        split = train_ds.train_test_split(test_size=0.1)
        train_ds = split["train"]
        eval_ds = split["test"]
else:
    # If it's just a Dataset object (unlikely directly from load_dataset unless split is specified)
    train_ds = dataset
    split = train_ds.train_test_split(test_size=0.1)
    train_ds = split["train"]
    eval_ds = split["test"]

print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

# 4. Data Conversion (Simple Check)
# Check column names of the first example
example = train_ds[0]
if "tokenized_text" not in example and "tokens" in example:
    print("Detected 'tokens' column. Renaming to 'tokenized_text' for GLiNER compatibility...")
    train_ds = train_ds.rename_column("tokens", "tokenized_text")
    eval_ds = eval_ds.rename_column("tokens", "tokenized_text")
    
# Check NER format
# GLiNER expects 'ner' to be a list of [start, end, label]
# If the dataset has 'ner_tags' (integers), we ideally need to convert them.
# However, without knowing the label map, we can't do it blindly.
# We will assume for now the user's dataset 'tabachain/synthetic-pii-ja' 
# might already be compatible or they will handle the mapping if it fails.
# Just in case, if there is 'ner' column, we trust it.

if "ner" not in train_ds.column_names:
    print("WARNING: 'ner' column not found. Training might fail if 'ner' data is missing or named differently.")
    print(f"Available columns: {train_ds.column_names}")

# 5. Data Collator
collator = SpanDataCollator(model.config, data_processor=model.data_processor) 

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="liquid_gliner_checkpoints",
    learning_rate=1e-5,
    weight_decay=0.01,
    others_lr=1e-4, 
    others_weight_decay=0.01,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    use_cpu=not torch.cuda.is_available(),
    report_to="none"
)

# 7. Train
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=collator
)

trainer.train()
print("Training finished!")

