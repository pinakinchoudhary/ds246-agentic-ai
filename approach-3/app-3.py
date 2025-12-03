import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# silencing the noise
warnings.filterwarnings('ignore')

# Quick hardware check
if torch.cuda.is_available():
    print(f"Running on: {torch.cuda.get_device_name(0)}")
else:
    print("Heads up: Running on CPU. This might be slow.")

# ---------------------------------------------------------
# 1. Data Loading & EDA
# ---------------------------------------------------------

# Grab the parquet files
train_df = pd.read_parquet('train.parquet')
query_pool = pd.read_parquet('query_pool.parquet')

# Just taking a quick look at how the prefixes map to queries
prefix_query_map = train_df.groupby('prefix')['query'].apply(list).to_dict()

# Let's check the distribution. 
# If we have too few queries per prefix, the model might struggle.
queries_per_prefix = train_df.groupby('prefix').size()

plt.figure(figsize=(10, 6))
queries_per_prefix.hist(bins=50, edgecolor='black')
plt.title('Queries per Prefix Distribution')
plt.show()

# ---------------------------------------------------------
# 2. Model Setup (GPT-2)
# ---------------------------------------------------------

# Sticking with GPT-2 Small for now. It's old but lightweight and fast enough for testing.
model_name = "gpt2"
print(f"Spinning up {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT-2 doesn't have a pad token by default, so we use EOS
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # FP16 to save memory
    device_map="auto"
)

# ---------------------------------------------------------
# 3. Data Formatting
# ---------------------------------------------------------

def create_training_prompt(prefix, queries, max_queries=10):
    """
    Formats the data into an instruction style prompt.
    Input: "sho"
    Output: "Prefix: sho\nCompletions: shoes, shorts, shoulder bag"
    """
    queries_str = ", ".join(queries[:max_queries])
    return f"Prefix: {prefix}\nCompletions: {queries_str}"

training_texts = []

# Formatting the raw data into something the LLM can understand
for prefix, queries in tqdm(prefix_query_map.items(), desc="Formatting data"):
    # Dedupe and sort by frequency if possible (simple list here)
    unique_queries = list(set(queries))
    training_texts.append(create_training_prompt(prefix, unique_queries))

# Standard split - keeping some data aside to check if we're overfitting
train_texts, val_texts = train_test_split(training_texts, test_size=0.15, random_state=42)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )

# Convert to HF Dataset objects
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

# Tokenize everything
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ---------------------------------------------------------
# 4. LoRA Configuration
# ---------------------------------------------------------

# We're using LoRA here because fine-tuning the whole model is overkill and expensive.
# This injects small trainable rank decomposition matrices into the attention layers.
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    target_modules=["c_attn", "c_proj"], # Specific to GPT-2 architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 
# You should see < 1% trainable params here, which is what we want.

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./gpt2-lora-autocomplete",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, # Simulating a larger batch size
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    save_strategy="steps",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Uncomment this to actually kick off the training
# trainer.train()

# ---------------------------------------------------------
# 6. Inference & Eval Helper Functions
# ---------------------------------------------------------

def generate_completions(prefix, model, tokenizer, max_new_tokens=30):
    """
    Runs the model inference. 
    Note: Sampling is turned on (do_sample=True) so results might vary.
    """
    prompt = f"Prefix: {prefix}\nCompletions:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=5, # Generate a few options to pick from
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Just grab the part after our prompt
        if "Completions:" in text:
            completion = text.split("Completions:")[-1].strip()
            results.append(completion)
    return results

def evaluate_quality(generated_list, actual_pool):
    """
    Checks if the model is hallucinating or actually generating 
    valid items from our inventory (query pool).
    """
    valid_cnt = 0
    pool_set = set(actual_pool['query'].str.lower().str.strip())
    
    for item in generated_list:
        if item.lower().strip() in pool_set:
            valid_cnt += 1
            
    return {
        "total": len(generated_list),
        "valid_in_pool": valid_cnt,
        "valid_pct": (valid_cnt / len(generated_list)) * 100 if generated_list else 0
    }

# ---------------------------------------------------------
# 7. Quick Smoke Test
# ---------------------------------------------------------

test_prefixes = ['sho', 'red dr', 'baby']

print("\n--- Running Smoke Test ---")
for prefix in test_prefixes:
    print(f"\nInput: {prefix}")
    # This will run on the untrained/loaded model unless you ran trainer.train() above
    comps = generate_completions(prefix, model, tokenizer) 
    for c in comps:
        print(f" -> {c}")