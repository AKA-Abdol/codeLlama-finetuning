import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


dataset = load_dataset("jtatman/python-code-dataset-500k", split="train")
train_dataset = dataset["train"].shuffle(seed=42).select(range(10000))
eval_dataset = dataset["train"].shuffle(seed=42).select(range(1000))

model_name = "meta-llama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# tuning tokenizer using less memory
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(examples):
    encoded = tokenizer(examples["text"], truncation=True, max_length=512, padding=False, return_tensors=None)
    encoded['labels'] = encoded['input_ids'].copy()
    return encoded
    
def generate_and_tokenize_prompt(data):
    prompt = f"""{data['system'].strip()}
### Instruction:
{data['instruciton'].strip()}
### Output:
{data['output'].strip()}
"""
    return tokenize(prompt)

train_dataset = train_dataset.map(generate_and_tokenize_prompt, remove_columns=["text"])
eval_dataset = eval_dataset.map(generate_and_tokenize_prompt, remove_columns=["text"])

model.train()
model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./python-code-llama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=3e-4,
    fp16=True,
    save_total_limit=2,
    report_to="none",
    group_by_length=True,
    load_best_model_at_end=True,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./python-code-llama/best_model")
tokenizer.save_pretrained("./python-code-llama/best_model")
