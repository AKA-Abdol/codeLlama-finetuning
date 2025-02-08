import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model_name = "meta-llama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token="",
    quantization_config=config
)

model = PeftModel.from_pretrained(model, 'python-code-llama/best_model').to(device)

# tuning tokenizer using less memory
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def generate(prompt):
    model_input = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(device)
    
    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=512)[0], skip_special_tokens=True))
    
def generate_prompt(data):
    prompt = f"""{data['system'].strip()}
### Instruction:
{data['instruciton'].strip()}
### Output:
"""
    return prompt

if __name__ == '__main__':
    generate(generate_prompt({'system': input('system:'), 'instruction': input('instruction:')}))