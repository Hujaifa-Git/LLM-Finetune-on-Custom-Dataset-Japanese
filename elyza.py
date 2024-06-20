from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset

model_name = "cyberagent/open-calm-7b"
dataset = 'fujiki/japanese_alpaca_data'

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto')
data = load_dataset(dataset)


def tokenize(prompt, tokenizer):
    result = tokenizer(prompt+tokenizer.eos_token, truncation=False, max_length=10000, padding=False)
    
    return {
        'input_ids' : result['input_ids'],
        'attention_mask' : result['attention_mask'],
    }
    

def generate_prompt(data_point):
    if data_point['input']:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


# exit()

train_val = data['train'].train_test_split(test_size=2000, shuffle=True, seed=42)
train_data = train_val['train']
val_data = train_val['test']
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

max_train, max_val = 0, 0

for data in train_data:
    max_train = max(max_train, len(data['input_ids']))
    
for data in val_data:
    max_val = max(max_val, len(data['input_ids']))
    
print(max_train, max_val)

