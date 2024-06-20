import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'cyberagent/open-calm-7b'
peft_name = 'lora-calm-7b-japanese-alpaca_v1/checkpoint-164000'
CUTOFF_LEN = 1024

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = PeftModel.from_pretrained(
    model, 
    peft_name, 
    device_map="auto"
)


model.eval()

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:"""

device = "cuda" if torch.cuda.is_available() else "cpu"
def generate(instruction,input=None,maxTokens=CUTOFF_LEN):

    prompt = generate_prompt({'instruction':instruction,'input':input})
    input_ids = tokenizer(prompt, truncation=False, max_length=CUTOFF_LEN, padding=False, return_tensors='pt').to(device)
    input_ids = {
        'input_ids' : input_ids['input_ids'],
        'attention_mask' : input_ids['attention_mask'],
    }
    
    outputs = model.generate(
        max_new_tokens=maxTokens, 
        do_sample=True,
        temperature=0.7,#0.7 
        top_p=0.75, 
        top_k=40,         
        no_repeat_ngram_size=2,
        **input_ids
    )
    outputs = outputs[0].tolist()
    

    # print(outputs)
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])
        # print(decoded)

        sentinel = "### Response:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            response = decoded[sentinelLoc+len(sentinel):]
            if '、' in response: response = response[1:]
            response = response.strip()
            print(response)
            return response
        else:
            print('Warning: Expected prompt template to be emitted.  Ignoring output.')
    else:
        print('Warning: no <eos> detected ignoring output')
        
# while True:
#     instruction = input('Enter Instruction in Japanese:::')
#     if instruction == '0':
#         break
#     input_ = input('Enter Input in Japanese::')
#     if input == '': input=None        
#     generate(instruction=instruction, input=input_)