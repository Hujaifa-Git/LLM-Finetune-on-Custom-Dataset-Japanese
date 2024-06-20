import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import json
import config as ctg

model_name = ctg.model_name
peft_name = ctg.peft_model_path
CUTOFF_LEN = ctg.CUTOFF_LEN
BATCH_SIZE = ctg.BATCH_SIZE
dataset_path = ctg.dataset_dir

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
def generate(data,maxTokens=CUTOFF_LEN):
    predictions = []
    ground_truth = []
    no_batches = len(data) // BATCH_SIZE
    for i in range(no_batches):
        print(f'Infering {i+1}/{no_batches}')
        batch_data = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        
        prompt_batch = [generate_prompt({'instruction':batch_data['instruction'][j],'input':batch_data['input'][j]}) for j in range(BATCH_SIZE)]
        input_ids = tokenizer(prompt_batch, truncation=True, max_length=CUTOFF_LEN, padding=True, return_tensors='pt').to(device)
        input_ids = {
        'input_ids' : input_ids['input_ids'],
        'attention_mask' : input_ids['attention_mask'],
        }
        outputs = model.generate(
        max_new_tokens=maxTokens, #Maximum token of output
        do_sample=True, #Randomple sample 15% of the time from the output
        temperature=0.7, 
        top_p=0.75, 
        top_k=40,         
        no_repeat_ngram_size=2,
        **input_ids
        )
        outputs = outputs.tolist()
    
        output_batch = []
        for output in outputs:
            if tokenizer.eos_token_id in output:
                eos_index = output.index(tokenizer.eos_token_id)
                decoded = tokenizer.decode(output[:eos_index], skip_special_tokens=True)

                sentinel = "### Response:"
                sentinelLoc = decoded.find(sentinel)
                if sentinelLoc >= 0:
                    response = decoded[sentinelLoc+len(sentinel):]
                    if '、' in response: response = response[1:]
                    response = response.strip()
                    
                    output_batch.append(response)
                    # print('Response:::')
                    # print(response)
                                        
                else:
                    print('Warning: Expected prompt template to be emitted.  Ignoring output.')
                    output_batch.append('')
            else:
                print('Warning: no <eos> detected ignoring output')
                output_batch.append('')
                
        predictions += output_batch
        ground_truth += data['output'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        if len(predictions) != len(ground_truth):
            print(f'Not equal Pred:{len(predictions)} GT:{len(ground_truth)} BatchNum: {i+1}')
                
            
    return ground_truth, predictions
        
        

print('Inference Done. Creating Dictionary for Evaluation')
dataset = load_from_disk(dataset_path)
test_data = dataset['test']
ground_truth, predictions = generate(test_data)

# with open('pred.pkl', 'wb') as file:
#     pickle.dump(predictions, file)
    
# with open('gt.pkl', 'wb') as file:
#     pickle.dump(ground_truth, file)
    
print(len(ground_truth), len(predictions))
inference_dict = {
    'Ground Truths' : [],
    'Predictions' : []
}
for i in range(len(ground_truth)):
    inference_dict['Ground Truths'].append(ground_truth[i])
    inference_dict['Predictions'].append(predictions[i])

with open(ctg.batch_inference_json_path, 'w', encoding='utf-8') as f:
    json.dump(inference_dict, f,ensure_ascii=False)
# print(test_data[1])
