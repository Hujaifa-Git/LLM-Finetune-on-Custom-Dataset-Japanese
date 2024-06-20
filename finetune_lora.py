from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import transformers
import config as ctg

model_name = ctg.model_name
# dataset = 'kunishou/databricks-dolly-15k-ja'
dataset = ctg.dataset
peft_name = ctg.peft_name
output_dir = ctg.output_dir
CUTOFF_LEN = ctg.CUTOFF_LEN
VAL_SET_SIZE = ctg.VAL_SET_SIZE
# idx = 5

eval_steps = ctg.eval_steps
save_steps = ctg.save_steps
logging_steps = ctg.logging_steps
EPOCHS = ctg.EPOCHS
LE = ctg.LE

R=ctg.R
ALPHA=ctg.ALPHA
DROPOUT=ctg.DROPOUT

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer.special_tokens_map)
print(f'eos_token : {tokenizer.eos_token}, {tokenizer.eos_token_id}')
print(f'bos_token : {tokenizer.bos_token}, {tokenizer.bos_token_id}')
print(f'unk_token : {tokenizer.unk_token}, {tokenizer.unk_token_id}')
print(f'pad_token : {tokenizer.pad_token}, {tokenizer.pad_token_id}')
# exit()

def tokenize(prompt, tokenizer):
    result = tokenizer(prompt+tokenizer.eos_token, truncation=True, max_length=CUTOFF_LEN, padding=False)
    
    return {
        'input_ids' : result['input_ids'],
        'attention_mask' : result['attention_mask'],
    }
    
# print(tokenize('hi there', tokenizer))
# # exit()

data = load_dataset(dataset)

# print(data['train'][idx])
# exit()

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

# print(generate_prompt(data['train'][idx]))
# # exit()

train_val = data['train'].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
train_val.save_to_disk(ctg.dataset_dir)

# dataset = load_from_disk('/media/nsl3090-3/hdd1/hujaifa/Open_Calm_7b_Japanese_Alpaca/Dataset/JP_Alpaca')
# exit()

train_data = dataset['train']
val_data = dataset['test']
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto')

print(model)


lora_config = LoraConfig(
    r=R,
    lora_alpha=ALPHA,
    target_modules=[ctg.target_modules],
    lora_dropout=DROPOUT,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# exit()
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
      num_train_epochs=EPOCHS,
      learning_rate=LE,
      logging_steps=logging_steps,
      evaluation_strategy='steps',
      save_strategy='steps',
      eval_steps=eval_steps,
      save_steps=save_steps,
      output_dir=output_dir,
      report_to=['tensorboard'],
      save_total_limit=5,
      push_to_hub=False,
      auto_find_batch_size=True,
      load_best_model_at_end=True,
      metric_for_best_model='eval_loss'    
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False
trainer.train(resume_from_checkpoint=True)
model.config.use_cache = True

trainer.model.save_pretrained(peft_name)





