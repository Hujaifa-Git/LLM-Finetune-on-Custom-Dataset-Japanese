model_name = 'cyberagent/open-calm-7b'
# dataset = 'kunishou/databricks-dolly-15k-ja'
dataset = 'fujiki/japanese_alpaca_data'
dataset_dir = 'Dataset/JP_Alpaca'
peft_name = 'lora-calm-7b'
output_dir = 'lora-calm-7b-japanese-alpaca_v1'
CUTOFF_LEN = 1024 #Max 1084
VAL_SET_SIZE = 2000
idx = 5

eval_steps = 200
save_steps = 200
logging_steps = 20
EPOCHS = 100
LE = 3e-4

R=8
ALPHA=16
DROPOUT=0.05
target_modules = 'query_key_value'

peft_model_path = 'lora-calm-7b-japanese-alpaca_v1/checkpoint-164000'

BATCH_SIZE = 16
batch_inference_json_path = 'test_inference_dict_japanese_alpaca_best_train-loss.json'