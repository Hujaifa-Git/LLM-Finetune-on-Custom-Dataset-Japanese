# Project Name

Welcome to a Japanese LLM created using base model Open-Calm and finetuned on Japanese Alpaca Dataset. You can use this model to do any Language related task
## Table of Contents

- Iroduction
- Installation
- Finetuning
- Inference Single
- Inference Batch
- Inference Demo
- Demo

## Introduction

This LLM Apllication was build to create a intrustion based Japanese LLM. After openning the application you can ask any question or give it any language related task and the model will reply in Japanese


## Installation

To get started, you need to set up the Conda environment.

### Step 1: Install Conda

If you haven't already, install Conda from the [official Anaconda website](https://www.anaconda.com/products/distribution) and follow the installation instructions.

### Step 2: Finetune

To Finetune the model you first need to change the 'config.py' file as necessary. You can change the base model, dataset or any other training configuration as needed. The 'model_name' and 'dataset' should be huggingface model and dataset

```python
model_name = 'cyberagent/open-calm-7b'
dataset = 'fujiki/japanese_alpaca_data'
dataset_dir = 'Dataset/JP_Alpaca'
peft_name = 'lora-calm-7b'
output_dir = 'lora-calm-7b-japanese-alpaca_v1'
CUTOFF_LEN = 1024 
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
```
After that you can simply run the 'finetune_lora.py' file. 
```bash
python finetune_lora.py
```
You can also see logs of training using tensorboard using the below command.
```bash
tensorboard --logdir lora-calm-7b-japanese-alpaca_v1/runs
```

## Inference Single

To do a single Inference you have to give the correct peft model path in 'config.py' and run 'inference_lora_single.py' file. After that give the necesary inputs to get reponse from LLM

```python
peft_model_path = 'lora-calm-7b-japanese-alpaca_v1/checkpoint-164000'
```
```bash
python inference_lora_single.py
```

## Inference Batch

To do a batch Inference on Test dataset of any custom dataset you have to select BATCH_SIZE and the path where you'll save the inference json in 'config.py'

```python
BATCH_SIZE = 16
batch_inference_json_path = 'test_inference_dict_japanese_alpaca_best_train-loss.json'
```
 and run 'inference_lora_batch.py' file. After some time this script will create a json file containing grounth truths and predictions so you can use it for evaluation
```bash
python inference_lora_batch.py'
```


## Inference
To run the app you just hape to run the following command,
```bash
python app.py
```

## Demo

<video width="800" height="360" controls>
  <source src="Demo/Finetune.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

