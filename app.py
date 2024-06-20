from flask import Flask, render_template, request
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import numpy as np
from datasets import load_from_disk
from inference_lora_single import generate

    

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def save_text():
    user_ins = request.form['user_ins']
    user_inp = request.form['user_inp']
    output_text = generate(user_ins, user_inp)
    return render_template('index.html', output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
