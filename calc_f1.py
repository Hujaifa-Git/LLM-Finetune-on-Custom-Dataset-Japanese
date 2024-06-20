from transformers import AutoTokenizer
import torch
from collections import Counter
CUTOFF_LEN = 1024

def compute_f1(pred_tokens, truth_tokens):
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    # print('gt', truth_tokens)
    # print('pred', pred_tokens)
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    truth_dict = dict(Counter(truth_tokens))
    pred_dict = dict(Counter(pred_tokens))
    common_tokens = 0
    for k, v in truth_dict.items():
        if k in pred_dict:
            common_tokens += min(truth_dict[k], pred_dict[k])
            
    if common_tokens == 0:
        return 0
    
    prec = common_tokens / len(pred_tokens)
    rec = common_tokens / len(truth_tokens)
    # print('prec recall', prec, rec)
    return 2 * (prec * rec) / (prec + rec)

def tokenize_sentence(arg):
    encoded_arg = t5_tokenizer(arg)
    return t5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'cyberagent/open-calm-7b'
t5_tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == '__main__':
    import json, csv
    json_path = 'test_inference_dict_japanese_alpaca_best_train-loss.json'
    json_data = json.load(open(json_path))
    gt_list, pred_list = json_data['Ground Truths'], json_data['Predictions']
    f1_score = []
    for i in range(len(gt_list)):
        gt, pred = gt_list[i], pred_list[i]
        gt_tokens = t5_tokenizer(gt, truncation=True, max_length=CUTOFF_LEN, padding=False)['input_ids']
        pred_tokens = t5_tokenizer(pred, truncation=True, max_length=CUTOFF_LEN, padding=False)['input_ids']
        f1 = compute_f1(pred_tokens, gt_tokens)
        f1_score.append([gt, pred, f1])
        # break
    # exit()
    with open('test_inference_dict_japanese_alpaca_best_train-loss.csv', 'w') as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerows(f1_score)

    