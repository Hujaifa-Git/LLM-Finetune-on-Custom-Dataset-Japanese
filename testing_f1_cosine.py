from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from collections import Counter
import numpy as np
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

def compute_cosine(list1, list2):
    max_length = max(len(list1), len(list2))
    list1_padded = np.pad(list1, (0, max_length - len(list1)))
    list2_padded = np.pad(list2, (0, max_length - len(list2)))

    # Convert padded lists to NumPy arrays
    array1 = np.array(list1_padded).reshape(1, -1)  # Reshape to a 2D array (1 row)
    array2 = np.array(list2_padded).reshape(1, -1)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(array1, array2)

    # Extract the similarity value from the matrix
    similarity_value = cosine_sim[0, 0]
    return similarity_value
   

model_name = 'cyberagent/open-calm-7b'
t5_tokenizer = AutoTokenizer.from_pretrained(model_name)

ground_truth = """代替エネルギー源は、気候変動の最も深刻な影響を軽減するために絶対に必要です。世界が化石燃料から風力、太陽光、水力などのより持続可能なエネルギー源に移行することで、大気中の温室効果ガスの量を減らし、地球のさらなる温暖化を防止することができます。さらに、再生可能エネルギー源や代替エネルギー源を使用することで、数千の雇用を創出し、健全で安定した経済を作り出すことができます。代替エネルギー源への投資は、気候危機に効果的に対処するために必要な重要な一歩であり、合理的な決定です。"""

prediction = """代替エネルギーのメリットは、気候変動による影響を緩和するだけでなく、汚染を減らし、資源を保護することにも役立ちます。さらに、代替エネルギーはコストを削減し、二酸化炭素やその他の汚染物質の排出を減らすことができます"""

ground_truth_tokens = t5_tokenizer(ground_truth, truncation=True, max_length=CUTOFF_LEN, padding=False)['input_ids']
prediction_tokens = t5_tokenizer(prediction, truncation=True, max_length=CUTOFF_LEN, padding=False)['input_ids']
print('F1:',compute_f1(prediction_tokens, ground_truth_tokens))
print('Cosine:',compute_cosine(prediction_tokens, ground_truth_tokens))



