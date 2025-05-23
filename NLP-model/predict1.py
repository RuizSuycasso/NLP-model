# src/predict1.py

import os
import sys
import torch
import re
from model import BiLSTM_CRF
from utils import load_pickle, tokenize_ocr, extract_entities

# Thiết lập đường dẫn
THIS_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
sys.path.append(THIS_DIR)

# Nạp vocab & label maps
word2idx = load_pickle(os.path.join(PROJECT_DIR, "word2idx.pkl"))
label2idx = load_pickle(os.path.join(PROJECT_DIR, "label2idx.pkl"))
idx2label = {v: k for k, v in label2idx.items()}

# Khởi tạo model và nạp weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(
    vocab_size=len(word2idx), tagset_size=len(label2idx),
    padding_idx=word2idx['<PAD>']
).to(device)
model.load_state_dict(torch.load(
    os.path.join(PROJECT_DIR, "bilstm_crf_best.pt"),
    map_location=device
))
model.eval()

# Hàm encode 1 câu
def encode(tokens, max_len=None):
    length = len(tokens)
    seq_len = max_len or length
    ids = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    mask = [1]*len(ids)
    if len(ids) < seq_len:
        pad = seq_len - len(ids)
        ids += [word2idx['<PAD>']]*pad
        mask += [0]*pad
    else:
        ids = ids[:seq_len]
        mask = mask[:seq_len]
    return torch.LongTensor([ids]).to(device), torch.BoolTensor([mask]).to(device)

# Đọc file OCR và tokenize
ocr_file = os.path.join(DATA_DIR, "Mau4.txt")
sentences = tokenize_ocr(ocr_file, lowercase=True)

# Predict và gom thực thể DRUG
found = []
for tokens in sentences:
    X, M = encode(tokens)
    pred_ids = model(X, mask=M)[0][:len(tokens)]
    pred_labels = [idx2label[i] for i in pred_ids]
    ents = extract_entities(tokens, pred_labels).get('DRUG', [])
    for ent in ents:
        clean = ent.lower()
        if clean not in found:
            found.append(clean)

# In kết quả
print("Danh sách thuốc phát hiện:")
for d in found:
    print(d)
