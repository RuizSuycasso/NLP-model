# src/predict_conll.py

import os
import sys
import torch

# --- Thiết lập đường dẫn dự án ---
THIS_DIR    = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
CKPT_DIR    = os.path.join(PROJECT_DIR, "checkpoints")
sys.path.append(THIS_DIR)
# ----------------------------------

from model import BiLSTM_CRF
from utils import load_pickle, read_data, extract_entities

# 1. Nạp vocab & nhãn
word2idx  = load_pickle(os.path.join(PROJECT_DIR, "word2idx.pkl"))
label2idx = load_pickle(os.path.join(PROJECT_DIR, "label2idx.pkl"))
idx2label = {idx: lab for lab, idx in label2idx.items()}

# 2. Khởi tạo model & load checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(
    vocab_size=len(word2idx),
    tagset_size=len(label2idx),
    padding_idx=word2idx['<PAD>']
).to(device)

ckpt = os.path.join(CKPT_DIR, "bilstm_crf_best.pt")
if not os.path.isfile(ckpt):
    raise FileNotFoundError(f"Không tìm thấy checkpoint tại {ckpt}")
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# 3. Hàm encode
def encode(tokens):
    ids  = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    mask = [1] * len(ids)
    return (
        torch.LongTensor([ids]).to(device),
        torch.BoolTensor([mask]).to(device)
    )

# 4. Đọc test CoNLL
sentences, _ = read_data(os.path.join(DATA_DIR, "test3.txt"), lowercase=False)

# 5. Predict và gom thực thể
found = []
for tokens in sentences:
    X, M = encode(tokens)
    pred_ids = model(X, mask=M)[0]
    pred_labels = [idx2label[i] for i in pred_ids[:len(tokens)]]

    ents = extract_entities(tokens, pred_labels).get('DRUG', [])
    for d in ents:
        if d not in found:
            found.append(d)

# 6. In kết quả
print("Danh sách thuốc phát hiện:")
for d in found:
    print(d)
