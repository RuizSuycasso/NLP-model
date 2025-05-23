# src/utils.py

import re
import pickle
from typing import List

def smart_tokenize(line: str) -> List[str]:
    """
    Tách token nâng cao:
    - \w+ bắt chữ/số
    - [^\w\s] bắt dấu câu riêng
    """
    return [t for t in re.findall(r"\w+|[^\w\s]", line, flags=re.UNICODE) if t.strip()]

def read_data(filepath: str, lowercase: bool = False):
    """
    Đọc file CoNLL mỗi dòng 'token tag', câu cách bằng dòng trống.
    Trả về:
      sentences: List[List[str]]
      labels:    List[List[str]]
    """
    sentences, labels = [], []
    sent, labs = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(labs)
                    sent, labs = [], []
                continue
            parts = line.split()
            token, tag = parts[0], parts[-1]
            if lowercase:
                token = token.lower()
            sent.append(token)
            labs.append(tag)
    if sent:
        sentences.append(sent)
        labels.append(labs)
    return sentences, labels

def build_maps(sentences, labels, min_freq: int = 1):
    """
    Tạo word2idx, label2idx:
    - word2idx: PAD=0, UNK=1
    - label2idx: 'O'=0, còn lại tự enumerate
    """
    from collections import Counter
    word_cnt = Counter(w for s in sentences for w in s)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, cnt in word_cnt.items():
        if cnt >= min_freq:
            word2idx[w] = len(word2idx)
    label_set = sorted({l for labs in labels for l in labs})
    label2idx = {l: i for i, l in enumerate(label_set)}
    # đảm bảo 'O'=0
    if 'O' in label2idx and label2idx['O'] != 0:
        other = [k for k,v in label2idx.items() if v==0][0]
        label2idx[other], label2idx['O'] = label2idx['O'], label2idx[other]
    return word2idx, label2idx

def save_pickle(obj, path: str):
    """Lưu obj ra file .pkl"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    """Đọc obj từ file .pkl"""
    with open(path, "rb") as f:
        return pickle.load(f)

def tokenize_ocr(filepath: str, lowercase: bool = False):
    """
    Đọc file văn bản OCR, tách token với smart_tokenize().
    Trả về List[List[str]] các câu.
    """
    sentences = []
    with open(filepath, encoding="utf-8") as f:
        for ln in f:
            line = ln.strip()
            if not line:
                continue
            toks = smart_tokenize(line)
            if lowercase:
                toks = [t.lower() for t in toks]
            if toks:
                sentences.append(toks)
    return sentences

def extract_entities(tokens: List[str], labels: List[str]):
    """
    Gom thực thể theo BIO:
    Trả về dict {entity_type: [entity_str]}
    """
    entities = {}
    cur, cur_type = [], None
    for tok, lab in zip(tokens, labels):
        if lab == 'O':
            if cur:
                ent = ' '.join(cur)
                entities.setdefault(cur_type, []).append(ent)
                cur, cur_type = [], None
        else:
            tag, ent_type = lab.split('-', 1)
            if tag == 'B' or ent_type != cur_type:
                if cur:
                    ent = ' '.join(cur)
                    entities.setdefault(cur_type, []).append(ent)
                cur, cur_type = [tok], ent_type
            else:  # I-*
                cur.append(tok)
    if cur:
        ent = ' '.join(cur)
        entities.setdefault(cur_type, []).append(ent)
    return entities
