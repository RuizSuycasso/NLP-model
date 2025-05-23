# src/train.py

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from model import BiLSTM_CRF
from utils import read_data, build_maps, save_pickle

# Thiết lập đường dẫn
THIS_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
sys.path.append(THIS_DIR)

# 1. Đọc dữ liệu
train_sents, train_labels = read_data(os.path.join(DATA_DIR, 'train4.txt'), lowercase=True)
valid_sents, valid_labels = read_data(os.path.join(DATA_DIR, 'valid4.txt'), lowercase=True)

# 2. Tạo maps và lưu
word2idx, label2idx = build_maps(train_sents + valid_sents, train_labels + valid_labels, min_freq=1)
save_pickle(word2idx, os.path.join(PROJECT_DIR, 'word2idx.pkl'))
save_pickle(label2idx, os.path.join(PROJECT_DIR, 'label2idx.pkl'))

# 3. Dataset
class DrugDataset(Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels
    def __len__(self):
        return len(self.sents)
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

# 4. Collate để pad và trả về lengths
def collate_fn(batch):
    sents, labs = zip(*batch)
    lengths = [len(s) for s in sents]
    max_len = max(lengths)
    X, Y, M = [], [], []
    for sent, lab in zip(sents, labs):
        ids = [word2idx.get(w, word2idx['<UNK>']) for w in sent]
        tags = [label2idx[t] for t in lab]
        mask = [1]*len(ids)
        pad_len = max_len - len(ids)
        ids += [word2idx['<PAD>']]*pad_len
        tags += [label2idx['O']]*pad_len
        mask += [0]*pad_len
        X.append(ids); Y.append(tags); M.append(mask)
    return (
        torch.LongTensor(X),
        torch.LongTensor(Y),
        torch.BoolTensor(M),
        torch.LongTensor(lengths)
    )

# 5. DataLoader
train_loader = DataLoader(DrugDataset(train_sents, train_labels),
                          batch_size=16, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(DrugDataset(valid_sents, valid_labels),
                          batch_size=16, shuffle=False, collate_fn=collate_fn)

# 6. Khởi tạo model, optimizer, scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM_CRF(
    vocab_size=len(word2idx),
    tagset_size=len(label2idx),
    emb_dim=100, hid_dim=128, dropout=0.5,
    padding_idx=word2idx['<PAD>']
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# 7. Vòng train/val và lưu checkpoint tốt nhất
best_val = float('inf')
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for Xb, Yb, Mb, lengths in train_loader:
        Xb, Yb, Mb, lengths = Xb.to(device), Yb.to(device), Mb.to(device), lengths.to(device)
        loss = model(Xb, tags=Yb, mask=Mb, lengths=lengths)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    avg_train = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, Yb, Mb, lengths in valid_loader:
            Xb, Yb, Mb, lengths = Xb.to(device), Yb.to(device), Mb.to(device), lengths.to(device)
            val_loss += model(Xb, tags=Yb, mask=Mb, lengths=lengths).item()
    avg_val = val_loss / len(valid_loader)
    scheduler.step(avg_val)

    print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), os.path.join(PROJECT_DIR, 'bilstm_crf_best.pt'))

print("Hoàn thành huấn luyện.")
