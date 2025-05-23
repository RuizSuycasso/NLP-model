# NLP-model/model.py
import torch
import torch.nn as nn
# Import lớp CRF tiêu chuẩn trực tiếp
from torchcrf import CRF # Xóa bỏ alias TorchCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# XÓA BỎ TOÀN BỘ ĐỊNH NGHĨA LỚP CRF TÙY CHỈNH TẠI ĐÂY
# Lớp CRF tùy chỉnh đã gây ra lỗi AttributeError

class BiLSTM_CRF(nn.Module):
    """
    Mô hình BiLSTM + CRF cho nhận diện thực thể:
    - Hỗ trợ packed sequences cho input biến độ dài
    - Dropout giảm overfitting
    - Khởi tạo Xavier cho embedding và FC
    """
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        emb_dim: int = 100,
        hid_dim: int = 128,
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hid_dim, tagset_size)
        # Sử dụng lớp CRF TIÊU CHUẨN từ torchcrf
        # Lớp này mong đợi tên tham số mới: start_transitions, end_transitions, transitions
        self.crf = CRF(num_tags=tagset_size, batch_first=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x_ids, tags=None, mask=None, lengths=None):
        emb = self.embedding(x_ids)
        emb = self.emb_dropout(emb)

        # Note: Although the predict loop doesn't provide lengths,
        # the model's forward still handles packed sequences if provided (e.g., during training).
        # For inference batch=1 without padding, lengths is not strictly needed
        # but keeping this branch doesn't hurt and maintains training compatibility.
        if lengths is not None and lengths.max() > 0: # Add check for empty sequences
            # Ensure lengths is on CPU for pack_padded_sequence
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
             # Handle cases with no lengths or empty sequences
             lstm_out, _ = self.lstm(emb) # LSTM can handle padded sequences directly if batch_first=True and no lengths


        lstm_out = self.lstm_dropout(lstm_out)
        emissions = self.fc(lstm_out)

        if tags is not None:
            # In training mode, calculate loss
            # mask and reduction='mean' are handled by torchcrf.CRF
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # In inference mode, decode the best tag sequence
            # crf.decode returns a list of lists of tag indices (batch_size, sequence_length)
            return self.crf.decode(emissions, mask=mask)