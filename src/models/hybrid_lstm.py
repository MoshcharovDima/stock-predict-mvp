# src/models/hybrid_lstm.py — гибридная модель LSTM + Embeddings
# Лучшая архитектура по результатам дипломного исследования
# (сравнение с ARIMA, Random Forest, LSTM base, LSTM+Sentiment,
#  LSTM+Attention на rolling-window валидации):
#   Поток 1: цены (batch, 20, 27)      → LSTM(64) → Dropout(0.2)
#   Поток 2: эмбеддинги (batch, 20, 5) → LSTM(16)
#   Конкатенация [64 ⊕ 16 = 80] → Dense(32)+ReLU → Dense(1)

import numpy as np
import torch
import torch.nn as nn

from config import (
    PRICE_FEATURES, EMBEDDING_FEATURES,
    HIDDEN_PRICE, HIDDEN_EMB, DENSE_HIDDEN, DROPOUT,
)


class LSTMEmbeddings(nn.Module):
    """Двухпоточная LSTM с конкатенацией скрытых состояний."""

    def __init__(self,
                 price_size: int = len(PRICE_FEATURES),
                 emb_size: int = len(EMBEDDING_FEATURES),
                 hidden_price: int = HIDDEN_PRICE,
                 hidden_emb: int = HIDDEN_EMB,
                 dense_hidden: int = DENSE_HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.price_size = price_size
        self.emb_size   = emb_size

        self.lstm_price = nn.LSTM(price_size, hidden_price, batch_first=True)
        self.dropout    = nn.Dropout(dropout)
        self.lstm_emb   = nn.LSTM(emb_size, hidden_emb, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_price + hidden_emb, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 1),
        )

    def forward(self, x_price: torch.Tensor,
                x_emb: torch.Tensor) -> torch.Tensor:
        out_p, _ = self.lstm_price(x_price)
        out_p    = self.dropout(out_p[:, -1, :])       # (batch, 64)
        out_e, _ = self.lstm_emb(x_emb)
        out_e    = out_e[:, -1, :]                     # (batch, 16)
        z = torch.cat([out_p, out_e], dim=1)           # (batch, 80)
        return self.head(z).squeeze(-1)


def split_streams(X: np.ndarray, feature_cols: list) -> tuple:
    """
    Делит массив (N, look_back, F) на два входа модели
    по спискам признаков из config.
    """
    p_idx = [feature_cols.index(f) for f in PRICE_FEATURES
             if f in feature_cols]
    e_idx = [feature_cols.index(f) for f in EMBEDDING_FEATURES
             if f in feature_cols]
    return X[:, :, p_idx], X[:, :, e_idx]


def create_sequences(X: np.ndarray, y: np.ndarray,
                     look_back: int) -> tuple:
    """(N, F) → (N-look_back, look_back, F)."""
    X_seq, y_seq = [], []
    for i in range(look_back, len(X)):
        X_seq.append(X[i - look_back:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)
