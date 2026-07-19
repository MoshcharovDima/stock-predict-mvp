# ============================================================
# tests/test_model.py — быстрые проверки модели и пайплайна
# Запуск: python -m pytest tests/ -v  (или python tests/test_model.py)
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import LOOK_BACK, PRICE_FEATURES, EMBEDDING_FEATURES
from src.models.hybrid_lstm import LSTMEmbeddings, create_sequences, split_streams

FEATURE_COLS = PRICE_FEATURES + EMBEDDING_FEATURES


def test_forward_shapes():
    model = LSTMEmbeddings()
    xp = torch.randn(4, LOOK_BACK, len(PRICE_FEATURES))
    xe = torch.randn(4, LOOK_BACK, len(EMBEDDING_FEATURES))
    out = model(xp, xe)
    assert out.shape == (4,), f'Неверная форма выхода: {out.shape}'


def test_create_sequences():
    n, f = 100, len(FEATURE_COLS)
    X, y = np.random.rand(n, f), np.random.rand(n)
    X_seq, y_seq = create_sequences(X, y, LOOK_BACK)
    assert X_seq.shape == (n - LOOK_BACK, LOOK_BACK, f)
    assert y_seq.shape == (n - LOOK_BACK,)
    # таргет i соответствует окну, заканчивающемуся строкой i-1
    assert np.allclose(X_seq[0], X[:LOOK_BACK])
    assert y_seq[0] == y[LOOK_BACK]


def test_split_streams():
    X = np.random.rand(8, LOOK_BACK, len(FEATURE_COLS))
    xp, xe = split_streams(X, FEATURE_COLS)
    assert xp.shape[-1] == len(PRICE_FEATURES)
    assert xe.shape[-1] == len(EMBEDDING_FEATURES)


def test_overfit_tiny_batch():
    """Модель должна суметь переобучиться на крошечной выборке."""
    torch.manual_seed(0)
    model = LSTMEmbeddings()
    xp = torch.randn(8, LOOK_BACK, len(PRICE_FEATURES))
    xe = torch.randn(8, LOOK_BACK, len(EMBEDDING_FEATURES))
    y  = torch.rand(8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    model.train()
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(xp, xe), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.1, 'Loss не падает — проблема в модели'


if __name__ == '__main__':
    test_forward_shapes()
    test_create_sequences()
    test_split_streams()
    test_overfit_tiny_batch()
    print('✅ Все тесты пройдены')
