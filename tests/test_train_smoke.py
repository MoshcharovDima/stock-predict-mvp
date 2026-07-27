# ============================================================
# tests/test_train_smoke.py — smoke-тест пайплайна обучения
# Работает ОФФЛАЙН: данные (цены, новости, эмбеддинги) синтетические.
# Проверяет: build_training_frame → обучение → сохранение артефактов
#            → загрузка артефактов → прогноз.
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import src.train as train_mod
from config import (
    ARTIFACT_DIR, LOOK_BACK, PRICE_FEATURES, EMBEDDING_FEATURES,
    SENTIMENT_FEATURES, STATIONARY_FEATURES, TARGET_COL,
)
from src.data.prices import add_indicators, make_stationary
from src.predict import load_artifacts
from src.models.hybrid_lstm import split_streams

TICKER = '_SMOKE'


def make_synthetic_data(n_days: int = 320, seed: int = 0):
    """Синтетический random-walk рынок + случайные новостные признаки."""
    rng = np.random.default_rng(seed)
    # даты заканчиваются сегодня, чтобы попасть в TRAIN_START из config
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(),
                           periods=n_days)
    n_days = len(dates)

    close = 100 + np.cumsum(rng.normal(0.05, 1.2, n_days))
    close = np.maximum(close, 5)
    high  = close + rng.uniform(0.1, 2.0, n_days)
    low   = close - rng.uniform(0.1, 2.0, n_days)
    opn   = low + rng.uniform(0, 1, n_days) * (high - low)
    vol   = rng.integers(1_000_000, 5_000_000, n_days).astype(float)

    df = pd.DataFrame({'Open': opn, 'High': high, 'Low': low,
                       'Close': close, 'Volume': vol}, index=dates)
    df.index.name = 'date'
    df = add_indicators(df)
    if STATIONARY_FEATURES:
        df = make_stationary(df)
        df[TARGET_COL] = np.log(df['raw_Close'].shift(-1) / df['raw_Close'])
    else:
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[f'raw_{c}'] = df[c]
        df[TARGET_COL] = df['Close'].shift(-1)
    df = df.dropna(subset=PRICE_FEATURES)

    sent = pd.DataFrame(
        rng.uniform(0, 1, (len(df), 3)),
        columns=['sent_pos', 'sent_neg', 'sent_neu'], index=df.index)
    sent = sent.div(sent.sum(axis=1), axis=0)
    sent['sentiment_score'] = sent['sent_pos'] - sent['sent_neg']

    emb = pd.DataFrame(rng.normal(0, 1, (len(df), 768)), index=df.index)
    return df, sent[SENTIMENT_FEATURES], emb


def test_train_smoke(tmp_run: bool = True):
    # подменяем загрузку данных на синтетику
    train_mod.build_raw_dataset = (
        lambda ticker, start, end, with_target=True, show_progress=True:
        make_synthetic_data())
    # ускоряем обучение
    train_mod.EPOCHS, train_mod.PATIENCE = 6, 3

    metrics = train_mod.train_ticker(TICKER, seeds=[0, 1])
    assert set(metrics) == {'MAE', 'RMSE', 'MAPE', 'DA'}

    # Синтетика — random walk, предсказать его нельзя, поэтому наивный
    # прогноз близок к оптимуму и модель обязана быть с ним одного
    # порядка. Проигрыш в разы означает, что прогноз упирается
    # в границы train-диапазона.
    import json
    meta = json.load(open(os.path.join(ARTIFACT_DIR, TICKER, 'meta.json')))
    naive_mae = meta['metrics_naive']['MAE']
    assert metrics['MAE'] < naive_mae * 2.0, (
        f'Модель в {metrics["MAE"] / naive_mae:.1f} раза хуже наивного '
        f'прогноза — вероятен срез экстраполяции или утечка масштаба')

    # артефакты загружаются и делают прогноз
    art = load_artifacts(TICKER)
    X = np.random.rand(1, LOOK_BACK,
                       len(PRICE_FEATURES) + len(EMBEDDING_FEATURES))
    xp, xe = split_streams(X, PRICE_FEATURES + EMBEDDING_FEATURES)
    with torch.no_grad():
        out = art['model'](torch.tensor(xp, dtype=torch.float32),
                           torch.tensor(xe, dtype=torch.float32))
    assert out.shape == (1,)
    print('✅ Smoke-тест пайплайна пройден')

    # чистим за собой
    import shutil
    shutil.rmtree(os.path.join(ARTIFACT_DIR, TICKER), ignore_errors=True)


if __name__ == '__main__':
    test_train_smoke()
