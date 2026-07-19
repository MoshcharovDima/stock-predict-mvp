# src/train.py — обучение гибридной модели на свежих данных
# Запуск:  python -m src.train --ticker AAPL
#          python -m src.train --all
# Артефакты (artifacts/{ticker}/):
#   model.pt, scaler_X.joblib, scaler_y.joblib,
#   emb_scaler.joblib, pca.joblib, meta.json, test_predictions.csv

import argparse
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ARTIFACT_DIR, BATCH_SIZE, DATA_DIR, DEVICE, EMBEDDING_FEATURES, END_DATE,
    EPOCHS, LOOK_BACK, LR, N_PCA_COMPONENTS, PATIENCE, PRICE_FEATURES,
    PRICE_START, SEED, TARGET_COL, TICKERS, TRAIN_FRAC, TRAIN_START, VAL_FRAC,
)
from src.features.dataset import build_raw_dataset
from src.models.hybrid_lstm import LSTMEmbeddings, create_sequences, split_streams

FEATURE_COLS = PRICE_FEATURES + EMBEDDING_FEATURES


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / Directional Accuracy — в реальных ценах."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred)
                                / (np.abs(y_true) + 1e-8))) * 100)
    da = float(np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100)
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4), 'DA': round(da, 2)}


def build_training_frame(ticker: str) -> tuple:
    """
    Собирает датафрейм признаков и обучает препроцессинг эмбеддингов.
    PCA и StandardScaler обучаются ТОЛЬКО на train-части (без утечки).
    """
    df_prices, sent_daily, emb_daily = build_raw_dataset(
        ticker, PRICE_START, END_DATE, with_target=True)

    # обрезаем прогрев индикаторов и последнюю строку (NaN в target)
    mask = df_prices.index >= pd.Timestamp(TRAIN_START)
    df_prices  = df_prices[mask].dropna(subset=[TARGET_COL])
    sent_daily = sent_daily.loc[df_prices.index]
    emb_matrix = emb_daily.loc[df_prices.index].values

    n = len(df_prices)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    # эмбеддинги: StandardScaler + PCA (fit на train) 
    emb_scaler = StandardScaler().fit(emb_matrix[:n_train])
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
    pca.fit(emb_scaler.transform(emb_matrix[:n_train]))
    emb_pc = pca.transform(emb_scaler.transform(emb_matrix))
    print(f'  PCA объясняет {pca.explained_variance_ratio_.sum()*100:.1f}% '
          f'дисперсии эмбеддингов (train)')

    df = df_prices.copy()
    for i in range(N_PCA_COMPONENTS):
        df[f'emb_pc{i+1}'] = emb_pc[:, i]
    df = pd.concat([df, sent_daily], axis=1)   # сентимент — для анализа/UI

    return df, n_train, n_val, emb_scaler, pca


def make_tensors(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    xp, xe = split_streams(X, FEATURE_COLS)
    return TensorDataset(torch.tensor(xp, dtype=torch.float32),
                         torch.tensor(xe, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))


def train_ticker(ticker: str) -> dict:
    set_seed()
    print(f'\n{"="*52}\n  Обучение: {ticker}\n{"="*52}')

    df, n_train, n_val, emb_scaler, pca = build_training_frame(ticker)
    n = len(df)
    print(f'  Датасет: {n} торговых дней '
          f'({df.index[0].date()} → {df.index[-1].date()})')
    print(f'  Сплит: train={n_train}, val={n_val}, test={n - n_train - n_val}')

    # сохраняем датасет для воспроизводимости
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, f'dataset_{ticker}.csv'))

    # нормализация (fit на train)
    X_raw = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values

    scaler_X = MinMaxScaler().fit(X_raw[:n_train])
    scaler_y = MinMaxScaler().fit(y_raw[:n_train].reshape(-1, 1))
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    # последовательности на всём ряду, сплит по позиции таргета
    # (контекст val/test — это прошлое, утечки нет; строки не теряются)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOK_BACK)
    pos = np.arange(LOOK_BACK, n)          # индекс строки-таргета
    tr = pos < n_train
    va = (pos >= n_train) & (pos < n_train + n_val)
    te = pos >= n_train + n_val

    loaders = {
        'train': DataLoader(make_tensors(X_seq[tr], y_seq[tr]),
                            batch_size=BATCH_SIZE, shuffle=True),
        'val':   DataLoader(make_tensors(X_seq[va], y_seq[va]),
                            batch_size=BATCH_SIZE),
    }

    # модель
    model = LSTMEmbeddings().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float('inf'), None, 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xp, xe, yb in loaders['train']:
            xp, xe, yb = xp.to(DEVICE), xe.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xp, xe), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= tr.sum()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xp, xe, yb in loaders['val']:
                xp, xe, yb = xp.to(DEVICE), xe.to(DEVICE), yb.to(DEVICE)
                va_loss += loss_fn(model(xp, xe), yb).item() * len(yb)
        va_loss /= va.sum()
        history.append({'epoch': epoch, 'train': float(tr_loss),
                        'val': float(va_loss)})

        marker = ''
        if va_loss < best_val:
            best_val, wait = va_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = ' ★'
        else:
            wait += 1

        if epoch % 5 == 0 or marker:
            print(f'  epoch {epoch:3d} | train {tr_loss:.5f} '
                  f'| val {va_loss:.5f}{marker}')
        if wait >= PATIENCE:
            print(f'  Early stopping (epoch {epoch})')
            break

    model.load_state_dict(best_state)

    #тест
    model.eval()
    xp, xe = split_streams(X_seq[te], FEATURE_COLS)
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(xp, dtype=torch.float32).to(DEVICE),
            torch.tensor(xe, dtype=torch.float32).to(DEVICE),
        ).cpu().numpy()

    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_seq[te].reshape(-1, 1)).flatten()
    metrics = calc_metrics(y_true, y_pred)
    print(f'\n  Метрики на тесте: {metrics}')

    # наивный бенчмарк: «завтра = сегодня»
    close_today = df['Close'].values[pos[te]]
    naive = calc_metrics(y_true, close_today)
    print(f'  Наивный прогноз : {naive}')

    # сохранение артефактов
    out = os.path.join(ARTIFACT_DIR, ticker)
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    joblib.dump(scaler_X,   os.path.join(out, 'scaler_X.joblib'))
    joblib.dump(scaler_y,   os.path.join(out, 'scaler_y.joblib'))
    joblib.dump(emb_scaler, os.path.join(out, 'emb_scaler.joblib'))
    joblib.dump(pca,        os.path.join(out, 'pca.joblib'))

    test_dates = df.index[pos[te]]
    pd.DataFrame({'date': test_dates, 'y_true': y_true,
                  'y_pred': y_pred}).to_csv(
        os.path.join(out, 'test_predictions.csv'), index=False)

    meta = {
        'ticker': ticker,
        'trained_at': pd.Timestamp.now().isoformat(timespec='seconds'),
        'period': [str(df.index[0].date()), str(df.index[-1].date())],
        'n_days': int(n),
        'split': {'train': int(n_train), 'val': int(n_val),
                  'test': int(n - n_train - n_val)},
        'feature_cols': FEATURE_COLS,
        'look_back': LOOK_BACK,
        'metrics_test': metrics,
        'metrics_naive': naive,
        'pca_explained_var': round(
            float(pca.explained_variance_ratio_.sum()), 4),
        'history': history,
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'  💾 Артефакты сохранены в {out}')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    tickers = TICKERS if (args.all or not args.ticker) else [args.ticker]
    results = {t: train_ticker(t) for t in tickers}

    print(f'\n{"="*52}\n  ИТОГ\n{"="*52}')
    for t, m in results.items():
        print(f'  {t}: {m}')
