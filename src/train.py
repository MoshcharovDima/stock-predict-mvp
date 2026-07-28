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
    PRICE_START, SEED, STATIONARY_FEATURES, TARGET_COL, TICKERS, TRAIN_FRAC,
    TRAIN_START, VAL_FRAC,
)
from src.features.dataset import build_raw_dataset
from src.models.hybrid_lstm import LSTMEmbeddings, create_sequences, split_streams

FEATURE_COLS = PRICE_FEATURES + EMBEDDING_FEATURES


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def price_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE — в реальных ценах (долларах)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred)
                                / (np.abs(y_true) + 1e-8))) * 100)
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4)}


def directional_accuracy(true_change: np.ndarray,
                         pred_change: np.ndarray) -> float:
    """
    Доля дней, когда угадано направление движения от сегодняшнего закрытия
    к завтрашнему: sign(pred - Close_T) против sign(true - Close_T).
    """
    return float(np.mean((true_change > 0) == (pred_change > 0)) * 100)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 close_today: np.ndarray) -> dict:
    """Полный набор метрик прогноза цены закрытия T+1."""
    m = price_metrics(y_true, y_pred)
    m['DA'] = round(directional_accuracy(y_true - close_today,
                                         y_pred - close_today), 2)
    return m


def build_training_frame(ticker: str) -> tuple:
    """
    Собирает датафрейм признаков и обучает препроцессинг эмбеддингов.
    PCA и StandardScaler обучаются ТОЛЬКО на train-части (без утечки).
    """
    df_prices, sent_daily, emb_daily = build_raw_dataset(
        ticker, PRICE_START, END_DATE, with_target=True)

    source = df_prices.attrs.get('source', 'unknown')

    # обрезаем прогрев индикаторов и последнюю строку (NaN в target)
    mask = df_prices.index >= pd.Timestamp(TRAIN_START)
    df_prices  = df_prices[mask].dropna(subset=[TARGET_COL])

    # дни до первой новости: эмбеддинги NaN (см. dataset.py)
    emb_ok = emb_daily.loc[df_prices.index].notna().all(axis=1)
    if not emb_ok.all():
        print(f'  Отброшено {int((~emb_ok).sum())} дней до первой новости')
        df_prices = df_prices[emb_ok.values]

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
    df.attrs['source'] = source

    return df, n_train, n_val, emb_scaler, pca


def make_tensors(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    xp, xe = split_streams(X, FEATURE_COLS)
    return TensorDataset(torch.tensor(xp, dtype=torch.float32),
                         torch.tensor(xe, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))


def _fit_one_seed(X_seq, y_seq, tr, va, seed: int, verbose: bool = True,
                  use_news: bool = True):
    """Обучает одну модель с заданным сидом. Возвращает (state_dict, история)."""
    set_seed(seed)

    loaders = {
        'train': DataLoader(make_tensors(X_seq[tr], y_seq[tr]),
                            batch_size=BATCH_SIZE, shuffle=True),
        'val':   DataLoader(make_tensors(X_seq[va], y_seq[va]),
                            batch_size=BATCH_SIZE),
    }

    model = LSTMEmbeddings(use_news=use_news).to(DEVICE)
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
            marker = ' *'
        else:
            wait += 1

        if verbose and (epoch % 10 == 0 or marker):
            print(f'    epoch {epoch:3d} | train {tr_loss:.5f} '
                  f'| val {va_loss:.5f}{marker}')
        if wait >= PATIENCE:
            if verbose:
                print(f'    Early stopping (epoch {epoch})')
            break

    return best_state, history, float(best_val)


def _predict_test(state, X_seq, te, use_news: bool = True) -> np.ndarray:
    """Прогноз модели на тестовой части (в шкале скейлера)."""
    model = LSTMEmbeddings(use_news=use_news).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    xp, xe = split_streams(X_seq[te], FEATURE_COLS)
    with torch.no_grad():
        return model(
            torch.tensor(xp, dtype=torch.float32).to(DEVICE),
            torch.tensor(xe, dtype=torch.float32).to(DEVICE),
        ).cpu().numpy()


def train_ticker(ticker: str, seeds: list | None = None,
                 use_news: bool = True, prebuilt: tuple | None = None,
                 out_suffix: str = '') -> dict:
    """
    Обучает модель для тикера.

    seeds — список сидов. При нескольких сидах обучается несколько моделей
    на одних и тех же данных: в артефакты идёт лучшая по валидации, в
    meta.json — среднее и разброс метрик по всем прогонам. Тест короткий
    (~38 дней), и разброс между сидами сопоставим с разницей между
    моделью и бенчмарком, поэтому одного прогона недостаточно.
    """
    seeds = seeds or [SEED]
    tag = 'гибрид (цены + новости)' if use_news else 'ablation (только цены)'
    print(f'\n{"="*52}\n  Обучение: {ticker} — {tag}\n{"="*52}')

    # prebuilt позволяет обучить несколько конфигураций на одном датасете:
    # FinBERT и выкачка новостей выполняются один раз, сплит идентичен.
    if prebuilt is None:
        prebuilt = build_training_frame(ticker)
    df, n_train, n_val, emb_scaler, pca = prebuilt
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

    # StandardScaler не имеет жёстких границ [0,1] и допускает
    # значения, выходящие за train-диапазон
    Scaler = StandardScaler if STATIONARY_FEATURES else MinMaxScaler
    scaler_X = Scaler().fit(X_raw[:n_train])
    scaler_y = Scaler().fit(y_raw[:n_train].reshape(-1, 1))
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    # последовательности на всём ряду, сплит по позиции таргета
    # (контекст val/test — это прошлое, утечки нет; строки не теряются)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOK_BACK)
    pos = np.arange(LOOK_BACK, n)          # индекс строки-таргета
    tr = pos < n_train
    va = (pos >= n_train) & (pos < n_train + n_val)
    te = pos >= n_train + n_val

    close_today = df['raw_Close'].values[pos[te]]
    true_target = scaler_y.inverse_transform(
        y_seq[te].reshape(-1, 1)).flatten()
    y_true = (close_today * np.exp(true_target) if STATIONARY_FEATURES
              else true_target)

    # прогоны по сидам
    runs, best = [], None
    for s in seeds:
        print(f'\n  --- сид {s} ---')
        state, history, best_val = _fit_one_seed(X_seq, y_seq, tr, va, s,
                                                 use_news=use_news)
        pred_target = scaler_y.inverse_transform(
            _predict_test(state, X_seq, te, use_news).reshape(-1, 1)).flatten()
        y_pred = (close_today * np.exp(pred_target) if STATIONARY_FEATURES
                  else pred_target)
        m = calc_metrics(y_true, y_pred, close_today)
        print(f'    метрики: {m}')
        runs.append({'seed': s, 'val_loss': best_val, **m})
        if best is None or best_val < best['val_loss']:
            best = {'seed': s, 'val_loss': best_val, 'state': state,
                    'history': history, 'y_pred': y_pred, 'metrics': m}

    metrics = best['metrics']

    # Наивный бенчмарк «завтра = сегодня». DA для него не определён:
    # прогноз изменения тождественно нулевой. Вместо него — доля
    # растущих дней, то есть результат стратегии «всегда вверх».
    naive = price_metrics(y_true, close_today)
    naive['DA'] = None
    up_share = float(np.mean(y_true > close_today) * 100)

    print(f'\n{"-"*52}')
    if len(runs) > 1:
        print(f'  Разброс по {len(runs)} сидам:')
        for k in ('MAE', 'RMSE', 'MAPE', 'DA'):
            vals = np.array([r[k] for r in runs], dtype=float)
            print(f'    {k:5s}: {vals.mean():8.4f} ± {vals.std():.4f}  '
                  f'(min {vals.min():.4f}, max {vals.max():.4f})')
        print(f'  В артефакты сохранён лучший по валидации сид '
              f'{best["seed"]}')
    print(f'  Метрики модели   : {metrics}')
    print(f'  Наивный прогноз  : {naive}')
    print(f'  Доля растущих дней на тесте: {up_share:.1f}% '
          f'(бенчмарк «всегда вверх» для DA)')

    mae_mean = float(np.mean([r['MAE'] for r in runs]))
    if mae_mean < naive['MAE']:
        print(f'  [OK] Модель точнее наивного прогноза по среднему MAE '
              f'({mae_mean:.3f} < {naive["MAE"]:.3f})')
    else:
        print(f'  [!!] Модель ХУЖЕ наивного прогноза по среднему MAE '
              f'({mae_mean:.3f} >= {naive["MAE"]:.3f})')

    # сохранение артефактов
    out = os.path.join(ARTIFACT_DIR, ticker + out_suffix)
    os.makedirs(out, exist_ok=True)
    torch.save(best['state'], os.path.join(out, 'model.pt'))
    joblib.dump(scaler_X,   os.path.join(out, 'scaler_X.joblib'))
    joblib.dump(scaler_y,   os.path.join(out, 'scaler_y.joblib'))
    joblib.dump(emb_scaler, os.path.join(out, 'emb_scaler.joblib'))
    joblib.dump(pca,        os.path.join(out, 'pca.joblib'))

    test_dates = df.index[pos[te]]
    pd.DataFrame({'date': test_dates, 'y_true': y_true,
                  'y_pred': best['y_pred'],
                  'close_today': close_today}).to_csv(
        os.path.join(out, 'test_predictions.csv'), index=False)

    seed_summary = {
        k: {'mean': round(float(np.mean([r[k] for r in runs])), 4),
            'std':  round(float(np.std([r[k] for r in runs])), 4)}
        for k in ('MAE', 'RMSE', 'MAPE', 'DA')
    }

    meta = {
        'ticker': ticker,
        'trained_at': pd.Timestamp.now().isoformat(timespec='seconds'),
        'period': [str(df.index[0].date()), str(df.index[-1].date())],
        'target_mode': 'logret' if STATIONARY_FEATURES else 'price_level',
        'use_news': use_news,
        'price_source': df.attrs.get('source', 'unknown'),
        'baseline_up_share': round(up_share, 2),
        'n_days': int(n),
        'split': {'train': int(n_train), 'val': int(n_val),
                  'test': int(n - n_train - n_val)},
        'feature_cols': FEATURE_COLS,
        'look_back': LOOK_BACK,
        'metrics_test': metrics,
        'metrics_naive': naive,
        'seeds': seeds,
        'best_seed': best['seed'],
        'seed_runs': runs,
        'seed_summary': seed_summary,
        'pca_explained_var': round(
            float(pca.explained_variance_ratio_.sum()), 4),
        'history': best['history'],
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'  Артефакты сохранены в {out}')
    return metrics


def ablation_ticker(ticker: str, seeds: list) -> dict:
    """
    Ablation «помогают ли новости»: две модели на ОДНОМ датасете —
    гибрид (цены + эмбеддинги новостей) и та же сеть только с ценовым
    потоком. Датасет и PCA строятся один раз, поэтому строки, сплит и
    сиды у обеих конфигураций идентичны, а FinBERT прогоняется однократно.

    Продакшн-артефакты artifacts/{ticker}/ не перезаписываются:
    результаты идут в {ticker}_hybrid и {ticker}_pricesonly.
    """
    print(f'\n{"#"*52}\n  ABLATION: {ticker}\n{"#"*52}')
    prebuilt = build_training_frame(ticker)

    metas = {}
    for key, use_news, suffix in (('hybrid', True,  '_hybrid'),
                                  ('prices', False, '_pricesonly')):
        train_ticker(ticker, seeds=seeds, use_news=use_news,
                     prebuilt=prebuilt, out_suffix=suffix)
        with open(os.path.join(ARTIFACT_DIR, ticker + suffix,
                               'meta.json')) as f:
            metas[key] = json.load(f)

    naive = metas['hybrid']['metrics_naive']
    hs, ps = (metas['hybrid']['seed_summary'],
              metas['prices']['seed_summary'])

    print(f'\n{"="*64}\n  ИТОГ ABLATION: {ticker}\n{"="*64}')
    print(f'  {"":22s} {"MAE":>18s} {"MAPE, %":>16s} {"DA, %":>16s}')
    for name, sm in (('Гибрид (цены+новости)', hs),
                     ('Только цены', ps)):
        print(f'  {name:22s} '
              f'{sm["MAE"]["mean"]:10.4f} ± {sm["MAE"]["std"]:<5.3f} '
              f'{sm["MAPE"]["mean"]:9.4f} ± {sm["MAPE"]["std"]:<4.3f} '
              f'{sm["DA"]["mean"]:9.2f} ± {sm["DA"]["std"]:<4.2f}')
    print(f'  {"Наивный прогноз":22s} {naive["MAE"]:10.4f} {"":6s} '
          f'{naive["MAPE"]:9.4f} {"":5s} {"—":>9s}')

    delta = ps['MAE']['mean'] - hs['MAE']['mean']
    pooled = float(np.sqrt(hs['MAE']['std'] ** 2 + ps['MAE']['std'] ** 2)) or 1e-9
    print(f'\n  Выигрыш гибрида по MAE: {delta:+.4f} $ '
          f'({delta / pooled:+.2f} σ разброса по сидам)')
    if abs(delta) < pooled:
        print('  Разница меньше разброса между сидами — новостной поток '
              'не даёт измеримого вклада.')

    summary = {'ticker': ticker, 'seeds': seeds,
               'hybrid': hs, 'prices_only': ps, 'naive': naive,
               'up_share': metas['hybrid']['baseline_up_share'],
               'delta_mae': round(delta, 4),
               'delta_sigma': round(delta / pooled, 3)}
    with open(os.path.join(ARTIFACT_DIR, f'ablation_{ticker}.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  Сводка: artifacts/ablation_{ticker}.json')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seeds', type=int, default=1,
                        help='сколько сидов обучить (для оценки разброса)')
    parser.add_argument('--ablation', action='store_true',
                        help='обучить гибрид и модель только на ценах '
                             'на одном датасете и сравнить')
    args = parser.parse_args()

    seed_list = [SEED + i for i in range(max(1, args.seeds))]
    tickers = TICKERS if (args.all or not args.ticker) else [args.ticker]

    if args.ablation:
        for t in tickers:
            ablation_ticker(t, seeds=seed_list)
    else:
        results = {t: train_ticker(t, seeds=seed_list) for t in tickers}
        print(f'\n{"="*52}\n  ИТОГ\n{"="*52}')
        for t, m in results.items():
            print(f'  {t}: '
                  + str({k: v for k, v in m.items()
                         if not k.startswith('_')}))
