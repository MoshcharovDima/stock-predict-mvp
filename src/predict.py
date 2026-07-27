# src/predict.py — live-инференс: свежие данные → прогноз Close(T+1)
# Запуск из консоли:  python -m src.predict --ticker AAPL

import argparse
import json
import os
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import torch

from config import (
    ARTIFACT_DIR, DEVICE, EMBEDDING_FEATURES, LOOK_BACK,
    PRICE_FEATURES, SENTIMENT_FEATURES, STATIONARY_FEATURES, WARMUP_DAYS,
)
from src.data.news import fetch_news, align_to_trading_days
from src.data.prices import build_price_frame
from src.features.dataset import NEUTRAL_SENTIMENT
from src.features.sentiment import (
    daily_sentiment_frame, finbert_scores_and_embeddings, select_top_news,
    IDX_NEU, IDX_POS, IDX_NEG,
)
from src.models.hybrid_lstm import LSTMEmbeddings

FEATURE_COLS = PRICE_FEATURES + EMBEDDING_FEATURES
NEWS_LOOKBACK_DAYS = 45   # календарных дней новостей для окна инференса


def load_artifacts(ticker: str) -> dict:
    """Загружает модель и препроцессинг, обученные src/train.py."""
    path = os.path.join(ARTIFACT_DIR, ticker)
    if not os.path.exists(os.path.join(path, 'model.pt')):
        raise FileNotFoundError(
            f'Нет артефактов для {ticker}. '
            f'Сначала обучите модель: python -m src.train --ticker {ticker}')

    model = LSTMEmbeddings()
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'),
                                     map_location='cpu'))
    model.to(DEVICE).eval()

    with open(os.path.join(path, 'meta.json')) as f:
        meta = json.load(f)

    # Скейлеры обучены под конкретную постановку задачи: при рассинхроне
    # с config.STATIONARY_FEATURES прогноз был бы некорректным.
    expected = 'logret' if STATIONARY_FEATURES else 'price_level'
    actual = meta.get('target_mode', 'price_level')
    if actual != expected:
        raise RuntimeError(
            f'Артефакты {ticker} обучены в режиме "{actual}", '
            f'а config.STATIONARY_FEATURES ожидает "{expected}". '
            f'Переобучите модель: python -m src.train --ticker {ticker}')

    return {
        'model': model,
        'scaler_X':   joblib.load(os.path.join(path, 'scaler_X.joblib')),
        'scaler_y':   joblib.load(os.path.join(path, 'scaler_y.joblib')),
        'emb_scaler': joblib.load(os.path.join(path, 'emb_scaler.joblib')),
        'pca':        joblib.load(os.path.join(path, 'pca.joblib')),
        'meta': meta,
    }


def score_recent_news(df_news: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Индивидуальная тональность последних новостей — для интерфейса."""
    df = select_top_news(df_news).sort_values('date').tail(top_n)
    if df.empty:
        return df
    probs, _ = finbert_scores_and_embeddings(df['text'].tolist(),
                                             show_progress=False)
    df = df.reset_index(drop=True)
    df['sent_pos'] = probs[:, IDX_POS]
    df['sent_neg'] = probs[:, IDX_NEG]
    df['sent_neu'] = probs[:, IDX_NEU]
    df['label'] = np.array(['neutral', 'positive', 'negative'])[
        probs.argmax(axis=1)]
    return df


def predict_next_close(ticker: str, artifacts: dict | None = None) -> dict:
    """
    Автоматически подтягивает свежие цены и новости,
    строит окно из последних LOOK_BACK торговых дней
    и прогнозирует цену закрытия следующего торгового дня.
    """
    art = artifacts or load_artifacts(ticker)

    today = date.today()
    price_start = (today - timedelta(days=WARMUP_DAYS + 60)).isoformat()
    news_start  = (today - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat()
    end = (today + timedelta(days=1)).isoformat()

    df_prices = build_price_frame(ticker, price_start, end, with_target=False)
    if len(df_prices) < LOOK_BACK:
        raise RuntimeError('Недостаточно торговых дней для окна LOOK_BACK')
    window = df_prices.iloc[-LOOK_BACK:].copy()

    df_news = fetch_news(ticker, news_start, end)
    if df_news.empty:
        raise RuntimeError(f'Нет новостей для {ticker} — проверьте API-ключ')
    df_news = df_news.copy()

    # Выравнивание на полный торговый индекс, а не на окно: иначе np.clip
    # в align_to_trading_days отнесёт все новости старше окна к его
    # первому дню.
    df_news['date'] = align_to_trading_days(df_news['date'], df_prices.index)
    df_news = df_news[df_news['date'] >= window.index[0]]
    if df_news.empty:
        raise RuntimeError(
            f'Нет новостей за последние {LOOK_BACK} торговых дней для {ticker}')

    sent_daily, emb_daily = daily_sentiment_frame(df_news,
                                                  show_progress=False)

    # Заполнение пропусков совпадает с обучением (см. dataset.py).
    sent_daily = sent_daily.reindex(window.index).ffill()
    for col, val in NEUTRAL_SENTIMENT.items():
        sent_daily[col] = sent_daily[col].fillna(val)

    emb_daily = emb_daily.reindex(window.index).ffill()
    # начало окна без новостей — первым известным вектором окна;
    # нулевой эмбеддинг после StandardScaler был бы выбросом
    emb_daily = emb_daily.bfill()

    emb_pc = art['pca'].transform(
        art['emb_scaler'].transform(emb_daily.values))
    for i in range(emb_pc.shape[1]):
        window[f'emb_pc{i+1}'] = emb_pc[:, i]

    X = art['scaler_X'].transform(window[FEATURE_COLS].values)
    xp = torch.tensor(X[None, :, :len(PRICE_FEATURES)],
                      dtype=torch.float32).to(DEVICE)
    xe = torch.tensor(X[None, :, len(PRICE_FEATURES):],
                      dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_scaled = art['model'](xp, xe).cpu().numpy()
    pred_target = float(art['scaler_y'].inverse_transform(
        pred_scaled.reshape(-1, 1))[0, 0])

    last_close = float(window['raw_Close'].iloc[-1])
    last_date  = window.index[-1]

    if art['meta'].get('target_mode', 'price_level') == 'logret':
        # модель предсказывает log(Close_{T+1}/Close_T)
        pred = last_close * float(np.exp(pred_target))
    else:
        pred = pred_target

    # источник цен на инференсе должен совпадать с обучением
    src_now   = df_prices.attrs.get('source', 'unknown')
    src_train = art['meta'].get('price_source', 'unknown')
    if src_train != 'unknown' and src_now != src_train:
        print(f'  ⚠️ Источник цен ({src_now}) не совпадает с источником '
              f'обучения ({src_train}) — шкала цен может отличаться')
    next_date  = (last_date + pd.tseries.offsets.BDay(1)).date()

    return {
        'ticker': ticker,
        'last_date': str(last_date.date()),
        'last_close': round(last_close, 2),
        'predicted_date': str(next_date),
        'predicted_close': round(pred, 2),
        'change_abs': round(pred - last_close, 2),
        'change_pct': round((pred / last_close - 1) * 100, 2),
        'sentiment_today': {
            k: round(float(sent_daily[k].iloc[-1]), 3)
            for k in SENTIMENT_FEATURES
        },
        'price_source': src_now,
        'window': window,          # DataFrame для графиков в UI (колонки raw_*)
        'news': df_news,           # сырые новости окна для UI
        'meta': art['meta'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    args = parser.parse_args()

    res = predict_next_close(args.ticker)
    print(f"\n{'='*44}")
    print(f"  {res['ticker']} — прогноз на {res['predicted_date']}")
    print(f"{'='*44}")
    print(f"  Close {res['last_date']}   : {res['last_close']:.2f} USD")
    print(f"  Прогноз Close      : {res['predicted_close']:.2f} USD")
    print(f"  Изменение          : {res['change_abs']:+.2f} "
          f"({res['change_pct']:+.2f}%)")
    print(f"  Сентимент сегодня  : {res['sentiment_today']}")
