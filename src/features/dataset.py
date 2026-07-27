# src/features/dataset.py — сборка датасета: цены ⊕ новости

import pandas as pd

from config import SENTIMENT_FEATURES
from src.data.prices import build_price_frame
from src.data.news import fetch_news, align_to_trading_days
from src.features.sentiment import daily_sentiment_frame

# нейтральный fallback на случай дней совсем без новостей
NEUTRAL_SENTIMENT = {'sent_pos': 0.1, 'sent_neg': 0.1,
                     'sent_neu': 0.8, 'sentiment_score': 0.0}


def build_raw_dataset(ticker: str,
                      price_start: str,
                      end: str,
                      with_target: bool = True,
                      show_progress: bool = True):
    """
    Возвращает:
        df_prices  : DataFrame торговых дней с 27 индикаторами (+target)
        sent_daily : DataFrame (торговые дни) sent_pos/neg/neu/score
        emb_daily  : DataFrame (торговые дни) — сырые 768-мерные
                     эмбеддинги. PCA применяется отдельно:
                     fit на train при обучении / загрузка сохранённого
                     объекта при инференсе.
    """
    # цены + индикаторы 
    df_prices = build_price_frame(ticker, price_start, end,
                                  with_target=with_target)
    trading_index = df_prices.index

    # новости
    df_news = fetch_news(ticker, price_start, end)
    if df_news.empty:
        raise RuntimeError(f'Не удалось получить новости для {ticker}')

    # новости выходных/праздников -> ближайший следующий торговый день
    df_news = df_news.copy()
    df_news['date'] = align_to_trading_days(df_news['date'], trading_index)

    #  FinBERT: тональность + эмбеддинги по датам
    sent_daily, emb_daily = daily_sentiment_frame(
        df_news, show_progress=show_progress)

    # Выравнивание на все торговые дни: только ffill, без bfill —
    # обратное заполнение принесло бы в день T тональность новостей,
    # вышедших позже (look-ahead-утечка).
    sent_daily = sent_daily.reindex(trading_index).ffill()
    for col, val in NEUTRAL_SENTIMENT.items():
        sent_daily[col] = sent_daily[col].fillna(val)

    # Дни до первой новости остаются NaN: заполнить их можно только
    # значением из будущего, поэтому train.py отбрасывает этот участок.
    emb_daily = emb_daily.reindex(trading_index).ffill()

    return df_prices, sent_daily[SENTIMENT_FEATURES], emb_daily
