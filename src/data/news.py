# src/data/news.py — загрузка новостей (Finnhub, fallback yfinance)

import time
from datetime import date, timedelta

import pandas as pd

from config import FINNHUB_API_KEY


def _fetch_finnhub_chunk(client, ticker: str,
                         start: date, end: date) -> list:
    """Один запрос company_news за интервал дат."""
    return client.company_news(
        ticker, _from=start.isoformat(), to=end.isoformat()
    ) or []


def fetch_news_finnhub(ticker: str, start: str, end: str,
                       chunk_days: int = 20,
                       pause: float = 1.1) -> pd.DataFrame:
    """
    Загружает новости по тикеру через Finnhub кусками по chunk_days,
    чтобы не упереться в лимит выдачи одного запроса.
    Free tier: ~1 год истории, 60 запросов/мин.

    Возвращает DataFrame: date | text
    """
    if not FINNHUB_API_KEY:
        raise RuntimeError(
            'Не задан FINNHUB_API_KEY. Получите бесплатный ключ на '
            'https://finnhub.io и выполните: export FINNHUB_API_KEY=...'
        )
    import finnhub
    client = finnhub.Client(api_key=FINNHUB_API_KEY)

    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)

    rows = []
    cur = start_d
    while cur <= end_d:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_d)
        try:
            items = _fetch_finnhub_chunk(client, ticker, cur, chunk_end)
        except Exception as e:                          # rate limit и пр.
            print(f'  Finnhub {cur}→{chunk_end}: {e}; повтор через 15с')
            time.sleep(15)
            items = _fetch_finnhub_chunk(client, ticker, cur, chunk_end)

        for it in items:
            headline = (it.get('headline') or '').strip()
            summary  = (it.get('summary') or '').strip()
            text = headline if len(headline) > len(summary) else summary
            if it.get('datetime') and text:
                rows.append({
                    'date': pd.Timestamp(it['datetime'], unit='s').normalize(),
                    'text': f'{headline}. {summary}'.strip('. ').strip(),
                })
        cur = chunk_end + timedelta(days=1)
        time.sleep(pause)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['date', 'text'])

    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 15]
    df = df.drop_duplicates(subset=['text'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'text']]


def fetch_news_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fallback без API-ключа: последние новости из yfinance
    (истории почти нет — годится только для live-инференса).
    """
    import yfinance as yf

    items = yf.Ticker(ticker).news or []
    rows = []
    for it in items:
        content = it.get('content', it)  # новый/старый формат yfinance
        title   = (content.get('title') or '').strip()
        summary = (content.get('summary') or '').strip()
        pub     = content.get('pubDate') or content.get('providerPublishTime')
        if not title or pub is None:
            continue
        ts = (pd.Timestamp(pub, unit='s') if isinstance(pub, (int, float))
              else pd.Timestamp(pub))
        rows.append({
            'date': ts.tz_localize(None) if ts.tzinfo else ts,
            'text': f'{title}. {summary}'.strip('. ').strip(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['date', 'text'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.drop_duplicates(subset=['text']).sort_values('date')
    return df.reset_index(drop=True)


def fetch_news(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Единая точка входа: Finnhub при наличии ключа, иначе yfinance."""
    if FINNHUB_API_KEY:
        return fetch_news_finnhub(ticker, start, end)
    print('  ⚠️ FINNHUB_API_KEY не задан — использую yfinance '
          '(только свежие новости, без истории)')
    return fetch_news_yfinance(ticker)


def align_to_trading_days(news_dates: pd.Series,
                          trading_index: pd.DatetimeIndex) -> pd.Series:
    """
    Относит каждую новость к БЛИЖАЙШЕМУ СЛЕДУЮЩЕМУ торговому дню:
    новости выходных влияют на понедельник, а не пропадают
    (улучшение относительно left-join из дипломного пайплайна).
    """
    import numpy as np

    idx = trading_index.sort_values()
    pos = np.searchsorted(idx.values, news_dates.values, side='left')
    pos = np.clip(pos, 0, len(idx) - 1)
    return pd.Series(idx.values[pos], index=news_dates.index)
