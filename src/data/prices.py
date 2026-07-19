# src/data/prices.py — загрузка OHLCV и технические индикаторы


import io
import time

import pandas as pd
import ta

from config import PRICE_FEATURES, TARGET_COL, TARGET_SHIFT

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def _from_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Основной источник: stooq.com — бесплатные дневные котировки,
    API-ключ не нужен. Формат тикера: aapl.us, tsla.us.
    """
    import requests

    symbol = f'{ticker.lower()}.us'
    url = (f'https://stooq.com/q/d/l/?s={symbol}&i=d'
           f'&d1={start.replace("-", "")}&d2={end.replace("-", "")}')

    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=30,
                             headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                return df.set_index('Date').sort_index()
            last_err = RuntimeError(f'пустой ответ Stooq: {r.text[:80]}')
        except Exception as e:
            last_err = e
        time.sleep(5 * attempt)

    raise RuntimeError(f'Stooq недоступен: {last_err}')


def _from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Запасной источник: Yahoo Finance через yfinance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        df = yf.Ticker(ticker).history(start=start, end=end,
                                       auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError('yfinance вернул пустые данные')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Скачивает OHLCV: сначала Stooq, при отказе — yfinance."""
    try:
        df = _from_yfinance(ticker, start, end)
        source = 'yfinance'
    except Exception as e:
        print(f'  {e}')
        print('  → переключаюсь на запасной источник stooq')
        df = _from_stooq(ticker, start, end)
        source = 'stooq'

    df.index = pd.to_datetime(df.index)
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = 'date'

    df = df[OHLCV].dropna()
    if df.empty:
        raise RuntimeError(f'Не удалось получить цены для {ticker}')
    print(f'  Цены {ticker}: {len(df)} строк '
          f'({df.index[0].date()} → {df.index[-1].date()}) [{source}]')
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """27 признаков из дипломной работы (библиотека ta)."""
    df = df.copy()
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    # тренд
    df['MA_5']   = ta.trend.sma_indicator(close, window=5)
    df['MA_20']  = ta.trend.sma_indicator(close, window=20)
    df['MA_50']  = ta.trend.sma_indicator(close, window=50)
    df['EMA_12'] = ta.trend.ema_indicator(close, window=12)
    df['EMA_26'] = ta.trend.ema_indicator(close, window=26)

    # моментум
    df['RSI'] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist']   = macd.macd_diff()
    df['ROC'] = ta.momentum.roc(close, window=10)
    df['MOM'] = close.diff(10)
    df['CCI'] = ta.trend.cci(high, low, close, window=20)
    df['Williams_R'] = ta.momentum.williams_r(high, low, close, lbp=14)
    stoch = ta.momentum.StochasticOscillator(high, low, close,
                                             window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # волатильность
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_upper']  = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower']  = bb.bollinger_lband()
    df['BB_width']  = bb.bollinger_wband()
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['ADX'] = ta.trend.adx(high, low, close, window=14)

    # объём
    df['OBV'] = ta.volume.on_balance_volume(close, volume)

    return df


def build_price_frame(ticker: str, start: str, end: str,
                      with_target: bool = True) -> pd.DataFrame:
    """
    Полный ценовой датафрейм: OHLCV + индикаторы (+ target для обучения).
    NaN от прогрева индикаторов удаляются.
    """
    df = download_prices(ticker, start, end)
    df = add_indicators(df)

    if with_target:
        df[TARGET_COL] = df['Close'].shift(TARGET_SHIFT)

    df = df.dropna(subset=PRICE_FEATURES)
    return df
