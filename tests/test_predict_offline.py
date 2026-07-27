# ============================================================
# tests/test_predict_offline.py — офлайн-проверка пути инференса
# Сеть и API-ключи не нужны: цены, новости и эмбеддинги — синтетические.
# Проверяет:
#   · восстановление цены из лог-доходности,
#   · согласованность окна признаков между train и predict,
#   · выравнивание новостей на окно инференса.
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import src.predict as predict_mod
import src.train as train_mod
from config import ARTIFACT_DIR, LOOK_BACK
from tests.test_train_smoke import TICKER, make_synthetic_data


def _fake_news(dates: pd.DatetimeIndex, per_day: int = 2) -> pd.DataFrame:
    """Синтетическая лента: несколько новостей на каждый день."""
    rows = []
    for d in dates:
        for k in range(per_day):
            rows.append({'date': d,
                         'text': f'Company reports quarterly update {d.date()} #{k}'})
    return pd.DataFrame(rows)


def _fake_sentiment(df_news, show_progress=False):
    """Заглушка FinBERT: детерминированные тональность и эмбеддинги."""
    rng = np.random.default_rng(1)
    dates = pd.DatetimeIndex(sorted(df_news['date'].unique()))
    sent = pd.DataFrame(
        {'sent_pos': rng.uniform(0.2, 0.5, len(dates)),
         'sent_neg': rng.uniform(0.1, 0.3, len(dates)),
         'sent_neu': rng.uniform(0.3, 0.6, len(dates))}, index=dates)
    sent['sentiment_score'] = sent['sent_pos'] - sent['sent_neg']
    emb = pd.DataFrame(rng.normal(0, 1, (len(dates), 768)), index=dates)
    return sent, emb


def test_predict_offline():
    # 1. обучаем модель на синтетике, чтобы получить артефакты
    train_mod.build_raw_dataset = (
        lambda ticker, start, end, with_target=True, show_progress=True:
        make_synthetic_data())
    train_mod.EPOCHS, train_mod.PATIENCE = 5, 3
    train_mod.train_ticker(TICKER)

    # 2. подменяем источники данных на пути инференса
    df_prices, _, _ = make_synthetic_data()
    df_prices = df_prices.drop(columns=['target'])

    predict_mod.build_price_frame = (
        lambda ticker, start, end, with_target=False: df_prices)
    predict_mod.fetch_news = (
        lambda ticker, start, end: _fake_news(df_prices.index[-30:]))
    predict_mod.daily_sentiment_frame = _fake_sentiment

    res = predict_mod.predict_next_close(TICKER)

    # 3. проверки
    assert set(res) >= {'predicted_close', 'last_close', 'change_pct'}

    last_close = df_prices['raw_Close'].iloc[-1]
    assert abs(res['last_close'] - last_close) < 0.01, \
        'last_close должен браться из сырых цен (raw_Close), не из признаков'

    # прогноз на один торговый день не может отличаться в разы
    assert 0.8 * last_close < res['predicted_close'] < 1.2 * last_close, (
        f'Прогноз {res["predicted_close"]:.2f} неправдоподобен при '
        f'последней цене {last_close:.2f} — вероятна ошибка восстановления '
        f'цены из лог-доходности')
    assert abs(res['change_pct']) < 20

    print(f'✅ Инференс офлайн: last={res["last_close"]:.2f} → '
          f'прогноз={res["predicted_close"]:.2f} '
          f'({res["change_pct"]:+.2f}%)')

    import shutil
    shutil.rmtree(os.path.join(ARTIFACT_DIR, TICKER), ignore_errors=True)


if __name__ == '__main__':
    test_predict_offline()
