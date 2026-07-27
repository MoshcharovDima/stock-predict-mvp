# app/streamlit_app.py — веб-интерфейс сервиса прогнозирования
# Запуск:  streamlit run app/streamlit_app.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ARTIFACT_DIR, TICKERS
from src.predict import load_artifacts, predict_next_close, score_recent_news

st.set_page_config(
    page_title='Stock Forecast — Hybrid LSTM + FinBERT',
    page_icon='📈',
    layout='wide',
)

SENT_COLORS = {'positive': '#2E7D32', 'negative': '#C62828',
               'neutral': '#F57F17'}
SENT_LABELS = {'positive': 'Позитив', 'negative': 'Негатив',
               'neutral': 'Нейтрально'}


# Кэширование тяжёлых объектов
@st.cache_resource(show_spinner='Загрузка модели...')
def get_artifacts(ticker: str):
    return load_artifacts(ticker)


@st.cache_data(ttl=3600, show_spinner='Загрузка свежих данных и прогноз...')
def get_prediction(ticker: str):
    art = get_artifacts(ticker)
    res = predict_next_close(ticker, artifacts=art)
    news_scored = score_recent_news(res['news'], top_n=8)
    return res, news_scored


# Сайдбар
with st.sidebar:
    st.title('📈 Stock Forecast')
    st.caption('Гибридная модель LSTM + FinBERT-эмбеддинги')

    available = [t for t in TICKERS
                 if os.path.exists(os.path.join(ARTIFACT_DIR, t, 'model.pt'))]
    if not available:
        st.error('Модели не обучены. Запустите: '
                 '`python -m src.train --all`')
        st.stop()

    ticker = st.selectbox('Акция', available,
                          format_func=lambda t: {
                              'AAPL': 'Apple (AAPL)',
                              'TSLA': 'Tesla (TSLA)',
                          }.get(t, t))

    if st.button('🔄 Обновить прогноз'):
        get_prediction.clear()

    st.divider()
    st.markdown(
        '**Как это работает**\n\n'
        '1. Загружаются свежие котировки (Yahoo Finance) '
        'и финансовые новости (Finnhub)\n'
        '2. FinBERT-tone оценивает тональность новостей и извлекает '
        'эмбеддинги (768 → PCA → 5)\n'
        '3. Двухпоточная LSTM объединяет 27 технических индикаторов '
        'и новостные эмбеддинги\n'
        '4. Модель прогнозирует цену закрытия следующего торгового дня'
    )
    st.caption('Учебный проект, не является инвестиционной '
               'рекомендацией.')

# Прогноз
try:
    res, news_scored = get_prediction(ticker)
except Exception as e:
    st.error(f'Не удалось получить прогноз: {e}')
    st.stop()

st.title(f'{ticker} — прогноз на {res["predicted_date"]}')

c1, c2, c3, c4 = st.columns(4)
c1.metric(f'Close {res["last_date"]}', f'${res["last_close"]:.2f}')
c2.metric('Прогноз Close (T+1)', f'${res["predicted_close"]:.2f}',
          delta=f'{res["change_abs"]:+.2f} ({res["change_pct"]:+.2f}%)')
score = res['sentiment_today']['sentiment_score']
c3.metric('Новостной фон сегодня', f'{score:+.3f}',
          delta='позитивный' if score > 0.02
          else ('негативный' if score < -0.02 else 'нейтральный'),
          delta_color='normal' if score > 0.02
          else ('inverse' if score < -0.02 else 'off'))
_mape = res['meta']['metrics_test']['MAPE']
_mape_naive = res['meta']['metrics_naive']['MAPE']
c4.metric('MAPE на тесте', f'{_mape:.2f}%',
          delta=f'{_mape - _mape_naive:+.2f} п.п. к наивному',
          delta_color='inverse',
          help='Средняя абсолютная процентная ошибка на отложенном '
               'тестовом периоде. Для сравнения приведена разница с '
               'наивным прогнозом «завтра = сегодня»: отрицательная '
               'величина означает, что модель точнее.')

# График: окно 20 дней + прогнозная точка
window = res['window']
# в модель идут стационарные признаки, для графика — сырые цены raw_*
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=window.index, open=window['raw_Open'], high=window['raw_High'],
    low=window['raw_Low'], close=window['raw_Close'], name='OHLC'))

# признак MA_20 хранится как log(Close / MA_20) → возвращаем уровень
ma20 = window['raw_Close'] / np.exp(window['MA_20'])
fig.add_trace(go.Scatter(
    x=window.index, y=ma20, name='MA 20',
    line=dict(color='#FF9800', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(
    x=[pd.Timestamp(res['predicted_date'])], y=[res['predicted_close']],
    mode='markers+text', name='Прогноз',
    text=[f'${res["predicted_close"]:.2f}'], textposition='top center',
    marker=dict(size=14, color='#7B1FA2', symbol='diamond')))
fig.update_layout(
    height=420, xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
    title='Окно модели (20 торговых дней) и прогноз',
    legend=dict(orientation='h', y=1.08))
st.plotly_chart(fig, use_container_width=True)

# Новости + качество модели
col_news, col_model = st.columns([1.1, 1])

with col_news:
    st.subheader('📰 Последние новости и их тональность')
    if news_scored.empty:
        st.info('Свежих новостей не найдено')
    else:
        for _, row in news_scored.sort_values('date',
                                              ascending=False).iterrows():
            color = SENT_COLORS[row['label']]
            label = SENT_LABELS[row['label']]
            conf = max(row['sent_pos'], row['sent_neg'], row['sent_neu'])
            st.markdown(
                f'<div style="border-left:4px solid {color};'
                f'padding:6px 10px;margin-bottom:8px;'
                f'background:rgba(128,128,128,0.06);border-radius:4px">'
                f'<span style="color:{color};font-weight:600">'
                f'{label} {conf:.0%}</span> · '
                f'<span style="color:gray;font-size:0.85em">'
                f'{row["date"].date()}</span><br>{row["text"][:180]}'
                f'</div>',
                unsafe_allow_html=True)

with col_model:
    st.subheader('🎯 Качество модели (тестовый период)')
    meta = res['meta']
    m, nv = meta['metrics_test'], meta['metrics_naive']
    table = pd.DataFrame({
        'Гибридная LSTM': m,
        'Наивный прогноз («завтра = сегодня»)': nv,
    }).T
    # DA наивного прогноза не определён — в таблице прочерк
    st.dataframe(
        table.style.format(lambda v: '—' if v is None or pd.isna(v)
                           else f'{v:.2f}'),
        use_container_width=True)
    ss = meta.get('seed_summary')
    if ss and len(meta.get('seeds', [])) > 1:
        st.caption(
            f'Разброс по {len(meta["seeds"])} сидам: '
            f'MAE {ss["MAE"]["mean"]:.2f} ± {ss["MAE"]["std"]:.2f}, '
            f'DA {ss["DA"]["mean"]:.1f} ± {ss["DA"]["std"]:.1f}%. '
            f'В артефакты сохранён лучший по валидации.')

    if meta.get('baseline_up_share') is not None:
        st.caption(f'Доля растущих дней на тесте: '
                   f'{meta["baseline_up_share"]:.1f}% — это DA стратегии '
                   f'«всегда вверх», с ней и нужно сравнивать DA модели.')
    st.caption(
        f'Обучена {meta["trained_at"][:10]} на данных '
        f'{meta["period"][0]} → {meta["period"][1]} '
        f'({meta["n_days"]} торговых дней, '
        f'PCA объясняет {meta["pca_explained_var"]*100:.0f}% '
        f'дисперсии эмбеддингов)')

    pred_path = os.path.join(ARTIFACT_DIR, ticker, 'test_predictions.csv')
    if os.path.exists(pred_path):
        dfp = pd.read_csv(pred_path, parse_dates=['date'])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dfp['date'], y=dfp['y_true'],
                                  name='Факт', line=dict(color='#1565C0')))
        fig2.add_trace(go.Scatter(x=dfp['date'], y=dfp['y_pred'],
                                  name='Прогноз модели',
                                  line=dict(color='#7B1FA2', dash='dot')))
        if 'close_today' in dfp.columns:
            fig2.add_trace(go.Scatter(
                x=dfp['date'], y=dfp['close_today'],
                name='Наивный («завтра = сегодня»)',
                line=dict(color='#9E9E9E', dash='dash', width=1)))
        fig2.update_layout(height=280,
                           margin=dict(l=10, r=10, t=30, b=10),
                           title='Факт vs прогноз на тесте',
                           legend=dict(orientation='h', y=1.15))
        st.plotly_chart(fig2, use_container_width=True)
