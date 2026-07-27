# config.py — единая конфигурация сервиса прогнозирования

import os
from datetime import date, timedelta

import torch

# Общие настройки
SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TICKERS = ['AAPL', 'TSLA']

# Периоды данных
# Finnhub (free tier) отдаёт историю новостей примерно за 1 год,
# поэтому обучаемся на последнем годе + прогрев индикаторов.
TODAY          = date.today()
NEWS_DAYS_BACK = 360                       # глубина истории новостей
WARMUP_DAYS    = 90                        # прогрев MA_50 и прочих индикаторов

TRAIN_START = (TODAY - timedelta(days=NEWS_DAYS_BACK)).isoformat()
PRICE_START = (TODAY - timedelta(days=NEWS_DAYS_BACK + WARMUP_DAYS)).isoformat()
END_DATE    = TODAY.isoformat()

# хронологический сплит train/val/test
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15   # test = остаток

# Постановка задачи
# True  — таргет log(Close_{T+1}/Close_T), признаки безразмерные
#         (см. src/data/prices.py::make_stationary)
# False — прогноз уровня цены
STATIONARY_FEATURES = True

# Целевая переменная
TARGET_COL   = 'target'
TARGET_SHIFT = -1

# Признаки
PRICE_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_5', 'MA_20', 'MA_50', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'ROC', 'MOM', 'CCI', 'Williams_R', 'Stoch_K', 'Stoch_D',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
    'ATR', 'ADX', 'OBV',
]

SENTIMENT_FEATURES = ['sent_pos', 'sent_neg', 'sent_neu', 'sentiment_score']

N_PCA_COMPONENTS   = 5
EMBEDDING_FEATURES = [f'emb_pc{i+1}' for i in range(N_PCA_COMPONENTS)]

# FinBERT
FINBERT_MODEL   = 'yiyanghkust/finbert-tone'  # 0=Neutral, 1=Positive, 2=Negative
MAX_NEWS_PER_DAY = 30                          # топ-N заголовков в день
FINBERT_BATCH    = 32
FINBERT_MAXLEN   = 128

# Модель / обучение
LOOK_BACK  = 20
HIDDEN_PRICE = 64
HIDDEN_EMB   = 16
DENSE_HIDDEN = 32
DROPOUT      = 0.2

EPOCHS     = 100
BATCH_SIZE = 32
LR         = 1e-3
PATIENCE   = 15

# Пути
ROOT         = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(ROOT, 'artifacts')
FIG_DIR      = os.path.join(ROOT, 'figures')
DATA_DIR     = os.path.join(ROOT, 'data')

for _d in (ARTIFACT_DIR, FIG_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

# Внешние API
# ключ Finnhub: https://finnhub.io (бесплатная регистрация)
# локально: export FINNHUB_API_KEY=...
# на HF Spaces: Settings → Variables and secrets
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
