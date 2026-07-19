# src/features/sentiment.py — FinBERT-tone: тональность + эмбеддинги
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from config import (
    DEVICE, FINBERT_MODEL, FINBERT_BATCH, FINBERT_MAXLEN,
    MAX_NEWS_PER_DAY, SENTIMENT_FEATURES,
)

# finbert-tone: 0=Neutral, 1=Positive, 2=Negative
IDX_NEU, IDX_POS, IDX_NEG = 0, 1, 2


def clean_text(text: str) -> str:
    """Очистка текста новости перед FinBERT."""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\$[A-Z]{1,5}', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?%\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@lru_cache(maxsize=1)
def load_finbert():
    """Загружает FinBERT-tone один раз на процесс."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def finbert_scores_and_embeddings(texts: list,
                                  batch_size: int = FINBERT_BATCH,
                                  show_progress: bool = True):
    """
    Возвращает:
        sentiments (N, 3) — вероятности [neu, pos, neg]
        embeddings (N, 768) — CLS-эмбеддинги последнего слоя
    """
    tokenizer, model = load_finbert()
    all_probs, all_emb = [], []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc='  FinBERT', unit='batch')

    for i in iterator:
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True,
                           padding=True, max_length=FINBERT_MAXLEN)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        all_probs.append(torch.softmax(out.logits, -1).cpu().numpy())
        all_emb.append(out.hidden_states[-1][:, 0, :].cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_emb)


def select_top_news(df_news: pd.DataFrame,
                    top_n: int = MAX_NEWS_PER_DAY) -> pd.DataFrame:
    """Очистка + топ-N самых информативных (длинных) текстов на дату."""
    df = df_news.copy()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 15]
    if df.empty:
        return df
    df['_len'] = df['text'].str.len()
    df = (df.sort_values(['date', '_len'], ascending=[True, False])
            .groupby('date', group_keys=False)
            .head(top_n)
            .drop(columns='_len')
            .reset_index(drop=True))
    return df


def daily_sentiment_frame(df_news: pd.DataFrame,
                          show_progress: bool = True) -> tuple:
    """
    Прогоняет новости через FinBERT и усредняет по датам.

    Возвращает:
        sent_daily (DataFrame): index=date, cols=sent_pos/neg/neu/score
        emb_daily  (DataFrame): index=date, 768 колонок — средние
                                CLS-эмбеддинги по датам
    """
    df = select_top_news(df_news)
    if df.empty:
        raise RuntimeError('После очистки не осталось ни одной новости')

    probs, emb = finbert_scores_and_embeddings(
        df['text'].tolist(), show_progress=show_progress)

    df = df.reset_index(drop=True)
    df['sent_neu'] = probs[:, IDX_NEU]
    df['sent_pos'] = probs[:, IDX_POS]
    df['sent_neg'] = probs[:, IDX_NEG]

    sent_daily = df.groupby('date')[['sent_pos', 'sent_neg', 'sent_neu']].mean()
    sent_daily['sentiment_score'] = (sent_daily['sent_pos']
                                     - sent_daily['sent_neg'])
    sent_daily = sent_daily[SENTIMENT_FEATURES]

    emb_daily = (pd.DataFrame(emb, index=df['date'])
                   .groupby(level=0).mean()
                   .loc[sent_daily.index])

    return sent_daily, emb_daily
