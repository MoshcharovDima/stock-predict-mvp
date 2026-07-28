# ============================================================
# tests/test_ablation_offline.py — офлайн-проверка ablation-режима
# Сеть и API-ключи не нужны: данные синтетические (переиспользуется
# генератор из test_train_smoke).
# Проверяет:
#   · модель с use_news=False игнорирует новостной поток,
#   · ablation_ticker обучает обе конфигурации на ОДНОМ датасете,
#   · продакшн-артефакты artifacts/{ticker}/ не перезаписываются,
#   · сводка artifacts/ablation_{ticker}.json корректна.
# ============================================================

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import src.train as train_mod
from config import ARTIFACT_DIR
from src.models.hybrid_lstm import LSTMEmbeddings
from tests.test_train_smoke import make_synthetic_data

TICKER = '_ABL'


def test_price_only_model_ignores_news():
    """use_news=False: выход не зависит от новостного входа."""
    xp = torch.randn(8, 20, 27)
    xe = torch.randn(8, 20, 5)

    model = LSTMEmbeddings(use_news=False).eval()
    with torch.no_grad():
        a = model(xp, xe)
        b = model(xp, torch.randn_like(xe) * 100)
    assert torch.allclose(a, b), 'прайс-онли модель реагирует на новости'

    hybrid = LSTMEmbeddings(use_news=True).eval()
    with torch.no_grad():
        c = hybrid(xp, xe)
        d = hybrid(xp, torch.randn_like(xe) * 100)
    assert not torch.allclose(c, d), 'гибрид не реагирует на новости'

    assert (sum(p.numel() for p in model.parameters())
            < sum(p.numel() for p in hybrid.parameters()))


def test_ablation_runs_and_writes_summary():
    train_mod.build_raw_dataset = (
        lambda ticker, start, end, with_target=True, show_progress=True:
        make_synthetic_data())
    train_mod.EPOCHS, train_mod.PATIENCE = 4, 2

    dirs = [os.path.join(ARTIFACT_DIR, TICKER + s)
            for s in ('_hybrid', '_pricesonly')]
    summary_path = os.path.join(ARTIFACT_DIR, f'ablation_{TICKER}.json')
    try:
        summary = train_mod.ablation_ticker(TICKER, seeds=[0, 1])

        for d in dirs:
            assert os.path.exists(os.path.join(d, 'meta.json'))
        assert os.path.exists(summary_path)

        metas = [json.load(open(os.path.join(d, 'meta.json'))) for d in dirs]
        hyb, pri = metas
        assert hyb['use_news'] is True and pri['use_news'] is False

        # обе конфигурации обучены на одном датасете и сплите
        assert hyb['split'] == pri['split']
        assert hyb['n_days'] == pri['n_days']
        assert hyb['metrics_naive']['MAE'] == pri['metrics_naive']['MAE']

        # продакшн-артефакты не тронуты
        assert not os.path.exists(os.path.join(ARTIFACT_DIR, TICKER))

        for key in ('hybrid', 'prices_only', 'naive', 'delta_mae'):
            assert key in summary
        assert 'MAE' in summary['hybrid'] and 'mean' in summary['hybrid']['MAE']
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
        if os.path.exists(summary_path):
            os.remove(summary_path)
