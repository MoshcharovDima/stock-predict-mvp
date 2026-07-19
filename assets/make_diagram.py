# assets/make_diagram.py — схема архитектуры для README
# Запуск: python assets/make_diagram.py → figures/architecture.png

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import FIG_DIR


def box(ax, x, y, w, h, text, color, fontsize=9, tc='white'):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle='round,pad=0.12',
                                facecolor=color, edgecolor='white',
                                linewidth=1.5, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=tc, zorder=4,
            multialignment='center')


def arr(ax, x1, y1, x2, y2, color='#444444'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.suptitle('Hybrid LSTM + FinBERT Embeddings — архитектура сервиса',
                 fontsize=14, fontweight='bold')

    # источники
    box(ax, 3.5, 9.3, 4.6, 0.75,
        'Yahoo Finance — котировки OHLCV', '#455A64', fontsize=8)
    box(ax, 12.0, 9.3, 4.6, 0.75,
        'Finnhub — финансовые новости', '#455A64', fontsize=8)

    # вход 1
    box(ax, 3.5, 8.2, 4.6, 0.85,
        '27 технических индикаторов\n(20 дней × 27)', '#1565C0', fontsize=8)
    box(ax, 3.5, 7.0, 3.4, 0.8, 'LSTM (hidden=64)', '#0D47A1')
    box(ax, 3.5, 5.9, 3.4, 0.65, 'Dropout (0.2)', '#1976D2', fontsize=8)
    box(ax, 3.5, 4.9, 3.4, 0.75, 'вектор (64,)', '#1E88E5', fontsize=8)
    arr(ax, 3.5, 8.9, 3.5, 8.65)
    arr(ax, 3.5, 7.75, 3.5, 7.42)
    arr(ax, 3.5, 6.6, 3.5, 6.25)
    arr(ax, 3.5, 5.55, 3.5, 5.3)

    # вход 2
    box(ax, 12.0, 8.2, 4.6, 0.85,
        'FinBERT-tone → CLS(768) → PCA(5)\n(20 дней × 5)', '#6A1B9A',
        fontsize=8)
    box(ax, 12.0, 7.0, 3.4, 0.8, 'LSTM (hidden=16)', '#4A148C')
    box(ax, 12.0, 5.9, 3.4, 0.75, 'вектор (16,)', '#7B1FA2', fontsize=8)
    arr(ax, 12.0, 8.9, 12.0, 8.65)
    arr(ax, 12.0, 7.75, 12.0, 7.42)
    arr(ax, 12.0, 6.6, 12.0, 6.3)

    # объединение
    box(ax, 7.8, 3.9, 5.2, 0.8, 'Конкатенация [64 ⊕ 16] = 80', '#2E7D32')
    arr(ax, 3.5, 4.5, 5.6, 4.15, '#1565C0')
    arr(ax, 12.0, 5.5, 10.0, 4.15, '#6A1B9A')

    box(ax, 7.8, 2.85, 3.6, 0.65, 'Dense(32) + ReLU', '#1B5E20', fontsize=8)
    box(ax, 7.8, 2.0, 3.6, 0.65, 'Dense(1)', '#33691E', fontsize=8)
    box(ax, 7.8, 1.0, 4.8, 0.8,
        'Прогноз Close(T+1)\nцена закрытия следующего дня', '#E65100',
        fontsize=8)
    arr(ax, 7.8, 3.5, 7.8, 3.2)
    arr(ax, 7.8, 2.5, 7.8, 2.35)
    arr(ax, 7.8, 1.65, 7.8, 1.45)

    # гиперпараметры
    box(ax, 7.8, 7.4, 3.8, 1.5,
        'Гиперпараметры\nLOOK_BACK=20\nLR=1e-3, batch=32\n'
        'early stopping (pat.=15)', '#37474F', fontsize=8)

    path = os.path.join(FIG_DIR, 'architecture.png')
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Сохранено: {path}')


if __name__ == '__main__':
    main()
