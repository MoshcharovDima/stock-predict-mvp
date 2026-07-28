# scripts/plot_test_predictions.py
# График «факт vs прогноз модели vs наивный прогноз» на тестовой выборке.
# Источник: artifacts/{ticker}/test_predictions.csv (пишется train.py).
#
# Запуск:  python -m scripts.plot_test_predictions
# Результат: figures/test_pred_{TICKER}.png  и  figures/test_pred_all.png

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARTIFACT_DIR, FIG_DIR, TICKERS

C_TRUE  = '#1b1b1b'
C_MODEL = '#7b3ff2'
C_NAIVE = '#9aa0a6'


def load(ticker: str):
    path = os.path.join(ARTIFACT_DIR, ticker, 'test_predictions.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    meta_path = os.path.join(ARTIFACT_DIR, ticker, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return df, meta


def draw(ax, ticker: str, df: pd.DataFrame, meta: dict):
    x = df['date']
    ax.plot(x, df['y_true'], color=C_TRUE, lw=2.4, label='Факт', zorder=3)
    ax.plot(x, df['y_pred'], color=C_MODEL, lw=1.9, ls='-',
            label='Гибридная LSTM', zorder=2)
    ax.plot(x, df['close_today'], color=C_NAIVE, lw=1.9, ls='--',
            label='Наивный (завтра = сегодня)', zorder=1)

    mae_m = float(np.mean(np.abs(df['y_true'] - df['y_pred'])))
    mae_n = float(np.mean(np.abs(df['y_true'] - df['close_today'])))
    sd = (meta.get('seed_summary') or {}).get('MAE', {}).get('std')
    sd_txt = f' ± {sd:.3f}' if sd else ''

    ax.set_title(f'{ticker} — тест, {len(df)} торговых дней',
                 fontsize=15, fontweight='600', pad=10)
    ax.text(0.015, 0.965,
            f'MAE модели  {mae_m:.3f}{sd_txt} $\n'
            f'MAE наивного {mae_n:.3f} $',
            transform=ax.transAxes, va='top', ha='left', fontsize=11.5,
            family='DejaVu Sans Mono',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#d5d5d5'))

    ax.set_ylabel('Close, $', fontsize=12)
    ax.grid(alpha=0.25, lw=0.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10.5)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = {}
    for t in TICKERS:
        try:
            data[t] = load(t)
        except FileNotFoundError:
            print(f'  нет артефактов для {t}, пропускаю')

    # отдельные картинки
    for t, (df, meta) in data.items():
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
        draw(ax, t, df, meta)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f'test_pred_{t}.png')
        fig.savefig(out, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  сохранено: {out}')

    # сводная на два тикера
    if len(data) > 1:
        fig, axes = plt.subplots(len(data), 1, figsize=(11, 4.6 * len(data)),
                                 dpi=200)
        for ax, (t, (df, meta)) in zip(np.atleast_1d(axes), data.items()):
            draw(ax, t, df, meta)
        np.atleast_1d(axes)[0].legend(loc='lower right', fontsize=11,
                                      framealpha=0.95)
        fig.suptitle('Прогноз модели совпадает с наивным прогнозом',
                     fontsize=17, fontweight='700', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out = os.path.join(FIG_DIR, 'test_pred_all.png')
        fig.savefig(out, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  сохранено: {out}')


if __name__ == '__main__':
    main()
