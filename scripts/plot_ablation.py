# scripts/plot_ablation.py
# Сводка ablation «помогают ли новости»: парное сравнение по сидам
# (одинаковые сиды и один датасет у обеих конфигураций) + график.
#
# Читает artifacts/{ticker}_hybrid/meta.json и {ticker}_pricesonly/meta.json
# Запуск:  python -m scripts.plot_ablation
# Результат: figures/ablation.png

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARTIFACT_DIR, FIG_DIR, TICKERS

C_HYB   = '#7b3ff2'
C_PRICE = '#e07b39'
C_NAIVE = '#9aa0a6'


def load(ticker: str):
    def meta(suffix):
        p = os.path.join(ARTIFACT_DIR, ticker + suffix, 'meta.json')
        return json.load(open(p))
    h, p = meta('_hybrid'), meta('_pricesonly')
    by_seed = lambda m: {r['seed']: r['MAE'] for r in m['seed_runs']}
    seeds = sorted(set(by_seed(h)) & set(by_seed(p)))
    return {
        'seeds': seeds,
        'hyb': np.array([by_seed(h)[s] for s in seeds]),
        'pri': np.array([by_seed(p)[s] for s in seeds]),
        'naive': h['metrics_naive']['MAE'],
        'n_test': h['split']['test'],
    }


def paired_stats(d: dict) -> dict:
    """Парная разница по сидам: >0 означает выигрыш гибрида."""
    diff = d['pri'] - d['hyb']
    n = len(diff)
    sd = float(np.std(diff, ddof=1))
    sem = sd / np.sqrt(n) if sd else 1e-12
    t = float(diff.mean() / sem)
    try:
        from scipy import stats
        pv = float(stats.ttest_rel(d['pri'], d['hyb']).pvalue)
    except ImportError:
        pv = float('nan')
    return {'delta': float(diff.mean()), 'sd': sd, 'sem': sem,
            't': t, 'p': pv, 'wins': int((diff > 0).sum()), 'n': n}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data, stats_ = {}, {}
    for t in TICKERS:
        try:
            data[t] = load(t)
            stats_[t] = paired_stats(data[t])
        except FileNotFoundError:
            print(f'  нет ablation-артефактов для {t}, пропускаю')
    if not data:
        return

    print(f'\n{"="*64}\n  ABLATION — парное сравнение по сидам\n{"="*64}')
    for t, d in data.items():
        s = stats_[t]
        print(f'  {t}: гибрид {d["hyb"].mean():.4f} | только цены '
              f'{d["pri"].mean():.4f} | наивный {d["naive"]:.4f}')
        print(f'       эффект новостей {s["delta"]:+.4f} $ '
              f'({s["delta"]/d["hyb"].mean()*100:+.2f}% MAE), '
              f'в пользу гибрида {s["wins"]}/{s["n"]} сидов')
        print(f'       парный t-тест: t={s["t"]:.2f}, p={s["p"]:.3f} → '
              f'{"значимо" if s["p"] < 0.05 else "НЕ значимо"}\n')

    fig, axes = plt.subplots(1, len(data), figsize=(5.6 * len(data), 4.3),
                             dpi=200)
    for ax, (t, d) in zip(np.atleast_1d(axes), data.items()):
        rows = [('Гибрид\n(цены + новости)', d['hyb'], C_HYB),
                ('Только цены', d['pri'], C_PRICE)]
        ys = [1, 0]
        for y, (label, vals, c) in zip(ys, rows):
            ax.errorbar(vals.mean(), y, xerr=vals.std(), fmt='o',
                        color=c, ms=11, capsize=6, lw=2.2, zorder=3)
            ax.scatter(vals, [y] * len(vals), color=c, alpha=0.35, s=28,
                       zorder=2)
        ax.axvline(d['naive'], color=C_NAIVE, ls='--', lw=2, zorder=1)
        ax.text(d['naive'], 1.55, ' наивный прогноз', color='#6b6b6b',
                fontsize=10.5, va='top', ha='left')

        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
        ax.set_ylim(-0.6, 1.7)
        ax.set_xlabel('MAE на тесте, $  (меньше — лучше)', fontsize=11.5)

        s = stats_[t]
        ax.set_title(f'{t} — тест {d["n_test"]} дней\n'
                     f'эффект новостей {s["delta"]:+.3f} $, p = {s["p"]:.2f}',
                     fontsize=13, fontweight='600', pad=10)
        ax.grid(axis='x', alpha=0.25, lw=0.7)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', length=0)

    fig.suptitle('Новостной поток не даёт устойчивого эффекта: '
                 'знак разный на двух тикерах, ни один не значим',
                 fontsize=14.5, fontweight='700', y=1.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'ablation.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f'  сохранено: {out}')


if __name__ == '__main__':
    main()
