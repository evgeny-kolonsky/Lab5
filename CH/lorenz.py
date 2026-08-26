"""
Lorenz maps and Lyapunov exponents from the maxima table produced by bifdata.py.

The Lorenz map plots each local maximum of v_C1 against the next one. If the
points fall on a curve instead of a cloud, the three-dimensional flow has
collapsed onto a one-dimensional map -- the experimental signature of a
horseshoe, and hence of chaos.

From the same map the Lyapunov exponent follows as

    lambda = <ln|f'(M)|> / <T>

with f' estimated by a sliding linear fit along M and <T> the MEAN return time.
Two details that change the answer by more than 10 %: the mean, not the median
(lobe switches take about twice as long as an ordinary winding, so the
distribution is skewed), and the average of ln|f'| taken over the points the
trajectory actually visits, not from one straight line per branch (near the
turning point the slope drops toward zero and pulls the average down).

Usage:
    python lorenz.py                       # every record in maxima.csv
    python lorenz.py --r 520 660 700       # only the records nearest these R
    python lorenz.py --grid                # one page with all of them
"""
import argparse
import collections
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    recs = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            recs[row['file']].append((int(row['n']), float(row['t']),
                                      float(row['M']), float(row['R'])))
    out = []
    for name, v in recs.items():
        v.sort()
        out.append(dict(file=name,
                        R=v[0][3],
                        t=np.array([x[1] for x in v]),
                        M=np.array([x[2] for x in v])))
    out.sort(key=lambda d: d['R'])
    return out


def local_slopes(x, y, window):
    """f'(M) by a sliding linear fit along sorted M_n."""
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    sl = np.full(len(xs), np.nan)
    for i in range(len(xs)):
        a, b = max(0, i - window), min(len(xs), i + window + 1)
        if b - a > 8 and xs[b - 1] - xs[a] > 1e-6:
            sl[i] = np.polyfit(xs[a:b], ys[a:b], 1)[0]
    return sl


def lyapunov(t, M, window):
    if len(M) < 60:
        return None
    x, y = M[:-1], M[1:]
    sl = local_slopes(x, y, window)
    good = np.isfinite(sl) & (np.abs(sl) > 1e-9)
    if good.sum() < 30:
        return None
    L = np.log(np.abs(sl[good]))
    dt = np.diff(t)
    T_mean = float(dt.mean())
    return dict(lam=float(L.mean() / T_mean),
                lnf=float(L.mean()),
                stretch=float(np.exp(L.mean())),
                T_mean=T_mean,
                T_median=float(np.median(dt)),
                skew=float(dt.mean() / np.median(dt)),
                frac_contracting=float(np.mean(np.abs(sl[good]) < 1)))


def draw(ax, rec, ly, window):
    M = rec['M']
    x, y = M[:-1], M[1:]
    ax.plot(x, y, '.', ms=1.0, color='tab:blue', rasterized=True)
    lo, hi = M.min() - 0.05 * np.ptp(M), M.max() + 0.05 * np.ptp(M)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    title = f"R = {rec['R']:.1f} $\\Omega$   ({len(M)} maxima)"
    if ly:
        title += f"\n$\\lambda$ = {ly['lam']:+.0f} s$^{{-1}}$"
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maxima', default='maxima.csv')
    p.add_argument('--r', type=float, nargs='*',
                   help='pick the records closest to these resistances')
    p.add_argument('--window', type=int, default=25,
                   help='half-width of the sliding fit, in points')
    p.add_argument('--grid', action='store_true',
                   help='all selected records on one page')
    p.add_argument('--min-maxima', type=int, default=200,
                   help='skip records with fewer maxima than this')
    p.add_argument('--outdir', default='.')
    a = p.parse_args()

    if not os.path.exists(a.maxima):
        sys.exit(f"{a.maxima} not found -- run bifdata.py first")
    recs = load(a.maxima)

    if a.r:
        chosen = []
        for target in a.r:
            k = min(range(len(recs)), key=lambda i: abs(recs[i]['R'] - target))
            if recs[k] not in chosen:
                chosen.append(recs[k])
        recs = chosen

    print('%-14s %-10s %-8s %-11s %-11s %-9s %-8s' % (
        'file', 'R, ohm', 'N max', 'T mean, us', 'mean/median', 'stretch', 'lambda'))
    results = []
    for rec in recs:
        ly = lyapunov(rec['t'], rec['M'], a.window)
        results.append((rec, ly))
        if ly:
            print('%-14s %-10.1f %-8d %-11.1f %-11.3f %-9.2f %+8.0f' % (
                rec['file'], rec['R'], len(rec['M']), ly['T_mean'] * 1e6,
                ly['skew'], ly['stretch'], ly['lam']))
        else:
            print('%-14s %-10.1f %-8d  (too few maxima)' % (
                rec['file'], rec['R'], len(rec['M'])))

    usable = [(r, l) for r, l in results if len(r['M']) >= a.min_maxima]
    if not usable:
        sys.exit('nothing with enough maxima to plot')

    if a.grid:
        n = len(usable)
        cols = min(4, n)
        rowsn = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rowsn, cols, figsize=(4 * cols, 3.6 * rowsn),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, (rec, ly) in zip(axes.ravel(), usable):
            ax.axis('on')
            draw(ax, rec, ly, a.window)
        fig.supxlabel('$M_n$, V')
        fig.supylabel('$M_{n+1}$, V')
        fig.tight_layout()
        out = os.path.join(a.outdir, 'lorenz_maps.png')
        fig.savefig(out, dpi=120)
        print(f'\n{out}')
    else:
        for rec, ly in usable:
            fig, ax = plt.subplots(figsize=(5.2, 5))
            draw(ax, rec, ly, a.window)
            ax.set_xlabel('$M_n$, V')
            ax.set_ylabel('$M_{n+1}$, V')
            fig.tight_layout()
            out = os.path.join(a.outdir,
                               'lorenz_R%04d.png' % round(rec['R']))
            fig.savefig(out, dpi=120)
            plt.close(fig)
        print(f'\n{len(usable)} figures written to {a.outdir}')

    with open(os.path.join(a.outdir, 'lyapunov.csv'), 'w') as f:
        f.write('file,R,n_max,T_mean,T_median,mean_over_median,'
                'ln_slope,stretch,lambda,frac_contracting\n')
        for rec, ly in results:
            if not ly:
                continue
            f.write(f"{rec['file']},{rec['R']:.4f},{len(rec['M'])},"
                    f"{ly['T_mean']:.6e},{ly['T_median']:.6e},{ly['skew']:.4f},"
                    f"{ly['lnf']:.4f},{ly['stretch']:.4f},{ly['lam']:.2f},"
                    f"{ly['frac_contracting']:.4f}\n")
    print('lyapunov.csv')


if __name__ == '__main__':
    main()
