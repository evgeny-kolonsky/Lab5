"""
Batch-process a folder of oscilloscope records into one table of local maxima.

For every *.csv in the current folder this script
  1. measures the potentiometer from the CH3 voltage divider (as rpot.py does),
  2. smooths v_C1 and extracts its local maxima with sub-sample interpolation,
  3. writes one row per maximum.

Output 1 -- maxima.csv, one row per maximum:

    file,R,n,t,M

    R  potentiometer resistance for that record, ohm
    n  index of the maximum within the record, 0,1,2,...
    t  time of the maximum, s
    M  value of v_C1 at the maximum, V

  This one table serves all three analyses:
    bifurcation diagram : scatter M against R over all rows
    Lorenz map          : within one file, plot M[n] against M[n+1]
    Lyapunov exponent   : slope of that map, divided by mean(diff(t))

Output 2 -- summary.csv, one row per record, with the carrier frequency, the
mean return time, the divider fit residual, and the ADC quantisation step of
each channel. The quantisation step is a fingerprint of the vertical scale:
if it differs between records, their V/div settings differed, and their R
values are NOT comparable without re-doing the channel intercalibration.

Usage:
    python bifdata.py                    # current folder
    python bifdata.py --dir run2 --plot
"""
import argparse
import glob
import os
import sys
import numpy as np
from scipy.signal import savgol_filter, find_peaks


# --------------------------------------------------------------------------
def read_record(path, c1, c2, mid):
    with open(path) as f:
        names = [h.split('(')[0].strip() for h in f.readline().split(',')]
    d = np.genfromtxt(path, delimiter=',', skip_header=1)
    for want in (c1, c2, mid):
        if want not in names:
            raise ValueError(f"column {want} not in {names}")
    t = d[:, 0]
    v1, v2, vm = d[:, names.index(c1)], d[:, names.index(c2)], d[:, names.index(mid)]
    ok = np.isfinite(t) & np.isfinite(v1) & np.isfinite(v2) & np.isfinite(vm)
    return t[ok], v1[ok], v2[ok], vm[ok]


def divider_resistance(v1, v2, vm, r0, g21):
    """Rpot = R0 * (B/A) * g21, from v_m = A v1 + B v2 + C. Returns (R, residual)."""
    M = np.column_stack([v1, v2, np.ones_like(v1)])
    coef, *_ = np.linalg.lstsq(M, vm, rcond=None)
    A, B, _ = coef
    resid = float(np.sqrt(np.mean((vm - M @ coef) ** 2)))
    span = float(vm.max() - vm.min())
    return r0 * (B / A) * g21, resid / max(span, 1e-12)


def quant_step(v):
    """Smallest voltage step present in the data = the ADC LSB for that range."""
    u = np.unique(v)
    dd = np.diff(u)
    dd = dd[dd > 1e-9]
    return float(np.median(dd)) if len(dd) else float('nan')


def carrier_frequency(t, v, fmin, fmax):
    """Winding frequency, searched only inside a plausible band.

    The plain spectral maximum is not usable here. In a double scroll the
    spectrum is broad and its largest peak can sit far below the winding
    frequency, at the rate of lobe switching. Deriving the smoothing window and
    the minimum peak separation from such a value wipes out the real windings:
    in this circuit that cost about 95 % of the maxima across the whole chaotic
    range. Restricting the search to fmin..fmax fixes it.
    """
    dt = t[1] - t[0]
    x = (v - v.mean()) * np.hanning(len(v))
    F = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(len(v), dt)
    band = (fr >= fmin) & (fr <= min(fmax, 0.4 / dt))
    if not band.any():
        raise ValueError("carrier search band is empty")
    return float(fr[band][np.argmax(F[band])])


def extract_maxima(t, v, f0, prominence_frac, drop):
    """Smooth, find peaks, refine each to sub-sample accuracy by a parabola.

    Parabolic interpolation matters: with a few tens of samples per cycle the
    sampled peak sits at a random phase relative to the true maximum, and the
    resulting jitter is what smears a clean Lorenz map into a fuzzy band.
    """
    dt = t[1] - t[0]
    n0 = int(drop * len(t))
    t, v = t[n0:], v[n0:]

    per = 1.0 / f0
    win = int(per / 20 / dt)
    win = max(5, win | 1)                        # odd, at least 5
    vs = savgol_filter(v, win, 2) if win < len(v) // 4 else v

    span = vs.max() - vs.min()
    pk, _ = find_peaks(vs, prominence=prominence_frac * span,
                       distance=max(1, int(0.4 * per / dt)))
    pk = pk[(pk > 0) & (pk < len(vs) - 1)]

    y0, y1, y2 = vs[pk - 1], vs[pk], vs[pk + 1]
    den = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(den) > 1e-15, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    tm = t[pk] + shift * dt
    Mm = y1 - 0.25 * (y0 - y2) * shift
    return tm, Mm


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='.', help='folder with the csv records')
    p.add_argument('--pattern', default='*.csv')
    p.add_argument('--r0', type=float, default=992.0, help='fixed resistor, ohm')
    p.add_argument('--g21', type=float, default=1.0,
                   help='gain ratio of the v_C2 channel to the v_C1 channel')
    p.add_argument('--c1', default='CH1', help='column with v_C1')
    p.add_argument('--c2', default='CH2', help='column with v_C2')
    p.add_argument('--mid', default='CH3', help='column with the midpoint')
    p.add_argument('--drop', type=float, default=0.10,
                   help='fraction of each record discarded as transient')
    p.add_argument('--prominence', type=float, default=0.02,
                   help='peak prominence as a fraction of the signal range')
    p.add_argument('--fmin', type=float, default=500.0,
                   help='lower edge of the band searched for the winding frequency, Hz')
    p.add_argument('--fmax', type=float, default=20000.0,
                   help='upper edge of that band, Hz')
    p.add_argument('--out', default='maxima.csv')
    p.add_argument('--summary', default='summary.csv')
    p.add_argument('--plot', action='store_true', help='also draw the diagram')
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    files = [f for f in files
             if os.path.basename(f) not in (a.out, a.summary)]
    if not files:
        sys.exit(f"no files matching {a.pattern} in {a.dir}")

    rows, summ = [], []
    for path in files:
        name = os.path.basename(path)
        try:
            t, v1, v2, vm = read_record(path, a.c1, a.c2, a.mid)
            R, resid = divider_resistance(v1, v2, vm, a.r0, a.g21)
            f0 = carrier_frequency(t, v1, a.fmin, a.fmax)
            tm, Mm = extract_maxima(t, v1, f0, a.prominence, a.drop)
        except Exception as e:
            print(f"  {name}: skipped ({e})")
            continue

        for n, (ti, Mi) in enumerate(zip(tm, Mm)):
            rows.append((name, R, n, ti, Mi))

        T = float(np.mean(np.diff(tm))) if len(tm) > 1 else float('nan')
        # how many windings should there have been, at the winding frequency?
        expected = (t[-1] - t[0]) * (1 - a.drop) * f0
        frac = len(tm) / expected if expected > 0 else float('nan')
        summ.append(dict(file=name, R=R, n_max=len(tm), f0=f0, T_mean=T,
                         found=frac, resid=resid,
                         q1=quant_step(v1), q2=quant_step(v2), q3=quant_step(vm),
                         span=t[-1] - t[0], fs=1.0 / (t[1] - t[0])))
        flag = ""
        if resid > 0.05:
            flag += "  <-- divider fit poor"
        if frac < 0.9:
            flag += f"  <-- only {100*frac:.0f} % of windings found"
        print(f"  {name:22s} R = {R:8.2f}  f0 = {f0:7.1f} Hz  "
              f"{len(tm):5d} maxima{flag}")

    if not rows:
        sys.exit("nothing extracted")

    with open(os.path.join(a.dir, a.out), 'w') as f:
        f.write("file,R,n,t,M\n")
        for name, R, n, ti, Mi in rows:
            f.write(f"{name},{R:.4f},{n},{ti:.9e},{Mi:.6e}\n")

    keys = ['file', 'R', 'n_max', 'f0', 'T_mean', 'found', 'resid',
            'q1', 'q2', 'q3', 'span', 'fs']
    with open(os.path.join(a.dir, a.summary), 'w') as f:
        f.write(','.join(keys) + "\n")
        for s in summ:
            f.write(','.join(str(s[k]) if k == 'file' else f"{s[k]:.6g}"
                             for k in keys) + "\n")

    print(f"\n{len(rows)} maxima from {len(summ)} records")
    print(f"  {a.out}    one row per maximum")
    print(f"  {a.summary}  one row per record")

    # --- settings fingerprint -------------------------------------------
    for ch, key in (('CH1', 'q1'), ('CH2', 'q2'), ('CH3', 'q3')):
        vals = np.array([s[key] for s in summ])
        # tolerate rounding noise; a real V/div change moves the step by >=20 %
        if np.nanmax(vals) - np.nanmin(vals) > 0.01 * np.nanmax(vals):
            uniq = sorted(set(np.round(vals / np.nanmin(vals), 3) * np.nanmin(vals)))
            print(f"\nWARNING: the vertical scale of {ch} was not the same for all "
                  f"records.\n  quantisation steps found: "
                  f"{', '.join('%.4f mV' % (u*1e3) for u in uniq)}")
            print("  Records taken at different V/div have a systematic offset in R")
            print("  of up to a percent of (R0+R) -- tens of ohms. Do not put them on")
            print("  the same diagram without re-doing the channel intercalibration.")

    Rs = np.array([s['R'] for s in summ])
    print(f"\nR covered: {Rs.min():.1f} to {Rs.max():.1f} ohm, "
          f"{len(Rs)} records, median gap {np.median(np.diff(np.sort(Rs))):.2f} ohm")

    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        R = np.array([r[1] for r in rows])
        M = np.array([r[4] for r in rows])
        plt.figure(figsize=(11, 6))
        plt.scatter(R, M, s=1.0, c='k', lw=0)
        plt.gca().invert_xaxis()
        plt.xlabel('R, ohm')
        plt.ylabel(r'local maxima of $v_{C1}$, V')
        plt.grid(alpha=0.25)
        plt.tight_layout()
        out = os.path.join(a.dir, 'bifurcation.png')
        plt.savefig(out, dpi=120)
        print(f"  bifurcation.png")


if __name__ == '__main__':
    main()
