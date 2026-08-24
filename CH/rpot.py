"""
Measure the potentiometer from the midpoint voltage, without opening the circuit.

R0 and the potentiometer are in series between the v_C1 and v_C2 nodes, with
CH3 at the junction. No current flows into the scope probe, so the midpoint is
a plain voltage divider:

    v_m = A * v_C1 + B * v_C2 + C

Fit A and B over the whole record, then

    Rpot = R0 * (B / A) * (g2 / g1)

This form uses both coefficients, so the midpoint channel's gain cancels out
entirely; only the relative gain of the two end channels survives, and that is
close to 1 when they sit on the same V/div. The fitted offset C absorbs any
DC offset in the channels.

Defaults match this bench: R0 = 992 ohm, CH1 = v_C1, CH2 = v_C2, CH3 = midpoint.

Usage:
    python rpot.py record.csv
    python rpot.py record.csv --g21 0.9760
    python rpot.py *.csv                              # whole sweep at once
"""
import argparse
import glob
import numpy as np


def rpot(path, r0, g21, c1='CH1', c2='CH2', mid='CH3'):
    """Return (Rpot, residual_fraction) for one record."""
    with open(path) as f:
        names = [h.split('(')[0].strip() for h in f.readline().split(',')]
    d = np.genfromtxt(path, delimiter=',', skip_header=1)
    v1, v2, vm = (d[:, names.index(c1)], d[:, names.index(c2)],
                  d[:, names.index(mid)])
    ok = np.isfinite(v1) & np.isfinite(v2) & np.isfinite(vm)
    v1, v2, vm = v1[ok], v2[ok], vm[ok]

    M = np.column_stack([v1, v2, np.ones_like(v1)])
    (A, B, _), *_ = np.linalg.lstsq(M, vm, rcond=None)
    resid = float(np.sqrt(np.mean((vm - M @ np.linalg.lstsq(M, vm, rcond=None)[0]) ** 2)))
    span = float(vm.max() - vm.min())
    return r0 * (B / A) * g21, resid / max(span, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', nargs='+')
    p.add_argument('--r0', type=float, default=992.0, help='fixed resistor, ohm')
    p.add_argument('--g21', type=float, default=1.0,
                   help='gain ratio of the v_C2 channel to the v_C1 channel')
    p.add_argument('--c1', default='CH1', help='column with v_C1')
    p.add_argument('--c2', default='CH2', help='column with v_C2')
    p.add_argument('--mid', default='CH3', help='column with the midpoint')
    a = p.parse_args()

    files = []
    for pat in a.csv:
        files.extend(sorted(glob.glob(pat)) or [pat])

    for f in files:
        try:
            r, res = rpot(f, a.r0, a.g21, a.c1, a.c2, a.mid)
        except Exception as e:
            print(f"{f}: {e}")
            continue
        warn = "   CHECK: not a clean divider" if res > 0.05 else ""
        print(f"{f}: Rpot = {r:7.1f} ohm   (fit residual {100*res:.2f} %){warn}")


if __name__ == '__main__':
    main()
