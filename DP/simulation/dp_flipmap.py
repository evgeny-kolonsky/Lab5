#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dp_flipmap.py  --  the "time to the first flip" map for THIS pendulum.

The classic picture (usually drawn for two equal point-mass rods) coloured by
how long it takes the lower rod to go over the top, as a function of the two
starting angles, both rods released at rest. The white region in the middle is
where a flip is impossible at all: the released potential energy is less than
what lifting rod 2 over its axle would cost,

    e1 (1 - cos theta1_0) + e2 (1 - cos theta2_0)  <  2 e2 ,

and its boundary is drawn as a black curve here - it lands exactly on the edge
of the white region, which is a good check that the map is right. Around that
boundary the flip time is fractal: buds and filaments of long survival soaked
in a sea of quick flips.

This script draws the same map for the rig described in dp_search.BENCH, plus
the one-dimensional cut theta2_0 = 0 that dp_search.py actually sweeps, so the
islands found there can be placed on the map.

    python dp_flipmap.py                       # frictionless, the classic map
    python dp_flipmap.py --friction            # with the measured decay times
    python dp_flipmap.py --n 300 --horizon 20  # finer, slower

Note that friction changes the picture qualitatively: a marginal flip that the
conservative system just manages is easily lost when a few percent of the
energy has gone, so a dissipative map grows extra non-flipping windows that
have nothing to do with the energy boundary.
"""

import argparse
import os
import time

import numpy as np

import dp_search as D


def flip_times(y, p, horizon, dt, progress=True):
    """First time |theta2| exceeds pi for each row of y; inf if it never does."""
    rhs = D.make_rhs(p)
    tf = np.full(y.shape[0], np.inf)
    steps = int(round(horizon / dt))
    t = 0.0
    t0 = time.time()
    for i in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        fresh = (np.abs(y[:, 1]) > np.pi) & ~np.isfinite(tf)
        if fresh.any():
            tf[fresh] = t
        if progress and i % max(1, steps // 10) == 0:
            print("   %3.0f%%  %5.0f s elapsed" % (100 * i / steps, time.time() - t0))
    return tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="grid points per axis")
    ap.add_argument("--horizon", type=float, default=12.0, help="watch time, s")
    ap.add_argument("--dt", type=float, default=3e-3, help="RK4 step, s")
    ap.add_argument("--friction", action="store_true",
                    help="use the measured decay times (default: conservative)")
    ap.add_argument("--mark", type=float, default=None,
                    help="mark this theta1_0 on the map and the cut, deg")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    p = D.params(friction=args.friction)
    a, b, d, e1, e2 = D.coefficients(p)
    edge = np.degrees(np.arccos(max(-1.0, 1 - 2 * e2 / e1)))
    print("e1 = %.4f N m, e2 = %.4f N m; a flip needs E > 2 e2 = %.4f J"
          % (e1, e2, 2 * e2))
    print("with theta2_0 = 0 that is theta1_0 > %.2f deg" % edge)
    print("friction: %s\n" % ("on" if args.friction else "off"))

    # ---- the map
    g = np.linspace(-180, 180, args.n)
    G1, G2 = np.meshgrid(g, g, indexing="xy")
    y = np.zeros((G1.size, 4))
    y[:, 0] = np.radians(G1.ravel())
    y[:, 1] = np.radians(G2.ravel())
    V = e1 * (1 - np.cos(y[:, 0])) + e2 * (1 - np.cos(y[:, 1]))
    print("map: %d x %d starting points" % (args.n, args.n))
    tf = flip_times(y, p, args.horizon, args.dt).reshape(args.n, args.n)

    # ---- the cut theta2_0 = 0, at ten times the resolution
    ang = np.arange(0.0, 180.01, 0.1)
    yc = np.zeros((len(ang), 4))
    yc[:, 0] = np.radians(ang)
    print("cut: %d points along theta2_0 = 0" % len(ang))
    tc = flip_times(yc, p, max(args.horizon, 30.0), 1e-3, progress=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    im = ax[0].pcolormesh(g, g, np.where(np.isfinite(tf), tf, np.nan),
                          cmap="turbo", shading="auto", vmin=0,
                          vmax=0.7 * args.horizon)
    ax[0].contour(g, g, V.reshape(args.n, args.n), levels=[2 * e2],
                  colors="k", linewidths=1)
    ax[0].axhline(0, color="w", lw=1.2)
    if args.mark is not None:
        ax[0].plot([args.mark], [0], "w*", ms=13, mec="k")
    ax[0].set_xlabel("theta1_0, deg")
    ax[0].set_ylabel("theta2_0, deg")
    ax[0].set_title("time to the first flip of rod 2, s   "
                    "(white = no flip in %.0f s, friction %s)"
                    % (args.horizon, "on" if args.friction else "off"))
    fig.colorbar(im, ax=ax[0], label="s")

    ok = np.isfinite(tc)
    ax[1].plot(ang, np.where(ok, tc, np.nan), ".", ms=2)
    if (~ok).any():
        ax[1].plot(ang[~ok], np.full((~ok).sum(), tc[ok].max() if ok.any() else 30),
                   "|", color="#d62728", ms=8, label="no flip")
    ax[1].axvline(edge, color="g", ls=":", label="energy threshold %.1f deg" % edge)
    if args.mark is not None:
        ax[1].axvline(args.mark, color="k", ls="--", lw=1,
                      label="%.1f deg" % args.mark)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("theta1_0, deg   (theta2_0 = 0)")
    ax[1].set_ylabel("time to the first flip, s")
    ax[1].set_title("the cut along the white line")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    png = os.path.join(args.out, "dp_flipmap.png")
    fig.savefig(png, dpi=110)

    nof = ang[~ok]
    nof = nof[nof > edge]
    if nof.size:
        runs, start = [], nof[0]
        for x, nxt in zip(nof, list(nof[1:]) + [None]):
            if nxt is None or nxt - x > 0.15:
                runs.append((start, x))
                start = nxt
        print("\nnon-flipping windows above the energy threshold, deg:")
        for lo, hi in runs:
            print("   %7.1f ... %-7.1f  (%.1f deg wide)" % (lo, hi, hi - lo))
    print("\nwritten: %s" % png)


if __name__ == "__main__":
    main()
