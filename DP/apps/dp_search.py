#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dp_search.py  --  find the starting angle at which the double pendulum stops
                  being predictable.

Question it answers
-------------------
The second rod always starts hanging still (theta2_0 = 0, both rates zero).
Only theta1_0 is varied. Two runs are made from every candidate angle:

    reference:  theta1_0
    perturbed:  theta1_0 + delta          (delta = 1 deg by default)

and we ask when the two theta2(t) traces part company by more than a threshold
(10 deg by default) within a horizon (30 s by default).

    t_cross(theta1_0)  = first time |theta2_pert - theta2_ref| > threshold
                         (inf if it never happens inside the horizon)

The answer is the smallest theta1_0 whose t_cross fits inside the horizon.

Method
------
1. a coarse sweep over theta1_0 (all angles integrated at once, vectorised),
2. bisection inside the first bracket where the answer changes,
3. a robustness check: the smallest angle from which EVERY sampled angle above
   it also crosses - the plain boundary can sit on a narrow tongue, and for a
   student experiment the robust one is the number worth quoting,
4. a nature check: the whole sweep is repeated with a ten times smaller
   perturbation. If the angle still crosses the threshold, the growth is
   exponential (real chaos) and the delay gives lambda = ln10 / (t_small -
   t_big). If it no longer crosses, the 1 deg case was merely a 1 deg error
   amplified by a smooth, non-chaotic factor of ten - worth telling apart.

Everything is written to a CSV and a PNG next to this script.

Usage
-----
    python dp_search.py
    python dp_search.py --from 20 --to 170 --step 1 --delta 1 --threshold 10
    python dp_search.py --horizon 30 --dt 0.001 --no-friction

The pendulum is described the same way as in dp_sim.py: one rod at a time on
the bench - mass, distance to the centre of mass, period of small oscillations,
and the time in which the amplitude halves. Edit BENCH below to match the rig.
The script is standalone - only numpy is needed (matplotlib for the picture).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- physics
# The same model as dp_sim.py, repeated here in a dozen lines so that this
# script runs on its own (no GUI, no tkinter).

def coefficients(p):
    a = p["I1"] + p["m1"] * p["l1"] ** 2 + p["m2"] * p["l3"] ** 2
    b = p["m2"] * p["l3"] * p["l2"]
    d = p["I2"] + p["m2"] * p["l2"] ** 2
    e1 = (p["m1"] * p["l1"] + p["m2"] * p["l3"]) * p["g"]
    e2 = p["m2"] * p["l2"] * p["g"]
    return a, b, d, e1, e2


def from_measurements(m, l, period, half, g=9.81):
    """One rod hanging alone -> (I about its centre of mass, tau)."""
    J = m * g * l * (period / (2.0 * np.pi)) ** 2
    tau = half / (2.0 * np.log(2.0)) if half and half > 0 else 0.0
    return J - m * l ** 2, tau


def damping_matrix(p):
    """C in Q = -C [w1, w2]; friction in the two bearings, c_i = J_i / tau_i."""
    t1, t2 = float(p.get("tau1", 0) or 0), float(p.get("tau2", 0) or 0)
    if t1 <= 0 and t2 <= 0:
        return None
    c1 = (p["I1"] + p["m1"] * p["l1"] ** 2) / t1 if t1 > 0 else 0.0
    c2 = (p["I2"] + p["m2"] * p["l2"] ** 2) / t2 if t2 > 0 else 0.0
    return np.array([[c1 + c2, -c2], [-c2, c2]])


def normal_modes(p):
    a, b, d, e1, e2 = coefficients(p)
    M = np.array([[a, b], [b, d]])
    K = np.array([[e1, 0.0], [0.0, e2]])
    w2 = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(M, K))))
    return np.sqrt(np.maximum(w2, 0.0))

# --------------------------------------------------------------- the rig
BENCH = dict(
    m1=0.438, l1=0.215, T1=1.000, half1=30.0,     # upper rod, alone
    m2=0.480, l2=0.215, T2=1.000, half2=30.0,     # lower rod, alone
    l3=0.255,                                     # pivot -> second axle
    g=9.81,
)


def params(friction=True):
    p = dict(m1=BENCH["m1"], m2=BENCH["m2"], l1=BENCH["l1"], l2=BENCH["l2"],
             l3=BENCH["l3"], g=BENCH["g"])
    p["I1"], p["tau1"] = from_measurements(
        BENCH["m1"], BENCH["l1"], BENCH["T1"],
        BENCH["half1"] if friction else 0.0, BENCH["g"])
    p["I2"], p["tau2"] = from_measurements(
        BENCH["m2"], BENCH["l2"], BENCH["T2"],
        BENCH["half2"] if friction else 0.0, BENCH["g"])
    return p


# ------------------------------------------------- vectorised integration
def make_rhs(p):
    """RK4 right-hand side for MANY systems at once; y has shape (N, 4).

    Same equations as dp_sim.derivs - kept vectorised here because the sweep
    integrates a few hundred pendulums, and one at a time would be slow.
    """
    a, b, d, e1, e2 = coefficients(p)
    C = damping_matrix(p)

    def rhs(y):
        th1, th2, w1, w2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        delta = th1 - th2
        cd, sd = np.cos(delta), np.sin(delta)
        if C is None:
            q1 = q2 = 0.0
        else:
            q1 = -(C[0, 0] * w1 + C[0, 1] * w2)
            q2 = -(C[1, 0] * w1 + C[1, 1] * w2)
        f1 = -b * w2 * w2 * sd - e1 * np.sin(th1) + q1
        f2 = b * w1 * w1 * sd - e2 * np.sin(th2) + q2
        det = a * d - (b * cd) ** 2
        out = np.empty_like(y)
        out[:, 0] = w1
        out[:, 1] = w2
        out[:, 2] = (d * f1 - b * cd * f2) / det
        out[:, 3] = (a * f2 - b * cd * f1) / det
        return out

    return rhs


def wrap(x):
    """Angle difference folded into (-pi, pi] - 350 deg apart is 10 deg apart."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def crossing_times(angles_deg, p, delta_deg, threshold_deg, horizon, dt):
    """For every theta1_0 in angles_deg: the first time |dtheta2| > threshold.

    Both members of each twin pair are integrated in the same array, so the two
    runs see bit-for-bit the same arithmetic and the only difference between
    them is the perturbation itself.
    """
    n = len(angles_deg)
    y = np.zeros((2 * n, 4))
    y[:n, 0] = np.radians(angles_deg)                       # reference
    y[n:, 0] = np.radians(angles_deg + delta_deg)           # perturbed
    rhs = make_rhs(p)

    thr = np.radians(threshold_deg)
    t_cross = np.full(n, np.inf)
    max_gap = np.zeros(n)
    steps = int(round(horizon / dt))
    t = 0.0
    for i in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        gap = np.abs(wrap(y[n:, 1] - y[:n, 1]))
        max_gap = np.maximum(max_gap, gap)
        fresh = (gap > thr) & ~np.isfinite(t_cross)
        if fresh.any():
            t_cross[fresh] = t
    return t_cross, np.degrees(max_gap)


def scan(angles_deg, p, args):
    return crossing_times(np.asarray(angles_deg, float), p, args.delta,
                          args.threshold, args.horizon, args.dt)


def bisect(lo, hi, p, args, tol=0.01):
    """Narrow the bracket [lo (no crossing), hi (crossing)] down to tol deg."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        t, _ = scan([mid], p, args)
        if np.isfinite(t[0]):
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="a0", type=float, default=1.0,
                    help="lowest theta1_0 to try, deg (default 1)")
    ap.add_argument("--to", dest="a1", type=float, default=179.0,
                    help="highest theta1_0 to try, deg (default 179)")
    ap.add_argument("--step", type=float, default=1.0,
                    help="sweep step, deg (default 1)")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="perturbation added to theta1_0, deg (default 1)")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="theta2 difference that counts as divergence, deg")
    ap.add_argument("--horizon", type=float, default=30.0,
                    help="how long we watch, s (default 30)")
    ap.add_argument("--dt", type=float, default=1e-3,
                    help="RK4 step, s (default 0.001)")
    ap.add_argument("--no-friction", action="store_true",
                    help="ignore the measured decay times")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="where to write the CSV and the PNG")
    args = ap.parse_args()

    p = params(friction=not args.no_friction)
    print("pendulum: I1=%.6f I2=%.6f kg m^2 (about each c.o.m.)"
          % (p["I1"], p["I2"]))
    print("          tau1=%.2f s tau2=%.2f s%s"
          % (p["tau1"], p["tau2"], "  (friction off)" if args.no_friction else ""))
    w = normal_modes(p)
    print("          small-oscillation periods %.3f s and %.3f s"
          % (2 * np.pi / w[0], 2 * np.pi / w[1]))
    print("\nsearching theta1_0 in [%.1f, %.1f] deg, step %.2f;"
          % (args.a0, args.a1, args.step))
    print("theta2_0 = 0, rates 0; perturbation +%.3g deg on theta1_0;" % args.delta)
    print("divergence = |dtheta2| > %.3g deg within %.1f s\n"
          % (args.threshold, args.horizon))

    angles = np.arange(args.a0, args.a1 + 1e-9, args.step)
    t_cross, max_gap = scan(angles, p, args)
    hit = np.isfinite(t_cross)

    # the same sweep with a ten times smaller perturbation: does it still cross?
    small = argparse.Namespace(**vars(args))
    small.delta = args.delta / 10.0
    t_small, gap_small = scan(angles, p, small)
    kind = []
    lam = np.full(len(angles), np.nan)
    for i in range(len(angles)):
        if not hit[i]:
            kind.append("-")
        elif np.isfinite(t_small[i]):
            kind.append("chaotic")
            if t_small[i] > t_cross[i]:
                lam[i] = np.log(10.0) / (t_small[i] - t_cross[i])
        else:
            kind.append("linear")

    print("%-9s %-11s %-11s %-9s %-9s %s"
          % ("theta1_0", "t_cross, s", "max|dth2|", "t(d/10)", "lambda", "kind"))
    for i, a in enumerate(angles):
        print("%8.2f  %10s  %10.2f  %8s  %8s  %s"
              % (a,
                 "%.2f" % t_cross[i] if hit[i] else "-",
                 max_gap[i],
                 "%.2f" % t_small[i] if np.isfinite(t_small[i]) else "-",
                 "%.2f" % lam[i] if np.isfinite(lam[i]) else "-",
                 kind[i]))

    print()
    if not hit.any():
        print("No angle in this range diverges within the horizon.")
        return

    first = int(np.argmax(hit))
    if first == 0:
        boundary = angles[0]
        print("The whole range already diverges - lower --from to find the edge.")
    else:
        boundary = bisect(angles[first - 1], angles[first], p, args)
        print("boundary        : theta1_0 = %.2f deg  (the first angle that "
              "diverges; below it, nothing happens within %.0f s)"
              % (boundary, args.horizon))

    # robust boundary: from here on, every sampled angle diverges
    robust = None
    for i in range(len(angles)):
        if hit[i:].all():
            robust = angles[i]
            break
    if robust is not None:
        if robust > boundary + 1e-9:
            lo = robust - args.step
            print("robust boundary : theta1_0 = %.2f deg  (from here up, EVERY "
                  "sampled angle diverges; between %.2f and %.2f deg the "
                  "regular and chaotic starts are interleaved)"
                  % (robust, boundary, lo))
        else:
            print("robust boundary : the same angle - no interleaving above it")
    t_at = t_cross[hit]
    print("time to diverge : %.1f s at the boundary, %.1f s median over the "
          "diverging angles, %.1f s at the fastest"
          % (t_cross[first], np.median(t_at), t_at.min()))

    chao = np.array([k == "chaotic" for k in kind])
    if chao.any():
        firstc = int(np.argmax(chao))
        good = np.isfinite(lam)
        print("truly chaotic   : from theta1_0 = %.2f deg upwards (a ten times "
              "smaller kick still diverges)" % angles[firstc])
        if good.any():
            print("                  lambda = %.2f 1/s median over those angles "
                  "(e-folding %.2f s)"
                  % (np.median(lam[good]), 1.0 / np.median(lam[good])))
    lin = np.array([k == "linear" for k in kind])
    if lin.any():
        print("merely amplified: %d of the crossing angles fail this test - "
              "there a 1 deg start error simply grows by a smooth factor of "
              "about ten, without exponential separation" % lin.sum())

    # ---- files
    csv = os.path.join(args.out, "dp_search.csv")
    with open(csv, "w", encoding="utf-8", newline="") as fh:
        fh.write("# dp_search: theta2_0 = 0, perturbation +%g deg on theta1_0,\n"
                 "# divergence = |dtheta2| > %g deg, horizon %g s, RK4 dt = %g s\n"
                 % (args.delta, args.threshold, args.horizon, args.dt))
        fh.write("theta1_0_deg,t_cross_s,max_dtheta2_deg,t_cross_delta_over_10_s,"
                 "lambda_1_s,kind,diverges\n")
        for i, a in enumerate(angles):
            fh.write("%.4f,%s,%.4f,%s,%s,%s,%d\n"
                     % (a,
                        "%.4f" % t_cross[i] if hit[i] else "",
                        max_gap[i],
                        "%.4f" % t_small[i] if np.isfinite(t_small[i]) else "",
                        "%.4f" % lam[i] if np.isfinite(lam[i]) else "",
                        kind[i], int(hit[i])))

    png = os.path.join(args.out, "dp_search.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                               layout="constrained")
        shown = np.where(hit, t_cross, np.nan)
        ax[0].plot(angles, shown, ".-", lw=.8, ms=3, color="#1f77b4",
                   label="perturbation %g deg" % args.delta)
        ax[0].plot(angles, np.where(np.isfinite(t_small), t_small, np.nan),
                   ".-", lw=.8, ms=3, color="#ff7f0e", alpha=.8,
                   label="perturbation %g deg" % small.delta)
        ax[0].axhline(args.horizon, color="#888", ls=":")
        ax[0].axvline(boundary, color="r", ls="--",
                      label="boundary %.2f deg" % boundary)
        if robust is not None and robust > boundary:
            ax[0].axvline(robust, color="g", ls="--",
                          label="robust %.2f deg" % robust)
        ax[0].set_ylabel("time to diverge, s")
        ax[0].set_title("theta2_0 = 0, perturbation +%g deg on theta1_0, "
                        "divergence = %g deg of theta2 within %g s"
                        % (args.delta, args.threshold, args.horizon))
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=.3)
        ax[1].plot(angles, max_gap, ".-", lw=.8, ms=3, color="#1f77b4")
        ax[1].plot(angles, gap_small, ".-", lw=.8, ms=3, color="#ff7f0e", alpha=.8)
        ax[1].axhline(args.threshold, color="r", ls=":", label="threshold")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("theta1_0, deg")
        ax[1].set_ylabel("largest |dtheta2| seen, deg")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=.3)
        fig.savefig(png, dpi=110)
        print("\nwritten: %s\n         %s" % (csv, png))
    except Exception as exc:
        print("\nwritten: %s   (no plot: %s)" % (csv, exc))


if __name__ == "__main__":
    main()
