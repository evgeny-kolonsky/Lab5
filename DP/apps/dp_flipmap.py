#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dp_flipmap.py  --  map of the time to the first flip (TTFF) of a double
                   pendulum, over any rectangle of starting angles.

Both rods are released from rest at (theta1_0, theta2_0) and the clock runs
until the lower rod goes over the top of its own axle. The darker the pixel,
the sooner that happened; white means it never happened inside the horizon.

The white region in the middle is not dynamics but arithmetic: a flip needs
more energy than lifting the lower rod over its axle costs,

    e1 (1 - cos theta1_0) + e2 (1 - cos theta2_0)  >  2 e2 ,

and the black curve drawn on the map is exactly that boundary. It should land
on the edge of the white region, which is a free check that the map is right.
Around it the flip time is fractal.

The pendulum is described in config.ini (see README.md), the same bench
measurements dp_sim.py uses - friction included, so a run with no friction is
a matter of putting zeros in that file. If dp_sim.py sits next to this file,
its model is imported and used, so the map cannot drift away from the
simulator.

Examples
--------
    python dp_flipmap.py
    python dp_flipmap.py --th1 90 180 --th2 -60 60 --n 320
    python dp_flipmap.py --th1 110 140 --th2 -10 10 --n1 600 --n2 400 \
                         --horizon 30 --mark 125 0
"""

import argparse
import configparser
import os
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.ini")
sys.path.insert(0, APP_DIR)

__version__ = "1.2.0"

# --------------------------------------------------------------------------
#  configuration
# --------------------------------------------------------------------------
DEFAULTS = {
    "m1": "0.433",      # kg    mass of the upper rod
    "m2": "0.479",      # kg    mass of the lower rod
    "l1": "0.212",      # m     axis of rod 1 -> its centre of mass
    "l2": "0.219",      # m     axis of rod 2 -> its centre of mass
    "l3": "0.255",      # m     axis of rod 1 -> axis of rod 2
    "T1": "1.000",      # s     period of small swings, rod 1 hanging alone
    "T2": "1.000",      # s     the same for rod 2
    "half1": "30",      # s     time for that free decay to halve in amplitude
    "half2": "30",      # s     0 in both switches the friction off
    "g": "9.8",         # m/s2
}


def write_default_config(path=CONFIG_PATH):
    cp = configparser.ConfigParser()
    cp["pendulum"] = DEFAULTS
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Double pendulum bench measurements, one rod at a time.\n"
                 "# Lengths in metres, masses in kilograms, times in seconds.\n"
                 "# half1 = half2 = 0 means no friction.\n")
        cp.write(fh)
    return path


def load_config(path=CONFIG_PATH):
    """Bench numbers from config.ini; the file is created if it is missing."""
    if not os.path.exists(path):
        write_default_config(path)
        print("config.ini was not found - a default one has been written to\n"
              "   %s\nedit it to match your rig and run again.\n" % path)
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    sec = cp["pendulum"] if cp.has_section("pendulum") else {}
    raw = {}
    for k, v in DEFAULTS.items():
        try:
            raw[k] = float(sec.get(k, v))
        except (TypeError, ValueError):
            raise SystemExit("config.ini: '%s' is not a number" % k)
    return raw


# --------------------------------------------------------------------------
#  physics  (identical to dp_sim.py; imported from it when available)
# --------------------------------------------------------------------------
def _coefficients(p):
    a = p["I1"] + p["m1"] * p["l1"] ** 2 + p["m2"] * p["l3"] ** 2
    b = p["m2"] * p["l3"] * p["l2"]
    d = p["I2"] + p["m2"] * p["l2"] ** 2
    e1 = (p["m1"] * p["l1"] + p["m2"] * p["l3"]) * p["g"]
    e2 = p["m2"] * p["l2"] * p["g"]
    return a, b, d, e1, e2


def _from_measurements(m, l, period, half, g=9.8):
    """One rod hanging alone -> (inertia about its centre of mass, tau)."""
    J = m * g * l * (period / (2.0 * np.pi)) ** 2
    tau = half / (2.0 * np.log(2.0)) if half and half > 0 else 0.0
    return J - m * l ** 2, tau


def _damping_matrix(p):
    """Friction in the two bearings: c_i = J_i / tau_i, the second one acting
    on the relative rate, so C = [[c1 + c2, -c2], [-c2, c2]]."""
    t1 = float(p.get("tau1", 0) or 0)
    t2 = float(p.get("tau2", 0) or 0)
    if t1 <= 0 and t2 <= 0:
        return None
    c1 = (p["I1"] + p["m1"] * p["l1"] ** 2) / t1 if t1 > 0 else 0.0
    c2 = (p["I2"] + p["m2"] * p["l2"] ** 2) / t2 if t2 > 0 else 0.0
    return np.array([[c1 + c2, -c2], [-c2, c2]])


SIM = None
try:                                            # prefer the simulator's model
    import dp_sim as SIM                        # noqa: E402
except Exception:                               # tkinter missing, file absent
    SIM = None


def coefficients(p):
    f = getattr(SIM, "coefficients", None)
    return f(p) if f else _coefficients(p)


def from_measurements(m, l, period, half, g=9.8):
    f = getattr(SIM, "from_measurements", None)
    return f(m, l, period, half, g) if f else _from_measurements(m, l, period,
                                                                half, g)


def damping_matrix(p):
    f = getattr(SIM, "damping_matrix", None)
    return f(p) if f else _damping_matrix(p)


def params(cfg):
    """config.ini numbers -> the parameter dict the model works with."""
    p = {k: cfg[k] for k in ("m1", "m2", "l1", "l2", "l3", "g")}
    p["I1"], p["tau1"] = from_measurements(cfg["m1"], cfg["l1"], cfg["T1"],
                                           cfg["half1"], cfg["g"])
    p["I2"], p["tau2"] = from_measurements(cfg["m2"], cfg["l2"], cfg["T2"],
                                           cfg["half2"], cfg["g"])
    return p


def normal_modes(p):
    a, b, d, e1, e2 = coefficients(p)
    M = np.array([[a, b], [b, d]])
    K = np.array([[e1, 0.0], [0.0, e2]])
    w2 = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(M, K))))
    return np.sqrt(np.maximum(w2, 0.0))


def make_rhs(p):
    """RK4 right-hand side for MANY pendulums at once; y has shape (N, 4)."""
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


def flip_times(y, p, horizon, dt, which=2):
    """First time |theta| passes pi, for every row of y; inf if it never does.

    which = 2 watches the lower rod (the usual definition), 1 the upper one,
    0 whichever goes first.
    """
    rhs = make_rhs(p)
    tf = np.full(y.shape[0], np.inf)
    steps = int(round(horizon / dt))
    t, t0, last = 0.0, time.time(), 0.0
    for i in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        over = (np.abs(y[:, 1]) > np.pi) if which == 2 else \
               (np.abs(y[:, 0]) > np.pi) if which == 1 else \
               ((np.abs(y[:, 0]) > np.pi) | (np.abs(y[:, 1]) > np.pi))
        fresh = over & ~np.isfinite(tf)
        if fresh.any():
            tf[fresh] = t
        if time.time() - last > 5:
            last = time.time()
            print("   %3.0f %%   %5.0f s elapsed, %.0f %% already flipped"
                  % (100 * (i + 1) / steps, last - t0,
                     100 * np.isfinite(tf).mean()))
    return tf


def auto_name(args, n1, n2, friction, cmap_name):
    """A file name that says what is in the picture, e.g.

        flipmap_th1_90to180_th2_-60to60_320x220_h15_dt3ms_fric_magma.png

    Only characters Windows is happy with, so the file can live in Downloads.
    """
    def num(x):
        return ("%g" % x).replace(".", "p")
    return ("flipmap_th1_%sto%s_th2_%sto%s_%dx%d_h%s_dt%sms_%s_%s.png"
            % (num(args.th1[0]), num(args.th1[1]),
               num(args.th2[0]), num(args.th2[1]), n1, n2,
               num(args.horizon), num(1000 * args.dt),
               "fric" if friction else "nofric", cmap_name))


def free_path(path):
    """The given path, or the first path_1, path_2 ... that does not exist."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for i in range(1, 1000):
        cand = "%s_%d%s" % (stem, i, ext)
        if not os.path.exists(cand):
            return cand
    return "%s_%d%s" % (stem, int(time.time()), ext)


def save_figure(fig, path, dpi=130):
    """Save without ever losing the run.

    A picture already open in a viewer cannot be overwritten on Windows, and
    the whole folder may be read-only. So: never overwrite an existing file,
    and if writing still fails, step aside to the next free name and finally
    to the temporary folder.
    """
    import tempfile
    tried = []
    for cand in (free_path(path), free_path(path), 
                 os.path.join(tempfile.gettempdir(),
                              os.path.basename(free_path(path)))):
        if cand in tried:
            continue
        tried.append(cand)
        try:
            fig.savefig(cand, dpi=dpi)
            return cand
        except OSError as exc:
            print("   could not write %s (%s)" % (cand, exc))
    return None


def grid_states(th1_deg, th2_deg):
    """Every combination of the two angle arrays, released from rest."""
    G1, G2 = np.meshgrid(th1_deg, th2_deg, indexing="xy")
    y = np.zeros((G1.size, 4))
    y[:, 0] = np.radians(G1.ravel())
    y[:, 1] = np.radians(G2.ravel())
    return y, G1, G2


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--th1", type=float, nargs=2, default=[-180.0, 180.0],
                    metavar=("MIN", "MAX"), help="theta1_0 range, deg")
    ap.add_argument("--th2", type=float, nargs=2, default=[-180.0, 180.0],
                    metavar=("MIN", "MAX"), help="theta2_0 range, deg")
    ap.add_argument("--n", type=int, default=200,
                    help="grid points per axis (default 200)")
    ap.add_argument("--n1", type=int, default=None, help="points along theta1")
    ap.add_argument("--n2", type=int, default=None, help="points along theta2")
    ap.add_argument("--horizon", type=float, default=15.0,
                    help="how long to watch, s (default 15)")
    ap.add_argument("--dt", type=float, default=3e-3, help="RK4 step, s")
    ap.add_argument("--flip-of", type=int, default=2, choices=(0, 1, 2),
                    help="which rod must go over the top: 2 (default), 1, "
                         "or 0 for whichever does it first")
    ap.add_argument("--mark", type=float, nargs=2, default=None,
                    metavar=("TH1", "TH2"), help="put a marker at this point")
    ap.add_argument("--colour", "--color", dest="colour", default="magma",
                    choices=("magma", "cubehelix", "ice", "mono"),
                    help="colour scheme; all of them run dark (early flip) to "
                         "pale (late), and no flip at all stays pure white")
    ap.add_argument("--out", default=None,
                    help="output PNG; by default the name is built from the "
                         "parameters and never overwrites an existing file")
    args = ap.parse_args()

    n1 = args.n1 or args.n
    n2 = args.n2 or args.n
    cfg = load_config()
    p = params(cfg)
    a, b, d, e1, e2 = coefficients(p)

    print("dp_flipmap %s   model: %s" % (__version__,
          "dp_sim.py" if SIM is not None else
          "built-in copy (dp_sim not imported)"))
    print("rods: m = %.3f / %.3f kg, l1 = %.3f, l2 = %.3f, l3 = %.3f m"
          % (p["m1"], p["m2"], p["l1"], p["l2"], p["l3"]))
    print("      I about the centres of mass: %.5f / %.5f kg m2"
          % (p["I1"], p["I2"]))
    print("      friction: %s"
          % ("none (half1 = half2 = 0)" if damping_matrix(p) is None else
             "tau = %.1f / %.1f s" % (p["tau1"], p["tau2"])))
    w = normal_modes(p)
    print("      small oscillations: %.3f s (%.3f Hz) and %.3f s (%.3f Hz)"
          % (2 * np.pi / w[0], w[0] / 2 / np.pi,
             2 * np.pi / w[1], w[1] / 2 / np.pi))
    edge = (np.degrees(np.arccos(1 - 2 * e2 / e1))
            if abs(1 - 2 * e2 / e1) <= 1 else float("nan"))
    print("      a flip needs E > 2 e2 = %.4f J; on the line theta2_0 = 0 that "
          "is theta1_0 > %.2f deg" % (2 * e2, edge))

    g1 = np.linspace(args.th1[0], args.th1[1], max(1, n1))
    g2 = np.linspace(args.th2[0], args.th2[1], max(1, n2))
    y, G1, G2 = grid_states(g1, g2)
    print("\nmap: theta1 %g..%g deg in %d steps, theta2 %g..%g deg in %d steps "
          "= %d runs of %.0f s"
          % (g1[0], g1[-1], len(g1), g2[0], g2[-1], len(g2), y.shape[0],
             args.horizon))
    tf = flip_times(y, p, args.horizon, args.dt, args.flip_of)
    TF = tf.reshape(G1.shape)
    V = e1 * (1 - np.cos(np.radians(G1))) + e2 * (1 - np.cos(np.radians(G2)))

    ok = np.isfinite(tf)
    print("\nflipped within %.0f s: %.1f %% of the rectangle"
          % (args.horizon, 100 * ok.mean()))
    if ok.any():
        print("earliest flip %.2f s, latest %.2f s" % (tf[ok].min(), tf[ok].max()))

    out = args.out or os.path.join(os.getcwd(),
                                   auto_name(args, len(g1), len(g2),
                                             damping_matrix(p) is not None,
                                             args.colour))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Every scheme runs dark (flips at once) to pale (flips just in time)
        # and leaves pure white for "never flipped", so the white region reads
        # as a category of its own rather than as the end of the scale.
        import matplotlib.colors as mcolors
        base, line = {
            "magma":     ("magma", "#00c2d4"),      # black-purple-orange-cream
            "cubehelix": ("cubehelix", "#e03c31"),  # brightness rises steadily
            "ice":       ("YlGnBu_r", "#d1495b"),   # deep blue-green to cream
            "mono":      ("gray", "#b2301f"),
        }[args.colour]
        src = plt.get_cmap(base)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            base + "_cut", src(np.linspace(0.0, 0.93, 256)))
        cmap.set_bad("white")
        fig, ax = plt.subplots(figsize=(7.6, 6.2), layout="constrained")
        # most of a map flips in the first second or two, so a plain linear
        # shade leaves everything black; the square root keeps "darker =
        # sooner" while spreading the early times out
        norm = matplotlib.colors.PowerNorm(0.5, vmin=0.0, vmax=args.horizon)
        im = ax.pcolormesh(g1, g2, np.where(ok.reshape(TF.shape), TF, np.nan),
                           cmap=cmap, shading="auto", norm=norm)
        if V.min() < 2 * e2 < V.max():
            ax.contour(g1, g2, V, levels=[2 * e2], colors=line, linewidths=1.2)
        if args.mark:
            ax.plot([args.mark[0]], [args.mark[1]], "o", ms=9,
                    mfc="none", mec=line, mew=1.8)
        ax.set_xlabel("theta1_0, deg")
        ax.set_ylabel("theta2_0, deg")
        ax.set_title("Time to the first flip\n"
                     "dark = flips at once, pale = flips late, "
                     "white = no flip within %.0f s" % args.horizon)
        cb = fig.colorbar(im, ax=ax, label="TTFF, s")
        cb.set_ticks([v for v in (0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60)
                      if v <= args.horizon] + [args.horizon])
        saved = save_figure(fig, out)
        print("\nwritten: %s" % saved if saved else
              "\nTHE PICTURE COULD NOT BE SAVED ANYWHERE - the numbers above "
              "are still good")
    except Exception as exc:
        print("\n(no plot: %s)" % exc)


if __name__ == "__main__":
    main()
