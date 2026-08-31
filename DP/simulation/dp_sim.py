#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dp_sim.py  --  Double Pendulum Simulator (GUI)

Numerical experiments with a compound double pendulum, aimed at one question:
how long does it take for two nearly identical initial conditions to diverge?

Model (compound bodies, not point masses):
    l1  pivot          -> centre of mass of body 1
    l2  second axle     -> centre of mass of body 2
    l3  pivot           -> second axle   (distance between the two hinges)

    body 1: mass m1, moment of inertia I1 about its own centre of mass
    body 2: mass m2, moment of inertia I2 about its own centre of mass
    theta is measured from the downward vertical, positive to the right.

    a = I1 + m1 l1^2 + m2 l3^2      b = m2 l3 l2      d = I2 + m2 l2^2
    e1 = (m1 l1 + m2 l3) g          e2 = m2 l2 g

    [ a            b cos(th1-th2) ] [th1'']   [ -b w2^2 sin(th1-th2) - e1 sin th1 + Q1 ]
    [ b cos(th1-th2)      d       ] [th2''] = [ +b w1^2 sin(th1-th2) - e2 sin th2 + Q2 ]

Everything about the two rods is entered as MEASURED on the bench, one rod at
a time, hanging from its own axle:

    T1, T2      period of small free oscillations of that rod alone [s]
    half1, half2  time in which the amplitude of that free decay halves [s]

from which the model takes

    J = m g l (T / 2 pi)^2      moment of inertia about the axle
    I = J - m l^2               about the centre of mass
    tau = half / (2 ln2)        amplitude ~ exp(-t / (2 tau))
    c = J / tau                 viscous coefficient of that bearing

The friction stays where it physically is - in the two hinges - so with
F = 1/2 c1 w1^2 + 1/2 c2 (w2 - w1)^2 the friction torques are Q = -C [w1, w2],
C = [[c1 + c2, -c2], [-c2, c2]].

Integration: classical RK4 with a fixed step. With friction off, the drift of
the total energy over the run is reported - it is the honest accuracy figure of
the integration.

Run:
    python dp_sim.py

Dependencies: numpy, matplotlib (tkinter comes with Python)

CHANGELOG
    1.0.0  first version: model tab with a draggable pendulum, settings tab,
           list of runs, overlay plots, twin runs with a controlled
           perturbation, divergence analysis with a Lyapunov fit, CSV export
    1.1.0  geometry renamed to the convention used on the real rig:
           l1 = pivot -> c.o.m. 1, l2 = 2nd axle -> c.o.m. 2,
           l3 = distance between the hinges; defaults 215/215/250 mm.
           Second rod drawn to its full length 2*l2; CSV now exports the
           centre of mass of body 2 (xc2_m, yc2_m) instead of a mixed point.
    1.2.0  zoom and pan on the time axis of the plot: buttons, numeric
           from/to window, mouse wheel zoom at the cursor, drag to pan,
           double click to fit; y is rescaled to the visible window.
    1.2.1  small-oscillation modes now reported as period, frequency and
           omega together (period alone invited a Hz/s mix-up); helper that
           converts a measured single-rod period into I about the centre of
           mass, since I1/I2 are asked for about the c.o.m., not the axle.
    1.3.0  two friction models: "bearings" (default; tau1, tau2 measured on
           each rod hanging alone, c_i = J_i/tau_i, second hinge acting on the
           relative rate) and the previous "normal modes". Helper converting a
           measured half-life into tau, a live reminder of what tau means, and
           a rescaling of tau1, tau2 to reproduce a measured energy decay of
           the assembled pendulum.
    1.4.0  simplified: the settings tab now asks only for what is measured on
           the bench with one rod at a time - mass, distance to the centre of
           mass, period T and the time in which the amplitude halves. Moments
           of inertia and friction coefficients follow from those, and the
           alternative friction models and conversion dialogs are gone.
"""

__version__ = "1.4.0"

import configparser
import os
import sys
import threading
import traceback

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "dp_sim.ini")

DEFAULTS = {
    "version": __version__,
    "m1": "0.5359", "m2": "0.4794",          # kg
    "T1": "1.000", "T2": "1.000",            # s, period of that rod hanging alone
    "half1": "30", "half2": "30",            # s, its amplitude halves in
    "l1": "0.215",                           # pivot -> centre of mass of body 1
    "l2": "0.215",                           # second axle -> centre of mass 2
    "l3": "0.250",                           # pivot -> second axle (hinge to hinge)
    "g": "9.81",
    "duration": "20", "dt": "0.001", "out_dt": "0.005",
    "theta1": "90.000", "theta2": "90.000",
    "perturb_deg": "0.001", "perturb_var": "theta1",
    "csv_dir": "",
}


# =====================================================================
#                            PHYSICS
# =====================================================================

def coefficients(p):
    """The five constants the equations of motion actually depend on."""
    a = p["I1"] + p["m1"] * p["l1"] ** 2 + p["m2"] * p["l3"] ** 2
    b = p["m2"] * p["l3"] * p["l2"]
    d = p["I2"] + p["m2"] * p["l2"] ** 2
    e1 = (p["m1"] * p["l1"] + p["m2"] * p["l3"]) * p["g"]
    e2 = p["m2"] * p["l2"] * p["g"]
    return a, b, d, e1, e2


def normal_modes(p):
    """Small-oscillation modes: (frequencies rad/s, mass-normalised shapes)."""
    a, b, d, e1, e2 = coefficients(p)
    M = np.array([[a, b], [b, d]])
    K = np.array([[e1, 0.0], [0.0, e2]])
    w2, V = np.linalg.eig(np.linalg.solve(M, K))
    order = np.argsort(np.real(w2))
    w2 = np.real(w2[order])
    V = np.real(V[:, order])
    for i in range(2):                      # normalise so that phi^T M phi = 1
        n = float(V[:, i] @ M @ V[:, i])
        V[:, i] /= np.sqrt(abs(n)) if n else 1.0
    return np.sqrt(np.maximum(w2, 0.0)), V


def damping_matrix(p):
    """Viscous damping matrix C; the friction torques are Q = -C [w1, w2].

    The friction sits in the two bearings, each measured on its own rod hanging
    alone: rod i obeys Ji th'' + ci th' + mi g li th = 0 with Ji = Ii + mi li^2,
    whose amplitude decays as exp(-t/(2 tau_i)), so ci = Ji / tau_i. The second
    bearing acts on the RELATIVE rate w2 - w1, hence

        F = 1/2 c1 w1^2 + 1/2 c2 (w2 - w1)^2
        C = [[c1 + c2, -c2], [-c2, c2]]

    tau = 0 means that bearing is frictionless.
    """
    c1, c2 = friction_coefficients(p)
    if c1 == 0.0 and c2 == 0.0:
        return None
    return np.array([[c1 + c2, -c2], [-c2, c2]])


def friction_coefficients(p):
    """c1, c2 [N m s] from the two decay times."""
    t1 = float(p.get("tau1", 0) or 0)
    t2 = float(p.get("tau2", 0) or 0)
    J1 = p["I1"] + p["m1"] * p["l1"] ** 2
    J2 = p["I2"] + p["m2"] * p["l2"] ** 2
    return (J1 / t1 if t1 > 0 else 0.0, J2 / t2 if t2 > 0 else 0.0)


def from_measurements(m, l, period, half, g=9.81):
    """One rod hanging alone -> (I about its c.o.m., tau).

    J = m g l (T/2pi)^2 is the inertia about the axle, I = J - m l^2 the one
    about the centre of mass, and an amplitude that halves in `half` seconds
    means exp(-t/(2 tau)) with tau = half / (2 ln2).
    """
    J = m * g * l * (period / (2.0 * np.pi)) ** 2
    tau = half / (2.0 * np.log(2.0)) if half and half > 0 else 0.0
    return J - m * l ** 2, tau


def derivs(y, p, co=None, C=None):
    """State derivative for y = [th1, th2, w1, w2]."""
    a, b, d, e1, e2 = co if co is not None else coefficients(p)
    th1, th2, w1, w2 = y
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
    if abs(det) < 1e-15:
        det = 1e-15 if det >= 0 else -1e-15
    dw1 = (d * f1 - b * cd * f2) / det
    dw2 = (a * f2 - b * cd * f1) / det
    return np.array([w1, w2, dw1, dw2])


def energies(y, p, co=None):
    """Kinetic, potential (zero hanging at rest) and total energy [J]."""
    a, b, d, e1, e2 = co if co is not None else coefficients(p)
    th1, th2, w1, w2 = (y[..., 0], y[..., 1], y[..., 2], y[..., 3]) \
        if y.ndim > 1 else (y[0], y[1], y[2], y[3])
    T = 0.5 * a * w1 ** 2 + 0.5 * d * w2 ** 2 + b * w1 * w2 * np.cos(th1 - th2)
    V = e1 * (1.0 - np.cos(th1)) + e2 * (1.0 - np.cos(th2))
    return T, V, T + V


def integrate(y0, p, duration, dt, out_dt, progress=None, stop=None):
    """Classical RK4 with a fixed step; samples the state every out_dt.

    Fixed step is deliberate: two runs started from slightly different states
    then take exactly the same steps, so their difference is physics and not a
    difference in step selection.
    """
    co = coefficients(p)
    C = damping_matrix(p)
    n_steps = max(1, int(round(duration / dt)))
    every = max(1, int(round(out_dt / dt)))
    n_out = n_steps // every + 1
    ts = np.empty(n_out)
    ys = np.empty((n_out, 4))
    y = np.array(y0, dtype=float)
    ts[0], ys[0] = 0.0, y
    k = 1
    for i in range(1, n_steps + 1):
        k1 = derivs(y, p, co, C)
        k2 = derivs(y + 0.5 * dt * k1, p, co, C)
        k3 = derivs(y + 0.5 * dt * k2, p, co, C)
        k4 = derivs(y + dt * k3, p, co, C)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % every == 0 and k < n_out:
            ts[k], ys[k] = i * dt, y
            k += 1
        if progress is not None and i % 20000 == 0:
            progress(i / n_steps)
        if stop is not None and stop.is_set():
            break
    return ts[:k], ys[:k]


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def divergence(run_a, run_b, use_velocity=False, p=None):
    """Phase-space distance between two runs on their common time grid."""
    n = min(len(run_a["t"]), len(run_b["t"]))
    t = run_a["t"][:n]
    d1 = wrap_pi(run_a["y"][:n, 0] - run_b["y"][:n, 0])
    d2 = wrap_pi(run_a["y"][:n, 1] - run_b["y"][:n, 1])
    if not use_velocity:
        return t, np.sqrt(d1 ** 2 + d2 ** 2)
    # velocities scaled by sqrt(g/l) so that the sum is dimensionally sane
    w0 = np.sqrt(p["g"] / max(p["l3"], 1e-6)) if p else 1.0
    v1 = (run_a["y"][:n, 2] - run_b["y"][:n, 2]) / w0
    v2 = (run_a["y"][:n, 3] - run_b["y"][:n, 3]) / w0
    return t, np.sqrt(d1 ** 2 + d2 ** 2 + v1 ** 2 + v2 ** 2)


def fit_lyapunov(t, dist, floor_factor=3.0, ceiling=0.5):
    """Fit ln(distance) = lambda t on the exponential stretch.

    The separation oscillates while it grows, so the window is not chosen by
    contiguity but by two crossings: it starts when the distance first rises a
    few times above its initial value (past the initial transient) and ends
    when it first reaches `ceiling`, where growth saturates at the size of the
    attractor and the exponential law necessarily stops.

    Returns (lambda, t_start, t_end, n_points); lambda is nan when the
    separation never grows - which is the correct answer for regular motion.
    """
    d = np.asarray(dist, dtype=float)
    t = np.asarray(t, dtype=float)
    good = np.isfinite(d) & (d > 0)
    if good.sum() < 20:
        return np.nan, np.nan, np.nan, 0
    d0 = float(np.median(d[good][:5]))
    lo_val = d0 * floor_factor
    up = np.where(good & (d > lo_val))[0]
    if up.size == 0:
        return np.nan, np.nan, np.nan, 0          # never grew: regular motion
    i0 = up[0]
    sat = np.where(good & (d > ceiling))[0]
    i1 = sat[0] if sat.size else len(d) - 1
    sl = np.arange(i0, i1 + 1)
    sl = sl[good[sl]]
    if sl.size < 20:
        return np.nan, np.nan, np.nan, int(sl.size)
    lam = float(np.polyfit(t[sl], np.log(d[sl]), 1)[0])
    return lam, float(t[sl[0]]), float(t[sl[-1]]), int(sl.size)


def time_to_threshold(t, dist, threshold_deg=10.0):
    """First time the separation exceeds the threshold."""
    thr = np.radians(threshold_deg)
    i = np.where(np.asarray(dist) > thr)[0]
    return float(t[i[0]]) if i.size else np.nan


# =====================================================================
#                              GUI
# =====================================================================

def block(parent, title):
    f = ttk.LabelFrame(parent, text=" " + title + " ", padding=6)
    f.pack(fill="x", pady=(0, 8))
    return f


def hint(parent, text):
    ttk.Label(parent, text=text, foreground="#666", justify="left",
              wraplength=330).pack(anchor="w", pady=(2, 0))


def field(parent, label, var, width=10):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=1)
    ttk.Label(row, text=label).pack(side="left")
    ttk.Entry(row, textvariable=var, width=width).pack(side="right")
    return row


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Double Pendulum Simulator  v{__version__}")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.minsize(min(1150, sw - 40), min(720, sh - 60))
        self.geometry(f"{min(1280, sw - 40)}x{min(820, sh - 60)}+20+20")

        self.cfg = load_config()
        self.runs = []            # list of dicts: name, t, y, params, ...
        self.counter = 0
        self.job = None
        self.drag_handle = None
        self.tview = None         # (t0, t1) of the time axis, None = whole run
        self._pan = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.draw_pendulum()

    # ------------------------------------------------------- layout
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        tab_main = ttk.Frame(nb, padding=4)
        tab_set = ttk.Frame(nb, padding=8)
        nb.add(tab_main, text="Model")
        nb.add(tab_set, text="Settings")

        left = ttk.Frame(tab_main)
        left.pack(side="left", fill="y")
        right = ttk.Frame(tab_main)
        right.pack(side="right", fill="both", expand=True, padx=(6, 0))

        # ---- schematic
        self.cw = self.ch = 380
        self.canvas = tk.Canvas(left, width=self.cw, height=self.ch, bg="#101014",
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag_handle", None))
        ttk.Label(left, foreground="#666", wraplength=self.cw, justify="left",
                  text="Drag the middle joint to set theta1, the lower end to set "
                       "theta2, or type the angles below.").pack(anchor="w", pady=(3, 0))

        b = block(left, "Initial conditions")
        self.var_th1 = tk.StringVar(value=self.cfg["theta1"])
        self.var_th2 = tk.StringVar(value=self.cfg["theta2"])
        for lab, v in (("theta1, deg", self.var_th1), ("theta2, deg", self.var_th2)):
            r = field(b, lab, v, 12)
            v.trace_add("write", lambda *a: self.draw_pendulum())
        row = ttk.Frame(b)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="hanging", width=9,
                   command=lambda: self.set_angles(0, 0)).pack(side="left")
        ttk.Button(row, text="90/90", width=8,
                   command=lambda: self.set_angles(90, 90)).pack(side="left", padx=2)
        ttk.Button(row, text="180/180", width=9,
                   command=lambda: self.set_angles(180, 180)).pack(side="left")

        b = block(left, "Run")
        self.var_dur = tk.StringVar(value=self.cfg["duration"])
        self.var_dt = tk.StringVar(value=self.cfg["dt"])
        self.var_outdt = tk.StringVar(value=self.cfg["out_dt"])
        field(b, "duration, s", self.var_dur)
        field(b, "RK4 step, s", self.var_dt)
        field(b, "output step, s", self.var_outdt)
        self.var_name = tk.StringVar()
        field(b, "name", self.var_name, 22)
        ttk.Button(b, text="RUN", command=self.run_sim).pack(fill="x", pady=(4, 2))
        row = ttk.Frame(b)
        row.pack(fill="x")
        self.var_pert = tk.StringVar(value=self.cfg["perturb_deg"])
        self.var_pvar = tk.StringVar(value=self.cfg["perturb_var"])
        ttk.Label(row, text="twin: +").pack(side="left")
        ttk.Entry(row, textvariable=self.var_pert, width=8).pack(side="left", padx=2)
        ttk.Label(row, text="deg to").pack(side="left")
        ttk.Combobox(row, textvariable=self.var_pvar, width=7, state="readonly",
                     values=("theta1", "theta2")).pack(side="left", padx=2)
        ttk.Button(b, text="RUN TWIN (perturbed copy of the selected run)",
                   command=self.run_twin).pack(fill="x", pady=(2, 0))
        self.progress = ttk.Progressbar(b, mode="determinate")
        self.progress.pack(fill="x", pady=(4, 0))

        # ---- plot + runs
        top = ttk.Frame(right)
        top.pack(fill="x")
        ttk.Label(top, text="plot:").pack(side="left")
        self.var_plot = tk.StringVar(value="theta2")
        ttk.Combobox(top, textvariable=self.var_plot, width=22, state="readonly",
                     values=("theta1", "theta2", "omega1", "omega2",
                             "kinetic energy", "potential energy", "total energy",
                             "divergence (2 runs)",
                             "divergence incl. velocities (2 runs)")
                     ).pack(side="left", padx=4)
        ttk.Button(top, text="redraw", command=self.draw_plot).pack(side="left")
        self.var_thr = tk.StringVar(value="10")
        ttk.Label(top, text="   threshold, deg:").pack(side="left")
        ttk.Entry(top, textvariable=self.var_thr, width=6).pack(side="left")
        self.lbl_lyap = ttk.Label(top, text="", foreground="#060")
        self.lbl_lyap.pack(side="left", padx=8)

        # ---- time axis zoom / pan
        zb = ttk.Frame(right)
        zb.pack(fill="x", pady=(2, 0))
        ttk.Label(zb, text="time axis:").pack(side="left")
        for txt, cmd, w in (("|<", lambda: self.pan_time(-1e9), 3),
                            ("<<", lambda: self.pan_time(-0.5), 3),
                            ("zoom -", lambda: self.zoom_time(2.0), 8),
                            ("zoom +", lambda: self.zoom_time(0.5), 8),
                            (">>", lambda: self.pan_time(0.5), 3),
                            (">|", lambda: self.pan_time(1e9), 3),
                            ("fit all", self.fit_time, 8)):
            ttk.Button(zb, text=txt, width=w, command=cmd).pack(side="left", padx=1)
        ttk.Label(zb, text="  from").pack(side="left")
        self.var_t0 = tk.StringVar(value="")
        self.var_t1 = tk.StringVar(value="")
        ttk.Entry(zb, textvariable=self.var_t0, width=8).pack(side="left", padx=2)
        ttk.Label(zb, text="to").pack(side="left")
        ttk.Entry(zb, textvariable=self.var_t1, width=8).pack(side="left", padx=2)
        ttk.Label(zb, text="s").pack(side="left")
        ttk.Button(zb, text="apply", width=6, command=self.apply_time_entry
                   ).pack(side="left", padx=3)
        ttk.Label(zb, text="  (mouse wheel = zoom at the cursor, drag = pan)",
                  foreground="#666").pack(side="left")

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.fig = Figure(figsize=(7, 4.6), dpi=100, layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=right)
        w = self.plot_canvas.get_tk_widget()
        w.pack(fill="both", expand=True, pady=(4, 4))
        w.bind("<MouseWheel>", self.on_plot_wheel)              # Windows / macOS
        w.bind("<Button-4>", lambda e: self.on_plot_wheel(e, +1))   # X11
        w.bind("<Button-5>", lambda e: self.on_plot_wheel(e, -1))
        w.bind("<ButtonPress-1>", self.on_plot_press)
        w.bind("<B1-Motion>", self.on_plot_drag)
        w.bind("<ButtonRelease-1>", lambda e: setattr(self, "_pan", None))
        w.bind("<Double-Button-1>", lambda e: self.fit_time())

        bottom = ttk.Frame(right)
        bottom.pack(fill="both")
        self.listbox = tk.Listbox(bottom, selectmode="extended", height=8,
                                  exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.draw_plot())
        sb = ttk.Scrollbar(bottom, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        side = ttk.Frame(bottom)
        side.pack(side="left", fill="y", padx=(6, 0))
        for text, cmd in (("rename", self.rename_run), ("delete", self.delete_run),
                          ("select all", self.select_all),
                          ("save CSV (selected)", self.save_csv),
                          ("clear all", self.clear_runs)):
            ttk.Button(side, text=text, command=cmd, width=20).pack(pady=1)
        self.status = ttk.Label(right, text="Set the angles and press RUN.",
                                anchor="w")
        self.status.pack(fill="x")

        self._build_settings(tab_set)

    def _build_settings(self, s):
        for c in range(2):
            s.columnconfigure(c, weight=1, uniform="s")
        col0 = ttk.Frame(s)
        col0.grid(row=0, column=0, sticky="nsew", padx=6)
        col1 = ttk.Frame(s)
        col1.grid(row=0, column=1, sticky="nsew", padx=6)

        b = block(col0, "Rods, as measured one at a time")
        self.var = {}
        rows = (("m1, kg              upper rod", "m1"),
                ("l1, m               pivot -> c.o.m. 1", "l1"),
                ("T1, s               period, that rod alone", "T1"),
                ("half1, s            amplitude halves in", "half1"),
                ("", None),
                ("m2, kg              lower rod", "m2"),
                ("l2, m               2nd axle -> c.o.m. 2", "l2"),
                ("T2, s               period, that rod alone", "T2"),
                ("half2, s            amplitude halves in", "half2"),
                ("", None),
                ("l3, m               pivot -> 2nd axle", "l3"),
                ("g, m/s2", "g"))
        for lab, key in rows:
            if key is None:
                ttk.Label(b, text="").pack()
                continue
            self.var[key] = tk.StringVar(value=self.cfg[key])
            field(b, lab, self.var[key], 12)
            self.var[key].trace_add("write", lambda *a: self.show_coefficients())
        hint(b, "Hang each rod ALONE from its own axle - the upper rod from the "
                "top pivot, the lower one from the middle pin - and measure two "
                "things: the period T of small oscillations, and the time in "
                "which the amplitude falls to half.\n"
                "From T comes the moment of inertia, J = m g l (T/2pi)^2 about "
                "the axle and I = J - m l^2 about the centre of mass. From the "
                "half-life comes the friction of that bearing: the amplitude "
                "decays as exp(-t/(2 tau)) with tau = half / (2 ln2), so "
                "c = J / tau. The second bearing acts on the relative rate.\n"
                "Set a half-life to 0 to switch that bearing off; with both at "
                "0 the energy drift of the integration is reported after every "
                "run - the accuracy figure of RK4.")

        b = block(col1, "Derived constants")
        self.lbl_meas = ttk.Label(b, text="", justify="left",
                                  font=("TkFixedFont", 9), foreground="#046")
        self.lbl_meas.pack(anchor="w", pady=(0, 6))
        self.lbl_coef = ttk.Label(b, text="", justify="left", font=("TkFixedFont", 9))
        self.lbl_coef.pack(anchor="w")
        ttk.Button(b, text="recompute", command=self.show_coefficients).pack(fill="x")
        hint(b, "Everything here follows from the fields on the left and "
                "updates as you type. a and d are the effective moments of "
                "inertia of the two coordinates, b their coupling, e1 and e2 "
                "the gravity torques. The small-oscillation modes are what the "
                "model predicts for tiny amplitudes - a quick check against the "
                "assembled rig.")

        b = block(col1, "Files")
        self.var["csv_dir"] = tk.StringVar(value=self.cfg["csv_dir"])
        row = ttk.Frame(b)
        row.pack(fill="x")
        ttk.Label(row, text="CSV folder:").pack(side="left")
        ttk.Entry(row, textvariable=self.var["csv_dir"]).pack(side="left", fill="x",
                                                              expand=True, padx=2)
        ttk.Button(row, text="...", width=3, command=self.pick_dir).pack(side="left")
        ttk.Button(b, text="Save settings to dp_sim.ini",
                   command=self.save_settings).pack(fill="x", pady=(6, 0))
        ttk.Label(b, text=CONFIG_PATH, foreground="#666",
                  wraplength=330).pack(anchor="w")
        ttk.Label(b, text=f"dp_sim.py version {__version__}",
                  font=("", 9, "bold")).pack(anchor="w", pady=(6, 0))
        self.show_coefficients()

    # ------------------------------------------------------- params
    def params(self, quiet=False):
        """Model parameters as floats, derived from the measured rod data."""
        try:
            p = {k: float(self.var[k].get()) for k in
                 ("m1", "m2", "l1", "l2", "l3", "g",
                  "T1", "T2", "half1", "half2")}
        except (ValueError, KeyError, AttributeError):
            if not quiet:
                messagebox.showerror("Bad parameter",
                                     "One of the model parameters is not a number.")
            return None
        if min(p["m1"], p["m2"], p["l1"], p["l2"], p["l3"], p["g"],
               p["T1"], p["T2"]) <= 0:
            if not quiet:
                messagebox.showerror(
                    "Bad parameter",
                    "Masses, lengths, g and the two periods must be positive.")
            return None
        p["I1"], p["tau1"] = from_measurements(p["m1"], p["l1"], p["T1"],
                                               p["half1"], p["g"])
        p["I2"], p["tau2"] = from_measurements(p["m2"], p["l2"], p["T2"],
                                               p["half2"], p["g"])
        return p

    def show_coefficients(self, *_a):
        p = self.params(quiet=True)
        if not p:
            return
        a, bb, d, e1, e2 = coefficients(p)
        # small-oscillation normal modes of the linearised system
        M = np.array([[a, bb], [bb, d]])
        K = np.array([[e1, 0.0], [0.0, e2]])
        try:
            w2 = np.linalg.eigvals(np.linalg.solve(M, K))
            w2 = np.sort(np.real(w2[np.isreal(w2)]))
            ww = [np.sqrt(w) for w in w2 if w > 0]
            modes = "\n".join(
                f"  {nm:5s} mode: T = {2*np.pi/w:6.3f} s   f = {w/2/np.pi:6.3f} Hz"
                f"   omega = {w:7.3f} rad/s"
                for nm, w in zip(("slow", "fast"), sorted(ww)))
        except Exception:
            modes = "  -"
        self.lbl_coef.config(
            text=(f"a  = {a:10.6f} kg m^2\nb  = {bb:10.6f} kg m^2\n"
                  f"d  = {d:10.6f} kg m^2\ne1 = {e1:10.6f} N m\n"
                  f"e2 = {e2:10.6f} N m\n\nsmall oscillations:\n" + modes))
        c1, c2 = friction_coefficients(p)
        self.lbl_meas.config(text=(
            "from the measured rods:\n"
            f"  I1  = {p['I1']:.6f} kg m^2 about c.o.m."
            f"   (J1 = {p['I1'] + p['m1'] * p['l1'] ** 2:.6f} about the axle)\n"
            f"  I2  = {p['I2']:.6f} kg m^2 about c.o.m."
            f"   (J2 = {p['I2'] + p['m2'] * p['l2'] ** 2:.6f} about the axle)\n"
            f"  tau1 = {p['tau1']:7.2f} s    c1 = {c1:.6f} N m s\n"
            f"  tau2 = {p['tau2']:7.2f} s    c2 = {c2:.6f} N m s"))

    def angles(self):
        try:
            return float(self.var_th1.get()), float(self.var_th2.get())
        except ValueError:
            return 0.0, 0.0

    def set_angles(self, a1, a2):
        self.var_th1.set(f"{a1:.3f}")
        self.var_th2.set(f"{a2:.3f}")
        self.draw_pendulum()

    # ------------------------------------------------------- drawing
    def geom(self):
        p = self.params(quiet=True) or {"l3": 0.25, "l2": 0.215}
        span = (p["l3"] + 2 * p["l2"]) * 1.10
        scale = min(self.cw, self.ch) / (2 * span)
        return p, scale, self.cw / 2, self.ch * 0.30

    def draw_pendulum(self):
        if not hasattr(self, "canvas"):
            return
        p, sc, ox, oy = self.geom()
        t1, t2 = np.radians(self.angles())
        l1 = p["l3"]                       # rod 1 is drawn hinge to hinge
        l2r = 2 * p.get("l2", 0.215)       # rod 2 is drawn to its full length
        x1, y1 = ox + sc * l1 * np.sin(t1), oy + sc * l1 * np.cos(t1)
        x2, y2 = x1 + sc * l2r * np.sin(t2), y1 + sc * l2r * np.cos(t2)
        c = self.canvas
        c.delete("all")
        c.create_line(ox, oy, ox, oy + sc * (l1 + l2r) * 1.05,
                      fill="#334", dash=(3, 3))
        c.create_line(ox, oy, x1, y1, fill="#ffd23f", width=5)
        c.create_line(x1, y1, x2, y2, fill="#3ad1ff", width=5)
        r = 6
        c.create_oval(ox - 4, oy - 4, ox + 4, oy + 4, fill="#ff3b30", outline="")
        c.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="#ffd23f", outline="#fff")
        c.create_oval(x2 - r, y2 - r, x2 + r, y2 + r, fill="#3ad1ff", outline="#fff")
        c.create_text(8, 8, anchor="nw", fill="#aaa", font=("TkFixedFont", 9),
                      text=f"theta1 = {self.angles()[0]:+8.3f} deg\n"
                           f"theta2 = {self.angles()[1]:+8.3f} deg")
        self._handles = ((x1, y1), (x2, y2))

    def on_press(self, e):
        if not hasattr(self, "_handles"):
            return
        (x1, y1), (x2, y2) = self._handles
        d1 = np.hypot(e.x - x1, e.y - y1)
        d2 = np.hypot(e.x - x2, e.y - y2)
        self.drag_handle = 1 if d1 <= d2 else 2
        self.on_drag(e)

    def on_drag(self, e):
        if not self.drag_handle:
            return
        p, sc, ox, oy = self.geom()
        if self.drag_handle == 1:
            a = np.degrees(np.arctan2(e.x - ox, e.y - oy))
            self.var_th1.set(f"{a:.3f}")
        else:
            t1 = np.radians(self.angles()[0])
            x1 = ox + sc * p["l3"] * np.sin(t1)
            y1 = oy + sc * p["l3"] * np.cos(t1)
            a = np.degrees(np.arctan2(e.x - x1, e.y - y1))
            self.var_th2.set(f"{a:.3f}")
        self.draw_pendulum()

    # ------------------------------------------------------- running
    def default_name(self, t1, t2):
        self.counter += 1
        return f"th1={t1:.3f} th2={t2:.3f} #{self.counter}"

    def run_sim(self, y0=None, name=None):
        p = self.params()
        if not p:
            return
        try:
            dur = float(self.var_dur.get())
            dt = float(self.var_dt.get())
            out_dt = float(self.var_outdt.get())
        except ValueError:
            messagebox.showerror("Bad input", "duration / step must be numbers.")
            return
        if dt <= 0 or dur <= 0 or out_dt < dt:
            messagebox.showerror("Bad input",
                                 "Need dt > 0, duration > 0 and output step >= dt.")
            return
        if y0 is None:
            t1, t2 = self.angles()
            y0 = np.array([np.radians(t1), np.radians(t2), 0.0, 0.0])
            name = name or self.var_name.get().strip() or self.default_name(t1, t2)
        self.progress["value"] = 0
        self.status.config(text=f"integrating '{name}' ...")

        def work():
            try:
                t, y = integrate(y0, p, dur, dt, out_dt,
                                 progress=lambda f: self.after(
                                     0, lambda: self.progress.config(value=100 * f)))
                self.after(0, lambda: self.finish(name, t, y, p, dt))
            except Exception:
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("Integration failed", tb))

        threading.Thread(target=work, daemon=True).start()

    def finish(self, name, t, y, p, dt):
        T, V, E = energies(y, p)
        drift = float(E.max() - E.min())
        rel = drift / max(abs(E[0]), 1e-12)
        self.runs.append(dict(name=name, t=t, y=y, p=dict(p), dt=dt,
                              T=T, V=V, E=E))
        self.listbox.insert("end", name)
        self.listbox.selection_set("end")
        self.progress["value"] = 100
        msg = f"'{name}': {len(t)} samples, {t[-1]:.2f} s"
        if p["tau1"] <= 0 and p["tau2"] <= 0:
            msg += (f";  energy drift {drift:.3e} J = {100*rel:.2e}% "
                    f"(RK4 accuracy at dt={dt:g} s)")
        else:
            msg += f";  E: {E[0]:.4f} -> {E[-1]:.4f} J (friction on)"
        self.status.config(text=msg)
        self.var_name.set("")
        self.draw_plot()

    def run_twin(self):
        sel = list(self.listbox.curselection())
        if len(sel) != 1:
            messagebox.showinfo("Select one run",
                                "Select exactly one run to clone with a "
                                "perturbation.")
            return
        base = self.runs[sel[0]]
        try:
            eps = float(self.var_pert.get())
        except ValueError:
            messagebox.showerror("Bad input", "The perturbation must be a number.")
            return
        y0 = base["y"][0].copy()
        if self.var_pvar.get() == "theta1":
            y0[0] += np.radians(eps)
        else:
            y0[1] += np.radians(eps)
        self.run_sim(y0=y0, name=f"{base['name']}  +{eps:g} deg {self.var_pvar.get()}")

    # ------------------------------------------------------- plotting
    def selected(self):
        return [self.runs[i] for i in self.listbox.curselection()]

    # --------------------------------------------------- time axis zoom
    def data_range(self):
        """Time span covered by the runs currently selected."""
        runs = self.selected()
        if not runs:
            return None
        return (min(float(r["t"][0]) for r in runs),
                max(float(r["t"][-1]) for r in runs))

    def view_range(self):
        full = self.data_range()
        if full is None:
            return None
        if self.tview is None:
            return full
        t0, t1 = self.tview
        return max(t0, full[0]), min(t1, full[1])

    def set_view(self, t0, t1):
        full = self.data_range()
        if full is None:
            return
        span = max(t1 - t0, 1e-6)
        t0 = max(full[0], min(t0, full[1] - span))
        t1 = t0 + span
        if t1 > full[1]:
            t1, t0 = full[1], max(full[0], full[1] - span)
        self.tview = None if (t0 <= full[0] + 1e-12 and t1 >= full[1] - 1e-12) \
            else (t0, t1)
        self.draw_plot()

    def zoom_time(self, factor, centre=None):
        v = self.view_range()
        if v is None:
            return
        t0, t1 = v
        c = t0 + 0.5 * (t1 - t0) if centre is None else centre
        half = 0.5 * (t1 - t0) * factor
        self.set_view(c - half, c + half)

    def pan_time(self, frac):
        v = self.view_range()
        if v is None:
            return
        t0, t1 = v
        d = (t1 - t0) * frac
        self.set_view(t0 + d, t1 + d)

    def fit_time(self):
        self.tview = None
        self.draw_plot()

    def apply_time_entry(self):
        try:
            self.set_view(float(self.var_t0.get()), float(self.var_t1.get()))
        except ValueError:
            pass

    def _event_time(self, e):
        """Data x under the mouse (the widget uses screen pixels, mpl flips y)."""
        try:
            h = self.plot_canvas.get_tk_widget().winfo_height()
            x, _y = self.ax.transData.inverted().transform((e.x, h - e.y))
            return float(x)
        except Exception:
            return None

    def on_plot_wheel(self, e, direction=None):
        if direction is None:
            direction = 1 if getattr(e, "delta", 0) > 0 else -1
        c = self._event_time(e)
        self.zoom_time(0.8 if direction > 0 else 1.25, centre=c)
        return "break"

    def on_plot_press(self, e):
        t = self._event_time(e)
        v = self.view_range()
        if t is not None and v is not None:
            self._pan = (t, v)

    def on_plot_drag(self, e):
        if not self._pan:
            return
        t_now = self._event_time(e)
        if t_now is None:
            return
        t_grab, (v0, v1) = self._pan
        # the grabbed point should stay under the cursor
        shift = t_grab - t_now
        self.set_view(v0 + shift, v1 + shift)
        self._pan = (t_grab, self.view_range())

    def _apply_view(self, logy):
        """Set the x window and rescale y to whatever is visible in it."""
        v = self.view_range()
        if v is None:
            return
        self.ax.set_xlim(v)
        self.var_t0.set(f"{v[0]:.3f}")
        self.var_t1.set(f"{v[1]:.3f}")
        lo, hi = np.inf, -np.inf
        for ln in self.ax.get_lines():
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if x.size < 2:
                continue
            m = (x >= v[0]) & (x <= v[1]) & np.isfinite(y)
            if logy:
                m &= y > 0
            if m.any():
                lo = min(lo, y[m].min())
                hi = max(hi, y[m].max())
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if logy:
            lo, hi = max(lo, 1e-18), max(hi, 1e-17)
            self.ax.set_ylim(lo / 3.0, hi * 3.0)
        else:
            pad = 0.06 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
            self.ax.set_ylim(lo - pad, hi + pad)

    def draw_plot(self):
        runs = self.selected()
        self.ax.clear()
        self.lbl_lyap.config(text="")
        mode = self.var_plot.get()
        if not runs:
            self.ax.text(.5, .5, "select runs in the list", ha="center",
                         transform=self.ax.transAxes, color="#888")
            self.plot_canvas.draw()
            return

        if mode.startswith("divergence"):
            if len(runs) != 2:
                self.ax.text(.5, .5, "select exactly two runs", ha="center",
                             transform=self.ax.transAxes, color="#a00")
                self.plot_canvas.draw()
                return
            t, dist = divergence(runs[0], runs[1],
                                 use_velocity="velocities" in mode, p=runs[0]["p"])
            self.ax.semilogy(t, np.maximum(dist, 1e-18), lw=1)
            lam, t0, t1, npts = fit_lyapunov(t, dist)
            try:
                thr = float(self.var_thr.get())
            except ValueError:
                thr = 10.0
            t_thr = time_to_threshold(t, dist, thr)
            if np.isfinite(lam):
                tt = np.linspace(t0, t1, 50)
                d0 = np.interp(t0, t, dist)
                self.ax.semilogy(tt, d0 * np.exp(lam * (tt - t0)), "r--", lw=2,
                                 label=f"fit: lambda = {lam:.3f} 1/s")
            if np.isfinite(t_thr):
                self.ax.axvline(t_thr, color="g", ls=":", lw=1.5)
                self.ax.axhline(np.radians(thr), color="g", ls=":", lw=1)
            self.ax.set_xlabel("t, s")
            self.ax.set_ylabel("|difference|, rad")
            self.ax.set_title("divergence of two runs")
            self.ax.legend(fontsize=8)
            txt = ""
            if np.isfinite(lam):
                txt += (f"lambda = {lam:.3f} 1/s   (e-folding {1/lam:.2f} s, "
                        f"fit {t0:.1f}-{t1:.1f} s)")
            if np.isfinite(t_thr):
                txt += f"    reaches {thr:g} deg at t = {t_thr:.2f} s"
            self.lbl_lyap.config(text=txt)
        else:
            key = {"theta1": lambda r: np.degrees(wrap_pi(r["y"][:, 0])),
                   "theta2": lambda r: np.degrees(wrap_pi(r["y"][:, 1])),
                   "omega1": lambda r: np.degrees(r["y"][:, 2]),
                   "omega2": lambda r: np.degrees(r["y"][:, 3]),
                   "kinetic energy": lambda r: r["T"],
                   "potential energy": lambda r: r["V"],
                   "total energy": lambda r: r["E"]}[mode]
            for r in runs:
                self.ax.plot(r["t"], key(r), lw=.9, label=r["name"])
            self.ax.set_xlabel("t, s")
            self.ax.set_ylabel({"theta1": "theta1, deg", "theta2": "theta2, deg",
                                "omega1": "omega1, deg/s", "omega2": "omega2, deg/s"}
                               .get(mode, mode + ", J"))
            if len(runs) <= 8:
                self.ax.legend(fontsize=7)
        self.ax.grid(alpha=.3)
        self._apply_view(logy=mode.startswith("divergence"))
        self.plot_canvas.draw()

    # ------------------------------------------------------- list ops
    def rename_run(self):
        sel = list(self.listbox.curselection())
        if len(sel) != 1:
            return
        top = tk.Toplevel(self)
        top.title("Rename")
        v = tk.StringVar(value=self.runs[sel[0]]["name"])
        ttk.Entry(top, textvariable=v, width=50).pack(padx=8, pady=8)

        def ok():
            self.runs[sel[0]]["name"] = v.get()
            self.listbox.delete(sel[0])
            self.listbox.insert(sel[0], v.get())
            self.listbox.selection_set(sel[0])
            top.destroy()
            self.draw_plot()

        ttk.Button(top, text="OK", command=ok).pack(pady=(0, 8))

    def delete_run(self):
        for i in sorted(self.listbox.curselection(), reverse=True):
            self.listbox.delete(i)
            del self.runs[i]
        self.draw_plot()

    def select_all(self):
        self.listbox.selection_set(0, "end")
        self.draw_plot()

    def clear_runs(self):
        if self.runs and messagebox.askyesno("Clear", "Delete all runs?"):
            self.runs.clear()
            self.listbox.delete(0, "end")
            self.draw_plot()

    # ------------------------------------------------------- export
    def save_csv(self):
        runs = self.selected()
        if not runs:
            messagebox.showinfo("Nothing selected", "Select runs to save.")
            return
        base = runs[0]["name"].replace(" ", "_").replace("=", "").replace("#", "n")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=base + ".csv",
            initialdir=self.var["csv_dir"].get() or APP_DIR,
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            write_csv(path, runs)
        except Exception as e:
            messagebox.showerror("Could not save", str(e))
            return
        if not self.var["csv_dir"].get():
            self.var["csv_dir"].set(os.path.dirname(path))
        self.status.config(text=f"saved {len(runs)} run(s) to {path}")

    # ------------------------------------------------------- config
    def pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.var["csv_dir"].get() or APP_DIR)
        if d:
            self.var["csv_dir"].set(d)

    def collect_config(self):
        c = {"version": __version__}
        for k in ("m1", "m2", "l1", "l2", "l3", "g",
                  "T1", "T2", "half1", "half2", "csv_dir"):
            c[k] = self.var[k].get()
        c.update({"duration": self.var_dur.get(), "dt": self.var_dt.get(),
                  "out_dt": self.var_outdt.get(), "theta1": self.var_th1.get(),
                  "theta2": self.var_th2.get(), "perturb_deg": self.var_pert.get(),
                  "perturb_var": self.var_pvar.get()})
        return c

    def save_settings(self, quiet=False):
        try:
            save_config(self.collect_config())
            if not quiet:
                self.status.config(text=f"settings saved to {CONFIG_PATH}")
        except Exception as e:
            if not quiet:
                messagebox.showerror("Could not save settings", str(e))

    def on_close(self):
        self.save_settings(quiet=True)
        self.destroy()


def write_csv(path, runs):
    """One CSV for any number of runs; the first column tells them apart."""
    import datetime
    p = runs[0]["p"]
    head = [f"dp_sim {__version__} simulation output",
            f"written: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"model: m1={p['m1']} m2={p['m2']} "
            f"I1={p['I1']:.6f} I2={p['I2']:.6f} (about each c.o.m.) "
            f"l1={p['l1']} l2={p['l2']} l3={p['l3']} g={p['g']} "
            f"(l1,l2 = pivot/axle to centre of mass, l3 = hinge to hinge)",
            f"friction: bearings, tau1={p['tau1']:.3f} s tau2={p['tau2']:.3f} s "
            f"(each rod alone decays as exp(-t/(2 tau)))",
            f"integrator: RK4, fixed step {runs[0]['dt']} s",
            "",
            "angles from the downward vertical, positive to the right;",
            "energy zero = both rods hanging at rest",
            "",
            "read with: pandas.read_csv(path, comment='#')"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for ln in head:
            fh.write(("# " + ln if ln else "#") + "\n")
        fh.write("name,t,theta1_deg,theta2_deg,omega1_deg_s,omega2_deg_s,"
                 "KE_J,PE_J,E_J,xc2_m,yc2_m\n")
        for r in runs:
            th1, th2 = r["y"][:, 0], r["y"][:, 1]
            # centre of mass of body 2
            x2 = r["p"]["l3"] * np.sin(th1) + r["p"]["l2"] * np.sin(th2)
            y2 = -(r["p"]["l3"] * np.cos(th1) + r["p"]["l2"] * np.cos(th2))
            nm = r["name"].replace(",", ";")
            for i in range(len(r["t"])):
                fh.write("%s,%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n"
                         % (nm, r["t"][i], np.degrees(th1[i]), np.degrees(th2[i]),
                            np.degrees(r["y"][i, 2]), np.degrees(r["y"][i, 3]),
                            r["T"][i], r["V"][i], r["E"][i], x2[i], y2[i]))


def load_config():
    cfg = dict(DEFAULTS)
    parser = configparser.ConfigParser()
    try:
        if parser.read(CONFIG_PATH, encoding="utf-8") and parser.has_section("sim"):
            for k in DEFAULTS:
                if parser.has_option("sim", k):
                    cfg[k] = parser.get("sim", k)
    except Exception as e:
        print("dp_sim.ini could not be read:", e, file=sys.stderr)
    return cfg


def save_config(values):
    parser = configparser.ConfigParser()
    parser["sim"] = {k: str(v) for k, v in values.items()}
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        parser.write(fh)


if __name__ == "__main__":
    App().mainloop()
