#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dp_gui.py  --  Double Pendulum Tracker (GUI)

Graphical front-end for processing double pendulum videos:
    MOV / MP4  ->  CSV:  Time, X1, Y1, X2, Y2, theta1_deg, theta2_deg,
                         L1, L2, L1_out, L2_out

L1 and L2 are the measured rod lengths (px); L1_out / L2_out are 1 for the
frames where the length left the tolerance band set on the Settings tab, i.e.
where the marker was mis-detected.

theta1 is the angle of the rod origin -> M1, theta2 the angle of the rod
M1 -> M2. Both are measured from the downward vertical: 0 when the rod hangs
straight down, positive towards the right, and the values stay inside
(-180, +180] deg. Settings has an optional "unwrap" mode that instead makes the
curve continuous through full turns (then the numbers grow without bound).

Workflow:
    open a video -> click the pivot -> pick the start frame -> drag a box over
    each mass -> RUN TRACKING -> check the control video that plays in the
    window (rods, angles and markers are drawn on it) -> SAVE CSV.

Settings (filter margins, folders, output options) live on the Settings tab and
are stored in config.ini next to this script.

Run:
    python dp_gui.py

Dependencies:
    numpy, pandas, opencv-python, matplotlib
    (Pillow is optional -- the preview is faster with it)

Based on the scripts by Anton S. (dp_traj_extract_main.py,
dp_post_processing.py, dp_extraction_utils.py)
Version: 3.0
"""

import configparser
import os
import sys
import threading
import traceback

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # pragma: no cover
    print("The opencv-python package is required:  pip install opencv-python")
    raise

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False


APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.ini")

DEFAULTS = {
    "fps": "120",
    "h_margin": "10",
    "s_margin": "75",
    "v_margin": "75",
    "min_area": "100",
    "len_tol": "10",
    "degrees": "1",
    "unwrap": "0",
    "interpolate": "1",
    "write_video": "1",
    "scale_down": "2",
    "playback_fps": "25",
    "movies_dir": "",
    "csv_dir": "",
}


# =====================================================================
#                    CORE: image processing
# =====================================================================

def avg_hsv_from_roi(frame_bgr, roi):
    """Mean HSV inside the rectangle roi = (x, y, w, h).

    (the original script had w and h swapped here)
    """
    x, y, w, h = [int(v) for v in roi]
    h_img, w_img = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    # circular mean for the hue, otherwise red (0/179) averages to garbage
    hue = hsv[:, :, 0].astype(np.float64) * (2.0 * np.pi / 180.0)
    mh = np.degrees(np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())) / 2.0
    if mh < 0:
        mh += 180.0
    return np.array([mh, hsv[:, :, 1].mean(), hsv[:, :, 2].mean()], dtype=np.float64)


def build_filter(hsv_target, h_margin=10, s_margin=75, v_margin=75):
    """HSV filter bounds around the target colour."""
    h, s, v = [float(t) for t in hsv_target]
    return {
        "h": h, "hm": float(h_margin),
        "s_lo": max(0.0, s - s_margin), "s_hi": min(255.0, s + s_margin),
        "v_lo": max(0.0, v - v_margin), "v_hi": min(255.0, v + v_margin),
    }


def color_mask(hsv_frame, filt):
    """Binary colour mask; handles the hue wrapping around 0 correctly."""
    h_lo, h_hi = filt["h"] - filt["hm"], filt["h"] + filt["hm"]
    if h_lo < 0:
        ranges = [(0.0, h_hi), (180.0 + h_lo, 179.0)]
    elif h_hi > 179:
        ranges = [(h_lo, 179.0), (0.0, h_hi - 180.0)]
    else:
        ranges = [(h_lo, h_hi)]

    mask = None
    for a, b in ranges:
        lo = np.array([a, filt["s_lo"], filt["v_lo"]], dtype=np.uint8)
        hi = np.array([b, filt["s_hi"], filt["v_hi"]], dtype=np.uint8)
        m = cv2.inRange(hsv_frame, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def detect_color(frame_bgr, filt, min_size=100, hsv_frame=None):
    """Return (cx, cy, found, bbox) for the largest blob of the given colour."""
    if hsv_frame is None:
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv_frame, filt)
    bbox = (0, 0, 0, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, False, bbox
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_size:
        return 0, 0, False, bbox
    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0, 0, False, bbox
    return M["m10"] / M["m00"], M["m01"] / M["m00"], True, cv2.boundingRect(c)


def draw_crosshair(img, center, size=20, color=(0, 0, 255), thickness=2):
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def add_marker(img, bbox, color):
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.circle(img, (x + w // 2, y + h // 2), 5, color, -1)


def compute_angles(x1, y1, x2, y2, unwrap=True):
    """Angles of the two rods measured from the downward vertical.

    rod 1 : origin -> (x1, y1)
    rod 2 : (x1, y1) -> (x2, y2)

    Both angles use exactly the same definition:

        theta = atan2( dx , dy_down )

    where dx points right and dy_down points along the downward vertical, i.e.
    the y axis of the image. theta = 0 for a rod hanging straight down, +90 deg
    pointing right, -90 deg pointing left; the branch cut sits at straight up.

    NOTE: the original post-processing script used
        theta = unwrap(atan2(-dy, dx)) + pi/2 ,
    which is the same angle modulo 2*pi but puts the branch cut on the
    horizontal-left direction and returns e.g. +270 deg instead of -90 deg for a
    rod pointing left. That is what made theta2 look odd.
    """
    th1 = np.arctan2(x1, y1)
    th2 = np.arctan2(x2 - x1, y2 - y1)
    if unwrap:
        th1 = np.unwrap(th1)
        th2 = np.unwrap(th2)
    return th1, th2


def frame_angles_deg(origin, p1, p2):
    """Instantaneous angles [deg] of one frame, for the on-screen overlay."""
    th1 = np.degrees(np.arctan2(p1[0] - origin[0], p1[1] - origin[1]))
    th2 = np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1]))
    return th1, th2


def draw_overlay(img, origin, p1, p2, t=None, bad1=False, bad2=False):
    """Draw the two rods, the vertical reference and the angle readout.

    A rod whose length is outside the tolerance is drawn in red: those frames
    are mis-detections, not physics.
    """
    p0 = (int(round(origin[0])), int(round(origin[1])))
    cv2.line(img, p0, (p0[0], p0[1] + 140), (170, 170, 170), 1)   # vertical ref
    draw_crosshair(img, p0)
    if p1 is None or p2 is None:
        return
    q1 = (int(round(p1[0])), int(round(p1[1])))
    q2 = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(img, p0, q1, (0, 0, 255) if bad1 else (0, 255, 255), 2)   # rod 1
    cv2.line(img, q1, q2, (0, 0, 255) if bad2 else (255, 255, 0), 2)   # rod 2
    th1, th2 = frame_angles_deg(origin, p1, p2)
    txt = f"th1={th1:+.1f} th2={th2:+.1f} deg"
    if t is not None:
        txt = f"t={t:.3f}s  " + txt
    if bad1 or bad2:
        txt += "  LENGTH!"
    # the text must stay readable after the control video is scaled down
    w = img.shape[1]
    fs = max(0.45, w / 1400.0)
    th = max(1, int(round(fs * 2)))
    while fs > 0.35 and cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                        fs, th)[0][0] > w - 20:
        fs *= 0.9
        th = max(1, int(round(fs * 2)))
    pos = (10, int(16 + 26 * fs))
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), th, cv2.LINE_AA)


# =====================================================================
#                       BACKGROUND TRACKING
# =====================================================================

class TrackerJob(threading.Thread):
    """Runs through the video in a separate thread so the window stays alive."""

    def __init__(self, cfg, progress_cb, done_cb):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.progress_cb = progress_cb
        self.done_cb = done_cb
        self.stop_flag = threading.Event()

    def run(self):
        try:
            self.done_cb(self._work(), None)
        except Exception:
            self.done_cb(None, traceback.format_exc())

    def _work(self):
        cfg = self.cfg
        cap = cv2.VideoCapture(cfg["video_path"])
        if not cap.isOpened():
            raise RuntimeError("Could not open the video file.")

        start = int(cfg["start_frame"])
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        out_video = None
        out_size = None
        if cfg["write_video"]:
            sd = max(1, int(cfg["scale_down"]))
            out_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // sd,
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // sd)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(cfg["video_out_path"], fourcc,
                                        min(60.0, cfg["fps"]), out_size)

        ox, oy = cfg["origin"]
        tol = max(0.0, float(cfg.get("len_tol", 10.0))) / 100.0
        ref1, ref2 = cfg.get("len_ref", (None, None))
        rows, found1, found2 = [], [], []
        k = 0
        ret, frame = cap.read()
        while ret and not self.stop_flag.is_set():
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cx1, cy1, f1, b1 = detect_color(frame, cfg["filt1"], cfg["min_size"], hsv)
            cx2, cy2, f2, b2 = detect_color(frame, cfg["filt2"], cfg["min_size"], hsv)

            rows.append([cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy])
            found1.append(bool(f1))
            found2.append(bool(f2))

            if out_video is not None:
                g = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                 cv2.COLOR_GRAY2BGR)
                if f1:
                    add_marker(g, b1, cfg["color1"])
                if f2:
                    add_marker(g, b2, cfg["color2"])
                b1 = b2 = False
                if f1 and f2 and ref1 and ref2:
                    b1 = abs(np.hypot(cx1 - ox, cy1 - oy) - ref1) > tol * ref1
                    b2 = abs(np.hypot(cx2 - cx1, cy2 - cy1) - ref2) > tol * ref2
                draw_overlay(g, (ox, oy),
                             (cx1, cy1) if f1 else None,
                             (cx2, cy2) if f2 else None,
                             t=k / cfg["fps"], bad1=b1, bad2=b2)
                out_video.write(cv2.resize(g, out_size))

            k += 1
            if k % 10 == 0:
                self.progress_cb(k, max(1, n_total - start),
                                 int(np.sum(found1)), int(np.sum(found2)))
            ret, frame = cap.read()

        cap.release()
        if out_video is not None:
            out_video.release()
        if k == 0:
            raise RuntimeError("No frames were read.")

        arr = np.asarray(rows, dtype=np.float64)
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        f1 = np.asarray(found1, dtype=bool)
        f2 = np.asarray(found2, dtype=bool)
        t = np.arange(k, dtype=np.float64) / cfg["fps"]

        # ---- rod length validation ------------------------------------
        # Both rods are rigid, so their length is a constant: any frame where
        # the measured length leaves the tolerance band is a mis-detection.
        # The reference lengths come from the marking frame; if they are not
        # available, the median of the detected frames is used instead.
        len1 = np.hypot(x1, y1)
        len2 = np.hypot(x2 - x1, y2 - y1)
        if not ref1:
            ref1 = float(np.median(len1[f1])) if f1.any() else 0.0
        if not ref2:
            ref2 = float(np.median(len2[f1 & f2])) if (f1 & f2).any() else 0.0

        ok1 = f1 & (np.abs(len1 - ref1) <= tol * ref1)
        ok2 = f2 & (np.abs(len2 - ref2) <= tol * ref2)

        if cfg["interpolate"]:
            if ok1.sum() < 2 or ok2.sum() < 2:
                raise RuntimeError(
                    "Too few valid frames (M1: %d, M2: %d out of %d). Check the "
                    "colour selection, the filter margins and the rod length "
                    "tolerance on the Settings tab." % (ok1.sum(), ok2.sum(), k))
            # M1 first, then M2 against the repaired M1: rod 2 is measured
            # from M1, so a bad M1 would also condemn a good M2
            x1 = np.interp(t, t[ok1], x1[ok1])
            y1 = np.interp(t, t[ok1], y1[ok1])
            len2 = np.hypot(x2 - x1, y2 - y1)
            ok2 = f2 & (np.abs(len2 - ref2) <= tol * ref2)
            if ok2.sum() < 2:
                raise RuntimeError("Too few valid frames for M2 after repairing "
                                   "M1. Check the tolerance and the margins.")
            x2 = np.interp(t, t[ok2], x2[ok2])
            y2 = np.interp(t, t[ok2], y2[ok2])

        # lengths and flags reported in the CSV refer to the exported points
        len1 = np.hypot(x1, y1)
        len2 = np.hypot(x2 - x1, y2 - y1)
        out1 = (~ok1).astype(int)
        out2 = (~ok2).astype(int)

        # principal values in (-180, 180] -- the angle between the rod and the
        # downward vertical. Unwrapping is optional: it makes the curve
        # continuous through full turns, but then the numbers grow without
        # bound (and every mis-detection near the top adds a spurious turn).
        th1, th2 = compute_angles(x1, y1, x2, y2, unwrap=cfg.get("unwrap", False))

        # diagnostics are always measured on the continuous version
        u1, u2 = compute_angles(x1, y1, x2, y2, unwrap=True)
        jump = (float(np.max(np.abs(np.diff(u1)))) if k > 1 else 0.0,
                float(np.max(np.abs(np.diff(u2)))) if k > 1 else 0.0)
        rods = ((float(len1.mean()), float(len1.min()), float(len1.max())),
                (float(len2.mean()), float(len2.min()), float(len2.max())))

        if cfg["degrees"]:
            th1, th2 = np.degrees(th1), np.degrees(th2)
        suffix = "_deg" if cfg["degrees"] else "_rad"

        df = pd.DataFrame({
            "Time": t, "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
            "theta1" + suffix: th1, "theta2" + suffix: th2,
            "L1": len1, "L2": len2, "L1_out": out1, "L2_out": out2,
        })
        return {
            "df": df, "n_frames": k,
            "len_ref": (ref1, ref2), "len_tol": tol * 100.0,
            "n_out": (int(out1.sum()), int(out2.sum())),
            "found1": int(f1.sum()), "found2": int(f2.sum()),
            "units": "deg" if cfg["degrees"] else "rad",
            "cols": ("theta1" + suffix, "theta2" + suffix),
            "track_video": cfg["video_out_path"] if cfg["write_video"] else None,
            "max_jump_deg": (np.degrees(jump[0]), np.degrees(jump[1])),
            "rods": rods,
            "unwrapped": bool(cfg.get("unwrap", False)),
            "turns2": float((u2[-1] - u2[0]) / (2 * np.pi)) if k > 1 else 0.0,
        }


# =====================================================================
#                              GUI
# =====================================================================

MAX_ZOOM = 20.0


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Double Pendulum Tracker")

        # the video area is sized to fit the actual screen (small laptops too)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.vw = int(clamp(sw - 400, 480, 860))
        self.vh = int(clamp(sh - 300, 300, 560))
        self.minsize(self.vw + 340, self.vh + 190)

        self.cfg = load_config()

        # --- media state
        self.video_path = None
        self.track_path = None
        self.viewing = "source"      # 'source' | 'tracking'
        self.cap = None
        self.n_frames = 0
        self.cur_idx = 0
        self.frame = None
        self.display_frame = None
        self.photo = None

        # --- view transform
        self.zoom = 1.0
        self.view_px = self.view_py = 0.0
        self.view_eff = 1.0
        self.view_x0 = self.view_y0 = 0
        self.off_x = self.off_y = 0

        # --- playback / interaction
        self.playing = False
        self._suppress_slider = False
        self._pan_anchor = None
        self.mode = None             # 'origin' | 'm1' | 'm2'
        self.drag = None

        # --- analysis state
        self.origin = None
        self.hsv1 = self.hsv2 = None
        self.start_frame = 0
        self.job = None
        self.result = None
        self.plot_win = None

        self._build_ui()
        self.bind_all("<MouseWheel>", self._on_wheel)     # Windows / macOS
        self.bind_all("<Button-4>", self._on_wheel)       # X11 scroll up
        self.bind_all("<Button-5>", self._on_wheel)       # X11 scroll down
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ================================================== layout =========
    def _build_ui(self):
        root = ttk.Frame(self, padding=4)
        root.pack(fill="both", expand=True)
        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(root, width=300)
        panel.pack(side="right", fill="y", padx=(6, 0))
        panel.pack_propagate(False)

        # ---------------- video area ----------------
        self.canvas = tk.Canvas(left, bg="#202020", width=self.vw, height=self.vh,
                                highlightthickness=0, cursor="tcross")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        for b in (2, 3):                                  # pan with right/middle
            self.canvas.bind(f"<Button-{b}>", self.on_pan_start)
            self.canvas.bind(f"<B{b}-Motion>", self.on_pan_move)
            self.canvas.bind(f"<ButtonRelease-{b}>", self.on_pan_end)

        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=(3, 0))
        self.btn_play = ttk.Button(nav, text="Play", width=6,
                                   command=self.toggle_play)
        self.btn_play.pack(side="left")
        ttk.Button(nav, text="|<", width=3,
                   command=lambda: self.goto(0)).pack(side="left", padx=(4, 1))
        ttk.Button(nav, text="<", width=3,
                   command=lambda: self.step(-1)).pack(side="left", padx=1)
        ttk.Button(nav, text=">", width=3,
                   command=lambda: self.step(1)).pack(side="left", padx=1)
        self.slider = ttk.Scale(nav, from_=0, to=0, orient="horizontal",
                                command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_frame = ttk.Label(nav, text="-/-", width=12, anchor="e")
        self.lbl_frame.pack(side="left")

        nav2 = ttk.Frame(left)
        nav2.pack(fill="x", pady=(2, 0))
        self.var_view = tk.StringVar(value="source")
        ttk.Radiobutton(nav2, text="source", value="source", variable=self.var_view,
                        command=lambda: self.set_view("source")).pack(side="left")
        self.rb_track = ttk.Radiobutton(nav2, text="control video", value="tracking",
                                        variable=self.var_view, state="disabled",
                                        command=lambda: self.set_view("tracking"))
        self.rb_track.pack(side="left", padx=(2, 10))
        ttk.Label(nav2, text="play fps").pack(side="left")
        self.var_playfps = tk.IntVar(value=int(self.cfg["playback_fps"]))
        ttk.Spinbox(nav2, from_=1, to=120, width=4,
                    textvariable=self.var_playfps).pack(side="left", padx=(2, 10))
        ttk.Label(nav2, text="zoom").pack(side="left")
        ttk.Button(nav2, text="-", width=2,
                   command=lambda: self.zoom_by(1 / 1.25)).pack(side="left", padx=1)
        ttk.Button(nav2, text="+", width=2,
                   command=lambda: self.zoom_by(1.25)).pack(side="left", padx=1)
        ttk.Button(nav2, text="Fit", width=4,
                   command=self.zoom_reset).pack(side="left", padx=1)
        self.lbl_zoom = ttk.Label(nav2, text="100%", width=6, anchor="w")
        self.lbl_zoom.pack(side="left", padx=(3, 0))

        self.status = ttk.Label(left, text="Open a video file to start.", anchor="w",
                                wraplength=self.vw)
        self.status.pack(fill="x", pady=(3, 0))
        ttk.Label(left, anchor="w", foreground="#666",
                  text="wheel = zoom   right-drag = pan   left click/drag = mark"
                  ).pack(fill="x")

        # ---------------- notebook: Main / Settings ----------------
        nb = ttk.Notebook(panel)
        nb.pack(fill="both", expand=True)
        main = ttk.Frame(nb, padding=4)
        self.tab_set = ttk.Frame(nb, padding=4)
        nb.add(main, text="Main")
        nb.add(self.tab_set, text="Settings")
        self._build_main_tab(main)
        self._build_settings_tab(self.tab_set)

    def _build_main_tab(self, m):
        ttk.Button(m, text="1. Open video...", command=self.open_video).pack(fill="x")
        self.lbl_video = ttk.Label(m, text="no file", wraplength=270,
                                   foreground="#555")
        self.lbl_video.pack(anchor="w")

        r = ttk.Frame(m)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="capture FPS:").pack(side="left")
        self.var_fps = tk.StringVar(value=self.cfg["fps"])
        ttk.Entry(r, textvariable=self.var_fps, width=8).pack(side="right")

        sep(m)
        ttk.Label(m, text="2. Geometry", font=("", 9, "bold")).pack(anchor="w")
        ttk.Button(m, text="Set origin (click on frame)",
                   command=lambda: self.set_mode("origin")).pack(fill="x", pady=1)
        self.lbl_origin = ttk.Label(m, text="origin: not set", foreground="#a00")
        self.lbl_origin.pack(anchor="w")
        ttk.Button(m, text="Start frame = current",
                   command=self.set_start_frame).pack(fill="x", pady=(3, 1))
        self.lbl_start = ttk.Label(m, text="start: frame 0")
        self.lbl_start.pack(anchor="w")

        sep(m)
        ttk.Label(m, text="3. Marker colours", font=("", 9, "bold")).pack(anchor="w")
        ttk.Button(m, text="Select M1 (drag a box)",
                   command=lambda: self.set_mode("m1")).pack(fill="x", pady=1)
        self.lbl_m1 = ttk.Label(m, text="M1: not set", foreground="#a00")
        self.lbl_m1.pack(anchor="w")
        ttk.Button(m, text="Select M2 (drag a box)",
                   command=lambda: self.set_mode("m2")).pack(fill="x", pady=1)
        self.lbl_m2 = ttk.Label(m, text="M2: not set", foreground="#a00")
        self.lbl_m2.pack(anchor="w")
        self.lbl_rods = ttk.Label(m, text="", wraplength=270, foreground="#555")
        self.lbl_rods.pack(anchor="w")

        sep(m)
        self.btn_run = ttk.Button(m, text="4. RUN TRACKING", command=self.run_tracking)
        self.btn_run.pack(fill="x")
        self.progress = ttk.Progressbar(m, mode="determinate")
        self.progress.pack(fill="x", pady=3)
        self.btn_stop = ttk.Button(m, text="Abort", command=self.stop_tracking,
                                   state="disabled")
        self.btn_stop.pack(fill="x")

        sep(m)
        ttk.Label(m, text="5. Output", font=("", 9, "bold")).pack(anchor="w")
        self.btn_save = ttk.Button(m, text="SAVE CSV...", command=self.save_csv,
                                   state="disabled")
        self.btn_save.pack(fill="x", pady=1)
        self.btn_plot = ttk.Button(m, text="Show angle plot", command=self.show_plot,
                                   state="disabled")
        self.btn_plot.pack(fill="x", pady=1)
        self.lbl_saved = ttk.Label(m, text="", wraplength=270, foreground="#070")
        self.lbl_saved.pack(anchor="w")

    def _build_settings_tab(self, s):
        ttk.Label(s, text="Folders", font=("", 9, "bold")).pack(anchor="w")
        self.var_movies = tk.StringVar(value=self.cfg["movies_dir"])
        self.var_csvdir = tk.StringVar(value=self.cfg["csv_dir"])
        for text, var in (("Movies folder", self.var_movies),
                          ("CSV output folder", self.var_csvdir)):
            ttk.Label(s, text=text + ":").pack(anchor="w", pady=(4, 0))
            row = ttk.Frame(s)
            row.pack(fill="x")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="...", width=3,
                       command=lambda v=var, t=text: self.pick_dir(v, t)
                       ).pack(side="left", padx=(2, 0))

        sep(s)
        ttk.Label(s, text="Colour filter margins",
                  font=("", 9, "bold")).pack(anchor="w")
        self.var_h = tk.IntVar(value=int(self.cfg["h_margin"]))
        self.var_s = tk.IntVar(value=int(self.cfg["s_margin"]))
        self.var_v = tk.IntVar(value=int(self.cfg["v_margin"]))
        self.var_min = tk.IntVar(value=int(self.cfg["min_area"]))
        for text, var, hi in (("hue H +/-", self.var_h, 90),
                              ("saturation S +/-", self.var_s, 255),
                              ("value V +/-", self.var_v, 255),
                              ("min blob area, px", self.var_min, 100000)):
            row = ttk.Frame(s)
            row.pack(fill="x")
            ttk.Label(row, text=text).pack(side="left")
            ttk.Spinbox(row, from_=1, to=hi, textvariable=var,
                        width=7).pack(side="right")

        sep(s)
        ttk.Label(s, text="Rod length check", font=("", 9, "bold")).pack(anchor="w")
        row = ttk.Frame(s)
        row.pack(fill="x")
        ttk.Label(row, text="tolerance, % of L on the marking frame").pack(side="left")
        self.var_lentol = tk.DoubleVar(value=float(self.cfg["len_tol"]))
        ttk.Spinbox(row, from_=1, to=100, increment=1, textvariable=self.var_lentol,
                    width=6).pack(side="right")
        ttk.Label(s, foreground="#666", wraplength=270, justify="left",
                  text="Both rods are rigid: frames whose measured length leaves "
                       "the band are mis-detections. They are flagged in the CSV "
                       "(L1_out / L2_out), drawn in red on the control video and, "
                       "with interpolation on, replaced by interpolated points."
                  ).pack(anchor="w")

        sep(s)
        ttk.Label(s, text="Output", font=("", 9, "bold")).pack(anchor="w")
        self.var_deg = tk.BooleanVar(value=self.cfg["degrees"] == "1")
        self.var_unwrap = tk.BooleanVar(value=self.cfg["unwrap"] == "1")
        self.var_interp = tk.BooleanVar(value=self.cfg["interpolate"] == "1")
        self.var_outvid = tk.BooleanVar(value=self.cfg["write_video"] == "1")
        ttk.Checkbutton(s, text="angles in degrees",
                        variable=self.var_deg).pack(anchor="w")
        ttk.Checkbutton(s, text="unwrap angles (continuous, allows >180 deg)",
                        variable=self.var_unwrap).pack(anchor="w")
        ttk.Checkbutton(s, text="interpolate missing points",
                        variable=self.var_interp).pack(anchor="w")
        ttk.Checkbutton(s, text="write control video",
                        variable=self.var_outvid).pack(anchor="w")
        row = ttk.Frame(s)
        row.pack(fill="x")
        ttk.Label(row, text="control video scale down").pack(side="left")
        self.var_scale = tk.IntVar(value=int(self.cfg["scale_down"]))
        ttk.Spinbox(row, from_=1, to=8, textvariable=self.var_scale,
                    width=7).pack(side="right")

        sep(s)
        ttk.Button(s, text="Save settings to config.ini",
                   command=self.save_settings).pack(fill="x")
        ttk.Label(s, text=CONFIG_PATH, wraplength=270,
                  foreground="#666").pack(anchor="w", pady=(2, 0))
        ttk.Label(s, foreground="#666", wraplength=270, justify="left",
                  text="Angles: theta = 0 when the rod hangs down, positive to "
                       "the right, range +/-180 deg. Rod 1 = origin -> M1, "
                       "rod 2 = M1 -> M2. Unwrapping only makes sense when the "
                       "rod really performs full turns.").pack(anchor="w",
                                                               pady=(6, 0))

    # ================================================== config =========
    def collect_config(self):
        return {
            "fps": self.var_fps.get(),
            "h_margin": str(self.var_h.get()),
            "s_margin": str(self.var_s.get()),
            "v_margin": str(self.var_v.get()),
            "min_area": str(self.var_min.get()),
            "len_tol": str(self.var_lentol.get()),
            "degrees": "1" if self.var_deg.get() else "0",
            "unwrap": "1" if self.var_unwrap.get() else "0",
            "interpolate": "1" if self.var_interp.get() else "0",
            "write_video": "1" if self.var_outvid.get() else "0",
            "scale_down": str(self.var_scale.get()),
            "playback_fps": str(self.var_playfps.get()),
            "movies_dir": self.var_movies.get(),
            "csv_dir": self.var_csvdir.get(),
        }

    def save_settings(self, quiet=False):
        try:
            save_config(self.collect_config())
            if not quiet:
                self.set_status(f"Settings saved to {CONFIG_PATH}")
        except Exception as e:
            if not quiet:
                messagebox.showerror("Could not save settings", str(e))

    def pick_dir(self, var, title):
        d = filedialog.askdirectory(title=title, initialdir=var.get() or APP_DIR)
        if d:
            var.set(d)

    def on_close(self):
        self.playing = False
        self.save_settings(quiet=True)
        if self.cap is not None:
            self.cap.release()
        self.destroy()

    # ================================================== media ==========
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            initialdir=self.var_movies.get() or APP_DIR,
            filetypes=[("Video", "*.MOV *.mov *.mp4 *.MP4 *.avi *.AVI *.mkv"),
                       ("All files", "*.*")])
        if not path:
            return
        self.stop_play()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the file.")
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.video_path = path
        self.track_path = None
        self.viewing = "source"
        self.var_view.set("source")
        self.rb_track.config(state="disabled")
        if not self.var_movies.get():
            self.var_movies.set(os.path.dirname(path))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.origin = self.hsv1 = self.hsv2 = None
        self.start_frame = 0
        self.result = None
        self.btn_save.config(state="disabled")
        self.btn_plot.config(state="disabled")
        self.lbl_saved.config(text="")
        self.lbl_origin.config(text="origin: not set", foreground="#a00")
        self.lbl_m1.config(text="M1: not set", foreground="#a00")
        self.lbl_m2.config(text="M2: not set", foreground="#a00")
        self.lbl_start.config(text="start: frame 0")
        self.lbl_video.config(text=os.path.basename(path))
        self.zoom_reset(render=False)
        self.slider.config(from_=0, to=max(0, self.n_frames - 1))
        self.goto(0)
        self.set_status(f"{os.path.basename(path)} - {self.n_frames} frames. "
                        f"Set the origin, the start frame and both marker colours.")

    def set_view(self, which):
        """Switch between the source video and the generated control video."""
        if which == "tracking" and not (self.track_path
                                        and os.path.exists(self.track_path)):
            self.var_view.set(self.viewing)
            return
        self.stop_play()
        path = self.video_path if which == "source" else self.track_path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open {path}")
            self.var_view.set(self.viewing)
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.viewing = which
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(from_=0, to=max(0, self.n_frames - 1))
        self.zoom_reset(render=False)
        self.goto(0)
        if which == "tracking":
            self.set_status("Control video: markers, rods and the angles of each "
                            "frame. Marking is disabled in this mode.")

    def read_frame(self, idx):
        idx = int(clamp(idx, 0, max(0, self.n_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return (frame if ret else None), idx

    def goto(self, idx):
        if self.cap is None:
            return
        frame, idx = self.read_frame(idx)
        if frame is None:
            return
        self.cur_idx = idx
        self.frame = frame
        self._set_slider(idx)
        self.render(frame)

    def _set_slider(self, idx):
        self._suppress_slider = True
        self.slider.set(idx)
        self._suppress_slider = False
        self.lbl_frame.config(text=f"{idx}/{max(0, self.n_frames-1)}")

    def on_slider(self, _v):
        if self.cap is None or self._suppress_slider:
            return
        idx = int(float(self.slider.get()))
        if idx != self.cur_idx:
            self.stop_play()
            self.goto(idx)

    def step(self, d):
        if self.cap is None:
            return
        self.stop_play()
        self.goto(self.cur_idx + d)

    # ------------------------------------------------- playback -------
    def toggle_play(self):
        if self.cap is None:
            return
        if self.playing:
            self.stop_play()
        else:
            self.playing = True
            self.btn_play.config(text="Pause")
            self._play_step()

    def stop_play(self):
        self.playing = False
        if hasattr(self, "btn_play"):
            self.btn_play.config(text="Play")

    def _play_step(self):
        if not self.playing or self.cap is None:
            return
        ret, frame = self.cap.read()     # sequential read: faster than seeking
        if not ret:
            self.stop_play()
            return
        self.cur_idx += 1
        self.frame = frame
        self._set_slider(self.cur_idx)
        self.render(frame)
        self.after(int(1000 / max(1, self.var_playfps.get())), self._play_step)

    # ================================================== rendering ======
    def render(self, frame_bgr=None):
        """Draw the current frame with the zoom/pan transform applied."""
        if frame_bgr is not None:
            self.display_frame = frame_bgr
        f = self.display_frame
        if f is None:
            return
        h, w = f.shape[:2]
        eff = min(self.vw / w, self.vh / h) * self.zoom

        sw = min(w, self.vw / eff)
        sh = min(h, self.vh / eff)
        self.view_px = clamp(self.view_px, 0, max(0, w - sw))
        self.view_py = clamp(self.view_py, 0, max(0, h - sh))
        x0, y0 = int(round(self.view_px)), int(round(self.view_py))
        x1 = min(w, x0 + int(round(sw)))
        y1 = min(h, y0 + int(round(sh)))
        crop = f[y0:y1, x0:x1]

        dw = max(1, int(round((x1 - x0) * eff)))
        dh = max(1, int(round((y1 - y0) * eff)))
        img = cv2.resize(crop, (dw, dh),
                         interpolation=cv2.INTER_NEAREST if eff > 1.2
                         else cv2.INTER_AREA)

        self.view_eff, self.view_x0, self.view_y0 = eff, x0, y0
        self.off_x = max(0, (self.vw - dw) // 2)
        self.off_y = max(0, (self.vh - dh) // 2)

        if self.origin is not None and self.viewing == "source":
            vx = (self.origin[0] - x0) * eff
            vy = (self.origin[1] - y0) * eff
            if -30 < vx < dw + 30 and -30 < vy < dh + 30:
                draw_crosshair(img, (vx, vy), size=max(10, int(14 * self.zoom)))

        if HAS_PIL:
            self.photo = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        else:
            _ok, buf = cv2.imencode(".ppm", img)
            self.photo = tk.PhotoImage(data=buf.tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, anchor="nw",
                                 image=self.photo)
        self.lbl_zoom.config(text=f"{self.zoom*100:.0f}%")

    def to_orig(self, x, y, clip=False):
        """Canvas coordinates -> original image coordinates."""
        ox = self.view_x0 + (x - self.off_x) / self.view_eff
        oy = self.view_y0 + (y - self.off_y) / self.view_eff
        if clip and self.display_frame is not None:
            h, w = self.display_frame.shape[:2]
            ox, oy = clamp(ox, 0, w - 1), clamp(oy, 0, h - 1)
        return ox, oy

    def inside_image(self, x, y):
        if self.display_frame is None:
            return False
        h, w = self.display_frame.shape[:2]
        ox, oy = self.to_orig(x, y)
        return 0 <= ox < w and 0 <= oy < h

    # ------------------------------------------------- zoom / pan -----
    def zoom_by(self, factor, cx=None, cy=None):
        if self.display_frame is None:
            return
        new = clamp(self.zoom * factor, 1.0, MAX_ZOOM)
        if abs(new - self.zoom) < 1e-9:
            return
        if cx is None:
            cx, cy = self.vw / 2, self.vh / 2
        ox, oy = self.to_orig(cx, cy)          # point that stays under cursor
        self.zoom = new
        h, w = self.display_frame.shape[:2]
        eff = min(self.vw / w, self.vh / h) * new
        off_x = max(0, (self.vw - min(self.vw, w * eff)) / 2)
        off_y = max(0, (self.vh - min(self.vh, h * eff)) / 2)
        self.view_px = ox - (cx - off_x) / eff
        self.view_py = oy - (cy - off_y) / eff
        self.render()

    def zoom_reset(self, render=True):
        self.zoom = 1.0
        self.view_px = self.view_py = 0.0
        if render:
            self.render()

    def _on_wheel(self, e):
        """One global wheel handler: zoom over the video, scroll elsewhere."""
        delta = 0
        if getattr(e, "num", None) == 4:
            delta = 1
        elif getattr(e, "num", None) == 5:
            delta = -1
        elif getattr(e, "delta", 0):
            delta = 1 if e.delta > 0 else -1
        if delta and e.widget is self.canvas:
            self.zoom_by(1.25 if delta > 0 else 1 / 1.25, e.x, e.y)

    def on_pan_start(self, e):
        self._pan_anchor = (e.x, e.y, self.view_px, self.view_py)
        self.canvas.config(cursor="fleur")

    def on_pan_move(self, e):
        if self._pan_anchor is None:
            return
        x0, y0, px, py = self._pan_anchor
        self.view_px = px - (e.x - x0) / self.view_eff
        self.view_py = py - (e.y - y0) / self.view_eff
        self.render()

    def on_pan_end(self, _e):
        self._pan_anchor = None
        self.canvas.config(cursor="tcross")

    # ================================================== marking ========
    def set_mode(self, mode):
        if self.frame is None:
            messagebox.showinfo("No video", "Open a video file first.")
            return
        if self.viewing != "source":
            messagebox.showinfo("Wrong view",
                                "Switch back to the source video to mark points.")
            return
        self.stop_play()
        self.mode = mode
        self.set_status({
            "origin": "Click on the pivot point (zoom in first for accuracy).",
            "m1": "Drag a box over mass M1.",
            "m2": "Drag a box over mass M2."}[mode])

    def on_press(self, e):
        if self.mode is None or self.frame is None:
            return
        if not self.inside_image(e.x, e.y):
            self.set_status("Click inside the frame, not on the black margin.")
            return
        if self.mode == "origin":
            ox, oy = self.to_orig(e.x, e.y)
            self.origin = (ox, oy)
            self.lbl_origin.config(text=f"origin: ({ox:.1f}, {oy:.1f})",
                                   foreground="#070")
            self.mode = None
            self.render()
            self.set_status("Origin set.")
        else:
            self.drag = (e.x, e.y, e.x, e.y)

    def on_drag(self, e):
        if self.drag is None:
            return
        x0, y0, _, _ = self.drag
        self.drag = (x0, y0, e.x, e.y)
        self.canvas.delete("roi")
        self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="yellow",
                                     width=2, tags="roi")

    def on_release(self, e):
        if self.drag is None:
            return
        x0, y0, x1, y1 = self.drag
        self.drag = None
        self.canvas.delete("roi")
        ax0, ay0 = self.to_orig(min(x0, x1), min(y0, y1), clip=True)
        ax1, ay1 = self.to_orig(max(x0, x1), max(y0, y1), clip=True)
        roi = (ax0, ay0, ax1 - ax0, ay1 - ay0)
        if roi[2] < 3 or roi[3] < 3:
            self.set_status("The box is too small - try again.")
            return
        hsv = avg_hsv_from_roi(self.frame, roi)
        if hsv is None:
            return
        txt = f"HSV {hsv[0]:.0f}/{hsv[1]:.0f}/{hsv[2]:.0f}"
        if self.mode == "m1":
            self.hsv1 = hsv
            self.lbl_m1.config(text="M1: " + txt, foreground="#070")
        else:
            self.hsv2 = hsv
            self.lbl_m2.config(text="M2: " + txt, foreground="#070")
        self.mode = None
        self.set_status("Colour stored.")

    def set_start_frame(self):
        if self.cap is None or self.viewing != "source":
            return
        self.start_frame = self.cur_idx
        fps = self.get_fps()
        self.lbl_start.config(
            text=f"start: frame {self.start_frame} (t={self.start_frame/fps:.3f} s)")

    def get_fps(self):
        try:
            f = float(self.var_fps.get())
            return f if f > 0 else 120.0
        except ValueError:
            return 120.0

    # ================================================== tracking =======
    def run_tracking(self):
        if self.video_path is None:
            messagebox.showinfo("No video", "Open a video file first.")
            return
        if self.hsv1 is None or self.hsv2 is None:
            messagebox.showinfo("No colours", "Select both masses with a box.")
            return
        if self.origin is None:
            if not messagebox.askyesno(
                    "No origin",
                    "The origin is not set; the top-left corner of the frame will "
                    "be used instead. Continue?"):
                return
            self.origin = (0.0, 0.0)

        self.stop_play()
        hm, sm, vm = self.var_h.get(), self.var_s.get(), self.var_v.get()
        out_vid = (os.path.splitext(self.video_path)[0] + "_tracking.mp4"
                   if self.var_outvid.get() else "")
        ref = self.measure_rods(build_filter(self.hsv1, hm, sm, vm),
                                build_filter(self.hsv2, hm, sm, vm))
        cfg = {
            "video_path": self.video_path,
            "start_frame": self.start_frame,
            "fps": self.get_fps(),
            "origin": self.origin,
            "filt1": build_filter(self.hsv1, hm, sm, vm),
            "filt2": build_filter(self.hsv2, hm, sm, vm),
            "min_size": self.var_min.get(),
            "len_tol": self.var_lentol.get(),
            "len_ref": ref,
            "interpolate": self.var_interp.get(),
            "degrees": self.var_deg.get(),
            "unwrap": self.var_unwrap.get(),
            "write_video": self.var_outvid.get(),
            "video_out_path": out_vid,
            "scale_down": self.var_scale.get(),
            "color1": (255, 0, 0), "color2": (0, 255, 0),
        }
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress["value"] = 0
        self.save_settings(quiet=True)
        self.job = TrackerJob(cfg, self._on_progress, self._on_done)
        self.job.start()

    def measure_rods(self, filt1, filt2):
        """Reference rod lengths, measured on the start (marking) frame.

        Returns (L1, L2) in pixels, or (None, None) if the masses cannot be
        detected there -- the tracker then falls back to the median length.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return (None, None)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_frame))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return (None, None)
        cx1, cy1, f1, _b = detect_color(frame, filt1, self.var_min.get())
        cx2, cy2, f2, _b = detect_color(frame, filt2, self.var_min.get())
        if not (f1 and f2):
            self.set_status("Warning: the masses were not detected on the marking "
                            "frame; rod lengths fall back to the median.")
            return (None, None)
        ox, oy = self.origin
        l1 = float(np.hypot(cx1 - ox, cy1 - oy))
        l2 = float(np.hypot(cx2 - cx1, cy2 - cy1))
        self.lbl_rods.config(text=f"rods on the marking frame: "
                                  f"L1={l1:.1f}, L2={l2:.1f} px")
        return (l1, l2)

    def _on_progress(self, k, total, n1, n2):
        def upd():
            self.progress["maximum"] = total
            self.progress["value"] = k
            self.set_status(f"Processed {k}/{total};  detected M1: {n1}, M2: {n2}")
        self.after(0, upd)

    def _on_done(self, result, err):
        def upd():
            self.btn_run.config(state="normal")
            self.btn_stop.config(state="disabled")
            if err:
                messagebox.showerror("Processing failed",
                                     err.strip().split("\n")[-1])
                self.set_status("Failed. See the console for the traceback.")
                print(err, file=sys.stderr)
                return
            self.result = result
            self.progress["value"] = self.progress["maximum"]
            n = result["n_frames"]
            j1, j2 = result["max_jump_deg"]
            (m1, lo1, hi1), (m2, lo2, hi2) = result["rods"]
            msg = (f"Done: {n} frames; M1 detected in {result['found1']} "
                   f"({100*result['found1']/n:.1f}%), M2 in {result['found2']} "
                   f"({100*result['found2']/n:.1f}%); max angle step per frame "
                   f"{j1:.1f} / {j2:.1f} deg; rod lengths "
                   f"{m1:.0f} ({lo1:.0f}-{hi1:.0f}) / "
                   f"{m2:.0f} ({lo2:.0f}-{hi2:.0f}) px.")
            if max(j1, j2) > 90:
                msg += ("  WARNING: angle steps near 180 deg - the frame rate is "
                        "too low.")
            o1, o2 = result["n_out"]
            r1, r2 = result["len_ref"]
            msg += (f"  Rod length check (+/-{result['len_tol']:.0f}% of "
                    f"L1={r1:.1f}, L2={r2:.1f} px): out of tolerance in "
                    f"{o1} / {o2} frames ({100*o1/n:.1f}% / {100*o2/n:.1f}%).")
            if max(o1, o2) > 0.1 * n:
                msg += ("  WARNING: more than 10% of the frames are mis-detected "
                        "- adjust the filter margins or the tolerance.")
            if result["unwrapped"] and abs(result["turns2"]) > 0.5:
                msg += (f"  Note: rod 2 winds {result['turns2']:+.1f} turns; with "
                        "unwrapping on, theta2 leaves the +/-180 deg range.")
            self.set_status(msg)
            self.btn_save.config(state="normal")
            self.btn_plot.config(state="normal")
            tv = result.get("track_video")
            if tv and os.path.exists(tv):
                self.track_path = tv
                self.rb_track.config(state="normal")
                self.var_view.set("tracking")
                self.set_view("tracking")
                self.toggle_play()
        self.after(0, upd)

    def stop_tracking(self):
        if self.job is not None:
            self.job.stop_flag.set()
            self.set_status("Stopping...")

    # ================================================== output =========
    def save_csv(self):
        if not self.result:
            return
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=base + "_angles.csv",
            initialdir=(self.var_csvdir.get()
                        or os.path.dirname(self.video_path)),
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        # the units are part of the column names: theta1_deg / theta1_rad
        self.result["df"].to_csv(path, index=False)
        if not self.var_csvdir.get():
            self.var_csvdir.set(os.path.dirname(path))
        png = os.path.splitext(path)[0] + ".png"
        try:
            self.save_figure(png)
        except Exception as e:
            png = "(plot failed)"
            print("Could not save the plot:", e, file=sys.stderr)
        self.save_settings(quiet=True)
        self.set_status(f"Saved: {path}")
        self.lbl_saved.config(text=f"Saved:\n{path}\n{png}")

    def _make_figure(self):
        """Build the angle figure.

        pyplot is deliberately not used: it keeps global state, and switching
        its backend to Agg for saving used to make every later plt.show() a
        silent no-op (that is why the plot window opened only once).
        """
        from matplotlib.figure import Figure
        df = self.result["df"]
        u = self.result["units"]
        c1, c2 = self.result["cols"]
        fig = Figure(figsize=(8, 5), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        for a, col, lab in ((ax1, c1, f"$\\theta_1$ [{u}]"),
                            (ax2, c2, f"$\\theta_2$ [{u}]")):
            a.plot(df.Time, df[col], color="tab:blue", lw=1)
            a.set_ylabel(lab)
            a.grid(alpha=.4)
        ax2.set_xlabel("Time [s]")
        fig.suptitle(f"{os.path.basename(self.video_path)}\n"
                     f"IC: $\\theta_1$={df[c1].iloc[0]:.2f}, "
                     f"$\\theta_2$={df[c2].iloc[0]:.2f} [{u}]")
        fig.tight_layout()
        return fig

    def save_figure(self, path):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = self._make_figure()
        FigureCanvasAgg(fig)
        fig.savefig(path, dpi=150)

    def show_plot(self):
        """Open the plot in its own window; works any number of times."""
        if not self.result:
            return
        from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                       NavigationToolbar2Tk)
        if self.plot_win is not None:
            try:
                if self.plot_win.winfo_exists():
                    self.plot_win.destroy()
            except tk.TclError:
                pass
        win = tk.Toplevel(self)
        win.title("Angles - " + os.path.basename(self.video_path))
        canvas = FigureCanvasTkAgg(self._make_figure(), master=win)
        canvas.draw()
        NavigationToolbar2Tk(canvas, win).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        win.protocol("WM_DELETE_WINDOW", lambda w=win: self._close_plot(w))
        self.plot_win = win

    def _close_plot(self, win):
        self.plot_win = None
        win.destroy()

    # ================================================== helpers ========
    def set_status(self, text):
        self.status.config(text=text)


def sep(parent):
    ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)


# =====================================================================
#                            CONFIG FILE
# =====================================================================

def load_config():
    """Read config.ini next to this script; missing keys fall back to defaults."""
    cfg = dict(DEFAULTS)
    parser = configparser.ConfigParser()
    try:
        if parser.read(CONFIG_PATH, encoding="utf-8") and parser.has_section("dp"):
            for key in DEFAULTS:
                if parser.has_option("dp", key):
                    cfg[key] = parser.get("dp", key)
    except Exception as e:
        print("config.ini could not be read:", e, file=sys.stderr)
    return cfg


def save_config(values):
    parser = configparser.ConfigParser()
    parser["dp"] = {k: str(v) for k, v in values.items()}
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        parser.write(fh)


if __name__ == "__main__":
    App().mainloop()
