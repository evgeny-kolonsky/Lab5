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

The file starts with a '#' comment block holding the version, the settings and
the diagnostics of that run. Read it back with
    pandas.read_csv(path, comment='#')          (pandas needs comment='#')
    numpy.genfromtxt(path, delimiter=',', names=True)

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

CHANGELOG
    3.11 the angular zero can be taken from a vertical line drawn on the rig:
         drag along it and its tilt in the image is subtracted from both angles.
         Works on any frame, needs no rest interval, and the line stays drawn on
         the frame for checking
    3.10 angular zero: the offsets of theta1 and theta2 can be measured on a
         frame with the pendulum at rest (or typed in) and are subtracted from
         the exported angles; the mean angle over the run is reported as a
         diagnostic. A zero error of a couple of degrees produces an energy
         beat at the swing frequency, which is easy to mistake for physics
    3.9.2 the control video now also shows the RAW detections (hollow circles)
          and the clicked pivot (grey cross) beside the corrected points that
          were actually used, so the geometric corrections are visible instead
          of looking like a drawing error
    3.9.1 after each run the energy scatter is evaluated for a range of
          smoothing windows and the best one is reported (not applied)
    3.9 smoothing window given in milliseconds instead of frames, so it adapts
        to the frame rate (13 frames was 108 ms at 120 fps but 433 ms at 30 fps,
        which clipped the velocity peaks); energy-based check of the time scale
        that reports the capture rate implied by energy conservation; the
        across-rod marker offset is frozen when the rods stay nearly collinear
    3.8 Settings moved onto a full-width tab: labelled blocks in three columns
        (two on a narrow screen) plus a scrollbar as a fallback, so nothing is
        clipped any more; the video lives on the Main tab only
    3.7.1 the control video is written in a second pass, after the corrections
          are known, so its numbers match the CSV exactly (before, the overlay
          used the clicked origin and was off by the origin correction - a few
          pixels, i.e. a couple of degrees of theta1); it now draws the
          exported points and angles, and marks frames with no data
    3.7 marker offsets and the origin fitted so that both rods measure the same
        constant length (the marker is glued on by hand and need not sit on the
        axle); the plot is no longer written to PNG next to the CSV
    3.6 Settings reordered around the pendulum parameters (detection thresholds
        moved to an advanced block at the bottom); compound-body energy model
        with centre of mass and moment of inertia per beam; reference rod
        lengths taken as the median over the run instead of one marking frame
    3.5 kinematic post-processing: angles fitted to both markers at once with
        the rod lengths fixed (weighted by each marker's measured scatter),
        Savitzky-Golay smoothing and velocities, intensity-weighted sub-pixel
        centroid; omega and total energy exported; pendulum parameters and the
        energy plot added (zero = both rods hanging at rest)
    3.4.2 control video overlay drawn after the downscale (the text was
          unreadable); angle plot no longer built through pyplot's tight_layout,
          which could raise and kill the window; plot errors are now reported
    3.4  time axis from the per-frame container timestamps, rescaled by the
         slow-motion factor (median frame interval); dropped frames detected
         and reported instead of silently compressing the record
    3.3  origin refined by a circle fit to the M1 trajectory; search gate for
         the detector; capture-FPS sanity check against the playback rate;
         interpolation gap limit; aborted / truncated runs marked
    3.2  rod length validation with a tolerance, L1/L2 and L1_out/L2_out
         columns, bad frames drawn in red on the control video
    3.1  angles kept in (-180, +180] by default, unwrapping optional; plot
         rebuilt without pyplot so it opens more than once
    3.0  Main / Settings tabs, config.ini, folders, control video carries the
         rods and the angle readout (preview button dropped), tighter layout
    2.2  omega columns dropped from the CSV, angle definition fixed (branch cut
         moved to straight up), frame-rate probing
    2.1  scrollable panel, zoom and pan, built-in player for the control video
    2.0  first GUI version: file dialogs, background tracking, CSV export
    1.x  original command line scripts by Anton S.
"""

__version__ = "3.11.0"

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
    "version": __version__,
    "fps": "120",
    "h_margin": "10",
    "s_margin": "75",
    "v_margin": "75",
    "min_area": "100",
    "gate": "120",
    "len_tol": "10",
    "zero1_deg": "0",
    "zero2_deg": "0",
    "origin_fit": "1",
    "equal_rods": "1",
    "max_gap": "5",
    "use_timestamps": "1",
    "joint_fit": "1",
    "sg_window_ms": "110",
    "pend_L1": "0.25",
    "pend_L2": "0.25",
    "pend_m1": "535.9",
    "pend_m2": "479.4",
    "pend_c1": "0.25",
    "pend_c2": "0.25",
    "pend_I1": "0",
    "pend_I2": "0",
    "pend_g": "9.8100",
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


def detect_color(frame_bgr, filt, min_size=100, hsv_frame=None,
                 prev=None, gate=0.0, subpixel=True):
    """Return (cx, cy, found, bbox) for the blob of the given colour.

    Without a gate the largest blob in the whole frame wins, which lets the
    track jump onto any similarly coloured object. With prev=(x, y) and
    gate > 0 only blobs whose centroid lies within `gate` pixels of the
    previous position are considered: a marker cannot teleport between two
    consecutive frames. If nothing is inside the gate the frame counts as a
    miss (and the caller widens the gate), instead of silently locking onto
    the wrong object.
    """
    if hsv_frame is None:
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv_frame, filt)
    bbox = (0, 0, 0, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_size:
            continue
        rect = cv2.boundingRect(c)
        cx = cy = None
        if subpixel:
            # Weight each pixel by how strongly coloured it is instead of 0/1:
            # the half-covered pixels on the rim then contribute in proportion,
            # which removes most of the threshold jitter from the centroid.
            x, y, w, h = rect
            pad = 2
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
            sub = mask[y0:y1, x0:x1]
            if sub.size:
                wgt = (hsv_frame[y0:y1, x0:x1, 1].astype(np.float64)
                       * (sub > 0)) + 1e-9
                tot = wgt.sum()
                if tot > 1e-6:
                    ys, xs = np.mgrid[y0:y1, x0:x1]
                    cx = float((wgt * xs).sum() / tot)
                    cy = float((wgt * ys).sum() / tot)
        if cx is None:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        cands.append((area, cx, cy, rect))
    if not cands:
        return 0, 0, False, bbox

    if prev is not None and gate and gate > 0:
        near = [c for c in cands
                if (c[1] - prev[0]) ** 2 + (c[2] - prev[1]) ** 2 <= gate * gate]
        if not near:
            return 0, 0, False, bbox
        cands = near

    area, cx, cy, rect = max(cands, key=lambda c: c[0])
    return cx, cy, True, rect


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


def sanitize_timestamps(ts_ms):
    """Raw container timestamps [ms] -> clean playback time axis [s] from zero.

    Occasional repeated or out-of-order stamps (they happen, especially around
    the first frame and with B-frames) are repaired by linear interpolation
    over the good ones. Returns (t_seconds, n_repaired) or (None, 0) if the
    stamps are hopeless.
    """
    ts = np.asarray(ts_ms, dtype=np.float64)
    n = ts.size
    if n < 5 or not np.all(np.isfinite(ts)):
        return None, 0
    good = np.zeros(n, dtype=bool)
    last = -np.inf
    for i in range(n):
        if ts[i] > last:
            good[i] = True
            last = ts[i]
    good[0] = True
    n_bad = int(n - good.sum())
    if good.sum() < max(5, 0.9 * n):
        return None, 0
    idx = np.arange(n, dtype=np.float64)
    ts = np.interp(idx, idx[good], ts[good])
    t = (ts - ts[0]) / 1000.0
    if t[-1] <= 0:
        return None, 0
    return t, n_bad


def playback_fps(path):
    """Frame rate at which the file plays back, measured, not copied.

    Seeks to the last frame and reads its timestamp, so this is
    (n_frames - 1) / duration rather than the nominal metadata value.
    Returns (fps, n_frames) or (None, n).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    fps = None
    ts = []
    for _ in range(600):            # nominal cadence from the first frames
        ret, _f = cap.read()
        if not ret:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    if len(ts) > 5:
        d = np.diff(np.asarray(ts, dtype=np.float64))
        d = d[(d > 1e-6) & np.isfinite(d)]
        if d.size > 3:
            # median, not mean: dropped frames must not bias the nominal rate
            fps = 1000.0 / float(np.median(d))
    if fps is None or not np.isfinite(fps) or fps <= 0:
        fps = fps_meta if fps_meta and fps_meta > 0 else None
    return fps, n


def check_fps(entered, play_fps):
    """Sanity-check the capture rate the user typed against the file.

    The ratio capture/playback is the slow-motion factor: 1 for an ordinary
    clip, an integer (2, 4, 5, 8, 10) for a clip exported already slowed down.
    Anything else is almost always a typo or the wrong file. This cannot prove
    the rate is right -- only a physical reference (a free-fall calibration
    clip) can -- but it catches gross mistakes.
    """
    if not play_fps or play_fps <= 0 or not entered or entered <= 0:
        return "playback rate unknown - the capture FPS cannot be checked"
    ratio = entered / play_fps
    txt = f"file plays at {play_fps:.4g} fps, entered {entered:g} -> factor x{ratio:.3g}"
    if abs(ratio - 1) < 0.02:
        return txt + " (real time, consistent)"
    for k in (2, 3, 4, 5, 8, 10, 16, 20):
        if abs(ratio - k) < 0.02 * k:
            return txt + f" (slow motion x{k}, consistent)"
    return "WARNING: " + txt + " - not a plausible slow-motion factor, check the "\
                               "capture FPS or the file"


def fit_circle(x, y, n_iter=2, clip_sigma=3.0):
    """Least-squares circle through the points; returns (cx, cy, R, sigma, n).

    The masses move on circles about their joints, so the centre of the circle
    fitted to the M1 trajectory IS the pivot -- far more accurate than a mouse
    click. Algebraic (Kasa) fit with a couple of sigma-clipping passes.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    cx = cy = R = np.nan
    for _ in range(max(1, n_iter)):
        if keep.sum() < 10:
            return np.nan, np.nan, np.nan, np.nan, int(keep.sum())
        xs, ys = x[keep], y[keep]
        A = np.column_stack([2 * xs, 2 * ys, np.ones(xs.size)])
        b = xs ** 2 + ys ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan, np.nan, int(keep.sum())
        cx, cy = float(sol[0]), float(sol[1])
        R = float(np.sqrt(max(0.0, sol[2] + cx ** 2 + cy ** 2)))
        res = np.hypot(x - cx, y - cy) - R
        sigma = float(np.std(res[keep]))
        if not np.isfinite(sigma) or sigma == 0:
            break
        keep = keep & (np.abs(res) < clip_sigma * sigma)
    res = np.hypot(x[keep] - cx, y[keep] - cy) - R
    return cx, cy, R, float(np.std(res)), int(keep.sum())


def angular_span_deg(x, y):
    """Angular extent covered by the points around their own mean centre.

    A circle fit needs a decent arc: a pendulum that barely moves gives an
    ill-conditioned fit and the 'refined' centre would be nonsense.
    """
    ang = np.unwrap(np.sort(np.arctan2(np.asarray(y), np.asarray(x))))
    if ang.size < 3:
        return 0.0
    gaps = np.diff(np.sort(np.mod(ang, 2 * np.pi)))
    largest_gap = max(gaps.max() if gaps.size else 0.0,
                      2 * np.pi - (np.sort(np.mod(ang, 2 * np.pi))[-1]
                                   - np.sort(np.mod(ang, 2 * np.pi))[0]))
    return float(np.degrees(2 * np.pi - largest_gap))


def interp_with_gap_limit(t, valid, values, max_gap):
    """Linear interpolation over invalid samples, but only across short gaps.

    Gaps longer than max_gap consecutive frames are left as NaN: silently
    bridging a long dropout invents a straight line where the real trajectory
    is curved, and that bias is invisible afterwards.
    """
    out = np.interp(t, t[valid], values[valid])
    if max_gap is not None and max_gap >= 0:
        bad = ~valid
        i = 0
        n = bad.size
        while i < n:
            if bad[i]:
                j = i
                while j < n and bad[j]:
                    j += 1
                if (j - i) > max_gap:
                    out[i:j] = np.nan
                i = j
            else:
                i += 1
    # nothing to extrapolate from before the first / after the last valid point
    first, last = np.argmax(valid), valid.size - 1 - np.argmax(valid[::-1])
    out[:first] = np.nan
    out[last + 1:] = np.nan
    return out


def savgol(y, window, poly=3, deriv=0, delta=1.0):
    """Savitzky-Golay filter / derivative, implemented on numpy alone.

    A sliding local polynomial fit. Smoothing uses the temporal side of the
    kinematics: the pendulum has mass, so theta(t) cannot jitter from frame to
    frame, and the fit separates that jitter from the motion. The derivative is
    taken from the same fitted polynomial instead of differencing neighbours,
    which is what makes the velocities (and therefore the energy) usable.
    No dynamical model is assumed, so the energy check stays independent.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    window = int(window)
    if window < 3 or n < window:
        if deriv == 0:
            return y.copy()
        return np.gradient(y, delta)
    if window % 2 == 0:
        window += 1
    poly = min(poly, window - 1)
    m = window // 2
    z = np.arange(-m, m + 1, dtype=np.float64)
    V = np.vander(z, poly + 1, increasing=True)
    P = np.linalg.pinv(V)                       # (poly+1, window)
    fact = float(np.math.factorial(deriv)) if hasattr(np, "math") else \
        float(np.prod(np.arange(1, deriv + 1))) if deriv else 1.0
    c = P[deriv] * (fact / delta ** deriv)
    out = np.empty(n, dtype=np.float64)
    out[m:n - m] = np.correlate(y, c, mode="valid")
    # edges: evaluate the polynomial fitted to the first / last full window
    for lo, hi, idx in ((0, window, np.arange(0, m)),
                        (n - window, n, np.arange(n - m, n))):
        zz = np.arange(lo, hi, dtype=np.float64) - (lo + hi - 1) / 2.0
        co = np.polyfit(zz, y[lo:hi], poly)          # highest power first
        d = np.polyder(co, deriv) if deriv else co
        out[idx] = np.polyval(d, idx - (lo + hi - 1) / 2.0) / delta ** deriv
    return out


def smooth_segments(t, y, window, poly=3, deriv=0):
    """Apply savgol() to each gap-free stretch of the signal separately."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full(y.size, np.nan)
    ok = np.isfinite(y)
    if not ok.any():
        return out
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    idx = np.where(ok)[0]
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if seg.size == 0:
            continue
        out[seg] = savgol(y[seg], window, poly, deriv, dt)
    return out


def fit_marker_offset(x1, y1, x2, y2, iters=40):
    """Nudge the origin and the M1 marker so that both rods measure the same
    constant length.

    The marker is stuck on by hand, so its centre need not sit exactly on the
    axle, and the pivot is clicked by hand too. Both offsets bias the lengths:
    an offset of M1 along rod 1 lengthens rod 1 and shortens rod 2 (and vice
    versa), which is why the two never come out equal even though the beams are.

    Fitted parameters: the residual origin shift (dx, dy), the M1 offset along
    and across rod 1, and the common rod length L. They are found by minimising

        sum_i (|M1' - O| - L)^2 + (|M2 - M1'| - L)^2

    over every frame, by Levenberg-Marquardt. Only meaningful when the two rods
    really are the same length by construction.

    Returns (dx, dy, a_along, b_across, L, rms_before, rms_after) in pixels.
    """
    P1 = np.vstack([x1, y1])
    P2 = np.vstack([x2, y2])
    good = np.isfinite(P1).all(axis=0) & np.isfinite(P2).all(axis=0)
    if good.sum() < 50:
        return (0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 0.0, False, False)
    P1, P2 = P1[:, good], P2[:, good]

    def apply(p):
        ox, oy, a, b = p[0], p[1], p[2], p[3]
        O = np.array([[ox], [oy]])
        v = P1 - O
        n = np.hypot(v[0], v[1])
        n = np.where(n > 1e-9, n, 1.0)
        u = v / n
        perp = np.vstack([-u[1], u[0]])
        M1 = P1 - a * u - b * perp
        return O, M1

    def resid(p):
        O, M1 = apply(p)
        return np.concatenate([np.hypot(*(M1 - O)) - p[4],
                               np.hypot(*(P2 - M1)) - p[4]])

    L0 = 0.5 * (float(np.median(np.hypot(P1[0], P1[1])))
                + float(np.median(np.hypot(P2[0] - P1[0], P2[1] - P1[1]))))
    rms0 = float(np.std(resid(np.array([0.0, 0.0, 0.0, 0.0, L0]))))

    # The across-rod offset of M1 is seen only through a sin(theta2 - theta1)
    # modulation of rod 2. If the two rods stay nearly collinear that signal is
    # absent and the parameter runs away, bending theta2 by degrees. Freeze it
    # unless the run actually bends enough.
    bend = np.degrees(np.arctan2(P2[0] - P1[0], P2[1] - P1[1])
                      - np.arctan2(P1[0], P1[1]))
    bend = (bend + 180) % 360 - 180
    span = float(np.percentile(bend, 97.5) - np.percentile(bend, 2.5))
    free = np.array([True, True, True, span >= 60.0, True])

    def run_lm(p):
        lam = 1e-3
        J = None
        for _ in range(iters):
            r = resid(p)
            J = np.zeros((r.size, p.size))
            for i in range(p.size):
                if not free[i]:
                    continue
                q = p.copy()
                h = 1e-4 * max(1.0, abs(p[i]))
                q[i] += h
                J[:, i] = (resid(q) - r) / h
            A = J.T @ J + lam * np.eye(p.size)
            A[~free, :] = 0.0
            A[:, ~free] = 0.0
            A[~free, ~free] = 1.0
            g = J.T @ r
            g[~free] = 0.0
            try:
                dp = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                break
            if np.sum(resid(p + dp) ** 2) < np.sum(r ** 2):
                p = p + dp
                lam = max(lam * 0.5, 1e-9)
            else:
                lam *= 5.0
                if lam > 1e6:
                    break
        return p, J

    p, J = run_lm(np.array([0.0, 0.0, 0.0, 0.0, L0]))

    # standard errors: an offset the data cannot resolve must not be applied
    sig = np.zeros(5)
    r = resid(p)
    dof = max(1, r.size - int(free.sum()))
    try:
        cov = np.linalg.pinv(J.T @ J) * (float(r @ r) / dof)
        sig = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        pass
    for i in (2, 3):                       # the two marker offsets
        if free[i] and abs(p[i]) < 2.0 * sig[i]:
            free[i] = False
            p[i] = 0.0
    if not free[2] or not free[3]:
        p, J = run_lm(p)

    return (float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]),
            rms0, float(np.std(resid(p))), float(span),
            bool(free[2]), bool(free[3]))


def apply_marker_offset(x1, y1, x2, y2, dx, dy, a, b):
    """Apply the offsets found by fit_marker_offset() to the tracked points."""
    v = np.vstack([x1 - dx, y1 - dy])
    n = np.hypot(v[0], v[1])
    n = np.where(n > 1e-9, n, 1.0)
    u = v / n
    perp = np.vstack([-u[1], u[0]])
    nx1 = x1 - dx - a * u[0] - b * perp[0]
    ny1 = y1 - dy - a * u[1] - b * perp[1]
    return nx1, ny1, x2 - dx, y2 - dy


def refine_angles_joint(x1, y1, x2, y2, len1, len2, iters=6, w1=1.0, w2=1.0):
    """Least-squares angles from BOTH markers and the two fixed rod lengths.

    theta1 alone would come from M1 only, yet the measured M2 also constrains
    it, because M2 = L1*u(theta1) + L2*u(theta2). Minimising

        w1*|P1 - L1*u1|^2 + w2*|P2 - L1*u1 - L2*u2|^2

    uses every measurement for every unknown. Each half of the problem has a
    closed form (the best unit vector is the normalised residual), so a few
    alternating passes converge. Purely geometric - no dynamics involved.

    The weights matter: on real footage M2 is usually the noisier marker, and
    with equal weights its error leaks into theta1 and cancels the gain. Pass
    w = 1/sigma^2 measured from the data.
    """
    P1 = np.vstack([x1, y1])
    P2 = np.vstack([x2, y2])

    def unit(v, L):
        n = np.hypot(v[0], v[1])
        n = np.where(n > 1e-9, n, 1.0)
        return np.vstack([v[0] / n * L, v[1] / n * L])

    B = unit(P2 - P1, len2)                     # start from the measurement
    A = unit(P1, len1)
    s = float(w1) + float(w2)
    for _ in range(max(1, iters)):
        A = unit((w1 * P1 + w2 * (P2 - B)) / s, len1)   # both markers vote
        B = unit(P2 - A, len2)
    return np.arctan2(A[0], A[1]), np.arctan2(B[0], B[1])


def total_energy(th1, th2, w1, w2, L1, L2, m1, m2, g=9.81,
                 c1=None, c2=None, I1=0.0, I2=0.0):
    """Total energy of the double pendulum, in joules.

    Compound bodies: beam k has mass m_k, its centre of mass sits c_k from its
    upper axle and its moment of inertia about that centre of mass is I_k. With
    c_k = L_k and I_k = 0 this collapses to point masses at the joints.

        T = 1/2 (I1 + m1 c1^2) w1^2
          + 1/2 m2 (L1^2 w1^2 + c2^2 w2^2 + 2 L1 c2 w1 w2 cos(th1 - th2))
          + 1/2 I2 w2^2
        V = -(m1 c1 + m2 L1) g cos(th1) - m2 g c2 cos(th2)

    Zero is both rods hanging straight down at rest: E = T + V - V(rest).
    """
    c1 = L1 if c1 is None else c1
    c2 = L2 if c2 is None else c2
    T = (0.5 * (I1 + m1 * c1 ** 2) * w1 ** 2
         + 0.5 * m2 * (L1 ** 2 * w1 ** 2 + c2 ** 2 * w2 ** 2
                       + 2 * L1 * c2 * w1 * w2 * np.cos(th1 - th2))
         + 0.5 * I2 * w2 ** 2)
    V = -(m1 * c1 + m2 * L1) * g * np.cos(th1) - m2 * g * c2 * np.cos(th2)
    V0 = -(m1 * c1 + m2 * L1) * g - m2 * g * c2
    return T + V - V0


def velocity_scale_check(t, th1, th2, w1, w2, pend):
    """Does the energy say the time axis is right?

    A pendulum driven only by gravity cannot have more kinetic energy than the
    potential energy it has given up, and its total energy must be flat apart
    from slow damping. If the capture rate entered is wrong by a factor k, the
    velocities are wrong by k and the kinetic term by k^2, which makes the
    energy swing wildly at the swing frequency. Scanning a scale factor s on
    the velocities and taking the one that makes E flattest recovers k
    directly: the suggested capture rate is fps * s.

    Returns (s, rms_at_1, rms_at_s) or (nan, nan, nan).
    """
    fin = np.isfinite(th1) & np.isfinite(th2) & np.isfinite(w1) & np.isfinite(w2)
    if fin.sum() < 100:
        return float("nan"), float("nan"), float("nan")
    t, th1, th2 = t[fin], th1[fin], th2[fin]
    w1, w2 = w1[fin], w2[fin]

    def rms(s):
        e = total_energy(th1, th2, s * w1, s * w2, pend["L1"], pend["L2"],
                         pend["m1"], pend["m2"], pend["g"], pend.get("c1"),
                         pend.get("c2"), pend.get("I1", 0.0), pend.get("I2", 0.0))
        return float(np.std(e - np.polyval(np.polyfit(t, e, 1), t)))

    grid = np.geomspace(0.05, 4.0, 60)          # coarse scan, then refine
    vals = [rms(s) for s in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(0, i - 1)], grid[min(grid.size - 1, i + 1)]
    for _ in range(40):                          # golden-section refinement
        m1_, m2_ = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if rms(m1_) < rms(m2_):
            hi = m2_
        else:
            lo = m1_
    s = 0.5 * (lo + hi)
    return float(s), rms(1.0), rms(s)


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
    if unwrap:                       # NaN gaps must not poison the rest
        for th in (th1, th2):
            m = np.isfinite(th)
            if m.sum() > 1:
                th[m] = np.unwrap(th[m])
    return th1, th2


def frame_angles_deg(origin, p1, p2):
    """Instantaneous angles [deg] of one frame, for the on-screen overlay."""
    th1 = np.degrees(np.arctan2(p1[0] - origin[0], p1[1] - origin[1]))
    th2 = np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1]))
    return th1, th2


def draw_overlay(img, origin, p1, p2, t=None, bad1=False, bad2=False,
                 th1=None, th2=None, note=""):
    """Draw the two rods, the vertical reference and the angle readout.

    A rod whose length is outside the tolerance is drawn in red: those frames
    are mis-detections, not physics.
    """
    p0 = (int(round(origin[0])), int(round(origin[1])))
    cv2.line(img, p0, (p0[0], p0[1] + 140), (170, 170, 170), 1)   # vertical ref
    draw_crosshair(img, p0)
    txt = f"t={t:.3f}s  " if t is not None else ""
    if p1 is None or p2 is None:
        txt += "NO DATA" + ("  " + note if note else "")
        bad1 = bad2 = True
    else:
        q1 = (int(round(p1[0])), int(round(p1[1])))
        q2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(img, p0, q1, (0, 0, 255) if bad1 else (0, 255, 255), 2)  # rod 1
        cv2.line(img, q1, q2, (0, 0, 255) if bad2 else (255, 255, 0), 2)  # rod 2
        # the angles come from the exported data, not recomputed here, so the
        # video and the CSV can never disagree
        if th1 is None or th2 is None:
            th1, th2 = frame_angles_deg(origin, p1, p2)
        txt += f"th1={th1:+.1f} th2={th2:+.1f} deg"
        if bad1 or bad2:
            txt += "  LENGTH!"
        if note:
            txt += "  " + note
    # The overlay is drawn at the OUTPUT resolution, so the font is sized to
    # this image and stays readable; a filled bar behind it guarantees contrast
    # whatever the frame looks like.
    w = img.shape[1]
    fs = max(0.5, w / 900.0)
    th = max(1, int(round(2 * fs)))
    while fs > 0.4:
        (tw, tht), _b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        if tw <= w - 16:
            break
        fs *= 0.92
        th = max(1, int(round(2 * fs)))
    (tw, tht), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, th)
    pad = max(3, int(4 * fs))
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, tht + base + 2 * pad),
                  (0, 0, 0), -1)
    cv2.putText(img, txt, (pad, tht + pad), cv2.FONT_HERSHEY_DUPLEX, fs,
                (0, 0, 255) if (bad1 or bad2) else (255, 255, 255), th,
                cv2.LINE_AA)


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

    def _write_control_video(self, cfg, start, n, origin, x1, y1, x2, y2,
                             a1, a2, out1, out2, t, origin_clicked=None,
                             raw=None, raw_found=None):
        """Second pass: draw exactly the exported geometry onto the frames.

        Filled dots and the red crosshair are the points actually used (after
        the origin circle fit and the marker-offset fit); hollow circles and the
        grey cross are the raw detection and the clicked pivot. The gap between
        them is the correction, not a drawing error.
        """
        cap = cv2.VideoCapture(cfg["video_path"])
        if not cap.isOpened():
            return
        sd = max(1, int(cfg["scale_down"]))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (max(2, fw // sd), max(2, fh // sd))
        sc = size[0] / float(fw)
        writer = cv2.VideoWriter(cfg["video_out_path"],
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 min(60.0, cfg["fps"]), size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ox, oy = origin
        for i in range(n):
            if self.stop_flag.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR)
            g = cv2.resize(g, size)
            # raw detections first, so the corrected ones are drawn on top
            if raw is not None:
                if origin_clicked is not None:
                    gx = int(round(origin_clicked[0] * sc))
                    gy = int(round(origin_clicked[1] * sc))
                    cv2.line(g, (gx - 7, gy), (gx + 7, gy), (140, 140, 140), 1)
                    cv2.line(g, (gx, gy - 7), (gx, gy + 7), (140, 140, 140), 1)
                for j, (rx, ry, col) in enumerate((
                        (raw[0][i], raw[1][i], cfg["color1"]),
                        (raw[2][i], raw[3][i], cfg["color2"]))):
                    if raw_found is not None and not raw_found[j][i]:
                        continue
                    if np.isfinite(rx) and np.isfinite(ry):
                        cv2.circle(g, (int(round((origin_clicked[0] + rx) * sc)),
                                       int(round((origin_clicked[1] + ry) * sc))),
                                   max(4, int(8 * sc)), col, 1)
            ok = np.isfinite(x1[i]) and np.isfinite(x2[i])
            p1 = p2 = None
            if ok:
                p1 = ((ox + x1[i]) * sc, (oy + y1[i]) * sc)
                p2 = ((ox + x2[i]) * sc, (oy + y2[i]) * sc)
                # small filled dot: it must not hide the marker underneath
                cv2.circle(g, (int(round(p1[0])), int(round(p1[1]))),
                           max(2, int(3 * sc)), cfg["color1"], -1)
                cv2.circle(g, (int(round(p2[0])), int(round(p2[1]))),
                           max(2, int(3 * sc)), cfg["color2"], -1)
            draw_overlay(g, (ox * sc, oy * sc), p1, p2, t=t[i],
                         bad1=bool(out1[i]), bad2=bool(out2[i]),
                         th1=a1[i] if ok else None, th2=a2[i] if ok else None)
            writer.write(g)
            if i % 25 == 0:
                self.progress_cb(i, max(1, n), -1, -1)
        cap.release()
        writer.release()

    def _work(self):
        cfg = self.cfg
        cap = cv2.VideoCapture(cfg["video_path"])
        if not cap.isOpened():
            raise RuntimeError("Could not open the video file.")

        start = int(cfg["start_frame"])
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        ox, oy = cfg["origin"]
        tol = max(0.0, float(cfg.get("len_tol", 10.0))) / 100.0
        ref1, ref2 = cfg.get("len_ref", (None, None))
        rows, found1, found2 = [], [], []
        k = 0

        gate0 = float(cfg.get("gate", 0) or 0)
        prev = [None, None]        # last accepted position of M1 / M2
        miss = [0, 0]              # consecutive misses -> the gate opens up
        stamps = []
        ret, frame = cap.read()
        # POS_MSEC read *after* a successful read belongs to that very frame;
        # read before the first grab it is stale by one frame
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        while ret and not self.stop_flag.is_set():
            stamps.append(t_ms)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # the gate grows with every missed frame; after 10 misses the search
            # goes global again so the track can re-acquire after an occlusion
            g = [0.0 if (gate0 <= 0 or prev[i] is None or miss[i] > 10)
                 else gate0 * (1 + miss[i]) for i in (0, 1)]
            cx1, cy1, f1, b1 = detect_color(frame, cfg["filt1"], cfg["min_size"],
                                            hsv, prev[0], g[0])
            cx2, cy2, f2, b2 = detect_color(frame, cfg["filt2"], cfg["min_size"],
                                            hsv, prev[1], g[1])
            for i, (fd, pos) in enumerate(((f1, (cx1, cy1)), (f2, (cx2, cy2)))):
                if fd:
                    prev[i], miss[i] = pos, 0
                else:
                    miss[i] += 1

            rows.append([cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy])
            found1.append(bool(f1))
            found2.append(bool(f2))

            k += 1
            if k % 10 == 0:
                self.progress_cb(k, max(1, n_total - start),
                                 int(np.sum(found1)), int(np.sum(found2)))
            ret, frame = cap.read()
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        cap.release()
        if k == 0:
            raise RuntimeError("No frames were read.")

        aborted = self.stop_flag.is_set()
        expected = max(0, n_total - start)
        truncated = (not aborted) and expected > 0 and k < expected - 1

        arr = np.asarray(rows, dtype=np.float64)
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        # keep the raw detections so the control video can show what was seen
        # next to what was used after the geometric corrections
        raw = (x1.copy(), y1.copy(), x2.copy(), y2.copy())
        raw_found = (np.asarray(found1, dtype=bool), np.asarray(found2, dtype=bool))
        f1 = np.asarray(found1, dtype=bool)
        f2 = np.asarray(found2, dtype=bool)
        # ---- time axis -------------------------------------------------
        # Uniform 1/fps is an assumption; the container timestamps are a
        # measurement of the PLAYBACK time. For an already-slowed slow-motion
        # clip the playback second is (fps_capture / fps_playback) physical
        # seconds, so the measured axis only has to be rescaled by that factor.
        # Shape from the file, scale from the capture rate the user entered.
        t = np.arange(k, dtype=np.float64) / cfg["fps"]
        time_note = f"time: uniform 1/{cfg['fps']:g} s"
        drop_note = ""
        if cfg.get("use_timestamps", True):
            t_pts, n_rep = (sanitize_timestamps(stamps)
                            if len(stamps) == k else (None, 0))
            if t_pts is not None:
                # The slow-motion factor comes from the MEDIAN frame interval of
                # this very run, not from an average over the file: a dropped
                # frame must not shift the scale, only stretch its own gap.
                med_pts = float(np.median(np.diff(t_pts))) if t_pts.size > 1 else 0.0
                slow = cfg["fps"] * med_pts if med_pts > 0 else 1.0
                t = t_pts / slow if slow > 0 else t_pts
                dt = np.diff(t)
                med = float(np.median(dt)) if dt.size else 0.0
                worst = float(np.max(np.abs(dt - med))) if dt.size else 0.0
                n_gap = int(np.sum(dt > 1.5 * med)) if med > 0 else 0
                time_note = (f"time: file timestamps / x{slow:.4g} slow motion "
                             f"-> {1/med if med else float('nan'):.5g} fps"
                             + (f", {n_rep} stamps repaired" if n_rep else ""))
                if n_gap:
                    drop_note = (f"  {n_gap} frame intervals are longer than 1.5x "
                                 f"the median (worst {1000*worst:.1f} ms): the "
                                 f"camera dropped frames - the timestamps keep "
                                 f"the timing right, a uniform grid would not.")
                elif worst > 0.25 * med:
                    drop_note = (f"  timestamp jitter up to {1000*worst:.1f} ms "
                                 f"({100*worst/med:.0f}% of the frame interval).")
            else:
                time_note = (f"time: timestamps unusable, fell back to uniform "
                             f"1/{cfg['fps']:g} s")

        # ---- refine the origin by fitting a circle to the M1 trajectory ----
        origin_note = "origin: as clicked"
        origin_shift = (0.0, 0.0)
        origin_final = (ox, oy)          # pivot in image coordinates
        if cfg.get("origin_fit", True):
            span = angular_span_deg(x1[f1], y1[f1]) if f1.sum() > 10 else 0.0
            cx, cy, R, sig, n_used = fit_circle(x1[f1], y1[f1])
            r_click = float(np.median(np.hypot(x1[f1], y1[f1]))) if f1.any() else 0.0
            shift = float(np.hypot(cx, cy)) if np.isfinite(cx) else np.inf
            if not np.isfinite(R) or R <= 0 or span < 25.0:
                origin_note = (f"origin: circle fit skipped, the arc covers only "
                               f"{span:.0f} deg")
            elif shift > 0.25 * max(R, r_click):
                origin_note = (f"origin: circle fit rejected, it moved the pivot by "
                               f"{shift:.1f} px (>25% of R={R:.1f})")
            else:
                x1, y1 = x1 - cx, y1 - cy
                x2, y2 = x2 - cx, y2 - cy
                ref1 = R
                origin_shift = (cx, cy)
                origin_final = (ox + cx, oy + cy)
                origin_note = (f"origin: circle fit moved the pivot by {shift:.2f} px "
                               f"(dx={cx:+.2f}, dy={cy:+.2f}), R={R:.2f} px, "
                               f"residual {sig:.2f} px over {n_used} frames, "
                               f"arc {span:.0f} deg")

        # ---- rod length validation ------------------------------------
        # Both rods are rigid, so their length is a constant: any frame where
        # the measured length leaves the tolerance band is a mis-detection.
        # The reference lengths come from the marking frame; if they are not
        # available, the median of the detected frames is used instead.
        len1 = np.hypot(x1, y1)
        len2 = np.hypot(x2 - x1, y2 - y1)
        # The reference is the MEDIAN over the whole run, not one marking frame:
        # a single frame catches whatever perspective and detection happen to do
        # in that pose, and on real footage that was off by several per cent.
        # The clicked-frame values are kept only for the report.
        ref1_frame, ref2_frame = ref1, ref2
        if f1.any():
            ref1 = float(np.median(len1[f1]))
        if (f1 & f2).any():
            ref2 = float(np.median(len2[f1 & f2]))
        if not ref1 or not ref2:
            raise RuntimeError("Could not establish the rod lengths - the masses "
                               "are almost never detected.")

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
            gap = cfg.get("max_gap", 5)
            x1 = interp_with_gap_limit(t, ok1, x1, gap)
            y1 = interp_with_gap_limit(t, ok1, y1, gap)
            len2 = np.hypot(x2 - x1, y2 - y1)
            ok2 = f2 & (np.abs(len2 - ref2) <= tol * ref2)
            if ok2.sum() < 2:
                raise RuntimeError("Too few valid frames for M2 after repairing "
                                   "M1. Check the tolerance and the margins.")
            x2 = interp_with_gap_limit(t, ok2, x2, gap)
            y2 = interp_with_gap_limit(t, ok2, y2, gap)

        # ---- equalise the rod lengths by fitting the marker offsets --------
        marker_note = "marker offsets: not fitted"
        if cfg.get("equal_rods", True):
            # A hand-placed marker is off by a pixel or two, which shows up as a
            # length difference of a per cent or so. A big difference means the
            # rods are simply NOT equal - forcing them together would corrupt
            # the data, so refuse instead.
            med1 = float(np.nanmedian(np.hypot(x1, y1)))
            med2 = float(np.nanmedian(np.hypot(x2 - x1, y2 - y1)))
            rel = abs(med1 - med2) / max(1e-9, 0.5 * (med1 + med2))
            (dx, dy, a_off, b_off, Lc, rms0, rms1,
             bend_span, used_a, used_b) = fit_marker_offset(x1, y1, x2, y2)
            if rel > 0.05:
                marker_note = (f"marker offsets: NOT applied - the rods measure "
                               f"{med1:.1f} and {med2:.1f} px ({100 * rel:.1f}% "
                               f"apart), far more than a marker offset can "
                               f"explain; switch the option off or check which "
                               f"points are being tracked")
            elif np.isfinite(rms1) and Lc > 0 and max(abs(a_off), abs(b_off),
                                                      abs(dx), abs(dy)) < 0.1 * Lc:
                x1, y1, x2, y2 = apply_marker_offset(x1, y1, x2, y2,
                                                     dx, dy, a_off, b_off)
                origin_final = (origin_final[0] + dx, origin_final[1] + dy)
                ref1 = ref2 = Lc
                frozen = ("" if used_b else
                          f" (across-rod offset frozen at 0: the rods bend by "
                          f"only {bend_span:.0f} deg over the run, too little to "
                          f"resolve it)")
                marker_note = (f"marker offsets: origin {dx:+.2f}, {dy:+.2f} px; "
                               f"M1 {a_off:+.2f} px along the rod, {b_off:+.2f} px "
                               f"across; common length {Lc:.2f} px; equal-length "
                               f"residual {rms0:.2f} -> {rms1:.2f} px" + frozen)
            else:
                marker_note = ("marker offsets: fit rejected (implausibly large "
                               "correction)")

        # lengths and flags reported in the CSV refer to the exported points
        len1 = np.hypot(x1, y1)
        len2 = np.hypot(x2 - x1, y2 - y1)
        out1 = (~ok1).astype(int)
        out2 = (~ok2).astype(int)

        # principal values in (-180, 180] -- the angle between the rod and the
        # downward vertical. Unwrapping is optional: it makes the curve
        # continuous through full turns, but then the numbers grow without
        # bound (and every mis-detection near the top adds a spurious turn).
        # ---- angles: geometric refinement, then temporal smoothing ---------
        win_probe = 5
        if cfg.get("joint_fit", True):
            # weight each marker by the scatter it actually shows: the noise of
            # M1 and M2 is neither equal nor white on real footage
            def hf(arr):
                r = arr - smooth_segments(t, arr, win_probe)
                r = r[np.isfinite(r)]
                return float(np.std(r)) if r.size > 10 else 1.0
            s1n = 0.5 * (hf(x1) + hf(y1))
            s2n = 0.5 * (hf(x2) + hf(y2))
            wa = 1.0 / max(s1n, 1e-6) ** 2
            wb = 1.0 / max(s2n, 1e-6) ** 2
            raw1, raw2 = refine_angles_joint(x1, y1, x2, y2, ref1, ref2,
                                             w1=wa, w2=wb)
            fit_note = (f"angles: joint fit of both markers, weights from the "
                        f"measured scatter (M1 {s1n:.2f} px, M2 {s2n:.2f} px)")
        else:
            raw1, raw2 = compute_angles(x1, y1, x2, y2, unwrap=False)
            fit_note = "angles: M1 and M2 taken independently"

        # everything below works on the continuous (unwrapped) angle
        u1, u2 = raw1.copy(), raw2.copy()
        for u in (u1, u2):
            mm = np.isfinite(u)
            if mm.sum() > 1:
                u[mm] = np.unwrap(u[mm])

        # The window is given as a TIME, not a frame count: 13 frames is 108 ms
        # at 120 fps but 433 ms at 30 fps, which would smooth away the motion
        # itself. Converted here using the actual frame interval.
        dt_now = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / cfg["fps"]
        win_ms = float(cfg.get("sg_ms", 110) or 0)
        win = int(round(win_ms / 1000.0 / max(dt_now, 1e-9)))
        win = win + 1 if win % 2 == 0 else win
        win = max(1, win)
        if win >= 3:
            s1 = smooth_segments(t, u1, win, 3, 0)
            s2 = smooth_segments(t, u2, win, 3, 0)
            w1 = smooth_segments(t, u1, win, 3, 1)
            w2 = smooth_segments(t, u2, win, 3, 1)
            dt_med = float(np.median(np.diff(t))) if t.size > 1 else 0.0
            fit_note += (f"; Savitzky-Golay smoothing, {win} frames "
                         f"({1000 * win * dt_med:.0f} ms), cubic")
        else:
            s1, s2 = u1, u2
            w1 = np.gradient(u1, t)
            w2 = np.gradient(u2, t)
            fit_note += "; no smoothing, velocities by finite differences"

        # angular zero: at rest the rods hang vertically, so a non-zero mean
        # over a free decay is the zero error (camera roll and geometry). It
        # biases the potential energy linearly in theta, which shows up as an
        # energy beat at the swing frequency.
        z1, z2 = cfg.get("zero_deg", (0.0, 0.0))
        zero_note = (f"angle zero: subtracted {z1:+.2f} / {z2:+.2f} deg"
                     if (z1 or z2) else "angle zero: not corrected")
        if z1 or z2:
            s1 = s1 - np.radians(z1)
            s2 = s2 - np.radians(z2)
        with np.errstate(invalid="ignore"):
            zero_note += (f"; mean angle over the run "
                          f"{np.degrees(np.nanmean(s1)):+.2f} / "
                          f"{np.degrees(np.nanmean(s2)):+.2f} deg "
                          f"(should be near zero for a free decay)")

        if cfg.get("unwrap", False):
            th1, th2 = s1, s2
        else:                               # back to (-180, +180]
            th1 = (s1 + np.pi) % (2 * np.pi) - np.pi
            th2 = (s2 + np.pi) % (2 * np.pi) - np.pi

        jump = tuple(float(np.nanmax(np.abs(np.diff(u)))) if k > 1 else 0.0
                     for u in (u1, u2))
        rods = ((float(np.nanmean(len1)), float(np.nanmin(len1)),
                 float(np.nanmax(len1))),
                (float(np.nanmean(len2)), float(np.nanmin(len2)),
                 float(np.nanmax(len2))))
        n_nan = int(np.sum(~np.isfinite(x1) | ~np.isfinite(x2)))

        # ---- energy (optional, needs the pendulum parameters) --------------
        pend = cfg.get("pendulum") or {}
        energy = None
        time_warning = ""
        window_note = ""
        energy_note = "energy: pendulum parameters not set"
        if pend and pend.get("L1") and pend.get("m1"):
            energy = total_energy(s1, s2, w1, w2, pend["L1"], pend["L2"],
                                  pend["m1"], pend["m2"], pend["g"],
                                  pend.get("c1"), pend.get("c2"),
                                  pend.get("I1", 0.0), pend.get("I2", 0.0))
            # which smoothing window would the energy have preferred? The best
            # window depends on how fast the pendulum moves, not on the frame
            # rate alone, so this is reported rather than applied silently.
            try:
                cand, best, cur = [], None, None
                trials = [w for w in (1, 3, 5, 7, 9, 13, 17, 21, 27)
                          if w <= max(5, k // 20)]
                if win not in trials:
                    trials.append(win)          # always measure the current one
                for wtry in trials:
                    if wtry >= 3:
                        q1 = smooth_segments(t, u1, wtry, 3, 0)
                        q2 = smooth_segments(t, u2, wtry, 3, 0)
                        p1_ = smooth_segments(t, u1, wtry, 3, 1)
                        p2_ = smooth_segments(t, u2, wtry, 3, 1)
                    else:
                        q1, q2 = u1, u2
                        p1_, p2_ = np.gradient(u1, t), np.gradient(u2, t)
                    ee = total_energy(q1, q2, p1_, p2_, pend["L1"], pend["L2"],
                                      pend["m1"], pend["m2"], pend["g"],
                                      pend.get("c1"), pend.get("c2"),
                                      pend.get("I1", 0.0), pend.get("I2", 0.0))
                    sc_ = float(np.nanstd(ee - smooth_segments(t, ee, 401, 2, 0)))
                    cand.append((sc_, wtry))
                    if wtry == win:
                        cur = sc_
                if cand:
                    best = min(cand)
                    if cur is None:
                        cur = best[0]
                    if best[1] != win and best[0] < 0.85 * cur:
                        window_note = (
                            f"  Smoothing: {best[1]} frames "
                            f"({1000 * best[1] * dt_now:.0f} ms) would leave "
                            f"{best[0]:.3f} J of energy scatter instead of "
                            f"{cur:.3f} J at the current "
                            f"{win} frames ({win_ms:.0f} ms).")
            except Exception as _e:
                print("window advisor failed:", _e, file=sys.stderr)

            # does the energy agree with the time axis?
            sc, rms_1, rms_s = velocity_scale_check(t, s1, s2, w1, w2, pend)
            if np.isfinite(sc) and abs(np.log(sc)) > np.log(1.15) and \
                    rms_s < 0.5 * rms_1:
                time_warning = (
                    f"TIME SCALE: the energy is far from constant at the entered "
                    f"{cfg['fps']:g} fps, but becomes flat ({rms_1:.2f} -> "
                    f"{rms_s:.2f} J) if the velocities are scaled by {sc:.3f}. "
                    f"That points at a true capture rate of about "
                    f"{cfg['fps'] * sc:.4g} fps - check the FPS field.")
            fin = np.isfinite(energy)
            energy_note = (f"energy: L={pend['L1']:.3f}/{pend['L2']:.3f} m, "
                           f"m={1000*pend['m1']:.1f}/{1000*pend['m2']:.1f} g, "
                           f"g={pend['g']:.4g}; E from "
                           f"{np.min(energy[fin]):.3f} to "
                           f"{np.max(energy[fin]):.3f} J"
                           if fin.any() else "energy: not computable")

        if cfg["degrees"]:
            th1, th2 = np.degrees(th1), np.degrees(th2)
            w1o, w2o = np.degrees(w1), np.degrees(w2)
        else:
            w1o, w2o = w1, w2
        suffix = "_deg" if cfg["degrees"] else "_rad"
        wsuffix = "_deg_s" if cfg["degrees"] else "_rad_s"

        data = {
            "Time": t, "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
            "theta1" + suffix: th1, "theta2" + suffix: th2,
            "omega1" + wsuffix: w1o, "omega2" + wsuffix: w2o,
        }
        if energy is not None:
            data["E_J"] = energy
        data.update({"L1": len1, "L2": len2, "L1_out": out1, "L2_out": out2})
        df = pd.DataFrame(data)
        # ---- control video, written in a SECOND pass -----------------------
        # The origin correction, the marker offsets, the interpolation and the
        # smoothing are only known once the whole run is processed. Drawing
        # during the first pass therefore showed raw numbers that disagreed
        # with the CSV (by the origin shift: a few px = a couple of degrees).
        wrote_video = False
        if cfg["write_video"] and not aborted:
            disp1 = np.degrees((s1 + np.pi) % (2 * np.pi) - np.pi)
            disp2 = np.degrees((s2 + np.pi) % (2 * np.pi) - np.pi)
            self._write_control_video(cfg, start, k, origin_final,
                                      x1, y1, x2, y2, disp1, disp2,
                                      out1, out2, t, (ox, oy), raw, raw_found)
            wrote_video = True

        return {
            "df": df, "n_frames": k,
            "len_ref": (ref1, ref2), "len_tol": tol * 100.0,
            "len_ref_frame": (ref1_frame or 0.0, ref2_frame or 0.0),
            "n_out": (int(out1.sum()), int(out2.sum())),
            "n_nan": n_nan,
            "origin_note": origin_note, "origin_shift": origin_shift,
            "marker_note": marker_note, "zero_note": zero_note,
            "time_note": time_note, "drop_note": drop_note,
            "time_warning": time_warning, "window_note": window_note,
            "aborted": aborted, "truncated": truncated,
            "expected_frames": expected,
            "found1": int(f1.sum()), "found2": int(f2.sum()),
            "units": "deg" if cfg["degrees"] else "rad",
            "cols": ("theta1" + suffix, "theta2" + suffix),
            "wcols": ("omega1" + wsuffix, "omega2" + wsuffix),
            "has_energy": energy is not None,
            "fit_note": fit_note, "energy_note": energy_note,
            "track_video": cfg["video_out_path"] if wrote_video else None,
            "max_jump_deg": (np.degrees(jump[0]), np.degrees(jump[1])),
            "rods": rods,
            "unwrapped": bool(cfg.get("unwrap", False)),
            "turns2": (float((u2[np.isfinite(u2)][-1] - u2[np.isfinite(u2)][0])
                             / (2 * np.pi)) if np.isfinite(u2).sum() > 1 else 0.0),
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
        self.title(f"Double Pendulum Tracker  v{__version__}")

        # the video area is sized to fit the actual screen (small laptops too)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.vw = int(clamp(sw - 400, 480, 860))
        self.vh = int(clamp(sh - 300, 300, 560))
        self.minsize(min(self.vw + 340, sw - 40), min(self.vh + 190, sh - 80))
        # open wide enough for the three-column Settings tab when the screen allows
        self.geometry(f"{min(max(self.vw + 340, 1180), sw - 40)}x"
                      f"{min(self.vh + 210, sh - 80)}+20+20")

        self.cfg = load_config()

        # --- media state
        self.video_path = None
        self.play_fps = None
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
        self.mode = None             # 'origin' | 'm1' | 'm2' | 'vert'
        self.drag = None
        self.vert_line = None        # two points of the drawn plumb line

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
        """Two tabs. Main carries the video and the per-run controls; Settings
        gets the whole window, laid out in three columns, so nothing is cut off
        on a small screen (a scrollbar appears only if it still does not fit).
        """
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        tab_main = ttk.Frame(nb, padding=4)
        tab_set = ttk.Frame(nb, padding=4)
        nb.add(tab_main, text="Main")
        nb.add(tab_set, text="Settings")

        left = ttk.Frame(tab_main)
        left.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(tab_main, width=300)
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

        self._build_main_tab(panel)

        self.set_canvas, set_inner = scrollable(tab_set)
        self._build_settings_tab(set_inner)

    def _build_main_tab(self, m):
        ttk.Button(m, text="1. Open video...", command=self.open_video).pack(fill="x")
        self.lbl_video = ttk.Label(m, text="no file", wraplength=270,
                                   foreground="#555")
        self.lbl_video.pack(anchor="w")

        r = ttk.Frame(m)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="capture FPS:").pack(side="left")
        self.var_fps = tk.StringVar(value=self.cfg["fps"])
        e = ttk.Entry(r, textvariable=self.var_fps, width=8)
        e.pack(side="right")
        e.bind("<FocusOut>", lambda _e: self.update_fps_check())
        e.bind("<Return>", lambda _e: self.update_fps_check())
        self.lbl_fpschk = ttk.Label(m, text="", wraplength=270, foreground="#555")
        self.lbl_fpschk.pack(anchor="w")

        sep(m)
        ttk.Label(m, text="2. Geometry", font=("", 9, "bold")).pack(anchor="w")
        ttk.Button(m, text="Set origin (click on frame)",
                   command=lambda: self.set_mode("origin")).pack(fill="x", pady=1)
        self.lbl_origin = ttk.Label(m, text="origin: not set", foreground="#a00")
        self.lbl_origin.pack(anchor="w")
        ttk.Button(m, text="Set vertical (drag along the plumb line)",
                   command=lambda: self.set_mode("vert")).pack(fill="x", pady=(3, 1))
        ttk.Button(m, text="...or zero on this frame (pendulum at rest)",
                   command=self.set_angle_zero).pack(fill="x", pady=(0, 1))
        self.lbl_zero = ttk.Label(m, text="angle zero: 0.00 / 0.00 deg",
                                  foreground="#555")
        self.lbl_zero.pack(anchor="w")
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
        self.btn_energy = ttk.Button(m, text="Energy plot",
                                     command=self.show_energy, state="disabled")
        self.btn_energy.pack(fill="x", pady=1)
        self.lbl_saved = ttk.Label(m, text="", wraplength=270, foreground="#070")
        self.lbl_saved.pack(anchor="w")

    def _build_settings_tab(self, s):
        """Three columns of labelled blocks, ordered by how often they change:
        what a run needs on the left and centre, thresholds bottom right.
        """
        # three columns on a wide window, two on a narrow one; a vertical
        # scrollbar is the last-resort fallback so nothing can be cut off
        ncols = 3 if (self.vw + 340) >= 1150 else 2
        for c in range(ncols):
            s.columnconfigure(c, weight=1, uniform="set")
        col = [ttk.Frame(s) for _ in range(ncols)]
        for c, f in enumerate(col):
            f.grid(row=0, column=c, sticky="nsew", padx=4)

        def where(i3, i2):
            """Which column a block goes to, for the 3- and 2-column layouts."""
            return col[i3 if ncols == 3 else i2]

        # ---------------- column 1 ----------------
        b = block(where(0, 0), "Folders")
        self.var_movies = tk.StringVar(value=self.cfg["movies_dir"])
        self.var_csvdir = tk.StringVar(value=self.cfg["csv_dir"])
        for text, var in (("Movies folder", self.var_movies),
                          ("CSV output folder", self.var_csvdir)):
            ttk.Label(b, text=text + ":").pack(anchor="w")
            row = ttk.Frame(b)
            row.pack(fill="x", pady=(0, 4))
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="...", width=3,
                       command=lambda v=var, t=text: self.pick_dir(v, t)
                       ).pack(side="left", padx=(2, 0))

        b = block(where(0, 0), "Time axis")
        self.var_ts = tk.BooleanVar(value=self.cfg["use_timestamps"] == "1")
        ttk.Checkbutton(b, text="use file timestamps (scaled by slow motion)",
                        variable=self.var_ts).pack(anchor="w")
        hint(b, "Timestamps measure PLAYBACK time and are divided by the "
                "slow-motion factor (capture FPS / playback FPS). Unlike a "
                "uniform grid they keep dropped frames in their true place. "
                "Off = uniform 1/FPS.")

        b = block(where(0, 0), "Geometry fit")
        self.var_ofit = tk.BooleanVar(value=self.cfg["origin_fit"] == "1")
        ttk.Checkbutton(b, text="refine origin by circle fit",
                        variable=self.var_ofit).pack(anchor="w")
        hint(b, "M1 moves on a circle about the pivot, so the fitted centre is "
                "the pivot - a click is only good to a couple of pixels, which "
                "is 1-2 deg of theta1.")
        row = ttk.Frame(b)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="angle zero, deg   th1").pack(side="left")
        self.var_z1 = tk.StringVar(value=self.cfg["zero1_deg"])
        self.var_z2 = tk.StringVar(value=self.cfg["zero2_deg"])
        ttk.Entry(row, textvariable=self.var_z1, width=7).pack(side="left", padx=2)
        ttk.Label(row, text="th2").pack(side="left")
        ttk.Entry(row, textvariable=self.var_z2, width=7).pack(side="left", padx=2)
        hint(b, "The zero of both angles must be the true vertical. An error of "
                "a couple of degrees biases the potential energy linearly in "
                "theta and shows up as an energy beat at the swing frequency - "
                "easy to mistake for physics. Best: drag along a plumb line "
                "drawn on the rig (button on the Main tab) - it works on any "
                "frame and both angles get the same correction. Alternative: a "
                "frame with the pendulum hanging still. Or type the values.")
        self.var_eqrod = tk.BooleanVar(value=self.cfg["equal_rods"] == "1")
        ttk.Checkbutton(b, text="equalise rod lengths (fit marker offsets)",
                        variable=self.var_eqrod).pack(anchor="w", pady=(4, 0))
        hint(b, "Only when both beams are the same length by construction. The "
                "marker is glued on by hand, so its centre need not sit on the "
                "axle: an offset of M1 along rod 1 lengthens rod 1 and shortens "
                "rod 2. The offsets and the origin are fitted so that both rods "
                "measure the same constant length. Refused automatically if the "
                "rods differ by more than 5%.")

        # ---------------- column 2 ----------------
        b = block(where(1, 1), "Pendulum (energy check)")
        self.var_pL1 = tk.StringVar(value=self.cfg["pend_L1"])
        self.var_pL2 = tk.StringVar(value=self.cfg["pend_L2"])
        self.var_pm1 = tk.StringVar(value=self.cfg["pend_m1"])
        self.var_pm2 = tk.StringVar(value=self.cfg["pend_m2"])
        self.var_pc1 = tk.StringVar(value=self.cfg["pend_c1"])
        self.var_pc2 = tk.StringVar(value=self.cfg["pend_c2"])
        self.var_pI1 = tk.StringVar(value=self.cfg["pend_I1"])
        self.var_pI2 = tk.StringVar(value=self.cfg["pend_I2"])
        self.var_pg = tk.StringVar(value=self.cfg["pend_g"])
        grid = ttk.Frame(b)
        grid.pack(fill="x")
        ttk.Label(grid, text="upper").grid(row=0, column=1, padx=2)
        ttk.Label(grid, text="lower").grid(row=0, column=2, padx=2)
        rows = (("L, m   axle to axle", self.var_pL1, self.var_pL2),
                ("m, g   mass", self.var_pm1, self.var_pm2),
                ("c, m   axle to c.o.m.", self.var_pc1, self.var_pc2),
                ("I, kg*m2  about c.o.m.", self.var_pI1, self.var_pI2))
        for r, (lab, va, vb) in enumerate(rows, start=1):
            ttk.Label(grid, text=lab).grid(row=r, column=0, sticky="w", pady=1)
            ttk.Entry(grid, textvariable=va, width=9).grid(row=r, column=1, padx=2)
            ttk.Entry(grid, textvariable=vb, width=9).grid(row=r, column=2, padx=2)
        ttk.Label(grid, text="g, m/s2").grid(row=5, column=0, sticky="w", pady=1)
        ttk.Entry(grid, textvariable=self.var_pg, width=9).grid(row=5, column=1, padx=2)
        hint(b, "c = L and I = 0 means point masses at the joints; enter the "
                "measured centre of mass and inertia for compound beams. Zero of "
                "energy = both rods hanging at rest. E_J goes into the CSV and "
                "into the Energy plot; a flat curve with slow decay means the "
                "tracking is sound.")

        b = block(where(1, 1), "Kinematic post-processing")
        self.var_joint = tk.BooleanVar(value=self.cfg["joint_fit"] == "1")
        ttk.Checkbutton(b, text="joint fit of both markers (rigid rods)",
                        variable=self.var_joint).pack(anchor="w")
        self.var_sg = tk.IntVar(value=int(float(self.cfg["sg_window_ms"])))
        spin_row(b, "smoothing window, ms (0 = off)", self.var_sg, 0, 2000, 10)
        hint(b, "The window is a TIME, so it adapts to the frame rate: 110 ms "
                "is 13 frames at 120 fps but 3 frames at 30 fps. A window "
                "longer than about a tenth of the fastest swing period clips "
                "the velocity peaks and the energy dips exactly where the "
                "motion is fastest.\n"
                "Geometry: both angles are fitted to both marker positions at "
                "once with the rod lengths fixed, weighted by the scatter each "
                "marker actually shows. Time: a cubic Savitzky-Golay window "
                "smooths the angles and supplies the velocities from the same "
                "fit. Kinematics only - no equations of motion - so the energy "
                "check stays independent.")

        # ---------------- column 3 ----------------
        b = block(where(2, 1), "Output")
        self.var_deg = tk.BooleanVar(value=self.cfg["degrees"] == "1")
        self.var_unwrap = tk.BooleanVar(value=self.cfg["unwrap"] == "1")
        self.var_interp = tk.BooleanVar(value=self.cfg["interpolate"] == "1")
        self.var_outvid = tk.BooleanVar(value=self.cfg["write_video"] == "1")
        ttk.Checkbutton(b, text="angles in degrees",
                        variable=self.var_deg).pack(anchor="w")
        ttk.Checkbutton(b, text="unwrap angles (continuous, allows >180 deg)",
                        variable=self.var_unwrap).pack(anchor="w")
        ttk.Checkbutton(b, text="interpolate missing points",
                        variable=self.var_interp).pack(anchor="w")
        ttk.Checkbutton(b, text="write control video",
                        variable=self.var_outvid).pack(anchor="w")
        self.var_scale = tk.IntVar(value=int(self.cfg["scale_down"]))
        spin_row(b, "control video scale down", self.var_scale, 1, 8)

        b = block(where(2, 1), "Rod length check")
        self.var_lentol = tk.DoubleVar(value=float(self.cfg["len_tol"]))
        spin_row(b, "tolerance, % of the median length", self.var_lentol, 1, 100)
        self.var_gap = tk.IntVar(value=int(self.cfg["max_gap"]))
        spin_row(b, "max interpolated gap, frames", self.var_gap, 0, 1000)
        hint(b, "The rods are rigid, so a frame whose length leaves the band is "
                "a mis-detection: flagged in the CSV (L1_out / L2_out), drawn "
                "red on the control video and, with interpolation on, replaced. "
                "Longer dropouts are left empty rather than bridged by a "
                "straight line.")

        b = block(where(2, 0), "Detection (advanced)")
        self.var_h = tk.IntVar(value=int(self.cfg["h_margin"]))
        self.var_s = tk.IntVar(value=int(self.cfg["s_margin"]))
        self.var_v = tk.IntVar(value=int(self.cfg["v_margin"]))
        self.var_min = tk.IntVar(value=int(self.cfg["min_area"]))
        self.var_gate = tk.IntVar(value=int(self.cfg["gate"]))
        spin_row(b, "hue H +/-", self.var_h, 0, 90)
        spin_row(b, "saturation S +/-", self.var_s, 0, 255)
        spin_row(b, "value V +/-", self.var_v, 0, 255)
        spin_row(b, "min blob area, px", self.var_min, 0, 100000)
        spin_row(b, "search gate, px (0 = off)", self.var_gate, 0, 4000)
        hint(b, "The gate keeps the tracker from jumping onto another object of "
                "a similar colour: only blobs within that distance of the "
                "previous position are accepted.")

        # ---------------- footer, full width ----------------
        foot = ttk.Frame(s)
        foot.grid(row=1, column=0, columnspan=ncols, sticky="ew", pady=(8, 4))
        ttk.Button(foot, text="Save settings to config.ini",
                   command=self.save_settings).pack(side="left")
        ttk.Label(foot, text=f"  dp_gui.py version {__version__}",
                  font=("", 9, "bold")).pack(side="left", padx=(8, 0))
        ttk.Label(foot, text="  " + CONFIG_PATH,
                  foreground="#666").pack(side="left")
        if self.cfg.get("version") and self.cfg["version"] != __version__:
            ttk.Label(foot, text=f"  (config.ini written by "
                                 f"{self.cfg['version']})",
                      foreground="#a60").pack(side="left")
        ttk.Label(s, foreground="#666", justify="left",
                  text="Angles: theta = 0 with the rod hanging down, positive to "
                       "the right, range +/-180 deg. "
                       "Rod 1 = origin -> M1, rod 2 = M1 -> M2. Settings are "
                       "saved automatically before every run and on exit."
                  ).grid(row=2, column=0, columnspan=ncols, sticky="w")

    # ================================================== config =========
    def collect_config(self):
        return {
            "version": __version__,
            "fps": self.var_fps.get(),
            "h_margin": str(self.var_h.get()),
            "s_margin": str(self.var_s.get()),
            "v_margin": str(self.var_v.get()),
            "min_area": str(self.var_min.get()),
            "gate": str(self.var_gate.get()),
            "len_tol": str(self.var_lentol.get()),
            "zero1_deg": self.var_z1.get(),
            "zero2_deg": self.var_z2.get(),
            "origin_fit": "1" if self.var_ofit.get() else "0",
            "equal_rods": "1" if self.var_eqrod.get() else "0",
            "max_gap": str(self.var_gap.get()),
            "use_timestamps": "1" if self.var_ts.get() else "0",
            "joint_fit": "1" if self.var_joint.get() else "0",
            "sg_window_ms": str(self.var_sg.get()),
            "pend_L1": self.var_pL1.get(),
            "pend_L2": self.var_pL2.get(),
            "pend_m1": self.var_pm1.get(),
            "pend_m2": self.var_pm2.get(),
            "pend_c1": self.var_pc1.get(),
            "pend_c2": self.var_pc2.get(),
            "pend_I1": self.var_pI1.get(),
            "pend_I2": self.var_pI2.get(),
            "pend_g": self.var_pg.get(),
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
        self.vert_line = None
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
        self.play_fps, _n = playback_fps(path)
        self.update_fps_check()
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

        if self.vert_line is not None and self.viewing == "source":
            a = ((self.vert_line[0][0] - x0) * eff, (self.vert_line[0][1] - y0) * eff)
            bb = ((self.vert_line[1][0] - x0) * eff, (self.vert_line[1][1] - y0) * eff)
            cv2.line(img, (int(a[0]), int(a[1])), (int(bb[0]), int(bb[1])),
                     (136, 255, 0), 2)
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
        if not delta:
            return
        if e.widget is self.canvas:
            self.zoom_by(1.25 if delta > 0 else 1 / 1.25, e.x, e.y)
            return
        # anywhere on the Settings tab the wheel scrolls that pane
        w = e.widget
        for _ in range(12):
            if w is self.set_canvas:
                self.set_canvas.yview_scroll(-delta, "units")
                return
            w = getattr(w, "master", None)
            if w is None:
                return

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
            "m2": "Drag a box over mass M2.",
            "vert": "Drag along the vertical line drawn on the disc, from top "
                    "to bottom. Its tilt is the angular zero and is subtracted "
                    "from both angles."}[mode])

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
        if self.mode == "vert":
            self.canvas.create_line(x0, y0, e.x, e.y, fill="#00ff88",
                                    width=2, tags="roi")
        else:
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="yellow",
                                         width=2, tags="roi")

    def on_release(self, e):
        if self.drag is None:
            return
        x0, y0, x1, y1 = self.drag
        self.drag = None
        self.canvas.delete("roi")
        if self.mode == "vert":
            self.set_vertical_from_drag(x0, y0, x1, y1)
            return
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

        chk = self.update_fps_check() or ""
        if chk.startswith("WARNING") and not messagebox.askyesno(
                "Frame rate looks wrong", chk + "\n\nRun anyway?"):
            return

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
            "gate": self.var_gate.get(),
            "len_tol": self.var_lentol.get(),
            "len_ref": ref,
            "origin_fit": self.var_ofit.get(),
            "zero_deg": self.angle_zero(),
            "equal_rods": self.var_eqrod.get(),
            "max_gap": self.var_gap.get(),
            "use_timestamps": self.var_ts.get(),
            "play_fps": self.play_fps,
            "joint_fit": self.var_joint.get(),
            "sg_ms": self.var_sg.get(),
            "pendulum": self.pendulum_params(),
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

    def pendulum_params(self):
        """Pendulum geometry and masses in SI, or None if not filled in."""
        try:
            p = {"L1": float(self.var_pL1.get()), "L2": float(self.var_pL2.get()),
                 "m1": float(self.var_pm1.get()) / 1000.0,
                 "m2": float(self.var_pm2.get()) / 1000.0,
                 "c1": float(self.var_pc1.get() or 0) or None,
                 "c2": float(self.var_pc2.get() or 0) or None,
                 "I1": float(self.var_pI1.get() or 0),
                 "I2": float(self.var_pI2.get() or 0),
                 "g": float(self.var_pg.get())}
        except ValueError:
            return None
        if min(p["L1"], p["L2"], p["m1"], p["m2"], p["g"]) <= 0:
            return None
        return p

    def update_fps_check(self):
        """Compare the typed capture rate with the file's playback rate."""
        if not self.video_path:
            return
        txt = check_fps(self.get_fps(), self.play_fps)
        self.lbl_fpschk.config(text=txt,
                               foreground="#a00" if txt.startswith("WARNING")
                               else "#555")
        return txt

    def angle_zero(self):
        """Angular zero offsets (deg) for theta1 and theta2."""
        try:
            return (float(self.var_z1.get() or 0), float(self.var_z2.get() or 0))
        except ValueError:
            return (0.0, 0.0)

    def set_vertical_from_drag(self, cx0, cy0, cx1, cy1):
        """Angular zero from a line drawn on the rig (a plumb line on the disc).

        The line IS the physical vertical, so its tilt in the image is exactly
        the zero error of both angles - camera roll plus any tilt of the whole
        rig. Unlike the rest-frame method it works on any frame, moving or not.
        """
        p0 = np.array(self.to_orig(cx0, cy0))
        p1 = np.array(self.to_orig(cx1, cy1))
        v = p1 - p0
        if v[1] < 0:                       # always take it pointing downwards
            v = -v
            p0, p1 = p1, p0
        length = float(np.hypot(*v))
        self.mode = None
        if length < 20:
            self.set_status("The line is too short to define a direction - draw "
                            "it along as much of the marked vertical as you can.")
            return
        roll = float(np.degrees(np.arctan2(v[0], v[1])))
        self.vert_line = (p0, p1)
        self.var_z1.set(f"{roll:.2f}")
        self.var_z2.set(f"{roll:.2f}")
        self.lbl_zero.config(text=f"angle zero: {roll:+.2f} deg from the drawn "
                                  f"vertical ({length:.0f} px long)",
                             foreground="#070")
        self.render()
        self.set_status(f"Vertical set: the drawn line is tilted {roll:+.2f} deg "
                        f"in the image, and that is subtracted from both angles. "
                        f"Accuracy is about {np.degrees(1.0 / length):.2f} deg "
                        f"per pixel of clicking error.")

    def set_angle_zero(self):
        """Measure the angular zero on the current frame (rods at rest)."""
        if self.frame is None or self.hsv1 is None or self.hsv2 is None:
            messagebox.showinfo("Not ready",
                                "Open a video, set the origin and both marker "
                                "colours first, then park the slider on a frame "
                                "where the pendulum hangs still.")
            return
        if self.origin is None:
            messagebox.showinfo("No origin", "Set the origin first.")
            return
        hm, sm, vm = self.var_h.get(), self.var_s.get(), self.var_v.get()
        f1 = build_filter(self.hsv1, hm, sm, vm)
        f2 = build_filter(self.hsv2, hm, sm, vm)
        cx1, cy1, ok1, _b = detect_color(self.frame, f1, self.var_min.get())
        cx2, cy2, ok2, _b = detect_color(self.frame, f2, self.var_min.get())
        if not (ok1 and ok2):
            messagebox.showwarning("Not detected",
                                   "Both masses must be visible on this frame.")
            return
        z1, z2 = frame_angles_deg(self.origin, (cx1, cy1), (cx2, cy2))
        self.var_z1.set(f"{z1:.2f}")
        self.var_z2.set(f"{z2:.2f}")
        self.lbl_zero.config(text=f"angle zero: {z1:+.2f} / {z2:+.2f} deg "
                                  f"(frame {self.cur_idx})", foreground="#070")
        self.set_status(f"Angle zero taken from frame {self.cur_idx}: "
                        f"{z1:+.2f} / {z2:+.2f} deg. Make sure the pendulum was "
                        f"really at rest there.")

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
            if n1 < 0:                      # second pass: the control video
                self.set_status(f"Writing the control video, frame {k}/{total}")
            else:
                self.set_status(f"Processed {k}/{total};  "
                                f"detected M1: {n1}, M2: {n2}")
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
            head = ""
            if result["aborted"]:
                head = "RUN ABORTED (partial data!)  "
            elif result["truncated"]:
                head = (f"DECODING STOPPED EARLY: {n} of "
                        f"{result['expected_frames']} frames!  ")
            msg = (head +
                   f"Done: {n} frames; M1 detected in {result['found1']} "
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
            msg += (f"  Rod length check (+/-{result['len_tol']:.0f}% of the "
                    f"median L1={r1:.1f}, L2={r2:.1f} px): out of tolerance in "
                    f"{o1} / {o2} frames ({100*o1/n:.1f}% / {100*o2/n:.1f}%).")
            if max(o1, o2) > 0.1 * n:
                msg += ("  WARNING: more than 10% of the frames are mis-detected "
                        "- adjust the filter margins or the tolerance.")
            if result["n_nan"]:
                msg += (f"  {result['n_nan']} frames left empty (gaps longer than "
                        f"{self.var_gap.get()} frames were not interpolated).")
            msg += ("  " + result["origin_note"] + ".  " +
                    result["marker_note"] + ".  " + result["zero_note"] + ".  " +
                    result["time_note"] + "." + result["drop_note"] +
                    "  " + result["fit_note"] + ".  " + result["energy_note"] +
                    result.get("window_note", "") +
                    ("  " + result["time_warning"] if result["time_warning"]
                     else ""))
            if result.get("time_warning"):
                messagebox.showwarning("Check the capture FPS",
                                       result["time_warning"])
            if result["aborted"] or result["truncated"]:
                messagebox.showwarning(
                    "Incomplete run",
                    "The run did not cover the whole video:\n\n" + head.strip() +
                    "\n\nThe data are usable but partial - the saved file name "
                    "gets a _PARTIAL suffix.")
            if result["unwrapped"] and abs(result["turns2"]) > 0.5:
                msg += (f"  Note: rod 2 winds {result['turns2']:+.1f} turns; with "
                        "unwrapping on, theta2 leaves the +/-180 deg range.")
            self.set_status(msg)
            self.btn_save.config(state="normal")
            self.btn_plot.config(state="normal")
            self.btn_energy.config(state="normal" if result.get("has_energy")
                                   else "disabled")
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
        if self.result.get("aborted") or self.result.get("truncated"):
            base += "_PARTIAL"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=base + "_angles.csv",
            initialdir=(self.var_csvdir.get()
                        or os.path.dirname(self.video_path)),
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        # the units are part of the column names: theta1_deg / theta1_rad;
        # the provenance block goes on top as '#' comment lines
        try:
            head = ["# " + ln if ln else "#" for ln in self.meta_lines()]
        except Exception as e:
            head = ["# meta block failed: %s" % e]
            print("Could not build the meta block:", e, file=sys.stderr)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(head) + "\n")
            self.result["df"].to_csv(fh, index=False, lineterminator="\n")
        if not self.var_csvdir.get():
            self.var_csvdir.set(os.path.dirname(path))
        self.save_settings(quiet=True)
        self.set_status(f"Saved: {path}  (header carries the run metadata; "
                        f"read it with pandas.read_csv(..., comment='#'))")
        self.lbl_saved.config(text=f"Saved:\n{path}")

    def meta_lines(self):
        """Provenance block written as '#' comments on top of the data file.

        A data file whose provenance is not recorded cannot be re-checked
        months later, so every export carries its version and settings.
        """
        import datetime
        r = self.result
        lines = [
            f"dp_gui.py version : {__version__}",
            f"written           : {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"source video      : {self.video_path}",
            "",
            f"frames processed  : {r['n_frames']} (start frame {self.start_frame})",
            f"detected M1 / M2  : {r['found1']} / {r['found2']}",
            f"length out of tol : {r['n_out'][0]} / {r['n_out'][1]}",
            f"left empty (NaN)  : {r['n_nan']}",
            f"rod refs L1 / L2  : {r['len_ref'][0]:.2f} / {r['len_ref'][1]:.2f} px"
            f"  (median over the run, tolerance {r['len_tol']:.1f}%)",
            f"  on the marking frame: {r['len_ref_frame'][0]:.2f} / "
            f"{r['len_ref_frame'][1]:.2f} px (reference only)",
            f"max angle step    : {r['max_jump_deg'][0]:.2f} / "
            f"{r['max_jump_deg'][1]:.2f} deg per frame",
            f"origin clicked    : {self.origin[0]:.2f}, {self.origin[1]:.2f} px",
            f"origin correction : {r['origin_shift'][0]:+.2f}, "
            f"{r['origin_shift'][1]:+.2f} px",
            f"{r['origin_note']}",
            f"{r['marker_note']}",
            f"{r['zero_note']}",
            f"{r['time_note']}",
            f"{r['fit_note']}",
            f"{r['energy_note']}",
        ]
        if r.get("window_note"):
            lines.append(r["window_note"].strip())
        if r.get("time_warning"):
            lines.append(r["time_warning"])
        if r["drop_note"]:
            lines.append(r["drop_note"].strip())
        if r["aborted"]:
            lines.append("RUN ABORTED - the data cover only part of the video")
        if r["truncated"]:
            lines.append(f"DECODING STOPPED EARLY - {r['n_frames']} of "
                         f"{r['expected_frames']} frames")
        lines += ["", "settings:"]
        lines += [f"  {k:14s} = {v}" for k, v in sorted(self.collect_config().items())]
        lines += [
            "",
            "columns: Time [s]; X1,Y1,X2,Y2 [px] from the origin, image axes "
            "(y down); theta1,theta2 from the downward vertical, + to the right;",
            "         omega1,omega2 angular velocities from the same local "
            "polynomial fit; E_J total energy, zero = both rods hanging at rest;",
            "         L1,L2 rod lengths [px]; L1_out,L2_out = 1 where the length "
            "left the tolerance band (mis-detected frame).",
            "",
            "read with:  pandas.read_csv(path, comment='#')",
            "            numpy.genfromtxt(path, delimiter=',', names=True, "
            "skip_header={n})",
            "            numpy.loadtxt(path, delimiter=',', skiprows={n} + 1)",
        ]
        # the very first line states how many '#' lines there are, so the numpy
        # readers (which need skip_header, not a comment character) can be told
        n = len(lines) + 1
        lines = [f"dp_gui {__version__} data file - {n} header lines start "
                 f"with '#', then the column names"] + \
                [ln.replace("{n}", str(n)) for ln in lines]
        return lines

    def _make_figure(self):
        """Build the angle figure.

        pyplot is deliberately not used: it keeps global state, and switching
        its backend to Agg for saving used to make every later plt.show() a
        silent no-op (that is why the plot window opened only once).
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        df = self.result["df"]
        u = self.result["units"]
        c1, c2 = self.result["cols"]
        # constrained layout instead of tight_layout(): the latter needs a
        # renderer, and on a bare Figure it can raise - which is what silently
        # killed the plot window
        fig = Figure(figsize=(8, 5), dpi=100, layout="constrained")
        FigureCanvasAgg(fig)          # a real canvas from the start
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        for a, col, lab in ((ax1, c1, f"$\\theta_1$ [{u}]"),
                            (ax2, c2, f"$\\theta_2$ [{u}]")):
            a.plot(df.Time, df[col], color="tab:blue", lw=1)
            a.set_ylabel(lab)
            a.grid(alpha=.4)
        ax2.set_xlabel("Time [s]")
        mark = ""
        if self.result.get("aborted"):
            mark = "  [RUN ABORTED - PARTIAL DATA]"
        elif self.result.get("truncated"):
            mark = "  [DECODING STOPPED EARLY - PARTIAL DATA]"
        fig.suptitle(f"{os.path.basename(self.video_path)}{mark}   "
                     f"(dp_gui {__version__})\n"
                     f"IC: $\\theta_1$={df[c1].iloc[0]:.2f}, "
                     f"$\\theta_2$={df[c2].iloc[0]:.2f} [{u}]")
        return fig

    def _energy_figure(self):
        """E(t) plus its scatter about a slow trend -- the quality indicator."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        df = self.result["df"]
        t = df.Time.values
        e = df["E_J"].values
        fin = np.isfinite(e)
        # the slow trend is the damping; what scatters around it is error
        win = max(101, (int(len(e) / 8) // 2) * 2 + 1)
        trend = smooth_segments(t, e, win, 2, 0) if fin.sum() > win else e
        resid = e - trend
        fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        FigureCanvasAgg(fig)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax1.plot(t, e, lw=0.7, color="tab:blue", label="E(t)")
        ax1.plot(t, trend, lw=2, color="k", label="slow trend (damping)")
        ax1.set_ylabel("E [J]   (0 = hanging at rest)")
        ax1.grid(alpha=.35)
        ax1.legend(loc="upper right", fontsize=8)
        ax2.plot(t, resid, lw=0.6, color="tab:red")
        ax2.set_ylabel("E - trend [J]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(alpha=.35)
        rms = float(np.nanstd(resid[fin])) if fin.any() else float("nan")
        span = float(np.nanmax(e[fin]) - np.nanmin(e[fin])) if fin.any() else 1.0
        drift = ((np.nanmean(trend[fin][-20:]) - np.nanmean(trend[fin][:20]))
                 / (t[-1] - t[0]) if fin.sum() > 40 else float("nan"))
        fig.suptitle(f"{os.path.basename(self.video_path)}  -  energy check\n"
                     f"scatter about the trend {rms:.4f} J "
                     f"({100 * rms / span:.1f}% of the {span:.2f} J range), "
                     f"damping {drift:+.4f} J/s")
        return fig

    def show_energy(self):
        if not self.result or not self.result.get("has_energy"):
            messagebox.showinfo("No energy",
                                "Fill in the pendulum parameters on the Settings "
                                "tab and run the tracking again.")
            return
        try:
            self._show_plot(self._energy_figure(), "Energy - ")
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            messagebox.showerror("Could not open the plot",
                                 tb.strip().split("\n")[-1])

    def show_plot(self):
        """Open the plot in its own window; works any number of times."""
        if not self.result:
            return
        try:
            self._show_plot()
        except Exception:
            # a failure inside a Tk callback only reaches the console, and a
            # GUI started from the file manager has none -- so report it here
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            messagebox.showerror("Could not open the plot",
                                 tb.strip().split("\n")[-1] +
                                 "\n\n(matplotlib is required for the plot; "
                                 "the CSV itself is unaffected)")

    def _show_plot(self, figure=None, title="Angles - "):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.plot_win is not None:
            try:
                if self.plot_win.winfo_exists():
                    self.plot_win.destroy()
            except tk.TclError:
                pass
            self.plot_win = None
        win = tk.Toplevel(self)
        win.title(title + os.path.basename(self.video_path))
        canvas = FigureCanvasTkAgg(figure if figure is not None
                                   else self._make_figure(), master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        try:                        # the toolbar is a nicety, not a necessity
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            NavigationToolbar2Tk(canvas, win).update()
        except Exception as e:
            print("toolbar unavailable:", e, file=sys.stderr)
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


def block(parent, title):
    """A titled group of settings."""
    f = ttk.LabelFrame(parent, text=" " + title + " ", padding=6)
    f.pack(fill="x", pady=(0, 8))
    return f


def hint(parent, text):
    ttk.Label(parent, text=text, foreground="#666", justify="left",
              wraplength=330).pack(anchor="w", pady=(2, 0))


def spin_row(parent, label, var, lo, hi, increment=1):
    row = ttk.Frame(parent)
    row.pack(fill="x")
    ttk.Label(row, text=label).pack(side="left")
    ttk.Spinbox(row, from_=lo, to=hi, increment=increment, textvariable=var,
                width=8).pack(side="right")
    return row


def scrollable(parent):
    """Frame inside a canvas that scrolls only when the content does not fit."""
    canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = ttk.Frame(canvas)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _resize(_e=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(win, width=canvas.winfo_width())

    inner.bind("<Configure>", _resize)
    canvas.bind("<Configure>", _resize)
    return canvas, inner


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
