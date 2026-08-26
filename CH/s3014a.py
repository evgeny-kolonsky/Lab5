"""
Oscilloscope viewer for the Agilent/Keysight InfiniiVision DSO-X 3014A.

Standalone version: needs only pyvisa, numpy, matplotlib and tkinter.
No oscilloscope_api.py, no logo file, no log server required.

    pip install pyvisa pyvisa-py numpy matplotlib

pyvisa-py is a pure-python VISA backend and works over USB on Windows if
the instrument driver is installed. If you already have the Keysight IO
Libraries or NI-VISA, the script will use those automatically.
"""

import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import sys
import socket
import time
import configparser
import logging
import logging.handlers

import pyvisa

VERSION = "2.1-standalone"

# ---------------------------------------------------------------------------
# Settings (all overridable from config.ini, section [settings])
# ---------------------------------------------------------------------------
DEFAULT_SAVE_PATH = os.path.join(os.path.expanduser("~"), "Documents")
RAW_POINTS = 1000000            # points to request per channel (1 Mpt on the 3014A)
MAX_DISPLAY_POINTS = 20000      # decimate to this many before plotting (Y-t view)
XY_MAX_POINTS = 150000          # max points drawn in the XY (phase portrait) view
TIMEBASE_CHOICES = [            # s/div offered in the toolbar; 10 divisions on screen
    ("100 us", 100e-6), ("200 us", 200e-6), ("500 us", 500e-6),
    ("1 ms", 1e-3), ("2 ms", 2e-3), ("5 ms", 5e-3),
    ("10 ms", 10e-3), ("20 ms", 20e-3), ("50 ms", 50e-3),
]
MIN_PERIODS_WARN = 50           # warn if a record holds fewer cycles than this
CHANNELS = (1, 2, 3, 4)         # all four inputs; only displayed ones are read
VISA_TIMEOUT_MS = 60000         # a 1 Mpt transfer is slow
LOG_SERVER_IP = "132.68.74.143" # lab log server; ignored if unreachable
USE_LOG_SERVER = False          # set True inside the lab network

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
hostname = socket.gethostname()


class HostnameFilter(logging.Filter):
    def filter(self, record):
        record.hostname = hostname
        return True


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addFilter(HostnameFilter())

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s %(levelname)s [%(hostname)s] %(message)s'))
logger.addHandler(console_handler)

if USE_LOG_SERVER:                                          # optional, never fatal
    try:
        sh = logging.handlers.SocketHandler(
            LOG_SERVER_IP, logging.handlers.DEFAULT_TCP_LOGGING_PORT)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
    except Exception as e:
        logger.warning("Log server unavailable: %s", e)


# ---------------------------------------------------------------------------
# Instrument layer  (replaces oscilloscope_api.py)
# ---------------------------------------------------------------------------
def find_scope():
    """Search VISA resources for an InfiniiVision scope.

    Returns (address, idn) or (None, None).
    """
    try:
        rm = pyvisa.ResourceManager()
    except Exception as e:
        logger.error("No VISA backend: %s", e)
        try:
            rm = pyvisa.ResourceManager('@py')              # fall back to pyvisa-py
        except Exception as e2:
            logger.error("pyvisa-py also unavailable: %s", e2)
            return None, None

    for addr in rm.list_resources():
        if not any(k in addr.upper() for k in ("USB", "TCPIP", "GPIB")):
            continue
        try:
            inst = rm.open_resource(addr)
            inst.timeout = 3000
            idn = inst.query("*IDN?").strip()
            inst.close()
        except Exception:
            continue
        logger.info("Found instrument at %s: %s", addr, idn)
        if any(k in idn.upper() for k in
               ("INFINIIVISION", "DSO-X", "MSO-X", "AGILENT", "KEYSIGHT")):
            return addr, idn
    logger.warning("No InfiniiVision oscilloscope found")
    return None, None


class Oscilloscope:
    """Minimal InfiniiVision driver: screen settings + deep-memory capture."""

    def __init__(self, address, raw_points=RAW_POINTS):
        try:
            rm = pyvisa.ResourceManager()
        except Exception:
            rm = pyvisa.ResourceManager('@py')
        self.inst = rm.open_resource(address)
        self.inst.timeout = VISA_TIMEOUT_MS
        try:
            self.inst.chunk_size = 1024 * 1024          # faster big binary reads
        except Exception:
            pass
        self.raw_points = raw_points
        self.idn = self.inst.query("*IDN?").strip()

    def close(self):
        try:
            self.inst.close()
        except Exception:
            pass

    def active_channels(self, channels=CHANNELS):
        out = []
        for ch in channels:
            try:
                if int(float(self.inst.query(f":CHANnel{ch}:DISPlay?"))) == 1:
                    out.append(ch)
            except Exception:
                pass
        return out

    def input_impedances(self, channels=CHANNELS):
        """Return {channel: impedance string} for displayed channels."""
        out = {}
        for ch in self.active_channels(channels):
            try:
                out[ch] = self.inst.query(f":CHANnel{ch}:IMPedance?").strip().upper()
            except Exception:
                pass
        return out

    def get_screen_settings(self, channels=CHANNELS):
        chs = self.active_channels(channels)
        tb = (float(self.inst.query(":TIMebase:RANGe?")),
              float(self.inst.query(":TIMebase:POSition?")))
        offsets, scales, names = [], [], []
        for ch in chs:
            offsets.append(float(self.inst.query(f":CHANnel{ch}:OFFSet?")))
            scales.append(float(self.inst.query(f":CHANnel{ch}:SCALe?")))
            names.append(f"CH{ch}")
        return {'timebase': tb, 'channels': names,
                'offsets': offsets, 'scales': scales}

    def set_timebase(self, s_per_div):
        """Set the horizontal scale in seconds per division."""
        self.inst.write(f":TIMebase:SCALe {s_per_div:.9g}")

    def measure_frequency(self, channels=CHANNELS):
        """Ask the scope's built-in counter for the signal frequency.

        Used only to tell the user how many cycles a record actually holds.
        Returns None if the scope cannot measure it.
        """
        for ch in self.active_channels(channels):
            try:
                f = float(self.inst.query(f":MEASure:FREQuency? CHANnel{ch}"))
                if 0 < f < 1e12:                    # 9.9e37 means "no signal"
                    return f
            except Exception:
                continue
        return None

    def _set_transfer_timeout(self, npoints):
        """Give the waveform transfer its own, much longer timeout.

        The acquisition timeout is scaled to the timebase and can be as short
        as 15 s. That is fine for ':DIGitize', but a deep-memory ':WAVeform:DATA?'
        moves 2 bytes per point over USB and easily needs a minute. Reusing the
        acquisition timeout here is what made Deep memory fail while the
        1000-point mode worked.
        """
        secs = 30.0 + (npoints * 2) / 50000.0        # pessimistic 50 kB/s link
        secs = min(600.0, secs)
        self.inst.timeout = int(secs * 1000)
        return secs

    def _acquire(self, active):
        """Take one acquisition without ever blocking forever.

        ':DIGitize' waits for a trigger event. With the trigger sweep set to
        Normal and no qualifying edge, it never completes and '*OPC?' blocks
        until the VISA timeout, leaving the session in a broken state. So:

          1. force the trigger sweep to AUTO, which guarantees the scope
             acquires even with no trigger edge;
          2. use a timeout scaled to the timebase, not a fixed 60 s;
          3. on timeout, clear the session, force a trigger and retry once;
          4. if that still fails, fall back to ':STOP', which simply freezes
             whatever is currently in acquisition memory. That path needs no
             trigger at all and always works.
        """
        v = self.inst
        prev_sweep = None
        try:
            prev_sweep = v.query(":TRIGger:SWEep?").strip()
            if "AUTO" not in prev_sweep.upper():
                logger.info("Trigger sweep was %s, switching to AUTO", prev_sweep)
                v.write(":TRIGger:SWEep AUTO")
        except Exception as e:
            logger.debug("Could not read/set trigger sweep: %s", e)

        # ':DIGitize' only works in a time-base mode. In XY the scope has no
        # time axis at all, and in ROLL it never reports the acquisition as
        # complete, so '*OPC?' blocks until the timeout. Switch to MAIN for
        # the duration of the capture and put the instrument back afterwards.
        self._prev_tb_mode = None
        try:
            mode = v.query(":TIMebase:MODE?").strip().upper()
            if not mode.startswith("MAIN"):
                logger.info("Scope timebase mode is %s — switching to MAIN to acquire", mode)
                self._prev_tb_mode = mode
                v.write(":TIMebase:MODE MAIN")
                v.query("*OPC?")
        except Exception as e:
            logger.debug("Could not read/set timebase mode: %s", e)

        try:
            tb_range = float(v.query(":TIMebase:RANGe?"))
        except Exception:
            tb_range = 0.01

        # one acquisition takes about tb_range; allow generous margin plus
        # transfer time for up to a few Mpts
        timeout_s = max(15.0, 4.0 * tb_range + 15.0)
        v.timeout = int(timeout_s * 1000)
        logger.info("Acquiring: timebase %.6g s, timeout %.1f s", tb_range, timeout_s)

        for attempt in (1, 2):
            try:
                v.write(":DIGitize " + ",".join(f"CHANnel{c}" for c in active))
                v.query("*OPC?")
                self._restore_sweep(prev_sweep)
                return True
            except Exception as e:
                logger.warning("DIGitize attempt %d timed out: %s", attempt, e)
                try:
                    v.clear()                       # device clear: unstick the session
                except Exception:
                    pass
                if attempt == 1:
                    try:
                        v.write(":TRIGger:FORCe")
                    except Exception:
                        pass

        # last resort: freeze whatever is already in memory
        logger.warning("Falling back to :STOP (no fresh trigger)")
        try:
            v.write(":STOP")
            v.query("*OPC?")
        except Exception as e:
            logger.error("Even :STOP failed: %s", e)
            try:
                v.clear()
            except Exception:
                pass
            self._restore_sweep(prev_sweep)
            self._restore_timebase_mode()
            raise RuntimeError(
                "Scope did not respond.\n\n"
                "Check that a signal is present and that the trigger is set to Auto "
                "(and that the channel input is 1 MΩ, not 50 Ω).")
        self._restore_sweep(prev_sweep)
        return False

    def _restore_sweep(self, prev_sweep):
        if prev_sweep and "AUTO" not in prev_sweep.upper():
            try:
                self.inst.write(f":TRIGger:SWEep {prev_sweep}")
            except Exception:
                pass

    def _restore_timebase_mode(self):
        """Put the scope back into XY or ROLL after the data has been read.

        This must happen after ':WAVeform:DATA?', not before: leaving MAIN
        restarts acquisition and can invalidate the memory we are reading.
        """
        mode = getattr(self, "_prev_tb_mode", None)
        if not mode:
            return
        try:
            self.inst.write(f":TIMebase:MODE {mode}")
            logger.info("Timebase mode restored to %s", mode)
        except Exception:
            pass
        self._prev_tb_mode = None

    def get_trace(self, channels=CHANNELS, deep=True):
        """Single acquisition, then read the waveform for each channel.

        With deep=True the full acquisition memory is transferred. The line
        that matters is ':WAVeform:POINts:MODE RAW' — without it the scope
        silently returns about 1000 points of screen resolution, no matter
        how many you ask for.
        """
        v = self.inst
        active = self.active_channels(channels)
        if not active:
            raise RuntimeError("No channels are displayed on the scope")

        fresh = self._acquire(active)

        tb = (float(v.query(":TIMebase:RANGe?")),
              float(v.query(":TIMebase:POSition?")))

        data, offsets, scales, names = [], [], [], []
        t_axis = None
        for ch in active:
            v.write(f":WAVeform:SOURce CHANnel{ch}")
            v.write(":WAVeform:FORMat WORD")
            v.write(":WAVeform:BYTeorder LSBFirst")
            v.write(":WAVeform:UNSigned 0")
            v.write(":WAVeform:POINts:MODE " + ("RAW" if deep else "NORMal"))
            if deep:
                # Ask for no more than the scope actually holds. Requesting a
                # million points when only 500k were acquired makes the reply
                # unpredictable on some firmware.
                try:
                    avail = int(float(v.query(":ACQuire:POINts?")))
                except Exception:
                    avail = int(self.raw_points)
                want = min(int(self.raw_points), max(1000, avail))
                v.write(f":WAVeform:POINts {want}")

            try:
                npts = int(float(v.query(":WAVeform:POINts?")))
            except Exception:
                npts = int(self.raw_points) if deep else 1000

            secs = self._set_transfer_timeout(npts)
            logger.info("CH%d: transferring %d points, timeout %.0f s", ch, npts, secs)

            pre = v.query(":WAVeform:PREamble?").strip().split(',')
            xinc, xorg, xref = float(pre[4]), float(pre[5]), float(pre[6])
            yinc, yorg, yref = float(pre[7]), float(pre[8]), float(pre[9])

            t_start = time.time()
            raw = v.query_binary_values(":WAVeform:DATA?", datatype='h',
                                        container=np.array)
            dt_transfer = max(1e-3, time.time() - t_start)
            logger.info("CH%d: got %d points in %.1f s (%.0f kB/s)",
                        ch, len(raw), dt_transfer, len(raw) * 2 / 1000 / dt_transfer)
            data.append((raw - yref) * yinc + yorg)

            if t_axis is None or len(data[-1]) < len(t_axis):
                t_axis = (np.arange(len(data[-1])) - xref) * xinc + xorg

            offsets.append(float(v.query(f":CHANnel{ch}:OFFSet?")))
            scales.append(float(v.query(f":CHANnel{ch}:SCALe?")))
            names.append(f"CH{ch}")

        self._restore_timebase_mode()
        try:
            v.write(":RUN")                                # leave the scope running
        except Exception:
            pass
        n = min(len(d) for d in data)
        y = np.column_stack([d[:n] for d in data])
        return t_axis[:n], y, names, offsets, scales, tb, fresh


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------
def envelope_decimate(t, y, max_points):
    """Reduce a long trace for display while preserving visible extremes.

    Plain slicing (y[::n]) hides spikes and makes a chaotic attractor look
    thinner than it is. This keeps the min and max of every bucket, so the
    drawn envelope matches what the scope screen shows.
    """
    n = len(t)
    if n <= max_points:
        return t, y
    buckets = max_points // 2
    step = n // buckets
    usable = buckets * step
    tb = t[:usable].reshape(buckets, step)
    yb = y[:usable].reshape(buckets, step, y.shape[1])
    idx_min = yb.argmin(axis=1)
    idx_max = yb.argmax(axis=1)
    out_t = np.empty(buckets * 2)
    out_y = np.empty((buckets * 2, y.shape[1]))
    rows = np.arange(buckets)
    for c in range(y.shape[1]):
        lo = np.minimum(idx_min[:, c], idx_max[:, c])
        hi = np.maximum(idx_min[:, c], idx_max[:, c])
        out_y[0::2, c] = yb[rows, lo, c]
        out_y[1::2, c] = yb[rows, hi, c]
    out_t[0::2] = tb[rows, 0]
    out_t[1::2] = tb[rows, step - 1]
    if usable < n:
        out_t = np.append(out_t, t[-1])
        out_y = np.vstack([out_y, y[-1]])
    return out_t, out_y


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class App:
    def __init__(self, master):
        self.master = master
        master.title(f"צופה באוסצילוסקופ (DSO-X 3014A) v{VERSION}")

        # config.ini is looked for next to the script and next to the exe
        parser = configparser.ConfigParser()
        candidates = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"),
                      os.path.join(os.path.dirname(sys.executable), "config.ini")]
        self.config_path = candidates[0]                  # where settings get written back
        for p in candidates:
            if os.path.exists(p):
                parser.read(p, encoding='utf-8')
                self.config_path = p
                logger.info("Config loaded from: %s", p)
                break
        else:
            logger.info("No config.ini found, using defaults")

        self.save_root = parser.get('settings', 'save_path', fallback=DEFAULT_SAVE_PATH)
        self.deep_memory = parser.getboolean('settings', 'deep_memory', fallback=True)
        self.raw_points = parser.getint('settings', 'raw_points', fallback=RAW_POINTS)
        logger.info("Save path: %s | deep memory: %s | points: %d",
                    self.save_root, self.deep_memory, self.raw_points)

        self.scope = None
        self.scope_idn = ""
        self.time = None
        self.y = None
        self.time_disp = None
        self.y_disp = None
        self.chs = []
        self.offsets = []
        self.scales = []
        self.timebase = None
        self.blink_job = None
        self.is_capturing = False

        self.xy_mode = False            # False = voltage vs time, True = phase portrait

        self.dark_mode = False
        self.themes = {
            'light': {'bg': 'white',   'grid': '#c0c0c0', 'grid_minor': '#e0e0e0',
                      'tick': '#333333', 'ch': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']},
            'dark':  {'bg': '#1a1a1a', 'grid': '#404040', 'grid_minor': '#2a2a2a',
                      'tick': '#b0b0b0', 'ch': ['#FFE040', '#40FF40', '#40D0FF', '#FF7070']},
        }

        self.status_var = tk.StringVar()
        self.status_var.set("...מחפש אוסצילוסקופ")

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.xaxis.set_major_formatter(FuncFormatter(self.format_time_english))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax2 = None
        self.setup_scope_grid(self.ax)
        self.canvas.draw()

        frame = tk.Frame(master)
        frame.pack()

        self.capture_btn = tk.Button(frame, text="מדידה", command=self.capture)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(frame, text="CSVשמירה כ־", command=self.save,
                                  state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.reconnect_btn = tk.Button(frame, text="חיבור מחדש", command=self.reconnect)
        self.reconnect_btn.pack(side=tk.LEFT, padx=5)

        self.deep_var = tk.BooleanVar(value=self.deep_memory)
        tk.Checkbutton(frame, text="Deep memory", variable=self.deep_var,
                       command=self.on_deep_toggle).pack(side=tk.LEFT, padx=5)

        self.xy_btn = tk.Button(frame, text="XY", width=6, command=self.toggle_xy)
        self.xy_btn.pack(side=tk.LEFT, padx=(15, 2))

        self.xsel_var = tk.StringVar(value="CH1")
        self.ysel_var = tk.StringVar(value="CH2")
        tk.Label(frame, text="X:").pack(side=tk.LEFT, padx=(6, 0))
        self.xsel = ttk.Combobox(frame, textvariable=self.xsel_var, width=5,
                                 state="disabled", values=["CH1", "CH2"])
        self.xsel.pack(side=tk.LEFT, padx=1)
        tk.Label(frame, text="Y:").pack(side=tk.LEFT, padx=(4, 0))
        self.ysel = ttk.Combobox(frame, textvariable=self.ysel_var, width=5,
                                 state="disabled", values=["CH1", "CH2"])
        self.ysel.pack(side=tk.LEFT, padx=1)
        for box in (self.xsel, self.ysel):
            box.bind("<<ComboboxSelected>>", lambda e: self.redraw_plot())

        tk.Label(frame, text="s/div:").pack(side=tk.LEFT, padx=(15, 2))
        self.tb_var = tk.StringVar(value="20 ms")
        self.tb_box = ttk.Combobox(frame, textvariable=self.tb_var, width=8,
                                   state="readonly",
                                   values=[n for n, _ in TIMEBASE_CHOICES])
        self.tb_box.pack(side=tk.LEFT, padx=2)
        self.tb_box.bind("<<ComboboxSelected>>", self.on_timebase_change)

        status_frame = tk.Frame(master, bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(status_frame, text=f"v{VERSION}", anchor=tk.W,
                 font=('Arial', 9, 'normal'), fg='gray').pack(side=tk.LEFT, padx=(6, 2), pady=2)

        self.theme_btn = tk.Button(status_frame, text="☀", font=('Arial', 10), width=2,
                                   bd=0, highlightthickness=0, command=self.toggle_theme)
        self.theme_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.status_icon = tk.Canvas(status_frame, width=12, height=12, highlightthickness=0)
        self.status_icon.pack(side=tk.RIGHT, padx=6, pady=2)
        self.status_circle = self.status_icon.create_oval(2, 2, 10, 10, fill="gray")

        tk.Label(status_frame, textvariable=self.status_var, anchor=tk.E,
                 justify='right', font=('Arial', 11, 'normal')
                 ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.master.after(100, self.reconnect)

    # ---------------- settings / checks ----------------
    def on_deep_toggle(self):
        self.deep_memory = self.deep_var.get()
        logger.info("Deep memory set to %s", self.deep_memory)

    def sync_xy_selectors(self):
        """Offer exactly the channels that were actually captured."""
        names = list(self.chs)
        if not names:
            return
        for box, var, default in ((self.xsel, self.xsel_var, 0),
                                  (self.ysel, self.ysel_var, min(1, len(names) - 1))):
            box.config(values=names)
            if var.get() not in names:
                var.set(names[default])

    def sync_timebase_box(self):
        """Make the combobox show what the scope is actually set to.

        Previously the box said '20 ms' while the scope sat at 1 ms/div,
        because the default value is never pushed to the instrument — only a
        user selection triggers that. Showing the truth is safer than silently
        changing the operator's setting on connect.
        """
        if not self.scope or not self.timebase:
            return
        per_div = self.timebase[0] / 10.0
        name = min(TIMEBASE_CHOICES, key=lambda kv: abs(kv[1] - per_div))[0]
        self.tb_var.set(name)
        # A record much shorter than a few hundred cycles cannot show a switch
        # between scrolls, so flag it here rather than after the capture.
        if per_div < 5e-3:
            logger.info("Scope is at %s/div — short records; 20 ms/div "
                        "recommended for the attractor", name)

    def on_timebase_change(self, event=None):
        """Push the chosen s/div to the scope and refresh the empty grid."""
        if not self.scope:
            return
        name = self.tb_var.get()
        value = dict(TIMEBASE_CHOICES).get(name)
        if value is None:
            return
        try:
            self.scope.set_timebase(value)
            logger.info("Timebase set to %s (%.6g s/div)", name, value)
            self.apply_scope_settings()
        except Exception as e:
            logger.warning("Could not set timebase: %s", e)
            messagebox.showwarning("Timebase", str(e))

    def remember_save_path(self, folder):
        """Store the last used folder in config.ini so it survives a restart."""
        self.save_root = folder
        try:
            parser = configparser.ConfigParser()
            if os.path.exists(self.config_path):
                parser.read(self.config_path, encoding='utf-8')
            if not parser.has_section('settings'):
                parser.add_section('settings')
            parser.set('settings', 'save_path', folder)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                parser.write(f)
            logger.info("Save path remembered: %s", folder)
        except Exception as e:
            logger.warning("Could not write config.ini: %s", e)

    def toggle_xy(self):
        """Switch between voltage-vs-time and the XY phase portrait."""
        if not self.xy_mode and len(self.chs) < 2:
            messagebox.showinfo("XY", "XY view needs two displayed channels.")
            return
        self.xy_mode = not self.xy_mode
        self.xy_btn.config(text="Y(t)" if self.xy_mode else "XY",
                           relief=tk.SUNKEN if self.xy_mode else tk.RAISED)
        state = "readonly" if self.xy_mode else "disabled"
        self.xsel.config(state=state)
        self.ysel.config(state=state)
        logger.info("Display mode: %s", "XY" if self.xy_mode else "Y(t)")
        self.redraw_plot()

    def check_input_impedance(self):
        """Warn if an active channel input is set to 50 ohm.

        A 50 ohm input loads a high-impedance circuit (such as a Chua
        oscillator) so heavily that the oscillation stops. The symptom looks
        like a dead channel, which is easy to mistake for a broken probe.
        """
        if not self.scope:
            return
        try:
            imps = self.scope.input_impedances()
        except Exception as e:
            logger.debug("Impedance check skipped: %s", e)
            return
        bad = [ch for ch, s in imps.items() if "FIFT" in s or "50" in s]
        if bad:
            chans = ", ".join(f"CH{c}" for c in bad)
            logger.warning("Channels at 50 ohm input: %s", chans)
            messagebox.showwarning(
                "50 Ω input",
                f"{chans} set to 50 Ω input.\n\n"
                "This loads the circuit and can stop the oscillation.\n"
                "Set the channel impedance to 1 MΩ.")

    # ---------------- plotting ----------------
    def format_time_english(self, x, pos):
        if self.timebase:
            div = self.timebase[0] / 10
        else:
            div = abs(x) if x != 0 else 1

        if div < 1e-6:
            val, unit = x * 1e9, "ns"
        elif div < 1e-3:
            val, unit = x * 1e6, "µs"
        elif div < 1:
            val, unit = x * 1e3, "ms"
        else:
            val, unit = x, "s"

        if val == 0:
            return f"0 {unit}"
        if abs(val) >= 100:
            return f"{val:.0f} {unit}"
        if abs(val) >= 10:
            return f"{val:.1f} {unit}"
        return f"{val:.2f} {unit}"

    def get_theme(self):
        return self.themes['dark'] if self.dark_mode else self.themes['light']

    def setup_scope_grid(self, ax, time_per_div=None, volts_per_div=None):
        t = self.get_theme()
        ax.set_facecolor(t['bg'])
        self.fig.set_facecolor(t['bg'])
        if time_per_div:
            ax.xaxis.set_major_locator(MultipleLocator(time_per_div))
            ax.xaxis.set_minor_locator(MultipleLocator(time_per_div / 5))
        if volts_per_div:
            ax.yaxis.set_major_locator(MultipleLocator(volts_per_div))
            ax.yaxis.set_minor_locator(MultipleLocator(volts_per_div / 5))
        ax.grid(True, which='major', color=t['grid'], linewidth=0.5, alpha=0.7)
        ax.grid(True, which='minor', color=t['grid_minor'], linewidth=0.3, alpha=0.5)
        ax.minorticks_on()
        ax.tick_params(colors=t['tick'], which='both')
        ax.xaxis.label.set_color(t['tick'])
        ax.yaxis.label.set_color(t['tick'])

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme_btn.config(text="🌙" if self.dark_mode else "☀")
        self.redraw_plot()

    def redraw_plot(self):
        """Dispatch to the active view."""
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None
        self.ax.clear()
        if self.xy_mode and len(self.chs) >= 2:
            self.draw_xy()
        else:
            self.draw_yt()
        self.fig.tight_layout()
        self.canvas.draw()

    def draw_xy(self):
        """Phase portrait: one channel against the other.

        Envelope decimation is wrong here — it is a time-domain trick and
        would smear the trajectory. Plain striding is correct instead:
        consecutive samples lie close together on the attractor, so every
        n-th sample still falls on it. Because striding breaks continuity,
        strided data is drawn as dots rather than a connected line.
        """
        t = self.get_theme()
        has_data = self.time is not None and self.y is not None and self.y.shape[1] >= 2

        try:
            ix = self.chs.index(self.xsel_var.get())
            iy = self.chs.index(self.ysel_var.get())
        except ValueError:
            ix, iy = 0, 1
        if ix == iy:                                   # degenerate: would be a line
            iy = (ix + 1) % len(self.chs)
        xname, yname = self.chs[ix], self.chs[iy]

        self.ax.set_xlim(self.offsets[ix] - 4 * self.scales[ix],
                         self.offsets[ix] + 4 * self.scales[ix])
        self.ax.set_ylim(self.offsets[iy] - 4 * self.scales[iy],
                         self.offsets[iy] + 4 * self.scales[iy])
        self.setup_scope_grid(self.ax, time_per_div=self.scales[ix],
                              volts_per_div=self.scales[iy])
        self.ax.set_xlabel(f"{xname} ({self.scales[ix]:.2f} V/div)", color=t['ch'][ix % 4])
        self.ax.set_ylabel(f"{yname} ({self.scales[iy]:.2f} V/div)", color=t['ch'][iy % 4])

        if not has_data:
            return

        n = len(self.time)
        stride = max(1, int(np.ceil(n / XY_MAX_POINTS)))
        x = self.y[::stride, ix]
        yv = self.y[::stride, iy]
        colour = t['ch'][1] if self.dark_mode else 'tab:red'

        # Thin, semi-transparent line: where the trajectory passes many times
        # the colour builds up, so the layered sheets of the attractor stay
        # visible instead of saturating into one solid blob. Alpha adapts so
        # that a simple limit cycle (few points) is still drawn solidly.
        m = len(x)
        alpha = float(np.clip(20000.0 / max(m, 1), 0.15, 0.9))
        lw = 0.3 if m > 20000 else 0.9
        self.ax.plot(x, yv, color=colour, linewidth=lw, alpha=alpha,
                     solid_capstyle='round', rasterized=True, zorder=2)
        logger.info("XY view: %d of %d points (stride %d, alpha %.2f)",
                    m, n, stride, alpha)

    def draw_yt(self):
        """Voltage against time.

        Up to two channels get their own calibrated y axis, left and right.
        With three or four that stops working — you cannot hang four axes on
        one plot legibly — so everything is drawn in units of scope divisions
        instead, exactly as the instrument itself displays it. Each trace keeps
        its own V/div and offset, shown in the legend.
        """
        t = self.get_theme()
        has_data = self.time_disp is not None and self.y_disp is not None
        num_ch = len(self.chs)
        cols = t['ch']

        self.ax.xaxis.set_major_formatter(FuncFormatter(self.format_time_english))

        time_per_div = None
        if self.timebase:
            tb_range, tb_pos = self.timebase
            time_per_div = tb_range / 10
            self.ax.set_xlim(tb_pos - tb_range / 2, tb_pos + tb_range / 2)
        elif has_data:
            self.ax.set_xlim(self.time_disp[0], self.time_disp[-1])

        if num_ch <= 2:
            volts_per_div = self.scales[0] if num_ch >= 1 else None
            self.setup_scope_grid(self.ax, time_per_div=time_per_div,
                                  volts_per_div=volts_per_div)
            if num_ch >= 1:
                self.ax.set_ylabel(f"{self.chs[0]} ({self.scales[0]:.2f} V/div)",
                                   color=cols[0])
                self.ax.set_ylim(self.offsets[0] - 4 * self.scales[0],
                                 self.offsets[0] + 4 * self.scales[0])
            if num_ch >= 2:
                self.ax2 = self.ax.twinx()
                self.ax2.set_ylabel(f"{self.chs[1]} ({self.scales[1]:.2f} V/div)",
                                    color=cols[1])
                self.ax2.set_ylim(self.offsets[1] - 4 * self.scales[1],
                                  self.offsets[1] + 4 * self.scales[1])
                self.ax2.yaxis.set_major_locator(MultipleLocator(self.scales[1]))
                self.ax2.yaxis.set_minor_locator(MultipleLocator(self.scales[1] / 5))
                self.ax2.tick_params(colors=t['tick'])
            if has_data:
                self.ax.plot(self.time_disp, self.y_disp[:, 0], color=cols[0],
                             linewidth=1.0, zorder=2)
                if num_ch >= 2 and self.ax2:
                    self.ax2.plot(self.time_disp, self.y_disp[:, 1], color=cols[1],
                                  linewidth=1.0, zorder=1)
            return

        # three or four channels: common axis in divisions
        self.setup_scope_grid(self.ax, time_per_div=time_per_div, volts_per_div=1.0)
        self.ax.set_ylim(-4, 4)
        self.ax.set_ylabel("делений от нуля канала")
        if has_data:
            for i in range(num_ch):
                d = (self.y_disp[:, i] - self.offsets[i]) / self.scales[i]
                self.ax.plot(self.time_disp, d, color=cols[i % len(cols)],
                             linewidth=1.0, zorder=2 + i,
                             label=f"{self.chs[i]}  {self.scales[i]:.2f} V/div")
            self.ax.legend(loc='upper right', fontsize=8, ncol=num_ch, framealpha=0.6)

    # ---------------- status ----------------
    def start_blink(self):
        if self.is_capturing:
            cur = self.status_icon.itemcget(self.status_circle, 'fill')
            self.status_icon.itemconfig(self.status_circle,
                                        fill="orange" if cur == "gray" else "gray")
            self.blink_job = self.master.after(300, self.start_blink)

    def stop_blink(self):
        if self.blink_job:
            self.master.after_cancel(self.blink_job)
            self.blink_job = None

    def set_connected_status(self, connected, idn_text=""):
        if connected:
            self.scope_idn = idn_text
            self.status_var.set(f"{idn_text} :מחובר")
            self.status_icon.itemconfig(self.status_circle, fill="green")
            logger.info("Connected: %s", idn_text)
        else:
            self.scope_idn = ""
            self.status_var.set("לא מחובר")
            self.save_btn.config(state=tk.DISABLED)
            self.status_icon.itemconfig(self.status_circle, fill="red")
            logger.info("Disconnected")

    # ---------------- connection ----------------
    def reconnect(self):
        logger.info("Searching for oscilloscope...")
        self.status_var.set("...מחפש אוסצילוסקופ")
        self.status_icon.itemconfig(self.status_circle, fill="gray")
        self.master.update_idletasks()

        if self.scope:
            self.scope.close()
            self.scope = None

        addr, idn = find_scope()
        if not addr:
            self.set_connected_status(False)
            return False

        try:
            self.scope = Oscilloscope(addr, raw_points=self.raw_points)
            self.set_connected_status(True, idn_text=self.scope.idn)
            self.apply_scope_settings()
            self.check_input_impedance()
            return True
        except Exception as e:
            logger.error("Connection error: %s", e, exc_info=True)
            self.scope = None
            self.set_connected_status(False)
            return False

    def apply_scope_settings(self):
        if not self.scope:
            return
        try:
            s = self.scope.get_screen_settings()
            self.timebase = s['timebase']
            self.chs = s['channels']
            self.offsets = s['offsets']
            self.scales = s['scales']
            self.time = self.y = self.time_disp = self.y_disp = None
            self.redraw_plot()
            self.sync_timebase_box()
            logger.info("Scope settings: timebase=%s, channels=%s", self.timebase, self.chs)
        except Exception as e:
            logger.warning("Could not read scope settings: %s", e)

    # ---------------- capture ----------------
    def capture(self):
        logger.info("'Capture' pressed")
        if not self.scope:
            if not self.reconnect():
                messagebox.showwarning("אין חיבור", "לא נמצא אוסצילוסקופ")
                return

        self.is_capturing = True
        self.start_blink()
        self.master.config(cursor="watch")
        self.status_var.set("...מקבל נתונים")
        self.master.update_idletasks()

        try:
            self.scope.raw_points = self.raw_points
            (self.time, self.y, self.chs, self.offsets,
             self.scales, self.timebase, fresh) = self.scope.get_trace(deep=self.deep_memory)
            logger.info("Captured %d points x %d channels (fresh trigger: %s)",
                        len(self.time), len(self.chs), fresh)
            if not fresh:
                self._no_trigger = True
            else:
                self._no_trigger = False
        except Exception as e:
            logger.warning("Unable to capture data: %s", e)
            try:
                self.scope.inst.clear()                    # unstick the VISA session
                logger.info("VISA session cleared, connection kept")
            except Exception:
                self.scope = None
                self.set_connected_status(False)
            messagebox.showerror("שגיאה בקריאה", f"{e} :שגיאה")
            self.is_capturing = False
            self.stop_blink()
            self.status_icon.itemconfig(self.status_circle, fill="orange")
            self.master.config(cursor="")
            return

        self.is_capturing = False
        self.stop_blink()
        self.status_icon.itemconfig(self.status_circle, fill="green")

        self.sync_xy_selectors()
        self.time_disp, self.y_disp = envelope_decimate(self.time, self.y,
                                                        MAX_DISPLAY_POINTS)
        if self.xy_mode and len(self.chs) < 2:                 # XY needs both channels
            self.xy_mode = False
            self.xy_btn.config(text="XY", relief=tk.RAISED)
            self.xsel.config(state="disabled")
            self.ysel.config(state="disabled")
        self.redraw_plot()
        self.save_btn.config(state=tk.NORMAL)

        for i in range(len(self.chs)):
            logger.info("CH%d: scale=%.4f V/div, offset=%.4f V, data=(%.4f, %.4f)",
                        i + 1, self.scales[i], self.offsets[i],
                        float(np.min(self.y[:, i])), float(np.max(self.y[:, i])))
        if self.timebase:
            logger.info("Timebase: range=%.6f s (%.6f s/div), pos=%.6f s",
                        self.timebase[0], self.timebase[0] / 10, self.timebase[1])
        logger.info("Data: %d points (%d drawn)", len(self.time), len(self.time_disp))

        # How many cycles did we actually capture? A record of only a few
        # cycles cannot show a switch between scrolls, so the phase portrait
        # looks like one lobe even when the scope screen shows two. That is a
        # timebase problem, not a plotting problem.
        cycles = None
        try:
            f0 = self.scope.measure_frequency()
            if f0:
                span = float(self.time[-1] - self.time[0])
                cycles = span * f0
                logger.info("Signal %.1f Hz, record %.4g s = %.0f cycles", f0, span, cycles)
        except Exception as e:
            logger.debug("Frequency measurement skipped: %s", e)

        if cycles is not None and cycles < MIN_PERIODS_WARN:
            need = MIN_PERIODS_WARN * 10 / f0 / 10          # s/div for MIN_PERIODS_WARN cycles
            nice = min((v for _, v in TIMEBASE_CHOICES if v >= need),
                       default=TIMEBASE_CHOICES[-1][1])
            label = next((n for n, v in TIMEBASE_CHOICES if v == nice), "")
            logger.warning("Only %.0f cycles captured", cycles)
            messagebox.showwarning(
                "Record too short",
                f"This record holds only about {cycles:.0f} cycles of a "
                f"{f0:.0f} Hz signal.\n\n"
                f"A chaotic attractor needs hundreds of cycles before it switches "
                f"between lobes, so the XY view will show one scroll even if the "
                f"scope screen shows two.\n\n"
                f"Set the timebase to {label or '20 ms'}/div or slower and capture again.")

        self.master.config(cursor="")
        warn = "⚠ no trigger, frozen memory | " if getattr(self, "_no_trigger", False) else ""
        self.status_var.set(
            f"{warn}({', '.join(self.chs)}, {len(self.time)} pts) נתונים התקבלו | {self.scope_idn}")
        self.status_icon.itemconfig(self.status_circle,
                                    fill="orange" if warn else "green")

    # ---------------- save ----------------
    def save(self):
        logger.info("Save pressed")
        if self.time is None:
            return

        if not os.path.isdir(self.save_root):
            self.save_root = DEFAULT_SAVE_PATH
            try:
                os.makedirs(self.save_root, exist_ok=True)
            except Exception:
                self.save_root = os.path.expanduser("~")

        base, idx = "trace", 1                               # suggest a free name
        while os.path.exists(os.path.join(self.save_root, f"{base}{idx}.csv")):
            idx += 1

        # A full file dialog: the user picks both the folder and the name, and
        # it opens in whatever folder was used last.
        file_path = filedialog.asksaveasfilename(
            parent=self.master,
            title="Save trace as CSV",
            initialdir=self.save_root,
            initialfile=f"{base}{idx}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            logger.info("Save cancelled")
            return

        self.master.config(cursor="watch")
        self.master.update_idletasks()
        try:
            header = "Time(s)," + ",".join(f"{ch}(V)" for ch in self.chs)
            block = np.column_stack([self.time, self.y])      # full resolution, not decimated
            # np.savetxt is far faster than a python loop; a 1 Mpt file would
            # otherwise take tens of seconds.
            np.savetxt(file_path, block, delimiter=',', header=header,
                       comments='', fmt='%.6e')
            size_mb = os.path.getsize(file_path) / 1e6
            # No modal dialog here: saving happens after almost every capture, and
            # a popup that must be dismissed each time breaks the rhythm of a sweep.
            # The full path goes to the console and to the status bar instead.
            logger.info("SAVED  %s  (%d points, %.1f MB)",
                        file_path, len(self.time), size_mb)
            self.remember_save_path(os.path.dirname(file_path))
            self.status_var.set(
                f"{os.path.basename(file_path)} \u2713 {len(self.time)} pts, "
                f"{size_mb:.1f} MB | {self.scope_idn}")
        except Exception as e:
            logger.error("File save failed: %s", e)
            messagebox.showerror("Save error", str(e))
        finally:
            self.master.config(cursor="")

    def on_close(self):
        logger.info("Closing")
        if self.scope:
            self.scope.close()
        self.master.destroy()
        os._exit(0)


if __name__ == "__main__":
    logger.info("Application started, version %s", VERSION)
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.bind("<Escape>", lambda e: app.on_close())
    root.mainloop()
