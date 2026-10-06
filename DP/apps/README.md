# Double pendulum: tracking, simulation, flip map

A small set of tools for the double-pendulum lab. Each one does a single job
and they share one description of the rig:

| file | what it does |
|---|---|
| `dp_gui.py` | tracks a video, exports angles, velocities and energy to CSV |
| `dp_sim.py` | integrates the model and compares runs |
| `dp_flipmap.py` | draws the map of the time to the first flip over a rectangle of starting angles |
| `config.ini` | the bench measurements `dp_flipmap.py` reads |

---

## 1. Environment

* Python 3.8 or newer.
* `numpy` — required by everything.
* `matplotlib` — needed for every picture.
* `tkinter` — only for the two windowed programs, `dp_sim.py` and `dp_gui.py`.
  It ships with the official Python installers on Windows and macOS; on Linux
  it is a separate package (`sudo apt install python3-tk`).
* `opencv-python` and `pandas` — only for `dp_gui.py`, which reads video.

```
pip install numpy matplotlib pandas opencv-python
```

`dp_flipmap.py` needs nothing but `numpy` and `matplotlib`. It imports
`dp_sim.py` when that file sits in the same folder, so that the map and the
simulator can never drift apart; if the import fails (file missing, or no
`tkinter` on a headless machine) it falls back to its own identical copy of
the equations and says so in the first line of its output.

Put all the files in one folder:

```
dp_gui.py
dp_sim.py
dp_flipmap.py
config.ini          <- created automatically on the first run if absent
```

---

## 2. config.ini

Every number is measured with **one rod at a time**, hanging from its own
axis. Lengths in metres, masses in kilograms, times in seconds.

```ini
# Double pendulum bench measurements, one rod at a time.
# Lengths in metres, masses in kilograms, times in seconds.
# half1 = half2 = 0 means no friction.
[pendulum]
m1 = 0.433      ; mass of the upper rod, kg
m2 = 0.479      ; mass of the lower rod, kg
l1 = 0.212      ; axis of rod 1 -> its centre of mass, m
l2 = 0.219      ; axis of rod 2 -> its centre of mass, m
l3 = 0.255      ; axis of rod 1 -> axis of rod 2, m
T1 = 1.000      ; period of small swings of rod 1 hanging alone, s
T2 = 1.000      ; the same for rod 2, s
half1 = 30      ; time in which that free decay halves in amplitude, s
half2 = 30      ; the same for rod 2, s
g = 9.8         ; m/s2
```

Everything else follows from those ten numbers:

```
J   = m g l (T / 2 pi)^2        moment of inertia about the axis
I   = J - m l^2                 about the centre of mass
tau = half / (2 ln 2)           amplitude decays as exp(-t / 2 tau)
c   = J / tau                   viscous coefficient of that bearing
```

`tau` is **not** the half-life: the amplitude halves after `1.386 tau`, the
energy after `0.693 tau`. Putting `half1 = half2 = 0` gives a frictionless
model — that is the only switch for friction, there is no command-line flag.

Note the units, which differ between the programs because each asks for what
is natural where it is used:

| | lengths | masses | inertia |
|---|---|---|---|
| `config.ini`, `dp_sim.ini` | metres | kilograms | derived from `T` |
| `dp_gui.py` settings tab | millimetres | grams | g·m², **about the axis** |

---

## 3. dp_flipmap.py

### The algorithm

1. Read `config.ini` and turn the bench measurements into the constants of the
   equations of motion (`I`, `tau`, then `a`, `b`, `d`, `e1`, `e2`, `c1`,
   `c2` — section 4 has the formulas).
2. Build a rectangular grid of starting angles. Every node is one pendulum,
   released **from rest**: `theta1 = theta1_0`, `theta2 = theta2_0`,
   both rates zero.
3. Integrate all of them at once with one vectorised RK4 of fixed step. After
   every step, check which pendulums have just had a rod pass over the top
   (`|theta| > pi`) and record the current time for those — each node keeps
   the **first** such time and is not written again.
4. Nodes that never flip inside the horizon keep `inf`.
5. Draw. The grey level is the flip time, black for an immediate flip, white
   for no flip at all. The shade follows the square root of the time, because
   most of a map flips within a second or two and a plain linear scale would
   leave everything black; the order is still monotone, so darker always means
   sooner.
6. Draw the energy threshold on top of it in red: a flip of rod 2 needs

   ```
   E = e1 (1 - cos theta1_0) + e2 (1 - cos theta2_0)  >  2 e2
   ```

   which is a curve in the plane of starting angles. Inside it no flip is
   possible whatever the dynamics does, so the curve must lie on the edge of
   the central white region. That it does is a free check that the whole
   computation is right.

### Parameters

| option | default | meaning |
|---|---|---|
| `--th1 MIN MAX` | `-180 180` | range of `theta1_0`, deg |
| `--th2 MIN MAX` | `-180 180` | range of `theta2_0`, deg |
| `--n N` | `200` | grid points per axis |
| `--n1 N`, `--n2 N` | — | points along each axis separately; override `--n` |
| `--horizon S` | `15` | how long each run is watched, s; also the white end of the grey scale |
| `--dt S` | `0.003` | fixed RK4 step, s |
| `--flip-of {2,1,0}` | `2` | which rod must go over the top: the lower one, the upper one, or whichever is first |
| `--mark TH1 TH2` | — | put a circle at this point |
| `--colour NAME` | `magma` | `magma`, `ice`, `cubehelix` or `mono` |
| `--out PATH` | built from the parameters | output picture |

### Colours and saving

All four schemes run **dark for an immediate flip to pale for a late one**,
and leave **pure white** for "never flipped", so the white region reads as a
category of its own instead of as the end of the scale. The shade follows the
square root of the time, because most of a map flips within a second or two
and a linear scale would leave everything dark; the order stays monotone.

* `magma` (default) — black, purple, orange, cream. Perceptually uniform, the
  most structure per pixel, and its pale end sits next to the white of the
  no-flip region without a jump.
* `ice` — deep blue through green to cream; calmer, good for printing.
* `cubehelix` — brightness rises steadily, so the picture survives being
  photocopied in black and white.
* `mono` — plain grey.

The energy-threshold curve and the `--mark` circle are drawn in a contrasting
colour chosen per scheme, so they stay visible on all of them.

The picture is **never written over an existing file**: if the name is taken,
`_1`, `_2` and so on are appended. If the file cannot be written at all —
typically because it is open in a viewer, which on Windows gives
`[Errno 22] Invalid argument` — the next free name is tried and then the
system temporary folder, so a long run is never lost. The default name
records the run:

```
flipmap_th1_90to180_th2_-60to60_320x220_h15_dt3ms_fric_magma.png
```

### Examples

```bash
# the classic full map
python dp_flipmap.py

# the experiment's own corner: rod 2 hanging, rod 1 lifted
python dp_flipmap.py --th1 90 180 --th2 -60 60 --n 320

# zoom on a suspected island, with a marker and a longer horizon
python dp_flipmap.py --th1 110 140 --th2 -10 10 --n1 600 --n2 400 \
                     --horizon 30 --mark 125 0

# a strip along theta2_0 = 0, which is the line the experiment works on
python dp_flipmap.py --th1 80 180 --th2 -3 3 --n1 800 --n2 40 --horizon 30
```

### What it prints

The parameters it ended up with, the two small-oscillation periods (check
these against the assembled pendulum before trusting anything else), the
energy threshold angle on the line `theta2_0 = 0`, progress every five
seconds, and the fraction of the rectangle that flipped with the earliest and
the latest flip time.

### Cost

Runtime is `n1 * n2 * horizon / dt` right-hand side evaluations, vectorised
over the whole grid. On an ordinary laptop a 200×200 grid with a 15 s horizon
and `dt = 3 ms` takes a couple of minutes; 600×400 with a 30 s horizon is
closer to an hour. Start coarse, then zoom. A smaller `--dt` changes the
answer only near the fractal boundary, where no finite grid is converged
anyway.

---

## 4. How dp_sim.py works

A window with two tabs that integrates the same equations the map uses.

**Settings tab** — the bench measurements, one rod at a time: mass, distance
from the axis to the centre of mass, the period `T` of small swings and the
time in which the amplitude halves. Everything else is derived and displayed
as you type: `I1`, `I2`, `tau1`, `tau2`, `c1`, `c2`, the five constants below
and the two small-oscillation modes, each shown as period, frequency and
`omega` together. The values live in `dp_sim.ini` beside the script.

**Model tab** — a schematic pendulum you drag with the mouse: grab the middle
joint to set `theta1`, the lower end to set `theta2`. Both angles are also
typed in, in degrees to three decimals, and the drawing follows. You set the
duration, the RK4 step and the output step, give the run a name and press RUN.

**The model.** Compound rods, not point masses:

```
a  = I1 + m1 l1^2 + m2 l3^2         b  = m2 l3 l2        d  = I2 + m2 l2^2
e1 = (m1 l1 + m2 l3) g              e2 = m2 l2 g

| a               b cos(th1-th2) | | th1'' |   | -b w2^2 sin(th1-th2) - e1 sin th1 + Q1 |
| b cos(th1-th2)  d              | | th2'' | = | +b w1^2 sin(th1-th2) - e2 sin th2 + Q2 |
```

Friction sits where it physically is, in the two bearings, with the Rayleigh
function `F = c1 w1^2 / 2 + c2 (w2 - w1)^2 / 2`, so `Q = -C [w1, w2]` with
`C = [[c1+c2, -c2], [-c2, c2]]`. The second bearing resists the **relative**
rate, which is why a free decay of the slow in-phase mode lasts much longer
than one of the fast mode.

The energy, with zero at both rods hanging at rest:

```
E = a w1^2 / 2 + d w2^2 / 2 + b w1 w2 cos(th1 - th2)
  + e1 (1 - cos th1) + e2 (1 - cos th2)
```

**Integration** is classical RK4 with a fixed step — fixed on purpose, so that
two runs started from slightly different states see exactly the same
arithmetic and the only difference between them is the one you introduced.
With friction off, the energy drift over the run is reported after every run:
it is the honest accuracy figure. At `dt = 1 ms` it is of order `10^-7` J.

**Runs list.** Every run is kept, can be renamed or deleted, and any selection
is drawn on one plot: `theta1`, `theta2`, `omega1`, `omega2`, kinetic,
potential or total energy. The time axis has its own zoom: buttons, a numeric
from/to window, mouse wheel over the plot, drag to pan, double click to fit.
RUN TWIN repeats the selected run with a small extra angle, so the two traces
can be overlaid and the moment they part company read off directly.

**CSV export** writes every selected run into one file with a `#` header
holding the parameters:

```
name, t, theta1_deg, theta2_deg, omega1_deg_s, omega2_deg_s,
KE_J, PE_J, E_J, xc2_m, yc2_m
```

`xc2`, `yc2` are the centre of mass of the lower rod, so the same scripts can
be pointed at both simulated and tracked data.
