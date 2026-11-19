"""Time-to-first-flip map of the rigid-rod double pendulum with viscous friction c1 = c2 = C.
Start at rest from (theta1, theta2); flip = |theta1| or |theta2| exceeds pi.
usage: python flipmap.py   (all settings below)
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Patch

# ---- settings ----
C = 4e-4          # viscous friction, both bearings, N*m*s
N = 600           # grid size N x N
DT = 0.001        # time step, s
TMAX = 30.0       # max simulated time, s
OUT = "flip_map"  # output name: OUT.npz and OUT.png

m1, m2 = 0.433, 0.479
l1, l2, l3 = 0.212, 0.219, 0.255
t1, t2 = 1.000, 1.000
g = 9.8
I1 = m1 * g * l1 * (t1 / 2 / np.pi) ** 2
I2 = m2 * g * l2 * (t2 / 2 / np.pi) ** 2
a, b, d = I1 + m2 * l3 ** 2, m2 * l3 * l2, I2
K1, K2 = m1 * g * l1 + m2 * g * l3, m2 * g * l2
c1 = c2 = C


def acc(th1, th2, w1, w2):
    wr = w2 - w1
    Q1 = -c1 * w1 + c2 * wr
    Q2 = -c2 * wr
    D = th1 - th2
    cD, sD = np.cos(D), np.sin(D)
    r1 = Q1 - b * sD * w2 * w2 - K1 * np.sin(th1)
    r2 = Q2 + b * sD * w1 * w1 - K2 * np.sin(th2)
    det = a * d - (b * cD) ** 2
    return (d * r1 - b * cD * r2) / det, (a * r2 - b * cD * r1) / det


# grid: cell centres, symmetric under (th1,th2) -> (-th1,-th2)
x = -np.pi + (np.arange(N) + 0.5) * 2 * np.pi / N
T1, T2 = np.meshgrid(x, x, indexing='xy')  # rows: theta2, cols: theta1
ttff = np.full((N, N), np.inf)

# energy filter: with friction energy only decreases; to flip rod 2 need V >= V(0, pi), rod 1 needs more
V = K1 * (1 - np.cos(T1)) + K2 * (1 - np.cos(T2))
Vflip = min(K2 * 2, K1 * 2)
half = np.zeros((N, N), bool)
half[: N // 2, :] = True                    # compute lower half of rows, mirror the rest
todo = half & (V >= Vflip)
idx = np.flatnonzero(todo)
th1, th2 = T1.ravel()[idx].copy(), T2.ravel()[idx].copy()
w1, w2 = np.zeros_like(th1), np.zeros_like(th1)
res = np.full(idx.size, np.inf)
alive = np.arange(idx.size)
print(f'{idx.size} trajectories to integrate (of {N*N})', flush=True)

t = 0.0
tic = time.time()
nsteps = int(round(TMAX / DT))
for k in range(nsteps):
    a1, a2 = acc(th1, th2, w1, w2)
    k1 = (w1, w2, a1, a2)
    s = [th1 + 0.5 * DT * k1[0], th2 + 0.5 * DT * k1[1], w1 + 0.5 * DT * k1[2], w2 + 0.5 * DT * k1[3]]
    a1, a2 = acc(*s)
    k2 = (s[2], s[3], a1, a2)
    s = [th1 + 0.5 * DT * k2[0], th2 + 0.5 * DT * k2[1], w1 + 0.5 * DT * k2[2], w2 + 0.5 * DT * k2[3]]
    a1, a2 = acc(*s)
    k3 = (s[2], s[3], a1, a2)
    s = [th1 + DT * k3[0], th2 + DT * k3[1], w1 + DT * k3[2], w2 + DT * k3[3]]
    a1, a2 = acc(*s)
    k4 = (s[2], s[3], a1, a2)
    th1 = th1 + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    th2 = th2 + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    w1 = w1 + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    w2 = w2 + DT / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    t += DT
    flipped = (np.abs(th1) > np.pi) | (np.abs(th2) > np.pi)
    if flipped.any():
        res[alive[flipped]] = t
        keep = ~flipped
        th1, th2, w1, w2, alive = th1[keep], th2[keep], w1[keep], w2[keep], alive[keep]
    if k % 2000 == 0:
        # drop trajectories whose energy can no longer reach a flip
        E = 0.5 * a * w1 ** 2 + 0.5 * d * w2 ** 2 + b * np.cos(th1 - th2) * w1 * w2 \
            + K1 * (1 - np.cos(th1)) + K2 * (1 - np.cos(th2))
        keep = E >= Vflip
        th1, th2, w1, w2, alive = th1[keep], th2[keep], w1[keep], w2[keep], alive[keep]
        print(f't={t:6.2f} s  alive={alive.size}  {time.time()-tic:.0f}s', flush=True)
    if alive.size == 0:
        break

flat = ttff.ravel()
flat[idx] = res
ttff = flat.reshape(N, N)
ttff[N // 2:, :] = ttff[: N // 2, :][::-1, ::-1]    # point symmetry
np.savez(OUT + ".npz", ttff=ttff, x=x, C=C, DT=DT, TMAX=TMAX)
print('done', time.time() - tic)

# ---- plot ----
xd = np.degrees(x)
ext = [xd[0], xd[-1], xd[0], xd[-1]]
cant = V < Vflip
nof = ~np.isfinite(ttff) & ~cant
fig, ax = plt.subplots(figsize=(7.5, 6.4))
im = ax.imshow(np.where(np.isfinite(ttff), ttff, np.nan), origin="lower", extent=ext,
               norm=LogNorm(vmin=0.1, vmax=TMAX), cmap="viridis", interpolation="nearest")
ax.imshow(np.where(cant, 1, np.nan), origin="lower", extent=ext, cmap=ListedColormap(["white"]), interpolation="nearest")
ax.imshow(np.where(nof, 1, np.nan), origin="lower", extent=ext, cmap=ListedColormap(["0.35"]), interpolation="nearest")
ax.contour(np.degrees(T1), np.degrees(T2), V, levels=[Vflip], colors="k", linewidths=.6)
fig.colorbar(im, ax=ax).set_label("time to first flip, s")
ax.set(xlabel="θ1(0), deg", ylabel="θ2(0), deg",
       title=f"Time to first flip, start at rest\nviscous friction c1 = c2 = {C:.1e} N·m·s")
ax.set_xticks(range(-180, 181, 60))
ax.set_yticks(range(-180, 181, 60))
ax.legend(handles=[Patch(color="white", ec="k", label="no flip possible (energy)"),
                   Patch(color="0.35", label=f"no flip within {TMAX:g} s")], loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=150)
plt.show()
