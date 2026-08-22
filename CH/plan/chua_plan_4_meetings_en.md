# Chua Oscillator: Plan for 4 Meetings (6 Hours Each)

Meeting 1 is the introduction to the topic. It is planned separately and is not described here. Below are meetings 2–4 for measurements and meeting 5 for the talk.

Setup: L = 18 mH, C2 = 100 nF, C1 = 10 nF, R = R0 (1.13 kΩ) + potentiometer (0–1 kΩ, 10 turns), shunt Rs = 216 Ω. Test points: 1 = v\_C1, 2 = v\_C2, 3–4 = across the shunt. K1 disconnects the resistors. K2 switches between oscillator mode and V–I measurement mode.

**A two-channel scope is enough.** The bifurcation diagram needs only one signal (v\_C1). The phase portrait needs two. The V–I curve needs two. 

\---

## Meeting 2. V–I Curve and First Survey (6 h)

Goal: **finish the day with a fitted V–I curve and a map of the regimes.** Without this, the next meeting is blind.

**0:00–0:45 — Build and warm up.** Build the circuit, switch on the power, let it warm up. While it warms, check all component values with a multimeter. Write down the real C1, C2, L, R0.

**0:45–1:00 — Set up the scope.** Set both channels to **DC coupling** (this is critical, see the end of this plan). Use ×10 probes and check their compensation. Set up the CSV export and test it on one trial record.

**1:00–2:15 — Measure the V–I curve.** Open K1. Put K2 in measurement mode. Feed a triangle wave, ±10 V, 20–100 Hz. Channel A = voltage across the diode. Channel B = voltage across the shunt, so the current is I = U\_Rs / 216 Ω. Check two things: the forward and backward sweeps must lie on top of each other (if they open into a loop, lower the frequency), and the sweep must reach both saturation parts beyond ±6 V. Export the full record.

**2:15–3:00 — Fit the curve on the spot.** Fit five straight segments: Ga (middle), Gb (outer), Gc (saturation), and the breakpoints E1 and E2. Check that the left and right branches are symmetric. A difference above 3% means the op-amp supply is unbalanced. Fix it now, not later.

**3:00–3:30 — Lunch.**

**3:30–4:00 — Find the working range.** Three equilibrium points exist when `1/|Ga| < R < 1/|Gb|`. Use your own slopes. Earlier data gave a range from about 300 Ω on the potentiometer up to the end of its travel.

**4:00–5:30 — Survey run.** Go from 1000 Ω down to 300 Ω in steps of 50 Ω. That is 15 records, each 200 ms long at 1 MSa/s. Measure the resistance with a multimeter every time (open K1, measure, close it again). That number is the file name. Write in the log: resistance, room temperature, regime by eye, file name.

**5:30–6:00 — Wrap up.** Mark on an axis where the oscillations start, where the single scroll appears, and where the double scroll appears. Decide where to use a fine step tomorrow.

**What you should find** (potentiometer values, from the model with the earlier V–I curve; real values may shift by 20–40 Ω):

|range|regime|
|-|-|
|above \~860|equilibrium, no oscillation|
|815–860|limit cycle around P₊|
|785–815|period-doubling cascade|
|765–785|single scroll|
|300–765|double scroll|
|below \~300|P± disappear, large symmetric orbit|

\---

## Meeting 3. Main Series (6 h)

Goal: **collect all the data for the diagram.**

**0:00–0:30 — Warm up and reference point.** Record the same setting you used yesterday and compare. The difference tells you how much the circuit drifts. You need to know this number.

**0:30–2:30 — Main run.** Go from the upper edge of the oscillation range down to 300 Ω in steps of 20 Ω. About 30 records. Plan for about 4 minutes per point, including the resistance measurement.

**2:30–3:00 — Lunch.**

**3:00–4:30 — Fine run through the period-doubling cascade.** Inside the window you found yesterday (roughly 765–815 Ω), use a step of **2 Ω**. About 25 records. A larger step will skip period-4 and period-8 completely.

**4:30–5:15 — Reverse run.** Go through the same range from the bottom up, step 20 Ω. If the boundaries move, the circuit has coexisting attractors. That is a real result, not a mistake.

**5:15–6:00 — Rough processing on the spot.** Build the diagram from what you already have, even roughly. The point is to see the gaps and know where to go back tomorrow. Do not leave all processing for meeting 4. By then there will be no time left to fill gaps.

**Total for the day:** about 80 records.

\---

## Meeting 4. Fill Gaps and Process (6 h)

Goal: **a finished diagram and all the pictures for the talk.**

**0:00–1:30 — Fill the gaps** you found yesterday. Also repeat 2–3 points where you were not sure about the regime.

**1:30–2:30 — Reference records for the talk.** Take long, clean records (500 ms each) of four regimes: limit cycle, period-2, single scroll, double scroll. These are the four pictures that will go on your slide.

**2:30–3:00 — Lunch.**

**3:00–4:30 — Bifurcation diagram.** For each record: drop the first 10%, smooth v\_C1 lightly (Savitzky–Golay, window about 1/20 of a period), find the local maxima with a prominence threshold of 2% of the range, and plot them as one vertical column at that value of R. Point the R axis so that resistance decreases to the right. Then the diagram reads left to right as chaos developing.

How to read it: one point per column = period-1; two = period-2; four = period-4. A solid band = chaos. **Two** separate bands, one positive and one negative = double scroll. A single band shifted away from zero = single scroll. White gaps inside the chaos = periodic windows.

**4:30–5:15 — Return maps and phase portraits.** Plot M\_n against M\_{n+1} for one chaotic record. A one-dimensional single-humped curve is direct experimental evidence of a horseshoe, that is, of the homoclinic tangle. Add the four phase portraits from the reference records.

**5:15–6:00 — Consistency checks.**

1. The range where three equilibria exist, computed from the V–I curve, matches the range where you actually see scrolls.
2. Inside the double-scroll range, the size of the attractor grows as R grows.
3. The change from single to double scroll happens when the minimum of v\_C1 in the single-scroll records reaches the breakpoint E1. This is the direct test of the geometric criterion. It is the reason DC coupling matters.
4. The ratios of the doubling intervals should move toward the Feigenbaum constant, 4.669. With two or three doublings you will only get the order of magnitude.

**If time is left:** compute the Lyapunov exponent (Rosenstein or Wolf method) for two or three points. It gives a number instead of a picture.

\---

## Meeting 5. The Talk (6 h)

**0:00–1:00 — Story line.** Agree on the narrative. One arc that works: Poincaré looks for stability of the Solar System and finds a homoclinic tangle → a hundred years later, spacecraft routes are planned along that same tangle → and you can see it on an oscilloscope in a single lab session.

**1:00–3:00 — Result slides.** Order: circuit diagram → V–I curve with the fit and the load lines → four phase portraits → bifurcation diagram → return map.

**3:00–3:30 — Lunch.**

**3:30–4:30 — Theory part.** Saddle-focus, the two scrolls, and how the double scroll is linked to a homoclinic orbit.

**4:30–5:30 — Full rehearsal with a timer.**

**5:30–6:00 — Fixes.**

\---

## Three Things That Can Ruin the Experiment

1. **AC coupling.** In the single-scroll regimes the attractor sits away from zero. AC coupling removes exactly the information that separates a single scroll from a symmetric orbit. Check the coupling at the start of every meeting.
2. **Reading the dial instead of measuring.** Ten turns plus backlash give an error of tens of ohms, and the whole doubling cascade is only 30–40 Ω wide.
3. **Leaving processing until the last day.** A rough diagram must exist by the end of meeting 3. Otherwise you will find the gaps when there is no time to fill them.

