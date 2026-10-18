"""
Required egg-to-chick survival phi*, i.e. the phi where r(phi) = r_obs.

Finds phi* for each scenario and four estimates of r_obs (Table 4, Fig 2), over the 378-set parameter grid
(Fig 3), and for different juvenile survival values (Fig 6).
r goes up with phi, so there is at most one phi*. If r < r_obs even at
phi = 1 there is no phi*.

Needs kakapo_agestructured.py in the same folder.
"""

import itertools
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from kakapo_agestructured import (SCENARIOS, COLOURS, OBS_STYLE, MEASURED_COLOUR, INK, MUTED,
                                  FIG_DIR, FULL_WIDTH, SINGLE_WIDTH, growth_rate,
                                  observed_growth_rate, fmt_range, panel_label, save)

# Measured survival from Savage et al. 2022 (Table 3)
HATCH_RATE = 1 - 401 / 662                 # 0.39
EGG_TO_150_DAYS = 72 / 252                  # 0.29
WITH_NESTING = 72 / 252 * 49 / 57           # 0.25
MEASURED_RANGE = (WITH_NESTING, HATCH_RATE)

# Grid (section 2.5)
MATURITY = range(6, 12)
CLUTCH = np.arange(1.0, 4.01, 0.5)
INTERVAL = [2, 3, 4]
LIFESPAN = [60, 75, 90]
BREEDING_SPAN = 50
MAX_CLUTCH = 4                              # biggest clutch recorded

JUVENILE_SURVIVAL = [0.99, 0.95, 0.91]

# Fig 3 colours: 6 yr = ceiling, 9 yr = central, 11 yr = floor
MATURITY_COLOURS = {6: COLOURS["Ceiling"], 9: COLOURS["Central"], 11: COLOURS["Floor"]}
ALL_SETS_COLOUR, TYPICAL_COLOUR = "#a9c4ea", "#5b3fc4"


def required_survival(params, r_obs, s_juv=None, n_halvings=50):
    """Bisection for phi* on [0, 1]. None if r(1) < r_obs."""
    if growth_rate(params, phi=1.0, s_juv=s_juv) < r_obs:
        return None
    low, high = 0.0, 1.0
    for _ in range(n_halvings):
        mid = (low + high) / 2
        if growth_rate(params, phi=mid, s_juv=s_juv) < r_obs:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def lowest_juvenile_survival(params, r_obs):
    """Juvenile survival where r(phi=1) = r_obs. Below this there's no phi*."""
    low, high = 0.5, 1 - 1 / params["L"]
    for _ in range(50):
        mid = (low + high) / 2
        if growth_rate(params, phi=1.0, s_juv=mid) < r_obs:
            low = mid
        else:
            high = mid
    return high


def in_measured_range(phi):
    return phi is not None and MEASURED_RANGE[0] <= phi <= MEASURED_RANGE[1]


def is_typical(Tm, c, y):
    # maturity 9-11, clutch 2-3, interval 3-4
    return Tm >= 9 and 2 <= c <= 3 and y >= 3


def run_grid(r_obs):
    rows = []
    for Tm, c, y, L in itertools.product(MATURITY, CLUTCH, INTERVAL, LIFESPAN):
        params = dict(Tm=Tm, c=c, y=y, f=0.5, L=L, Ta=BREEDING_SPAN)
        rows.append(dict(Tm=Tm, c=c, y=y, L=L, typical=is_typical(Tm, c, y),
                         phi_star=required_survival(params, r_obs)))
    return rows


def summarise(values, label):
    values = np.array(values)
    inside = (values >= MEASURED_RANGE[0]) & (values <= MEASURED_RANGE[1])
    print(f"  {label}: {len(values)} sets, phi* {fmt_range(values.min(), values.max(), 2)}, "
          f"median {np.median(values):.2f}, {inside.sum()} inside measured range "
          f"({100 * inside.mean():.0f}%)")


# ---------------- figures ----------------
def marker(ax, x, y, colour, size=6.5):
    ax.plot(x, y, "o", color=colour, ms=size, mec="white", mew=1.2, zorder=5)


def figure2(r_obs):
    """Fig 2: r against phi for each scenario."""
    phis = np.linspace(0.005, 1, 400)
    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH, 3.5))

    ax.axvspan(*MEASURED_RANGE, color=MEASURED_COLOUR, lw=0, zorder=0)
    ax.text(np.mean(MEASURED_RANGE), 0.232, "Measured\nsurvival", ha="center", va="top",
            fontsize=8, linespacing=1.15)
    ax.axhline(0, color="#8c8c8c", lw=0.8, zorder=1)
    ax.axhline(r_obs, **OBS_STYLE, zorder=3)
    ax.text(0.99, r_obs + 0.004, f"Observed r = {r_obs:.3f} yr$^{{-1}}$", ha="right",
            va="bottom", fontsize=8)

    for name in ("Ceiling", "Central", "Floor"):
        params = SCENARIOS[name]
        rates = np.array([growth_rate(params, phi=p) for p in phis])
        ax.plot(phis, rates, color=COLOURS[name], lw=1.8, zorder=2)
        ax.annotate(name, (1, rates[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, annotation_clip=False)
        phi_star = required_survival(params, r_obs)
        if phi_star is not None:
            marker(ax, phi_star, r_obs, COLOURS[name])
            # put the label where it doesn't overlap the curve
            if name == "Ceiling":
                offset, ha, va = (-7, 5), "right", "bottom"
            else:
                offset, ha, va = (7, -5), "left", "top"
            ax.annotate(f"φ* = {phi_star:.2f}", (phi_star, r_obs), xytext=offset,
                        textcoords="offset points", ha=ha, va=va, fontsize=8)

    ax.text(0.99, -0.033, "Floor stays below the observed\nrate even at φ = 1",
            ha="right", va="center", fontsize=7.5, color=MUTED, linespacing=1.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 0.24)
    ax.set_xlabel("Egg-to-chick survival φ")
    ax.set_ylabel("Growth rate r (yr$^{-1}$)")
    save(fig, "figure2_growth_vs_survival.png")


def figure3(rows, r_obs):
    """Fig 3: phi* against clutch size (a) and histogram over the grid (b)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0),
                                   gridspec_kw={"wspace": 0.32})

    # (a) y = 3, L = 75
    clutches = np.linspace(1, MAX_CLUTCH, 61)
    ax1.axhspan(*MEASURED_RANGE, color=MEASURED_COLOUR, lw=0, zorder=0)
    ax1.text(1.04, np.mean(MEASURED_RANGE), "Measured\nsurvival", ha="left", va="center",
             fontsize=7.5, linespacing=1.15)
    for Tm in (6, 9, 11):
        phis = [required_survival(dict(Tm=Tm, c=c, y=3, f=0.5, L=75, Ta=BREEDING_SPAN), r_obs)
                for c in clutches]
        ax1.plot(clutches, [np.nan if p is None else p for p in phis],
                 color=MATURITY_COLOURS[Tm], lw=1.8, label=f"First breeding at {Tm} yr")
    central = required_survival(SCENARIOS["Central"], r_obs)
    marker(ax1, 2.5, central, INK)
    ax1.plot([], [], "o", color=INK, ms=6.5, mec="white", mew=1.2,
             label=f"Central scenario (φ* = {central:.2f})")
    ax1.set_xlim(1, MAX_CLUTCH)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax1.set_xlabel("Eggs per breeding attempt, c")
    ax1.set_ylabel("Required egg-to-chick survival φ*")
    ax1.legend(loc="upper right", title="Interval 3 yr, lifespan 75 yr",
               title_fontsize=7.5, handlelength=1.6, borderaxespad=0.2)
    panel_label(ax1, "a")

    # (b)
    all_phi = np.array([r["phi_star"] for r in rows if r["phi_star"] is not None])
    typical_phi = np.array([r["phi_star"] for r in rows
                            if r["typical"] and r["phi_star"] is not None])
    bins = np.linspace(0, 1, 26)
    ax2.axvspan(*MEASURED_RANGE, color=MEASURED_COLOUR, lw=0, zorder=0)
    counts, *_ = ax2.hist(all_phi, bins=bins, color=ALL_SETS_COLOUR, edgecolor="white",
                          lw=0.6, zorder=2)
    ax2.hist(typical_phi, bins=bins, color=TYPICAL_COLOUR, edgecolor="white", lw=0.6, zorder=3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, counts.max() * 1.15)
    ax2.grid(axis="x", visible=False)
    ax2.set_xlabel("Required egg-to-chick survival φ*")
    ax2.set_ylabel("Number of parameter sets")
    handles = [Patch(color=ALL_SETS_COLOUR, label=f"All sets (n = {len(all_phi)})"),
               Patch(color=TYPICAL_COLOUR, label=f"Typical biology\n(n = {len(typical_phi)})"),
               Patch(color=MEASURED_COLOUR, label="Measured survival")]
    ax2.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0, handlelength=1.2)
    panel_label(ax2, "b")

    save(fig, "figure3_required_survival.png")


def figure6(r_obs):
    """Fig 6: central phi* against juvenile survival."""
    central = SCENARIOS["Central"]
    s_adult = 1 - 1 / central["L"]
    s_min = lowest_juvenile_survival(central, r_obs)
    s_values = np.linspace(s_min, 0.995, 200)
    phis = np.array([required_survival(central, r_obs, s_juv=s) for s in s_values], float)

    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH, 3.3))
    ax.axvspan(0.85, s_min, color="#efefef", lw=0, zorder=0)
    ax.text((0.85 + s_min) / 2, 0.5, "No φ* ≤ 1:\ngrowth too\nslow even\nat φ = 1",
            ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.2)
    ax.axhspan(*MEASURED_RANGE, color=MEASURED_COLOUR, lw=0, zorder=0)
    ax.text(s_min + 0.003, np.mean(MEASURED_RANGE), "Measured survival", ha="left",
            va="center", fontsize=7.5)
    ax.plot(s_values, phis, color=COLOURS["Central"], lw=1.8, zorder=2)

    for s in (0.91, 0.95):
        p = required_survival(central, r_obs, s_juv=s)
        marker(ax, s, p, COLOURS["Central"])
        ax.annotate(f"{s:.2f}: φ* = {p:.2f}", (s, p), xytext=(7, 4),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8)
    p_adult = required_survival(central, r_obs)
    marker(ax, s_adult, p_adult, INK)
    ax.annotate(f"Same as adults ({s_adult:.3f}):\nφ* = {p_adult:.2f}",
                (s_adult, p_adult), xytext=(s_adult - 0.004, 0.12), textcoords="data",
                ha="right", va="center", fontsize=8, linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))

    ax.set_xlim(0.85, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Juvenile survival (yr$^{{-1}}$), ages 0 to {central['Tm'] - 1} yr")
    ax.set_ylabel("Required egg-to-chick survival φ*")
    save(fig, "figure6_juvenile_survival.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # so φ prints on Windows
    r_obs, _, two_point, ci_99 = observed_growth_rate()
    estimates = [round(ci_99[0], 3), r_obs, two_point, round(ci_99[1], 3)]

    print(f"Measured range (Table 3): {fmt_range(*MEASURED_RANGE)} "
          f"(egg -> 150-day chick {EGG_TO_150_DAYS:.3f})\n")

    print("Table 4 - required egg-to-chick survival")
    print(f"  {'Scenario':9s} {'r(phi=1)':>9s} {'Doubling':>9s} "
          + " ".join(f"phi*({e:.3f})" for e in estimates) + "  In range?")
    for name, params in SCENARIOS.items():
        r1 = growth_rate(params)
        stars = [required_survival(params, e) for e in estimates]
        cells = " ".join(f"{'none':>11s}" if s is None else f"{s:11.2f}" for s in stars)
        main = stars[1]                        # the least-squares estimate
        if main is None:
            verdict = "-"
        elif in_measured_range(main):
            verdict = "Yes"
        else:
            verdict = "No (below)" if main < MEASURED_RANGE[0] else "No (above)"
        print(f"  {name:9s} {r1:9.3f} {np.log(2) / r1:9.1f} {cells}  {verdict}")

    central = SCENARIOS["Central"]
    r_low = growth_rate(central, phi=MEASURED_RANGE[0])
    r_high = growth_rate(central, phi=MEASURED_RANGE[1])
    print(f"\n  Central growth at the measured survival: {fmt_range(r_low, r_high)} per yr")

    print("\nSection 3.3 - parameter grid")
    rows = run_grid(r_obs)
    reachable = [r["phi_star"] for r in rows if r["phi_star"] is not None]
    typical = [r["phi_star"] for r in rows if r["typical"] and r["phi_star"] is not None]
    print(f"  {len(rows)} parameter sets; {len(rows) - len(reachable)} cannot reach r_obs at phi = 1")
    summarise(reachable, "All reachable sets")
    summarise(typical, "Typical biology   ")
    for Tm in (6, 9, 11):
        p = required_survival(dict(Tm=Tm, c=MAX_CLUTCH, y=3, f=0.5, L=75, Ta=BREEDING_SPAN), r_obs)
        print(f"  Clutch of {MAX_CLUTCH}, first breeding at {Tm:2d} yr (y = 3, L = 75): phi* = {p:.2f}")

    print("\nSection 3.5 - central scenario with a separate juvenile survival")
    print(f"  Same as adults ({1 - 1 / central['L']:.3f}): phi* = {required_survival(central, r_obs):.2f}")
    for s in JUVENILE_SURVIVAL:
        p = required_survival(central, r_obs, s_juv=s)
        print(f"  Juvenile survival {s:.2f}: phi* = {'none' if p is None else f'{p:.2f}'}")
    print(f"  Below juvenile survival {lowest_juvenile_survival(central, r_obs):.3f} "
          f"no phi* <= 1 exists")
    low, high = lowest_juvenile_survival(central, r_obs), 1 - 1 / central["L"]
    for _ in range(50):                        # juvenile survival where phi* = top of range
        mid = (low + high) / 2
        if required_survival(central, r_obs, s_juv=mid) > MEASURED_RANGE[1]:
            low = mid
        else:
            high = mid
    print(f"  phi* stays inside the measured range for juvenile survival >= {high:.3f}")

    figure2(r_obs)
    figure3(rows, r_obs)
    figure6(r_obs)
    print(f"\nSaved figure2_growth_vs_survival.png, figure3_required_survival.png and "
          f"figure6_juvenile_survival.png to {FIG_DIR}")