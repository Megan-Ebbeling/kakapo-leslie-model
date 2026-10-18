"""
Extra analyses for sections 2.5, 2.6, 3.2, 3.3, 3.5, 3.6 and 4.3.

1. Closed-form required survival (eq. 8, section 3.6). Growth depends on phi,
   c and y only through R = phi*c/y, the surviving chicks per female per year,
   so the Euler-Lotka equation can be solved for R* directly (a geometric
   series). Checked against the bisection results.
2. Required survival at the lower end of the 99% CI of r_obs (sections 3.2, 3.3).
3. Elasticity of lambda to fecundity, juvenile survival and adult survival,
   plus net reproductive rate R0 and generation time (section 3.5).
4. Starting age structure: phi needed to grow from 51 birds in 1995 to 325 in
   2026 if the 20 females in 1995 were all of breeding age (section 3.6).
5. Growth and doubling time of the central scenario for a few values of phi
   (section 4.3).

Makes figure 5. Needs the other two scripts in the same folder.
"""

import itertools
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from kakapo_agestructured import (SCENARIOS, COLOURS, INK, MUTED, FULL_WIDTH, YEARS, COUNTS,
                                  APPROX_YEAR, FEMALES_1995, TOTAL_1995, OBS_STYLE,
                                  MEASURED_COLOUR, growth_rate, leslie_matrix, dominant_eigen,
                                  observed_growth_rate, fmt_range, panel_label, save)
from kakapo_required_survival import (required_survival, MEASURED_RANGE, MATURITY, CLUTCH,
                                      INTERVAL, LIFESPAN, BREEDING_SPAN, is_typical)

# 2019 season (Savage et al. 2022, Table 3)
CHICKS_2019 = 72          # chicks alive at 150 days
MATURE_FEMALES_2019 = 57
NESTING_FEMALES_2019 = 49
EGGS_2019 = 252


# ---------------- 1. closed form ----------------
def required_recruitment(Tm, L, Ta, f, r):
    """R* = phi*c/y at which the growth rate equals r (eq. 8).

    Euler-Lotka: 1 = R f sum_{a=Tm}^{Tm+Ta-1} s^a e^{-r(a+1)}. With q = s e^{-r}
    the sum is a geometric series, e^{-r} q^Tm (1 - q^Ta) / (1 - q).
    """
    s = 1 - 1 / L
    q = s * np.exp(-r)
    return (1 - q) / (f * np.exp(-r) * q**Tm * (1 - q**Ta))


def required_survival_closed(params, r):
    """phi* = R* y / c. Can be above 1, which means no phi* exists."""
    R = required_recruitment(params["Tm"], params["L"], params["Ta"], params["f"], r)
    return R * params["y"] / params["c"]


# ---------------- 3. elasticity ----------------
def elasticities(params, phi):
    """Elasticity of lambda to each entry of A (Caswell 2001, ch. 9).
    Returns lambda, total for fecundity, juvenile survival (ages < Tm),
    adult survival, and the generation time 1/(fecundity total)."""
    A = leslie_matrix(**params, phi=phi)
    lam, w = dominant_eigen(A)
    _, v = dominant_eigen(A.T)                 # reproductive values
    E = (v[:, None] * w[None, :]) * A / (lam * (v @ w))
    Tm = params["Tm"]
    fec = E[0, :].sum()
    sub = np.array([E[a + 1, a] for a in range(A.shape[0] - 1)])
    juv, adult = sub[:Tm].sum(), sub[Tm:].sum()
    return lam, fec, juv, adult, 1 / fec


def net_reproductive_rate(params, phi):
    """R0: daughters a newborn female is expected to produce in her life."""
    s = 1 - 1 / params["L"]
    ages = np.arange(params["Tm"], params["Tm"] + params["Ta"])
    ages = ages[ages < 100]
    return phi * params["c"] / params["y"] * params["f"] * np.sum(s**ages)


# ---------------- 4. starting age structure ----------------
def start_vector(params, phi, adults_only):
    """20 females in 1995, in the stable age distribution, or the same
    distribution with every age below Tm removed (all at breeding age)."""
    _, w = dominant_eigen(leslie_matrix(**params, phi=phi))
    if adults_only:
        w = w.copy()
        w[:params["Tm"]] = 0
    return FEMALES_1995 * w / w.sum()


def project_from(params, phi, adults_only, n_years):
    A = leslie_matrix(**params, phi=phi)
    n = start_vector(params, phi, adults_only)
    totals = [n.sum()]
    for _ in range(n_years):
        n = A @ n
        totals.append(n.sum())
    return np.array(totals) * TOTAL_1995 / FEMALES_1995


def survival_to_reach_2026(params, adults_only):
    """phi at which the projection reaches the 2026 count (bisection)."""
    target, n_years = COUNTS[-1], YEARS[-1] - YEARS[0]
    low, high = 0.0, 1.0
    for _ in range(50):
        mid = (low + high) / 2
        if project_from(params, mid, adults_only, n_years)[-1] < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# ---------------- figure 6 ----------------
def figure5(r_obs, ci_99, phi_stable, phi_adult):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0),
                                   gridspec_kw={"wspace": 0.34})

    # (a) required chicks per female per year against age at first breeding
    ages = np.arange(6, 12)
    lifespan_colours = {60: COLOURS["Floor"], 75: COLOURS["Central"], 90: COLOURS["Ceiling"]}
    central = SCENARIOS["Central"]
    for L, colour in lifespan_colours.items():
        R = [required_recruitment(T, L, BREEDING_SPAN, 0.5, r_obs) for T in ages]
        ax1.plot(ages, R, "o-", color=colour, lw=1.6, ms=4, label=f"Lifespan {L} yr")
    measured = [MEASURED_RANGE[0] * central["c"] / central["y"],
                MEASURED_RANGE[1] * central["c"] / central["y"]]
    ax1.axhspan(*measured, color=MEASURED_COLOUR, lw=0, zorder=0)
    ax1.text(5.85, measured[0] - 0.008, "Measured φ × 2.5 eggs / 3 yr", ha="left",
             va="top", fontsize=7)
    per_season = CHICKS_2019 / MATURE_FEMALES_2019
    for y in (3, 4):
        ax1.axhline(per_season / y, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
        if y == 3:
            ax1.text(11.15, per_season / y + 0.005, f"2019 output, breeding every {y} yr",
                     ha="right", va="bottom", fontsize=7, color=MUTED)
        else:
            ax1.text(5.85, per_season / y + 0.005, f"2019 output, breeding every {y} yr",
                     ha="left", va="bottom", fontsize=7, color=MUTED)
    ax1.set_xlim(5.8, 11.2)
    ax1.set_ylim(0, 0.5)
    ax1.set_xticks(ages)
    ax1.set_xlabel("Age at first breeding, Tₘ (yr)")
    ax1.set_ylabel("Required chicks per female per year, R*")
    ax1.legend(loc="lower right", handlelength=1.6, fontsize=7.5)
    panel_label(ax1, "a")

    # (b) projections from the two starting age structures
    years = np.arange(1995, 2027)
    for adults_only, phi, colour, ls in ((False, phi_stable, COLOURS["Central"], "-"),
                                        (True, phi_adult, "#7a4fd0", "-")):
        N = project_from(central, phi, adults_only, len(years) - 1)
        label = ("Stable age structure" if not adults_only else "All females adult") \
            + f", φ = {phi:.2f}"
        ax2.plot(years, N, color=colour, lw=1.8, ls=ls, label=label)
    exact = YEARS != APPROX_YEAR
    ax2.scatter(YEARS[exact], COUNTS[exact], s=24, color=INK, edgecolor="white",
                linewidth=0.8, zorder=4, label="Census")
    ax2.scatter(YEARS[~exact], COUNTS[~exact], s=24, facecolor="white", edgecolor=INK,
                linewidth=1.1, zorder=4)
    ax2.set_xlim(1994, 2027)
    ax2.set_ylim(0, 360)
    ax2.set_xticks([1995, 2005, 2015, 2025])
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Number of birds")
    ax2.legend(loc="upper left", handlelength=1.6, fontsize=7.5,
               title="Central scenario", title_fontsize=7.5)
    panel_label(ax2, "b")

    save(fig, "figure5_recruitment_and_start.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r_obs, _, two_point, ci_99 = observed_growth_rate()
    r_low = round(ci_99[0], 3)
    central = SCENARIOS["Central"]

    print("1. Closed-form required survival (eq. 8, section 3.6)")
    worst = 0.0
    for Tm, c, y, L in itertools.product(MATURITY, CLUTCH, INTERVAL, LIFESPAN):
        p = dict(Tm=Tm, c=c, y=y, f=0.5, L=L, Ta=BREEDING_SPAN)
        closed = required_survival_closed(p, r_obs)
        bisect = required_survival(p, r_obs)
        if bisect is None:
            assert closed > 1
        else:
            worst = max(worst, abs(closed - bisect))
    print(f"  Largest difference from bisection over the grid: {worst:.1e}")
    for name, p in SCENARIOS.items():
        R = required_recruitment(p["Tm"], p["L"], p["Ta"], p["f"], r_obs)
        print(f"  {name:8s} R* = {R:.3f} chicks/female/yr, phi* = {R * p['y'] / p['c']:.3f}, "
              f"chicks per attempt phi*c = {R * p['y']:.2f}")
    s = 1 - 1 / central["L"]
    q = s * np.exp(-r_obs)
    print(f"  q = s e^-r = {q:.4f}; each extra year before first breeding multiplies "
          f"phi* by 1/q = {1 / q:.4f} (x{q**-5:.2f} over 5 yr); 1 - q^Ta = {1 - q**central['Ta']:.3f}")
    print("  R* (chicks per female per year) by Tm and L, Ta = 50:")
    R_grid = []
    for Tm in MATURITY:
        row = [required_recruitment(Tm, L, BREEDING_SPAN, 0.5, r_obs) for L in LIFESPAN]
        R_grid += row
        print(f"    Tm = {Tm:2d}: " + "  ".join(f"L={L}: {R:.3f}" for L, R in zip(LIFESPAN, row)))
    egg_rates = [c / y for c in CLUTCH for y in INTERVAL]
    print(f"  {len(R_grid)} values of R*, {fmt_range(min(R_grid), max(R_grid))}, "
          f"factor {max(R_grid) / min(R_grid):.1f}; c/y varies by a factor of "
          f"{max(egg_rates) / min(egg_rates):.0f}")
    print(f"  2019: {CHICKS_2019}/{MATURE_FEMALES_2019} = {CHICKS_2019 / MATURE_FEMALES_2019:.2f}"
          f" chicks per mature female; {CHICKS_2019}/{NESTING_FEMALES_2019} = "
          f"{CHICKS_2019 / NESTING_FEMALES_2019:.2f} per nesting female; "
          f"{EGGS_2019}/{NESTING_FEMALES_2019} = {EGGS_2019 / NESTING_FEMALES_2019:.1f} eggs per nesting female")
    R_central = required_recruitment(central["Tm"], central["L"], central["Ta"], central["f"], r_obs)
    for y in (2, 3, 4):
        per_year = CHICKS_2019 / MATURE_FEMALES_2019 / y
        print(f"    spread over {y} yr: {per_year:.3f} chicks/female/yr "
              f"({per_year / R_central:.2f} x central R*)")

    print(f"\n2. Lower end of the 99% CI, r = {r_low:.3f} (sections 3.2, 3.3)")
    for name, p in SCENARIOS.items():
        ps = required_survival(p, r_low)
        print(f"  {name:8s} phi* = {ps:.2f}")
    rows = []
    for Tm, c, y, L in itertools.product(MATURITY, CLUTCH, INTERVAL, LIFESPAN):
        p = dict(Tm=Tm, c=c, y=y, f=0.5, L=L, Ta=BREEDING_SPAN)
        rows.append((is_typical(Tm, c, y), required_survival_closed(p, r_low)))
    reach = np.array([v for _, v in rows if v <= 1])
    typ = np.array([v for t, v in rows if t and v <= 1])
    inside = lambda a: np.sum((a >= MEASURED_RANGE[0]) & (a <= MEASURED_RANGE[1]))
    print(f"  Grid: {len(reach)} of {len(rows)} reach r; phi* {fmt_range(reach.min(), reach.max(), 2)}, "
          f"median {np.median(reach):.2f}; {inside(reach)} inside ({100 * inside(reach) / len(reach):.0f}%)")
    print(f"  Typical: {len(typ)} sets; phi* {fmt_range(typ.min(), typ.max(), 2)}, median {np.median(typ):.2f}; "
          f"{inside(typ)} inside ({100 * inside(typ) / len(typ):.0f}%); "
          f"{np.sum(typ < MEASURED_RANGE[0])} below the range")
    p_floor_low = required_survival(SCENARIOS["Floor"], r_low)
    print(f"  Floor phi* at lower end {p_floor_low:.2f} (inside range: "
          f"{MEASURED_RANGE[0] <= p_floor_low <= MEASURED_RANGE[1]})")

    print("\n3. Elasticity of lambda at phi* (section 3.5)")
    for name in ("Central", "Ceiling"):
        p = SCENARIOS[name]
        phi = required_survival(p, r_obs)
        lam, fec, juv, adult, T = elasticities(p, phi)
        R0 = net_reproductive_rate(p, phi)
        print(f"  {name:8s} phi* = {phi:.3f}: lambda = {lam:.4f}; fecundity {fec:.3f}, "
              f"juvenile survival {juv:.3f}, adult survival {adult:.3f} (sum {fec + juv + adult:.3f})")
        print(f"           all survival / fecundity = {(juv + adult) / fec:.1f}; "
              f"each juvenile rate {juv / p['Tm']:.3f}")
        print(f"           generation time 1/E_F = {T:.1f} yr; R0 = {R0:.2f}; ln(R0)/r = "
              f"{np.log(R0) / np.log(lam):.1f} yr")
    p = SCENARIOS["Floor"]
    lam, fec, juv, adult, T = elasticities(p, 1.0)
    print(f"  Floor at phi = 1: fecundity {fec:.3f}, juvenile {juv:.3f}, adult {adult:.3f}, T = {T:.1f}")

    print("\n4. Starting age structure (central scenario), phi to reach 325 in 2026 (section 3.6)")
    phi_stable = survival_to_reach_2026(central, adults_only=False)
    phi_adult = survival_to_reach_2026(central, adults_only=True)
    w_stable = start_vector(central, phi_stable, False)
    print(f"  Stable start: {w_stable[central['Tm']:].sum():.1f} of 20 females at breeding age; "
          f"phi = {phi_stable:.3f}")
    print(f"  All-adult start: phi = {phi_adult:.3f}")
    for adults_only, phi in ((False, phi_stable), (True, phi_adult)):
        N = project_from(central, phi, adults_only, YEARS[-1] - YEARS[0])
        fit = N[YEARS - 1995]
        rmse = np.sqrt(np.mean((np.log(COUNTS) - np.log(fit)) ** 2))
        print(f"    {'adult' if adults_only else 'stable'}: projected "
              + ", ".join(f"{y}:{v:.0f}" for y, v in zip(YEARS, fit)) + f"; RMSE(log) {rmse:.3f}")
    r_long = growth_rate(central, phi=phi_adult)
    print(f"  Long-run r at phi = {phi_adult:.3f}: {r_long:.4f}")

    print("\n5. Central scenario growth for selected phi (section 4.3)")
    for phi in (MEASURED_RANGE[0], CHICKS_2019 / EGGS_2019, required_survival(central, r_obs),
                MEASURED_RANGE[1], 0.5, 0.75):
        r = growth_rate(central, phi=phi)
        print(f"  phi = {phi:.3f}: r = {r:.4f}, doubling {np.log(2) / r:.1f} yr")

    figure5(r_obs, ci_99, phi_stable, phi_adult)
    print("\nSaved figure5_recruitment_and_start.png")