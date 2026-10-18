"""
Leslie matrix model for the kakapo (females only).

Works out r_obs from the census, the growth rate for the floor, central and
ceiling scenarios, the logistic check from section 2.3 and the sensitivity
analysis. Makes figures 1 and 4.

kakapo_required_survival.py imports from this file.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save figures without opening a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Census counts (Table 1). After 1995 these are total birds, not just females.
# 2016 is approximate (DOC says the population doubled between 2016 and 2022).
YEARS = np.array([1995, 2005, 2009, 2012, 2016, 2018, 2022, 2026])
COUNTS = np.array([51, 86, 124, 126, 126, 149, 252, 325])
APPROX_YEAR = 2016
FEMALES_1995 = 20
TOTAL_1995 = 51

# Scenarios from Table 2
# Tm = age at first breeding, c = eggs per attempt, y = years between breeding,
# f = female fraction, L = lifespan, Ta = breeding span
N_AGES = 100   # ages 0 to 99

SCENARIOS = {
    "Floor":   dict(Tm=11, c=1.0, y=4, f=0.5, L=60, Ta=40),
    "Central": dict(Tm=9,  c=2.5, y=3, f=0.5, L=75, Ta=50),
    "Ceiling": dict(Tm=6,  c=4.0, y=2, f=0.5, L=90, Ta=60),
}

# Plot settings (used by all three scripts)
COLOURS = {"Floor": "#2a78d6", "Central": "#eb6834", "Ceiling": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#e6e6e6"
OBS_STYLE = dict(color=INK, lw=1.4, ls=(0, (5, 2.5)))   # observed r line
OBS_BAND = "#d9d9d9"          # 99% CI of r_obs
MEASURED_COLOUR = "#f6cdd8"   # measured survival range
FULL_WIDTH, SINGLE_WIDTH = 6.7, 6.0   # inches, so the fonts print at the right size

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#8c8c8c",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "text.color": INK,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": "white",
})


# ---------------- model ----------------
def survival_by_age(Tm, L, s_juv=None):
    """Survival for each age. s = 1 - 1/L, and ages below Tm get s_juv if given."""
    s = 1.0 - 1.0 / L
    s_ages = np.full(N_AGES - 1, s)
    if s_juv is not None:
        s_ages[:Tm] = s_juv
    return s_ages


def fecundity_by_age(Tm, c, y, f, Ta, phi=1.0):
    """First row of A: phi*(c/y)*f from age Tm to Tm+Ta-1, zero otherwise."""
    F = np.zeros(N_AGES)
    F[Tm:min(Tm + Ta, N_AGES)] = phi * (c / y) * f
    return F


def leslie_matrix(Tm, c, y, f, L, Ta, phi=1.0, s_juv=None):
    """Build A (equation 3)."""
    A = np.zeros((N_AGES, N_AGES))
    A[0, :] = fecundity_by_age(Tm, c, y, f, Ta, phi)
    A[np.arange(1, N_AGES), np.arange(N_AGES - 1)] = survival_by_age(Tm, L, s_juv)
    return A


def dominant_eigen(A):
    """Largest eigenvalue of A and its eigenvector, scaled to sum to 1."""
    values, vectors = np.linalg.eig(A)
    k = np.argmax(np.abs(values))
    w = np.abs(np.real(vectors[:, k]))
    return np.real(values[k]), w / w.sum()


def growth_rate(params, phi=1.0, s_juv=None, method="euler-lotka"):
    """r = ln(lambda).

    Solves the Euler-Lotka equation (eq. 7) for r by bisection, which gives the
    same lambda as the eigenvalue but runs a lot faster on the grid.
    method="eigen" uses np.linalg.eig instead, to check it.
    """
    if phi <= 0:
        return -np.inf
    if method == "eigen":
        lam, _ = dominant_eigen(leslie_matrix(**params, phi=phi, s_juv=s_juv))
        return np.log(lam)

    F = fecundity_by_age(params["Tm"], params["c"], params["y"], params["f"],
                         params["Ta"], phi)
    l = np.concatenate(([1.0], np.cumprod(survival_by_age(params["Tm"], params["L"], s_juv))))
    ages = np.nonzero(F)[0]
    weights, powers = F[ages] * l[ages], ages + 1.0

    def net_replacement(r):            # sum in eq. 7 minus 1
        return np.sum(weights * np.exp(-r * powers)) - 1.0

    low, high = -3.0, 3.0
    for _ in range(80):
        mid = (low + high) / 2
        if net_replacement(mid) > 0:
            low = mid                  # r too small
        else:
            high = mid
    return (low + high) / 2


def observed_growth_rate():
    """Fit ln N against year (eq. 1). Returns slope, intercept, two-point
    estimate and the 99% CI of the slope."""
    t = YEARS - 1995
    logN = np.log(COUNTS)
    slope, intercept = np.polyfit(t, logN, 1)

    residuals = logN - (intercept + slope * t)
    n = len(t)
    se = np.sqrt(np.sum(residuals**2) / (n - 2) / np.sum((t - t.mean())**2))
    t_crit_99 = 3.7074   # t value for 99%, 6 df
    ci_99 = (slope - t_crit_99 * se, slope + t_crit_99 * se)

    two_point = np.log(COUNTS[-1] / COUNTS[0]) / (YEARS[-1] - YEARS[0])
    return slope, intercept, two_point, ci_99


def project(params, years, phi=1.0):
    """Run the model forward from 1995, starting from the stable age
    distribution with 20 females. Returns total birds (x 51/20)."""
    A = leslie_matrix(**params, phi=phi)
    _, w = dominant_eigen(A)
    n = FEMALES_1995 * w
    totals = []
    for _ in years:
        totals.append(n.sum() * TOTAL_1995 / FEMALES_1995)
        n = A @ n
    return np.array(totals)


SENSITIVITY_LABELS = {
    "c":  "Clutch size, 1 to 4 eggs",
    "Tm": "Age at first breeding, 11 to 6 yr",
    "y":  "Breeding interval, 4 to 2 yr",
    "L":  "Lifespan, 60 to 90 yr",
    "Ta": "Breeding span, 40 to 60 yr",
}


def sensitivity():
    """Change one parameter at a time from floor to ceiling, rest at central."""
    results = {}
    for key in SENSITIVITY_LABELS:
        rates = []
        for scenario in ("Floor", "Ceiling"):
            params = dict(SCENARIOS["Central"])
            params[key] = SCENARIOS[scenario][key]
            rates.append(growth_rate(params))
        results[key] = tuple(rates)
    return results


# ---------------- figures ----------------
def fmt_range(lo, hi, digits=3):
    # "a to b" instead of a dash
    return f"{lo:.{digits}f} to {hi:.{digits}f}"


def panel_label(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def figure1(r_obs, lower, upper):
    """Fig 1: projections vs census (a), growth rates (b)."""
    years = np.arange(1995, 2027)
    rates = {name: growth_rate(p) for name, p in SCENARIOS.items()}
    curves = {name: project(p, years) for name, p in SCENARIOS.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0),
                                   gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.38})

    # (a)
    ax1.fill_between(years, curves["Floor"], curves["Ceiling"], color="#000000",
                     alpha=0.05, lw=0)
    for name in ("Ceiling", "Central", "Floor"):
        ax1.plot(years, curves[name], color=COLOURS[name], lw=1.8)
    exact = YEARS != APPROX_YEAR
    ax1.scatter(YEARS[exact], COUNTS[exact], s=26, color=INK, edgecolor="white",
                linewidth=0.8, zorder=4, clip_on=False)
    ax1.scatter(YEARS[~exact], COUNTS[~exact], s=26, facecolor="white", edgecolor=INK,
                linewidth=1.1, zorder=4, clip_on=False)
    ax1.set_yscale("log")
    ax1.set_ylim(40, 2000)
    ax1.yaxis.set_major_locator(FixedLocator([50, 100, 200, 500, 1000, 2000]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xlim(1994, 2027)
    ax1.set_xticks([1995, 2000, 2005, 2010, 2015, 2020, 2025])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of birds (log scale)")
    panel_label(ax1, "a")

    # (b)
    names = list(SCENARIOS)
    ax2.axhspan(lower, upper, color=OBS_BAND, alpha=0.7, lw=0)
    ax2.axhline(r_obs, **OBS_STYLE, zorder=3)
    bars = ax2.bar(names, [rates[n] for n in names], width=0.6,
                   color=[COLOURS[n] for n in names], zorder=2)
    for bar, n in zip(bars, names):
        x = bar.get_x() + bar.get_width() / 2
        if rates[n] > upper + 0.02:      # label above tall bars
            ax2.text(x, rates[n] + 0.005, f"{rates[n]:.3f}", ha="center", va="bottom",
                     fontsize=8)
        else:                            # inside short bars
            ax2.text(x, rates[n] - 0.005, f"{rates[n]:.3f}", ha="center", va="top",
                     fontsize=8, color="white", fontweight="bold")
    ax2.set_ylim(0, 0.25)
    ax2.set_ylabel("Growth rate r (yr$^{-1}$), φ = 1")
    ax2.grid(axis="x", visible=False)
    ax2.tick_params(axis="x", length=0)
    panel_label(ax2, "b")

    # shared legend under both panels
    handles = [Line2D([], [], color=COLOURS[n], lw=1.8, label=n) for n in names]
    handles += [Line2D([], [], ls="none", marker="o", ms=5, color=INK, label="Census"),
                Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK,
                       label="Census, approximate (2016)"),
                Line2D([], [], **OBS_STYLE, label=f"Observed r ({r_obs:.3f})"),
                Patch(color=OBS_BAND, alpha=0.7, label=f"99% CI ({lower:.3f} to {upper:.3f})")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               handlelength=1.8, columnspacing=2.5)

    save(fig, "figure1_envelope.png")


def figure4(results, r_obs, lower, upper):
    """Fig 4: tornado plot of the sensitivity results."""
    r_central = growth_rate(SCENARIOS["Central"])
    order = sorted(results, key=lambda k: abs(results[k][1] - results[k][0]))

    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH, 2.7))
    ax.axvspan(lower, upper, color=OBS_BAND, alpha=0.7, lw=0)
    ax.axvline(r_obs, **OBS_STYLE)
    for i, key in enumerate(order):
        lo, hi = sorted(results[key])
        ax.barh(i, max(hi - lo, 0.0006), left=lo, height=0.56,
                color=COLOURS["Central"], zorder=2)
        text = f"{lo:.3f} (no change)" if round(lo, 3) == round(hi, 3) else fmt_range(lo, hi)
        ax.text(hi + 0.0025, i, text, va="center", fontsize=8)
    ax.axvline(r_central, color=INK, lw=0.9, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SENSITIVITY_LABELS[k] for k in order])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.03, 0.17)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Growth rate r (yr$^{-1}$), φ = 1")
    handles = [Line2D([], [], **OBS_STYLE, label=f"Observed rate ({r_obs:.3f})"),
               Patch(color=OBS_BAND, alpha=0.7, label=f"99% CI ({lower:.3f} to {upper:.3f})"),
               Line2D([], [], color=INK, lw=0.9, label=f"Central scenario ({r_central:.3f})")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.8, columnspacing=1.2)

    save(fig, "figure4_sensitivity.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # so φ prints on Windows
    r_obs, _, two_point, ci_99 = observed_growth_rate()
    lower, upper = round(ci_99[0], 3), round(ci_99[1], 3)
    print("Section 2.2 - observed growth rate")
    print(f"  Least squares: r_obs = {r_obs:.4f} per yr "
          f"({100 * np.expm1(r_obs):.1f}% per yr, doubling {np.log(2) / r_obs:.1f} yr)")
    print(f"  Two-point:     {two_point:.4f} per yr")
    print(f"  99% CI:        {fmt_range(*ci_99)} per yr\n")

    ceiling = SCENARIOS["Ceiling"]
    r_logistic = (ceiling["c"] / ceiling["y"]) * ceiling["f"] - 1 / ceiling["L"]
    print("Section 2.3 - logistic model with ceiling values (c = 4, y = 2, L = 90)")
    print(f"  r = b - d = {r_logistic:.3f} per yr, {r_logistic / r_obs:.1f} times r_obs, "
          f"population x{np.exp(r_logistic):.2f} per yr\n")

    print("Section 3.1 - growth with no losses (phi = 1)")
    rates = {}
    for name, params in SCENARIOS.items():
        r = growth_rate(params)
        r_eig = growth_rate(params, method="eigen")
        rates[name] = r
        print(f"  {name:8s} r = {r:.4f} per yr, doubling {np.log(2) / r:.1f} yr"
              f"   (eigenvalue check: {r_eig:.4f})")
    position = (r_obs - rates["Floor"]) / (rates["Ceiling"] - rates["Floor"])
    print(f"  Observed rate is {100 * position:.0f}% of the way from floor to ceiling")
    floor_curve = project(SCENARIOS["Floor"], np.arange(1995, 2027))[YEARS - 1995]
    print("  Census above the floor projection: "
          + ", ".join(f"{y} {100 * (n / p - 1):.0f}%"
                      for y, n, p in zip(YEARS[1:], COUNTS[1:], floor_curve[1:])))
    for name, r in rates.items():
        inside = ci_99[0] <= r <= ci_99[1]
        print(f"  {name:8s} r inside the 99% CI of r_obs: {inside}")
    print()

    results = sensitivity()
    print("Section 3.4 - sensitivity (others central, phi = 1)")
    for key, (r_floor, r_ceiling) in results.items():
        print(f"  {SENSITIVITY_LABELS[key]:38s} {r_floor:.3f} -> {r_ceiling:.3f}")

    figure1(r_obs, lower, upper)
    figure4(results, r_obs, lower, upper)
    print(f"\nSaved figure1_envelope.png and figure4_sensitivity.png to {FIG_DIR}")