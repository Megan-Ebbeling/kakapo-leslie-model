# Required egg-to-chick survival in the recovering kākāpō

Code for the Biomathematics 374 report _"What egg-to-chick survival does the
Kākāpō's recovery require? An age-structured model compared with measured
hatching success"_ (Ebbeling & Ndlovu, Stellenbosch University, 2026).

A female-only Leslie matrix model is built for floor, central and ceiling
life-history scenarios. The egg-to-chick survival φ\* needed to reproduce the
observed growth of the population (1995 to 2026) is found by bisection and in
closed form, and compared with measured hatching success and chick production
(Savage et al. 2022).

## Requirements

Python 3.9 or newer, NumPy and Matplotlib:

```
pip install -r requirements.txt
```

## How to run

Run each script from this folder. Every number quoted in the report is printed
to the terminal and every figure is saved to `figures/`.

| Script                        | What it does                                                                                                                                                                               | Output                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `kakapo_agestructured.py`     | Observed growth rate from the census (Section 2.2), the logistic check (2.3), the Leslie model and the floor/central/ceiling growth rates (2.4, 3.1), one-at-a-time sensitivity (2.6, 3.4) | `figure1_envelope.png`, `figure4_sensitivity.png`                                                  |
| `kakapo_required_survival.py` | Required survival φ\* by bisection for each scenario and four estimates of r_obs (Table 4, Section 3.2), the 378-set parameter grid (3.3), and the effect of juvenile survival on φ\* (3.5)  | `figure2_growth_vs_survival.png`, `figure3_required_survival.png`, `figure6_juvenile_survival.png` |
| `kakapo_extensions.py`        | Closed-form required survival and chicks per female R\* (eq. 8, 3.6), the lower 99% limit of r_obs (3.2, 3.3), elasticity, R₀ and generation time (2.6, 3.5), starting age structure (3.6), growth for selected φ (4.3) | `figure5_recruitment_and_start.png`                                                                |

```
python kakapo_agestructured.py        # about 2 s
python kakapo_required_survival.py    # about 15 s
python kakapo_extensions.py           # about 15 s
```

`kakapo_required_survival.py` imports the model from `kakapo_agestructured.py`,
and `kakapo_extensions.py` imports from both, so keep all three files in the
same folder.

## Model summary

- Age classes 0 to 99 years, one-year time steps: **n**(t+1) = A **n**(t).
- Fecundity F′ = φ (c / y) f for ages Tₘ to Tₘ + Tₐ − 1; survival s = 1 − 1/L.
- Growth rate r = ln λ, where λ is the dominant eigenvalue of A. It is computed
  from the Euler-Lotka equation 1 = Σ F′ₐ lₐ λ^−(a+1), which is the characteristic
  equation of A; the script checks it against NumPy's eigenvalues.
- φ\* solves r(φ\*) = r_obs and is found by 50 bisection steps on [0, 1].
- Because F and s are the same at every breeding age, the Euler-Lotka sum is a
  geometric series and can also be solved directly for R\* = φ\*c/y, the
  surviving chicks per female per year (equation 8 in the report). This matches
  the bisection results to within 10⁻¹⁵.

| Parameter                     | Floor | Central | Ceiling |
| ----------------------------- | ----- | ------- | ------- |
| Age at first breeding Tₘ (yr) | 11    | 9       | 6       |
| Eggs per breeding attempt c   | 1     | 2.5     | 4       |
| Years between breeding y      | 4     | 3       | 2       |
| Female fraction f             | 0.5   | 0.5     | 0.5     |
| Lifespan L (yr)               | 60    | 75      | 90      |
| Breeding span Tₐ (yr)         | 40    | 50      | 60      |

The ceiling clutch is 4 eggs, the largest clutch recorded in field studies.

## Data

All data are hard-coded at the top of the scripts and are listed in Tables 1
and 3 of the report: census counts from the New Zealand Department of
Conservation and BirdLife International, and egg survival and chick production
from Savage et al.
(2022), _Animal Conservation_, doi:10.1111/acv.12746.
