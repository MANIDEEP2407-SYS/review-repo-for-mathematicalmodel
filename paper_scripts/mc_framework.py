"""Monte Carlo framework study for the coupled-model paper: distributional assumptions, convergence,
robustness to width / family, and a formal model-form versus parametric decomposition.
Read-only import of ../solar_desalination_simulator.   python -B mc_framework.py   -> mc_framework.json"""
import json, os, sys
from copy import deepcopy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src.physics.system import SystemConfig, simulate                      # noqa: E402
from src.config.defaults import PARAMETER_REGISTRY                          # noqa: E402
from src.scaling.monte_carlo import (COUPLED_UNCERTAINTIES, DEFAULT_UNCERTAINTIES,  # noqa: E402
                                     monte_carlo)

L, C = "legacy", "coupled"
N = 2000
SEED = 42
ASSUMED_NEW = ("diffusion_path_length", "gap_convection_coefficient", "condensation_coefficient", "glass_absorptance")


def draw(rng, dist, sigma):
    """Multiplicative factor with mean 1 and relative sd sigma."""
    if dist == "gaussian":
        return float(rng.normal(1.0, sigma))
    if dist == "uniform":                       # same sd: half-width sqrt(3) sigma
        h = np.sqrt(3.0) * sigma
        return float(rng.uniform(1 - h, 1 + h))
    if dist == "lognormal":                     # same mean and cv
        s2 = np.log(1 + sigma ** 2)
        return float(rng.lognormal(-0.5 * s2, np.sqrt(s2)))
    raise ValueError(dist)


def mc(coupling, unc, n=N, seed=SEED, dist="gaussian", scale=1.0, ignorance=False):
    base = SystemConfig(coupling=coupling)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        cfg = deepcopy(base)
        for name, s in unc.items():
            nominal = getattr(cfg, name)
            if ignorance and name in ASSUMED_NEW:           # uniform over the whole registered range
                lo, hi = PARAMETER_REGISTRY[name].allowed_range
                v = float(rng.uniform(lo, hi))
            else:
                v = nominal * draw(rng, dist, s * scale)
                v = max(v, nominal * 1e-3)
                lo, hi = PARAMETER_REGISTRY[name].allowed_range
                v = min(max(v, lo), hi)
            setattr(cfg, name, v)
        out[i] = simulate(cfg).daily_freshwater_L
    return out


def stats(x):
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), cv=float(x.std(ddof=1) / x.mean()),
                p05=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)))


def compare(a, b, rng, B=2000):
    """a = legacy sample, b = coupled sample (independent).  Formal model-form vs parametric split."""
    def one(a, b):
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        within = 0.5 * (va + vb)
        between = 0.25 * (mb - ma) ** 2           # equal model weights: Var(E[Y|M])
        pooled_sd = np.sqrt(within)
        return dict(delta_mean=mb - ma, ratio_mean=mb / ma, d=(mb - ma) / pooled_sd,
                    within=within, between=between, frac_model=between / (within + between),
                    sd_total=np.sqrt(within + between))
    pt = one(a, b)
    bs = {k: [] for k in pt}
    for _ in range(B):
        r = one(rng.choice(a, a.size), rng.choice(b, b.size))
        for k in bs: bs[k].append(r[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    # exceedance and overlap
    # P(Y_C > Y_L) for independent draws via ranks
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort() + 1
    u = ranks[a.size:].sum() - b.size * (b.size + 1) / 2
    p_exc = float(u / (a.size * b.size))
    edges = np.linspace(allv.min(), allv.max(), 61)
    ha, _ = np.histogram(a, edges); hb, _ = np.histogram(b, edges)
    ovl = float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())
    # how many legacy 95th percentiles are below the coupled 5th: separation of the bands
    return dict(point={k: float(v) for k, v in pt.items()}, ci95=ci, P_coupled_exceeds_legacy=p_exc,
                overlap_coeff=ovl, bands_overlap=bool(np.percentile(a, 95) > np.percentile(b, 5)))


out = {"N": N, "seed": SEED}
rng_b = np.random.default_rng(7)

# 0. widths actually used
out["widths_legacy"] = DEFAULT_UNCERTAINTIES
out["widths_coupled"] = COUPLED_UNCERTAINTIES
out["registry"] = {n: dict(nominal=getattr(SystemConfig(), n), range=list(PARAMETER_REGISTRY[n].allowed_range),
                           tier=PARAMETER_REGISTRY[n].source.value, unit=PARAMETER_REGISTRY[n].unit)
                   for n in COUPLED_UNCERTAINTIES}

# 1. the tool's own runs (reference; must equal paper_numbers.json mc_*)
ref = {}
for k in (L, C):
    m = monte_carlo(N, SystemConfig(coupling=k), seed=SEED)
    ref[k] = m.daily_L
out["reference"] = {k: stats(v) for k, v in ref.items()}
out["reference_compare"] = compare(ref[L], ref[C], rng_b)

# 2. like-for-like: both solvers perturb the same 19 parameters
same = {k: mc(k, COUPLED_UNCERTAINTIES) for k in (L, C)}
out["same_params"] = {k: stats(v) for k, v in same.items()}
out["same_params_compare"] = compare(same[L], same[C], rng_b)

# 3. width scaling and distribution family (same parameter set)
rob = []
for dist in ("gaussian", "uniform", "lognormal"):
    for sc in (0.5, 1.0, 2.0, 3.0):
        if dist != "gaussian" and sc not in (1.0, 2.0):
            continue
        a, b = mc(L, COUPLED_UNCERTAINTIES, dist=dist, scale=sc), mc(C, COUPLED_UNCERTAINTIES, dist=dist, scale=sc)
        cmp_ = compare(a, b, rng_b, B=500)
        rob.append(dict(dist=dist, scale=sc, legacy=stats(a), coupled=stats(b),
                        bands_overlap=cmp_["bands_overlap"], d=cmp_["point"]["d"],
                        frac_model=cmp_["point"]["frac_model"], P_exc=cmp_["P_coupled_exceeds_legacy"]))
out["robustness"] = rob

# 4. ignorance prior: the four new assumed coupled parameters uniform over their whole registered range,
#    everything else at its stated width.
ign = mc(C, COUPLED_UNCERTAINTIES, ignorance=True)
out["ignorance_coupled"] = stats(ign)
out["ignorance_P_below_legacy_mean"] = float((ign < ref[L].mean()).mean())
out["ignorance_P_below_legacy_p95"] = float((ign < np.percentile(ref[L], 95)).mean())
out["ignorance_compare"] = compare(ref[L], ign, rng_b, B=500)

# 4b. width multiplier at which the 5-95 % bands first touch (gaussian, same parameter set)
grid = []
for sc in np.round(np.arange(1.0, 2.01, 0.1), 2):
    a, b = mc(L, COUPLED_UNCERTAINTIES, scale=float(sc)), mc(C, COUPLED_UNCERTAINTIES, scale=float(sc))
    grid.append(dict(scale=float(sc), p95_legacy=float(np.percentile(a, 95)), p05_coupled=float(np.percentile(b, 5))))
out["band_touch_grid"] = grid
bt = next((g["scale"] for g in grid if g["p95_legacy"] >= g["p05_coupled"]), None)
out["band_touch_scale"] = bt

# 5. Monte Carlo error: sample size and seed
conv = {}
for n in (250, 500, 1000, 2000, 5000, 10000):
    conv[str(n)] = stats(mc(C, COUPLED_UNCERTAINTIES, n=n))
out["convergence_coupled"] = conv
seeds = [stats(mc(C, COUPLED_UNCERTAINTIES, seed=s)) for s in range(1, 11)]
out["seed_spread_coupled"] = {k: dict(mean=float(np.mean([s[k] for s in seeds])), sd=float(np.std([s[k] for s in seeds], ddof=1)))
                              for k in ("mean", "p05", "p95")}
seeds_l = [stats(mc(L, COUPLED_UNCERTAINTIES, seed=s)) for s in range(1, 11)]
out["seed_spread_legacy"] = {k: dict(mean=float(np.mean([s[k] for s in seeds_l])), sd=float(np.std([s[k] for s in seeds_l], ddof=1)))
                             for k in ("mean", "p05", "p95")}

json.dump(out, open(os.path.join(HERE, "mc_framework.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("reference", "same_params", "ignorance_coupled", "seed_spread_coupled", "seed_spread_legacy")}, indent=1))
print("ref compare", json.dumps(out["reference_compare"], indent=1))
print("same compare", json.dumps(out["same_params_compare"]["point"], indent=1), out["same_params_compare"]["bands_overlap"])
for r in rob: print(r["dist"], r["scale"], round(r["legacy"]["p95"], 2), round(r["coupled"]["p05"], 2), r["bands_overlap"], round(r["d"], 2), round(r["frac_model"], 3))
print("ign", out["ignorance_P_below_legacy_mean"], out["ignorance_P_below_legacy_p95"], out["ignorance_compare"]["bands_overlap"])
