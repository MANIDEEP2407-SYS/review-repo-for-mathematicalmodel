"""Figures for the ITSEMC-2026 version (single-column size, 300 dpi).
Numeric figures read paper_numbers.json (written by paper_numbers.py from the simulator); nothing is typed in."""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = json.load(open(os.path.join(HERE, "paper_numbers.json")))
OUT = os.path.join(HERE, "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.linewidth": 0.6})


# ---------- Fig: system schematic ----------
def schematic():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6); ax.axis("off")
    # basin + water
    ax.add_patch(Rectangle((1.6, 0.9), 5.6, 1.6, fc="#dfe9f3", ec="k", lw=0.8))
    ax.add_patch(Rectangle((1.6, 0.9), 5.6, 1.1, fc="#9ec5e8", ec="none"))
    ax.add_patch(Rectangle((1.45, 0.5), 5.9, 0.4, fc="#f3e6a8", ec="k", lw=0.6, hatch="////"))
    ax.text(4.4, 0.18, "polystyrene insulation", ha="center", fontsize=6)
    ax.text(1.8, 1.05, "brackish water", fontsize=6, color="#103a63")
    # foam raft + flowers
    ax.add_patch(Rectangle((2.5, 2.0), 3.8, 0.28, fc="#cfcfcf", ec="k", lw=0.6))
    for x in (3.0, 3.9, 4.8, 5.7):
        ax.add_patch(Polygon([[x - 0.35, 2.28], [x + 0.35, 2.28], [x + 0.5, 3.0], [x, 2.7], [x - 0.5, 3.0]],
                             fc="#222222", ec="k", lw=0.5))
    ax.annotate("FeS-cotton\n3D flowers", (4.4, 2.85), (4.6, 4.55), fontsize=6.2, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.annotate("foam raft", (3.4, 2.0), (3.9, 1.35), fontsize=5.8, ha="center", arrowprops=dict(arrowstyle="-", lw=0.4))
    # glass cover (sloped)
    ax.plot([1.6, 7.2], [5.0, 3.55], c="#1f77b4", lw=2.0)
    ax.text(5.9, 4.85, "sloped glass cover (A$_g$)", fontsize=6.2, rotation=-14, ha="center", color="#1f77b4")
    ax.plot([1.6, 1.6], [2.5, 5.0], c="k", lw=0.8); ax.plot([7.2, 7.2], [2.5, 3.55], c="k", lw=0.8)
    # condensate gutter + bottle
    ax.add_patch(Rectangle((7.2, 3.3), 0.7, 0.25, fc="#8fb8d8", ec="k", lw=0.6))
    ax.add_patch(Rectangle((8.1, 1.5), 0.9, 1.5, fc="#e9f3fa", ec="k", lw=0.8))
    ax.add_patch(Rectangle((8.1, 1.5), 0.9, 0.7, fc="#6aa8d8", ec="none"))
    ax.plot([7.9, 8.5, 8.5], [3.4, 3.4, 3.0], c="k", lw=0.6)
    ax.text(8.55, 0.6, "fresh\nwater", ha="center", fontsize=6)
    # reflectors + rays
    ax.plot([0.2, 1.2], [1.5, 3.7], c="#888888", lw=2.2)
    ax.text(0.15, 4.0, "reflector\n(G$_{opt}$)", fontsize=6, ha="left")
    for x in (1.9, 2.9, 3.9):
        ax.add_patch(FancyArrowPatch((x - 0.3, 7.3), (x + 0.2, 5.5), arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#e69f00"))
    ax.text(0.6, 7.2, "solar irradiance G", fontsize=6.2, color="#b36b00")
    ax.add_patch(FancyArrowPatch((1.0, 3.7), (2.2, 3.1), arrowstyle="-|>", mutation_scale=5, lw=0.7, color="#e69f00"))
    # venturi
    ax.add_patch(FancyBboxPatch((7.45, 4.6), 2.2, 0.9, boxstyle="round,pad=0.05", fc="#ffe3e3", ec="k", lw=0.6))
    ax.text(8.55, 5.05, "Venturi\nejector", ha="center", va="center", fontsize=6)
    ax.plot([7.2, 7.45], [4.0, 4.9], c="k", lw=0.6, ls="--")
    fig.savefig(os.path.join(OUT, "schematic.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


# ---------- Fig: model flow (coupled solver) ----------
def flow():
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")

    def box(x, y, w, h, t, fc="#eef3fa", fs=6.0, ls="-", lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec="k", lw=lw, ls=ls))
        if t:
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def arr(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=0.7))

    box(0.1, 9.7, 9.8, 1.2, "Inputs: G, G$_{opt}$, τ$_g$, η$_{opt}$, A$_{geo}$, A$_g$, t$_{sun}$, T$_{amb}$, v$_w$, C$_0$;\nassumed: δ, h$_{c,gap}$, α$_g$, h$_{cond}$, P$_{leak}$", "#fff4d6", 5.6)
    box(0.1, 8.0, 3.0, 1.3, "Solar input\nP$_{abs}$, P$_{g}$", fs=5.8)
    box(3.5, 4.35, 6.4, 4.95, "", "#f4f4f4", lw=0.8)
    ax.text(6.7, 8.95, "COUPLED nested solve (default)", ha="center", fontsize=5.9, fontweight="bold")
    ax.text(6.7, 8.55, "unknowns T$_e$, T$_{gi}$, T$_c$, ṁ; brentq, residuals", ha="center", fontsize=5.3)
    box(3.7, 7.1, 6.0, 1.15, "(a) evaporator node:\nP$_{abs}$ = ṁ e$_{evap}$ + Q$_{gap}$ + Q$_{ins}$", fs=5.4)
    box(3.7, 5.75, 6.0, 1.15, "(b) Stefan diffusion through air film δ\n(boiling limit handled)", fs=5.4)
    box(3.7, 4.5, 6.0, 1.05, "(c) glass node + condensate film\n→ T$_{gi}$, T$_c$", fs=5.4)
    box(0.1, 5.65, 3.0, 2.0, "Ejector: P$_{ch}$ =\nmax(P$_{atm}$−ΔP, P$_{sat}$(T$_m$),\nP$_{leak}$); floats up to\nP$_{sat}$(T$_c$)", "#ffe3e3", 4.9)
    box(0.1, 3.3, 3.0, 1.8, "pump power,\nvacuum load,\nboiling flag", "#ffe3e3", 5.4)
    box(3.5, 2.55, 6.4, 1.3, "yield = ṁ t$_{sun}$ η$_{coll}$ → salt balance\n→ sizing, sensitivity, Monte Carlo", "#e6f4e6", 5.5)
    box(0.1, 0.1, 9.8, 2.1, "LEGACY solver (coupling = \"legacy\", kept for comparison):\nη$_{th}$ is an input; T$_g$ = T$_{amb}$ + 0.35 (T$_e$ − T$_{amb}$);\nyield = min(evaporation, condensation cap);\nVenturi is a diagnostic only", "#ffffff", 5.3, ls="--")
    arr((1.6, 9.7), (1.6, 9.35)); arr((3.15, 8.65), (3.45, 8.65)); arr((3.1, 6.65), (3.65, 6.4))
    arr((6.7, 4.35), (6.7, 3.9)); arr((1.6, 5.6), (1.6, 5.15))
    fig.savefig(os.path.join(OUT, "model_flow.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


# ---------- Fig: glass-to-evaporator ratio and delta (coupled vs legacy) ----------
def ratio():
    d = NUM["ratio_curves"]
    r = d["r"]
    rv = NUM["ratio"]["r_valid"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.axvspan(0.4, rv, color="#f3d6d6", alpha=0.7, lw=0)
    ax.text(0.42, 1.8, "cover would\nexceed boiling\n(steam venting)", fontsize=5.0, ha="left", va="bottom", color="#8a1c1c")
    ax.plot(r, d["legacy"], c="#7f7f7f", lw=1.2, ls="-.", label="legacy (decoupled cap)")
    ax.plot(r, d["A"], c="k", lw=1.0, ls="--", label="coupled, ideal evaporator (A)")
    ax.plot(r, d["B"], c="#999999", lw=1.0, ls=":", label="coupled, J$_{emp}$ cap (B)")
    for key, lab, col, lw in (("C_0.005", "diffusion (C), δ = 0.5 cm", "#9ec5e8", 1.2),
                              ("C_0.02", "diffusion (C), δ = 2 cm", "#1f6fb4", 1.8),
                              ("C_0.05", "diffusion (C), δ = 5 cm", "#0b3d6e", 1.2)):
        ax.plot(r, d[key], c=col, lw=lw, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.4, 20); ax.set_ylim(1.5, 11.0)
    ax.set_xlabel("Cover-to-evaporator area ratio A$_g$/A$_{geo}$", fontsize=6.5)
    ax.set_ylabel("Freshwater (L/m$^2$/day)", fontsize=6.5)
    ax.tick_params(labelsize=6, width=0.5, length=2); ax.grid(alpha=0.3, lw=0.4)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.legend(fontsize=5.1, frameon=False, loc="lower right", ncol=1, handlelength=2.2, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "ratio_curves.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


# ---------- Fig: Monte Carlo, both solvers ----------
def mc():
    import numpy as np
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    bins = np.linspace(0, 20, 81)
    for k, col in (("legacy", "#7f7f7f"), ("coupled", "#1f6fb4")):
        m = NUM["mc_" + k]
        ax.hist(m["samples"], bins=bins, color=col, alpha=0.65, lw=0,
                label=f"{k}: mean {m['mean']:.2f}, 5–95 % band {m['p05']:.1f}–{m['p95']:.1f}")
        ax.axvline(m["mean"], c=col, lw=0.9)
    ax.set_xlabel("Freshwater (L/m$^2$/day)", fontsize=6.5); ax.set_ylabel("Samples (N = 2000)", fontsize=6.5)
    ax.tick_params(labelsize=6, width=0.5, length=2); ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=5.4, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "mc_both.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


# ---------- Fig: software architecture ----------
def architecture():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, title, lines, fc, fs=5.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec="k", lw=0.6))
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center", fontsize=6.4, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.7) / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, linespacing=1.15)

    box(0.1, 7.9, 9.8, 2.0, "Streamlit dashboard (11 tabs, coupling toggle)",
        ["Module sizer, Cost & sustainability, OpenFOAM, Real-world limits, Results,",
         "Modes, Scaling, Sensitivity, Uncertainty, Validation, Parameters;",
         "header: T$_e$, T$_{gi}$, T$_c$, P$_{ch}$, vacuum load, boiling flag"], "#fff4d6", 5.2)
    box(0.1, 5.0, 3.1, 2.6, "physics/", ["coupled solver (default)", "legacy solver (frozen)", "ejector + cover load",
                                          "4 evaporation modes", "real-world derating"], "#eef3fa")
    box(3.45, 5.0, 3.1, 2.6, "scaling/", ["module sizer", "household / village", "tornado sensitivity", "Monte Carlo"], "#e6f4e6")
    box(6.8, 5.0, 3.1, 2.6, "sustainability/", ["bill of materials", "cost (learning curve)", "CO$_2$, green cover", "(placeholder inputs)"], "#fbe9e7")
    box(0.1, 1.9, 4.7, 2.7, "config/ registry", ["every parameter: symbol,", "unit, default, range,", "source tier (measured /", "literature / assumed)"], "#f3e8fb")
    box(5.2, 1.9, 4.7, 2.7, "openfoam/ and validation/", ["WSL OpenFOAM runner", "(heat transfer only)", "consistency checks", "(not validation); 310 tests"], "#eeeeee")
    ax.text(5.0, 0.7, "all modules use SI units; L/day only at display", ha="center", fontsize=5.8, style="italic")
    for x in (1.65, 5.0, 8.35):
        ax.add_patch(FancyArrowPatch((x, 7.62), (x, 7.88), arrowstyle="<|-", mutation_scale=6, lw=0.7))
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    schematic(); flow(); ratio(); mc(); architecture()
    print("ok")
