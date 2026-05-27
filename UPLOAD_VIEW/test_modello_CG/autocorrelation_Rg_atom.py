import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os

# =========================
# ARGOMENTI
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--run_tag", default="5mvb_400K")
parser.add_argument("--xvg", default=None,
                    help="Path al file .xvg (default: <run_tag>/rg_<run_tag>.xvg)")
args = parser.parse_args()

run_tag = args.run_tag
xvg_path = args.xvg if args.xvg else os.path.join(run_tag, f"rg_hybrid2_unf_400K.xvg")

plot_dir = run_tag
os.makedirs(plot_dir, exist_ok=True)

print(f"Leggendo: {xvg_path}")

# =========================
# LETTURA .XVG
# =========================

def read_xvg(path):
    """
    Legge un file .xvg GROMACS, salta le righe di header
    che iniziano con '#' o '@'.
    Ritorna array numpy con tutte le colonne numeriche.
    """
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('@'):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)

data = read_xvg(xvg_path)

# Colonna 0: tempo (ps), colonna 1: Rg totale (nm)
time_ps = data[:, 0]
rg = data[:, 1]
N = len(rg)
# =========================
# TAGLIO PRIMI 50 ns
# =========================
cutoff_ps = 50000  # 50 ns = 50000 ps

mask = time_ps >= cutoff_ps

time_ps = time_ps[mask]
rg = rg[mask]

# opzionale: resetta il tempo a partire da 0
time_ps = time_ps - time_ps[0]

N = len(rg)

print(f"Dopo taglio 50 ns:")
print(f"Frame rimasti: {N}")
print(f"Nuovo tempo totale: {time_ps[-1]/1000:.3f} ns")
print(f"Rg tagliato : {np.mean(rg):.6f} ± {np.std(rg):.6f} nm")
# dt tra frame in ps
dt = float(time_ps[1] - time_ps[0]) if N > 1 else 1.0

print(f"Frame totali : {N}")
print(f"dt           : {dt:.3f} ps")
print(f"Tempo totale : {time_ps[-1]:.1f} ps = {time_ps[-1]/1000:.3f} ns")
print(f"Rg grezzo    : {np.mean(rg):.6f} ± {np.std(rg):.6f} nm")

# =========================
# FUNZIONI
# =========================

def autocorrelation(x):
    """Autocorrelazione unbiased normalizzata a 1 al lag 0."""
    x = x - np.mean(x)
    n = len(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]           # metà destra (lag >= 0)
    counts = np.arange(n, 0, -1)                 # correzione unbiased
    result = result / counts
    result = result / result[0]
    return result


def fit_tau(C, dt, max_lag_fraction=0.1):
    """
    Fit esponenziale C(τ) = exp(-τ/τ_c) sul primo max_lag_fraction dei lag.
    Ritorna tau_ps, tau_frame, curva fittata sull'intero range.
    """
    n = len(C)
    n_fit = max(10, int(n * max_lag_fraction))
    lags = np.arange(n_fit) * dt

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    try:
        idx = np.where(C[:n_fit] < 1 / np.e)[0]
        p0 = [idx[0] * dt] if len(idx) > 0 else [lags[-1] / 2]
    except Exception:
        p0 = [lags[-1] / 2]

    popt, _ = curve_fit(exp_decay, lags, C[:n_fit], p0=p0, maxfev=10000)
    tau_ps = popt[0]
    tau_frame = tau_ps / dt

    t_full = np.arange(n) * dt
    C_fit = exp_decay(t_full, tau_ps)
    return tau_ps, tau_frame, C_fit

# =========================
# AUTOCORRELAZIONE GLOBALE
# =========================

print("\nCalcolo autocorrelazione (unbiased)...")
C = autocorrelation(rg)
lags = np.arange(len(C))
time_lag = lags * dt

# --- Metodo 1: threshold 1/e ---
thresh_idx = np.where(C < 1 / np.e)[0]
if len(thresh_idx) == 0:
    print("ATTENZIONE: C(τ) non scende mai sotto 1/e — aumenta il numero di frame.")
    tau_thresh_ps = None
else:
    tau_thresh_ps = thresh_idx[0] * dt
    print(f"\nMetodo threshold 1/e  : τ = {tau_thresh_ps:.2f} ps ({thresh_idx[0]} frame)")

# --- Metodo 2: fit esponenziale ---
tau_fit_ps, tau_fit_frame, C_fit = fit_tau(C, dt, max_lag_fraction=0.1)
print(f"Metodo fit esponenziale: τ = {tau_fit_ps:.2f} ps ({tau_fit_frame:.1f} frame)")

tau_index = int(round(tau_fit_frame))
tau_ps = tau_fit_ps

# =========================
# EQUILIBRATION (2 * tau)
# =========================

equil_index = int(2 * tau_index)
print(f"\nTaglio equilibration a frame {equil_index} ({equil_index * dt:.2f} ps)")

rg_eq = rg[equil_index:]

# =========================
# SOTTOCAMPIONAMENTO
# =========================

step = max(1, tau_index)
rg_decorrelated = rg_eq[::step]

print("\n========== RISULTATO FINALE ==========")
print(f"Rg = {np.mean(rg_decorrelated):.6f} ± {np.std(rg_decorrelated):.6f} nm")
print(f"Campioni indipendenti : {len(rg_decorrelated)}")
print(f"τ (fit)               : {tau_fit_ps:.2f} ps = {tau_fit_ps/1000:.4f} ns")
print("======================================")

# =========================
# AUTOCORR DOPO EQUILIBRIO
# =========================

C_eq = autocorrelation(rg_eq)
time_eq = np.arange(len(C_eq)) * dt

# =========================
# PLOT
# =========================

n_plot = max(1, len(C) // 5)       # primo 20% dei lag
n_plot_eq = max(1, len(C_eq) // 5)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# --- Panel 0: Rg vs time ---
ax = axes[0]
ax.plot(time_ps / 1000, rg, color='steelblue', linewidth=0.6, alpha=0.8)
if equil_index < N:
    ax.axvline(equil_index * dt / 1000, linestyle='--', color='tomato',
               label=f"Equilibration ({equil_index * dt / 1000:.2f} ns)")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Rg (nm)")
ax.set_title("Radius of gyration vs time")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Panel 1: autocorr globale ---
ax = axes[1]
ax.plot(time_lag[:n_plot] / 1000, C[:n_plot],
        color='steelblue', label="C(τ) unbiased")
ax.plot(time_lag[:n_plot] / 1000, C_fit[:n_plot],
        color='tomato', linestyle='--',
        label=f"Fit exp (τ = {tau_fit_ps:.1f} ps)")
ax.axhline(1 / np.e, linestyle=':', color='gray', label='1/e')
if tau_thresh_ps is not None:
    ax.axvline(tau_thresh_ps / 1000, linestyle=':', color='orange',
               label=f"τ thresh = {tau_thresh_ps:.1f} ps")
ax.axvline(tau_fit_ps / 1000, linestyle='--', color='tomato', alpha=0.5)
ax.set_xlabel("Lag τ (ns)")
ax.set_ylabel("C(τ)")
ax.set_title("Rg autocorrelation (full)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Panel 2: autocorr dopo equilibration ---
ax = axes[2]
ax.plot(time_eq[:n_plot_eq] / 1000, C_eq[:n_plot_eq],
        color='seagreen', label="After equilibration (unbiased)")
ax.axhline(1 / np.e, linestyle=':', color='gray', label='1/e')
ax.axvline(tau_fit_ps / 1000, linestyle='--', color='tomato',
           label=f"τ = {tau_fit_ps:.1f} ps")
ax.set_xlabel("Lag τ (ns)")
ax.set_ylabel("C(τ)")
ax.set_title("Rg autocorrelation (after equil.)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(plot_dir, f"Rg_autocorrelation_{run_tag}.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nPlot salvato in: {out_path}")