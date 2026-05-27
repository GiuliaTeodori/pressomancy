import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# ARGOMENTI
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--RUN_TAG", required=True)
args = parser.parse_args()

# =========================
# PARAMETRI TEMPORALI
# =========================

dt_md = 0.003          # ns (timestep simulazione)
save_stride = 100      # frame ogni 100 step

dt = dt_md * save_stride   # ns tra due frame salvati = 0.3 ns

# =========================
# PARAMETRI SISTEMA
# =========================
N_avog = 6.02214076e23
rho_si = 0.6 * N_avog
no_obj = 45
N = int(no_obj / 3)
vol = N / rho_si
box_l = pow(vol, 1/3)
_box_l = box_l / 0.4e-09
box_dim = _box_l * np.ones(3)

context_string = "15mer"
base_root = "/home/stekajack/UPLOAD_VIEW"
sim_root = os.path.join(base_root, context_string)
run_dir = os.path.join(sim_root, args.RUN_TAG)
plot_dir = os.path.join(run_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

h5_path = os.path.join(run_dir, "data.h5")

print(f"Leggendo da: {h5_path}")

exclude_types = [200, 27, 100, 24, 25]

# =========================
# FUNZIONI
# =========================

def calculate_CM(positions, box):
    CM = np.zeros(3)
    num_particles = len(positions)
    for axis in range(3):
        xi, zeta = 0.0, 0.0
        for pos in positions:
            pos_axis = pos[axis] + 0.5 * box[axis]
            theta = pos_axis * 2.0 * np.pi / box[axis]
            xi += np.cos(theta)
            zeta += np.sin(theta)
        xi /= num_particles
        zeta /= num_particles
        theta_mean = np.arctan2(-zeta, -xi) + np.pi
        CM[axis] = box[axis] * theta_mean / (2.0 * np.pi) - 0.5 * box[axis]
    return CM


def calculate_rg(positions, box):
    CM = calculate_CM(positions, box)
    pos_centered = np.array([
        pos - CM - box * np.round((pos - CM) / box)
        for pos in positions
    ])

    N = len(pos_centered)
    S = np.zeros((3, 3))

    for i in range(N):
        r = pos_centered[i].reshape(3, 1)
        S += np.dot(r, r.T)

    S /= N
    evals = np.sort(np.linalg.eigh(S)[0])

    Rg = np.sqrt(np.sum(evals)) * 0.4  # nm
    return Rg


def autocorrelation(x):
    x = x - np.mean(x)
    N = len(x)

    # Correlazione completa (lunghezza 2N-1)
    res = np.correlate(x, x, mode='full')

    # Teniamo solo la metà destra: lag 0, 1, 2, ...
    res = res[res.size // 2:]

    # CORREZIONE: dividi ogni lag tau per (N - tau), il numero di coppie
    # effettivamente disponibili. Senza questo, i lag grandi sono sovrastimati
    # rispetto a quelli piccoli (bias statistico).
    counts = np.arange(N, 0, -1)  # [N, N-1, N-2, ..., 1]
    res = res / counts

    # Normalizza a 1 al lag 0
    res = res / res[0]

    return res



def fit_tau(C, dt, max_lag_fraction=0.1):
    """
    Stima tau con fit esponenziale C(tau) = exp(-tau/tau_c).

    Parametri
    ----------
    C               : array normalizzato a 1 al lag 0
    dt              : intervallo temporale tra frame (ns)
    max_lag_fraction: frazione dei lag usata per il fit (default 10%)

    Ritorna
    -------
    tau_ns   : tempo di correlazione in ns
    tau_frame: tempo di correlazione in frame
    C_fit    : curva fittata sull'intero range di C
    """
    N = len(C)
    n_fit = max(10, int(N * max_lag_fraction))

    lags = np.arange(n_fit) * dt

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    # p0: stima iniziale grossolana con il threshold 1/e
    try:
        p0_index = np.where(C[:n_fit] < 1 / np.e)[0]
        p0 = [p0_index[0] * dt] if len(p0_index) > 0 else [lags[-1] / 2]
    except Exception:
        p0 = [lags[-1] / 2]

    popt, pcov = curve_fit(exp_decay, lags, C[:n_fit], p0=p0, maxfev=5000)
    tau_ns = popt[0]
    tau_frame = tau_ns / dt

    # Curva fittata su tutto il range per il plot
    t_full = np.arange(N) * dt
    C_fit = exp_decay(t_full, tau_ns)

    return tau_ns, tau_frame, C_fit


# =========================
# LETTURA DATI
# =========================

with h5py.File(h5_path, 'r') as f:
    positions_all = f['particles/TelSeq/pos/value'][:]
    types_all = f['particles/TelSeq/type/value'][:]

discard_frames = 0

positions_all = positions_all[discard_frames:]
types_all = types_all[discard_frames:]
total_frames = positions_all.shape[0]
print(f"Totale frame: {total_frames}")

# =========================
# Rg SU TUTTI I FRAME
# =========================

rg_list = []

for i in range(total_frames):
    types = types_all[i].flatten()
    positions = positions_all[i]

    mask = ~np.isin(types, exclude_types)
    positions_filtered = positions[mask]

    if len(positions_filtered) == 0:
        continue

    rg_list.append(calculate_rg(positions_filtered, box_dim))

    if i % 1000 == 0:
        print(f"Frame {i}/{total_frames}")

rg_array = np.array(rg_list)

plt.figure()
plt.plot(rg_array)
plt.xlabel("Frame")
plt.ylabel("Rg (nm)")
plt.title("Radius of gyration vs time")
plt.grid()
plt.savefig(os.path.join(plot_dir, "Rg_time.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\nRg medio grezzo (tutti i frame):")
print(f"{np.mean(rg_array):.6f} ± {np.std(rg_array):.6f} nm")

# =========================
# AUTOCORRELAZIONE GLOBALE (con funzione corretta)
# =========================

print("\nCalcolo autocorrelazione (versione corretta, unbiased)...")

C = autocorrelation(rg_array)
lags = np.arange(len(C))
time = lags * dt

# =========================
# STIMA tau: METODO 1 - threshold 1/e (come prima, ma su C corretta)
# =========================

threshold_indices = np.where(C < 1 / np.e)[0]
if len(threshold_indices) == 0:
    raise RuntimeError("La autocorrelazione non scende mai sotto 1/e. "
                       "Aumenta il numero di frame.")

tau_index_threshold = threshold_indices[0]
tau_time_threshold = tau_index_threshold * dt

print("\n--- Metodo threshold 1/e ---")
print(f"tau ≈ {tau_time_threshold:.4f} ns")
print(f"tau ≈ {tau_index_threshold} frame")

# =========================
# STIMA tau: METODO 2 - fit esponenziale (più robusto)
# =========================

print("\n--- Metodo fit esponenziale (su primi 10% dei lag) ---")
tau_fit_ns, tau_fit_frame, C_fit = fit_tau(C, dt, max_lag_fraction=0.1)
print(f"tau ≈ {tau_fit_ns:.4f} ns")
print(f"tau ≈ {tau_fit_frame:.1f} frame")
print(f"tau ≈ {tau_fit_frame * save_stride:.0f} step MD")

# Usiamo il tau dal fit (più robusto) per tutti i passi successivi
tau_index = int(round(tau_fit_frame))
tau_time = tau_fit_ns

# =========================
# EQUILIBRATION
# =========================

equil_index = int(2 * tau_index)
print(f"\nTaglio equilibration a frame: {equil_index} ({equil_index * dt:.2f} ns)")

rg_eq = rg_array[equil_index:]

# =========================
# AUTOCORR DOPO EQUILIBRIO
# =========================

C_eq = autocorrelation(rg_eq)
lags_eq = np.arange(len(C_eq))
time_eq = lags_eq * dt

# =========================
# SOTTOCAMPIONAMENTO
# =========================

step = max(1, tau_index)
rg_decorrelated = rg_eq[::step]

print("\n========== RISULTATO FINALE ==========")
print(f"Rg = {np.mean(rg_decorrelated):.6f} ± {np.std(rg_decorrelated):.6f} nm")
print(f"Campioni indipendenti: {len(rg_decorrelated)}")
print(f"tau usato (fit): {tau_fit_ns:.4f} ns = {tau_index} frame")
print("======================================")

# =========================
# PLOT AUTOCORRELAZIONE
# =========================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel sinistro: autocorr globale ---
ax = axes[0]
n_plot = len(C) // 5   # mostra solo il primo 20% per leggibilità

ax.plot(time[:n_plot], C[:n_plot], color='steelblue', label="Full (unbiased)")
ax.plot(time[:n_plot], C_fit[:n_plot], color='tomato', linestyle='--',
        label=f"Fit exp (τ = {tau_fit_ns:.3f} ns)")
ax.axhline(1 / np.e, linestyle=':', color='gray', label='1/e')
ax.axvline(tau_time_threshold, linestyle=':', color='orange',
           label=f"τ threshold = {tau_time_threshold:.3f} ns")
ax.axvline(tau_fit_ns, linestyle='--', color='tomato', alpha=0.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("C(τ)")
ax.set_title("Rg autocorrelation (full)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Panel destro: autocorr dopo equilibration ---
ax = axes[1]
n_plot_eq = len(C_eq) // 5

ax.plot(time_eq[:n_plot_eq], C_eq[:n_plot_eq], color='seagreen',
        label="After equilibration (unbiased)")
ax.axhline(1 / np.e, linestyle=':', color='gray', label='1/e')
ax.axvline(tau_fit_ns, linestyle='--', color='tomato',
           label=f"τ = {tau_fit_ns:.3f} ns")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("C(τ)")
ax.set_title("Rg autocorrelation (after equil.)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Rg_autocorrelation.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlot salvati in: {plot_dir}")