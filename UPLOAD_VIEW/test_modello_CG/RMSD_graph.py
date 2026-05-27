import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--RUN_TAG", required=True)
args = parser.parse_args()

context_string = "monomer_nocat"
base_root = "/home/stekajack/UPLOAD_VIEW"
run_dir = os.path.join(base_root, context_string, args.RUN_TAG)

txt_filename = os.path.join(run_dir, f"RMSD_{context_string}.txt")
print(f"Leggendo da: {txt_filename}")

data = np.loadtxt(txt_filename, skiprows=1)
frames = data[:, 0]
rmsd_values = data[:, 1]
time_us = frames * 200 * 0.003 / 1000  # in μs

fig, ax = plt.subplots(figsize=(10, 4))
#ax.axvline(x=20000 * 200 * 0.003 / 1000, color='red', linestyle='--', lw=1, label='switch T')
#ax.legend()
ax.plot(time_us, rmsd_values, lw=0.8, color='tomato')
ax.set_xlabel("Tempo (μs)")
ax.set_ylabel("RMSD (nm)")
ax.set_title(f"RMSD — {context_string} {args.RUN_TAG}")
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_filename = os.path.join(run_dir, f"RMSD_{context_string}_{args.RUN_TAG}.png")
plt.savefig(plot_filename, dpi=150)
print(f"Grafico salvato in {plot_filename}")