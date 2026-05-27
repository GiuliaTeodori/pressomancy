import h5py
import numpy as np
import matplotlib.pyplot as plt

# Lista dei file HDF5
file_paths = [
    "/home/stekajack/DATA/H5/modello_semirigido.h5",
    "/home/stekajack/DATA/H5/modello_semirigido_con_pi.h5",
    "/home/stekajack/DATA/H5/modello_semirigido_con_pi_20.h5"
]

# Etichette per il grafico
labels = ["no pi", "con pi 5.0",  "con pi 20.0"]

# ID delle particelle da tracciare
ID_i, ID_j = 8, 33

# Colori diversi per ogni linea
colors = ["blue", "green", "red"]

plt.figure(figsize=(10,6))

for fpath, label, color in zip(file_paths, labels, colors):
    with h5py.File(fpath, "r") as h5file:
        # Leggi posizioni e ID
        pos = h5file['particles/TelSeq/pos_folded/value'][:]
        ids_all = h5file['particles/TelSeq/id/value'][:]
        ids = ids_all[0, :, 0]  # prendi ID al primo step

        # Trova gli indici delle particelle
        i = np.where(ids == ID_i)[0][0]
        j = np.where(ids == ID_j)[0][0]

        # Calcola le distanze step per step
        pos_i = pos[:, i, :]
        pos_j = pos[:, j, :]
        distances = np.linalg.norm(pos_i - pos_j, axis=1)

        # Traccia nel grafico
        plt.plot(distances, label=label, color=color)

plt.xlabel("Step")
plt.ylabel("Distanza tra particelle")
plt.title(f"Distanza tra particelle {ID_i} e {ID_j} in più simulazioni")
plt.legend()
plt.tight_layout()
plt.savefig("distances_all_models_1.png")
plt.show()



# Lista dei file HDF5
file_paths = [
    "/home/stekajack/DATA/H5/modello_semirigido.h5",
    "/home/stekajack/DATA/H5/modello_semirigido_anglemaggiore.h5",
    "/home/stekajack/DATA/H5/modello_semirigido_anglemaggiore_pi5.h5"
]

# Etichette per il grafico
labels = [ "no pi, angle maggiore", "pi 5, angle maggiore"]

# ID delle particelle da tracciare
ID_i, ID_j = 8, 33

# Colori diversi per ogni linea
colors = [ "green", "red"]

plt.figure(figsize=(10,6))

for fpath, label, color in zip(file_paths, labels, colors):
    with h5py.File(fpath, "r") as h5file:
        # Leggi posizioni e ID
        pos = h5file['particles/TelSeq/pos_folded/value'][:]
        ids_all = h5file['particles/TelSeq/id/value'][:]
        ids = ids_all[0, :, 0]  # prendi ID al primo step

        # Trova gli indici delle particelle
        i = np.where(ids == ID_i)[0][0]
        j = np.where(ids == ID_j)[0][0]

        # Calcola le distanze step per step
        pos_i = pos[:, i, :]
        pos_j = pos[:, j, :]
        distances = np.linalg.norm(pos_i - pos_j, axis=1)

        # Traccia nel grafico
        plt.plot(distances, label=label, color=color)

plt.xlabel("Step")
plt.ylabel("Distanza tra particelle")
plt.title(f"Distanza tra particelle {ID_i} e {ID_j} in più simulazioni")
plt.legend()
plt.tight_layout()
plt.savefig("distances_all_models_2.png")
plt.show()