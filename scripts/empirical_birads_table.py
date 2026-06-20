"""
Verificacion empirica H2: tabla 5x5 de probabilidades medias predichas por exp08.

Filas  = BI-RADS verdadero (1-5)
Columnas = indice de salida del modelo (0-4)
Celda   = probabilidad media predicha en el split de TEST de VinDr

Si H2 (birads_to_index = b-1) es correcta, la diagonal debe tener los
valores mas altos en cada fila.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading import load_vindr_records, MammoCLIPTransform, MammoDataset
from models import MammoVLM

# --- Rutas ---
TESIS_ROOT = os.path.join(os.path.dirname(__file__), "..")
VINDR_ROOT = os.path.join(TESIS_ROOT, "data", "vindr-mammo")
MAMMOCLIP_CKPT = os.path.join(TESIS_ROOT, "models", "mammo_clip_b5.tar")
EXP08_CKPT = os.path.join(
    TESIS_ROOT,
    "outputs", "experiments",
    "exp08_ordinal_sord_qwk_descongelado",
    "model.pt",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4

print(f"Device: {DEVICE}")
print(f"Cargando registros VinDr desde {VINDR_ROOT} ...")

# --- Cargar registros y filtrar test split ---
all_records = load_vindr_records(VINDR_ROOT)
test_records = all_records[all_records["split"] == "test"].reset_index(drop=True)
print(f"Test records: {len(test_records)}")
print("Distribucion BI-RADS (test):")
print(test_records["birads"].value_counts().sort_index())

# --- Construir dataset y dataloader ---
transform = MammoCLIPTransform(augment=False)
dataset = MammoDataset(test_records, transform=transform, augment=False)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

print(f"\nCargando modelo exp08 desde {EXP08_CKPT} ...")

# --- Construir modelo con arquitectura exp08 ---
model = MammoVLM(
    checkpoint_path=MAMMOCLIP_CKPT,
    num_birads_classes=5,
    num_density_classes=4,
    freeze_encoder=False,
    unfreeze_last_n_blocks=2,
    hidden_dim=256,
    dropout=0.2,
)

# Cargar pesos del checkpoint exp08
ckpt = torch.load(EXP08_CKPT, map_location="cpu")
state_dict = ckpt["model_state_dict"]
missing, unexpected = model.load_state_dict(state_dict, strict=True)
print(f"  Pesos cargados: missing={missing}, unexpected={unexpected}")

model.to(DEVICE)
model.eval()

# --- Inferencia ---
print("\nCorriendo inferencia sobre el split de test ...")
all_probs = []   # [N, 5]
all_true  = []   # [N] indice verdadero (0-4)

with torch.no_grad():
    for batch in tqdm(loader, desc="Inferencia"):
        images = batch["image"].to(DEVICE)
        birads_true = batch["birads"].numpy()  # ya es indice 0-4

        outputs = model.forward(images)
        probs = torch.softmax(outputs["birads"], dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_true.append(birads_true)

all_probs = np.concatenate(all_probs, axis=0)  # [N, 5]
all_true  = np.concatenate(all_true, axis=0)   # [N]

print(f"\nTotal imagenes procesadas: {len(all_true)}")

# --- Tabla 5x5 ---
print("\n" + "="*72)
print("TABLA 5x5: probabilidad media predicha por clase verdadera")
print("Filas = BI-RADS verdadero (1-5) | Columnas = indice predicho (0-4)")
print("="*72)

header = f"{'BI-RADS':>8} | {'n':>5} | " + " ".join(f"idx{c}  " for c in range(5))
print(header)
print("-" * len(header))

table = np.zeros((5, 5))
counts = np.zeros(5, dtype=int)

for idx in range(5):  # true index 0-4 → BI-RADS idx+1
    mask = (all_true == idx)
    n = mask.sum()
    counts[idx] = n
    if n > 0:
        mean_probs = all_probs[mask].mean(axis=0)
        table[idx] = mean_probs
        row_str = f"{'BR ' + str(idx+1):>8} | {n:>5} | " + " ".join(f"{p:.4f}" for p in mean_probs)
        # Marcar el maximo con asterisco
        max_col = mean_probs.argmax()
        diag_ok = "  <-- diag OK" if max_col == idx else f"  <-- PICO en idx{max_col} !"
        print(row_str + diag_ok)
    else:
        print(f"{'BR ' + str(idx+1):>8} | {n:>5} | (sin muestras)")

print("="*72)

# --- Resumen de H2 ---
print("\nVEREDICTO H2 (birads_to_index = b-1 → diagonal):")
diagonal_is_max = all(
    (counts[i] == 0 or table[i].argmax() == i)
    for i in range(5)
)
if diagonal_is_max:
    print("  CONFIRMADA: la diagonal tiene el valor maximo en todas las filas.")
else:
    for i in range(5):
        if counts[i] > 0 and table[i].argmax() != i:
            print(f"  REFUTADA en BI-RADS {i+1}: pico en idx{table[i].argmax()} (esperado idx{i})")

print("\nHecho.")
