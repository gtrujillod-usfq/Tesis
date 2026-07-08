## evaluate_density_head.py
## Evaluacion de la cabeza de DENSIDAD ACR (4 clases: A, B, C, D) de exp08
## Nunca se persistio esta evaluacion: solo se reportaba BI-RADS (Area 3)
## No entrena ni modifica pesos: carga el checkpoint congelado y evalua
## Usa la misma logica de carga/preprocesamiento que la evaluacion de BI-RADS
## (main.ipynb celda 5.7 / 5.5), cambiando unicamente la cabeza objetivo

import os

## Fijar la GPU antes de importar torch. Servidor compartido con 4x H200:
## se usa la GPU con mas memoria libre en el momento de correr esto para no
## interferir con otros procesos.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

TESIS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TESIS_ROOT / "src"))

from models import MammoVLM
from data_loading import MammoCLIPTransform, MammoDataset
from medical_metrics import BIRADSClassificationMetrics

## ============================================================
## Constantes congeladas (alineadas con XAI/xai/config_xai.py,
## unica fuente de verdad de rutas y arquitectura de exp08)
## ============================================================

EXPERIMENT_FINAL = "exp08_ordinal_sord_qwk_descongelado"
MAMMOCLIP_CKPT = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT = str(TESIS_ROOT / "outputs" / "experiments" / EXPERIMENT_FINAL / "model.pt")
TEST_CSV = TESIS_ROOT / "outputs" / "test_sets" / "test_set_vindr.csv"
OUT_DIR = TESIS_ROOT / "results"

## Resolucion de entrada del encoder, confirmada en config_xai.py
IMAGE_HEIGHT = 1520
IMAGE_WIDTH = 912
## Bloques finales descongelados del encoder en exp08 (config_xai.py)
UNFREEZE_LAST_N_BLOCKS = 2

## Densidad ACR: indice 0-3 = DENSITY A-D (data_loading.py DENSITY_TO_INDEX)
DENSITY_LABELS = ["A", "B", "C", "D"]
NUM_DENSITY_CLASSES = 4

BATCH_SIZE = 8
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Esta evaluacion requiere GPU; no se detecto CUDA."


def git_hash() -> str:
    ## Hash del commit HEAD para trazabilidad del resultado
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(TESIS_ROOT)
        ).decode().strip()
    except Exception:
        return "N/A"


def file_sha256(path: str) -> str:
    ## Hash del checkpoint para confirmar que se evaluo exp08 y no otro experimento
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


CKPT_SHA256 = file_sha256(EXP08_CKPT)
COMMIT_HASH = git_hash()
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print("=" * 70)
print("EVALUACION CABEZA DE DENSIDAD ACR - exp08_ordinal_sord_qwk_descongelado")
print("=" * 70)
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"Checkpoint exp08: {EXP08_CKPT}")
print(f"SHA256 checkpoint: {CKPT_SHA256}")
print(f"Commit HEAD: {COMMIT_HASH}")

## ============================================================
## Cargar exp08: misma arquitectura y misma logica de carga que
## se usa para evaluar la cabeza BI-RADS (main.ipynb celda 7.2 / 5.5)
## ============================================================
vlm_model = MammoVLM(
    checkpoint_path=MAMMOCLIP_CKPT,
    efficientnet_name="efficientnet-b5",
    num_birads_classes=5,
    num_density_classes=4,
    freeze_encoder=True,
    unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
    hidden_dim=256,
    dropout=0.2,
)
ckpt = torch.load(EXP08_CKPT, map_location="cpu", weights_only=False)
missing, unexpected = vlm_model.load_state_dict(ckpt["model_state_dict"], strict=True)
assert not missing and not unexpected, f"claves no cargadas: missing={missing} unexpected={unexpected}"
vlm_model = vlm_model.to(DEVICE)
vlm_model.eval()
print("Checkpoint exp08 cargado correctamente (strict=True, sin claves faltantes)")

## ============================================================
## Test set de VinDr (4000 filas, mismo archivo que la evaluacion BI-RADS)
## ============================================================
test_records = pd.read_csv(TEST_CSV)
print(f"Test set VinDr: {len(test_records)} muestras ({TEST_CSV})")

## Verificar que ninguna fila tiene densidad faltante (IGNORE_INDEX)
n_sin_densidad = test_records["density_index"].isna().sum()
if n_sin_densidad > 0:
    print(f"AVISO: {n_sin_densidad} filas sin densidad valida; se excluyen de la evaluacion")

## Transform de evaluacion identico al de BI-RADS (alta resolucion, sin augmentation)
eval_transform = MammoCLIPTransform(
    height=IMAGE_HEIGHT, width=IMAGE_WIDTH, augment=False, use_clahe=True
)

test_ds = MammoDataset(test_records, eval_transform, augment=False)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

## ============================================================
## Inferencia: recolectar predicciones de la cabeza de DENSIDAD
## (misma logica que la cabeza BI-RADS, cambiando solo la clave del output)
## ============================================================
y_true, y_pred, y_probs = [], [], []

print("Evaluando cabeza de densidad sobre el test set...")
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images = batch["image"].to(DEVICE)
        outputs = vlm_model(images)
        probs = torch.softmax(outputs["density"], dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(batch["density"].numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy().tolist())

        if i % 100 == 0:
            print(f"  batch {i}/{len(test_loader)}", flush=True)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

## Descartar filas con densidad faltante (IGNORE_INDEX = -100), si las hay
valid_mask = y_true != MammoDataset.IGNORE_INDEX
n_descartadas = int((~valid_mask).sum())
if n_descartadas > 0:
    print(f"Descartadas {n_descartadas} filas sin densidad valida")
y_true = y_true[valid_mask]
y_pred = y_pred[valid_mask]
y_probs = y_probs[valid_mask]

print(f"Predicciones recolectadas: {len(y_true)}")

## ============================================================
## Metricas: reutiliza BIRADSClassificationMetrics (generica en num_classes,
## calcula accuracy, F1 macro/weighted, MCC, quadratic weighted kappa,
## matriz de confusion y AUC one-vs-rest), aplicada a 4 clases de densidad
## ============================================================
density_metrics = BIRADSClassificationMetrics(num_classes=NUM_DENSITY_CLASSES)
report = density_metrics.compute_all(y_true, y_pred, y_probs)

## Remapear las claves genericas "birads_{cls}" del AUC ovr a los labels ACR A-D
auc_ovr_raw = report["auc_ovr"]
auc_por_clase = {
    DENSITY_LABELS[cls]: auc_ovr_raw[f"birads_{cls}"] for cls in range(NUM_DENSITY_CLASSES)
}
auc_macro = auc_ovr_raw["macro_avg"]

## Soporte real por clase en el test (conteo de casos reales de A, B, C, D)
support_por_clase = {
    DENSITY_LABELS[cls]: int((y_true == cls).sum()) for cls in range(NUM_DENSITY_CLASSES)
}

## Matriz de confusion con filas/columnas etiquetadas A-D
confusion = report["confusion_matrix"]

print()
print("=" * 70)
print("RESULTADOS - CABEZA DE DENSIDAD ACR (4 clases)")
print("=" * 70)
print(f"AUC macro (one-vs-rest): {auc_macro:.4f}")
for label in DENSITY_LABELS:
    print(f"  AUC {label}: {auc_por_clase[label]:.4f}  (soporte={support_por_clase[label]})")
print()
print(f"Accuracy:              {report['accuracy']:.4f}")
print(f"F1 macro:               {report['macro_f1']:.4f}")
print(f"Quadratic weighted kappa: {report['quadratic_kappa']:.4f}")
print()
print("Matriz de confusion (filas=real, columnas=predicho), orden A,B,C,D:")
for label, row in zip(DENSITY_LABELS, confusion):
    print(f"  {label}: {row}")

## ============================================================
## Guardar JSON de resultados
## ============================================================
results = {
    "metadata": {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": TIMESTAMP,
        "commit": COMMIT_HASH,
        "device": DEVICE,
        "checkpoint_path": EXP08_CKPT,
        "checkpoint_sha256": CKPT_SHA256,
        "test_csv_path": str(TEST_CSV),
        "n_test_total": int(len(test_records)),
        "n_evaluated": int(len(y_true)),
        "n_descartadas_sin_densidad": n_descartadas,
        "class_labels": DENSITY_LABELS,
        "class_order": "0=DENSITY A, 1=DENSITY B, 2=DENSITY C, 3=DENSITY D (ACR, ordinal)",
        "input_resolution": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
    },
    "support_per_class": support_por_clase,
    "auc_macro_ovr": auc_macro,
    "auc_per_class": auc_por_clase,
    "accuracy": report["accuracy"],
    "macro_f1": report["macro_f1"],
    "weighted_f1": report["weighted_f1"],
    "mcc": report["mcc"],
    "quadratic_weighted_kappa": report["quadratic_kappa"],
    "confusion_matrix": {
        "order": DENSITY_LABELS,
        "matrix": confusion,
        "note": "filas=clase real, columnas=clase predicha",
    },
    "per_class_metrics_raw": report["per_class"],
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"density_head_evaluation_{EXPERIMENT_FINAL}_{TIMESTAMP}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print()
print(f"JSON guardado en: {out_path}")
