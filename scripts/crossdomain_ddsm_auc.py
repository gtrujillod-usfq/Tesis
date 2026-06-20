"""
Test cross-domain DDSM de exp08.
Disparo unico: sin iterar preprocesamiento ni umbral.

Umbral VinDr: 0.120 (Youden J sobre validacion de VinDr, congelado).
malignancy_score = birads_probs[3] + birads_probs[4]  (MALIGNANT_INDICES_5CLS = [3,4]).
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loading import load_image_as_pil, MammoCLIPTransform
from models import MammoVLM
from ddsm_overlay import load_ddsm_records

TESIS_ROOT     = Path(__file__).parent.parent
DDSM_ROOT      = TESIS_ROOT / "data" / "6 DDSM"
MAMMOCLIP_CKPT = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT     = str(TESIS_ROOT / "outputs" / "experiments" /
                     "exp08_ordinal_sord_qwk_descongelado" / "model.pt")
OUT_DIR        = TESIS_ROOT / "outputs" / "crossdomain_ddsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 16
VINDR_THRESHOLD = 0.120    # congelado, no se toca
MALIGNANT_IDX   = [3, 4]   # MALIGNANT_INDICES_5CLS

# ---------------------------------------------------------------------------
# Confirmar constantes antes de cualquier calculo
# ---------------------------------------------------------------------------
print("=" * 70)
print("CONFIRMACION DE PARAMETROS")
print("=" * 70)
print(f"  Checkpoint    : {EXP08_CKPT}")
print(f"  Umbral VinDr  : {VINDR_THRESHOLD}  (congelado, Youden J validacion VinDr)")
print(f"  malignancy_score = birads_probs[{MALIGNANT_IDX[0]}] + birads_probs[{MALIGNANT_IDX[1]}]")
print(f"  MALIGNANT_INDICES_5CLS = {MALIGNANT_IDX}")
print(f"  Device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Cargar registros DDSM y agregar a nivel de imagen
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("CARGA DE DATOS")
print("=" * 70)

import logging; logging.disable(logging.WARNING)
lesion_records = load_ddsm_records(DDSM_ROOT)
logging.disable(logging.NOTSET)
print(f"  Registros de lesion cargados: {len(lesion_records)}")

n_sin_imagen = sum(1 for r in lesion_records if r["image_path"] is None)
records_con  = [r for r in lesion_records if r["image_path"] is not None]
print(f"  Sin imagen (sin overlay match): {n_sin_imagen}")
print(f"  Con imagen: {len(records_con)}")

# Agregar a nivel de imagen: regla de peor lesion
img_agg = defaultdict(lambda: {
    "pathology": "BENIGN", "case_id": None,
    "prefix": None, "image_path": None
})
for r in records_con:
    img = str(r["image_path"])
    img_agg[img]["image_path"] = r["image_path"]
    img_agg[img]["case_id"]    = r["case_id"]
    img_agg[img]["prefix"]     = Path(img).name.split("_")[0]
    if "MALIGNANT" in r["pathology"].upper():
        img_agg[img]["pathology"] = "MALIGNANT"

df = pd.DataFrame([
    {"image_path": k, "pathology": v["pathology"],
     "prefix": v["prefix"], "case_id": v["case_id"]}
    for k, v in img_agg.items()
])
df = df[df["pathology"].isin(["MALIGNANT", "BENIGN"])].reset_index(drop=True)
n_mal = (df["pathology"] == "MALIGNANT").sum()
n_ben = (df["pathology"] == "BENIGN").sum()

print(f"\n  Imagenes para inferencia:")
print(f"    MALIGNANT : {n_mal}")
print(f"    BENIGN    : {n_ben}")
print(f"    TOTAL     : {len(df)}")
print(f"    Excluidas por UNKNOWN/fallo de parser: 0 (todas son MAL o BEN)")
print(f"\n  Prefijos presentes: {sorted(df['prefix'].unique())}")
for pref in sorted(df['prefix'].unique()):
    sub = df[df["prefix"] == pref]
    print(f"    {pref}: {len(sub)} imgs  "
          f"({(sub['pathology']=='MALIGNANT').sum()} MAL  "
          f"{(sub['pathology']=='BENIGN').sum()} BEN)")

# ---------------------------------------------------------------------------
# 2. Cargar modelo exp08
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("CARGA DEL MODELO")
print("=" * 70)

model = MammoVLM(
    checkpoint_path=MAMMOCLIP_CKPT,
    num_birads_classes=5, num_density_classes=4,
    freeze_encoder=False, unfreeze_last_n_blocks=2,
    hidden_dim=256, dropout=0.2,
)
ckpt = torch.load(EXP08_CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
print(f"  Pesos cargados: missing={missing}  unexpected={unexpected}")
model.to(DEVICE)
model.eval()

transform = MammoCLIPTransform(augment=False, use_clahe=True)

# ---------------------------------------------------------------------------
# 3. Inferencia por batches
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("INFERENCIA")
print("=" * 70)

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DDSMSimpleDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            pil = load_image_as_pil(str(self.paths[i]))
            return self.transform(pil), True
        except Exception:
            return torch.zeros(3, 1520, 912), False

dataset = DDSMSimpleDataset(df["image_path"].tolist(), transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=4, pin_memory=(DEVICE=="cuda"))

all_probs = []
all_ok    = []

with torch.no_grad():
    for imgs, ok_flags in tqdm(loader, desc="Inferencia"):
        imgs  = imgs.to(DEVICE)
        out   = model.forward(imgs)
        probs = F.softmax(out["birads"], dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_ok.append(ok_flags.numpy())

all_probs = np.concatenate(all_probs, axis=0)   # [N, 5]
all_ok    = np.concatenate(all_ok).astype(bool)

n_fail_load = int((~all_ok).sum())
print(f"\n  Total procesadas : {len(all_ok)}")
print(f"  Fallos de carga  : {n_fail_load}  (imagenes con tensor cero — excluidas del AUC)")

# Excluir fallos de carga
df_ok   = df[all_ok].reset_index(drop=True)
probs_ok = all_probs[all_ok]

malignancy_scores = probs_ok[:, MALIGNANT_IDX[0]] + probs_ok[:, MALIGNANT_IDX[1]]
y_true = (df_ok["pathology"] == "MALIGNANT").astype(int).values

print(f"\n  Imagenes validas para metricas: {len(df_ok)}")
print(f"    MALIGNANT: {y_true.sum()}")
print(f"    BENIGN   : {(1-y_true).sum()}")

# ---------------------------------------------------------------------------
# 4. Sanity de la distribucion del malignancy_score
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SANITY DEL MALIGNANCY_SCORE")
print("=" * 70)

q = np.percentile(malignancy_scores, [0, 25, 50, 75, 100])
print(f"  N = {len(malignancy_scores)}")
print(f"  Media    : {malignancy_scores.mean():.4f}")
print(f"  Mediana  : {np.median(malignancy_scores):.4f}")
print(f"  Std      : {malignancy_scores.std():.4f}")
print(f"  Min      : {q[0]:.4f}")
print(f"  Q1       : {q[1]:.4f}")
print(f"  Q2/Med   : {q[2]:.4f}")
print(f"  Q3       : {q[3]:.4f}")
print(f"  Max      : {q[4]:.4f}")

# Verificar degeneracion
is_degenerate = malignancy_scores.std() < 0.01
print(f"\n  Degenerado (std < 0.01): {'SI — PARAR' if is_degenerate else 'NO'}")
if is_degenerate:
    print("  ERROR: scores casi identicos — el arreglo no llego a estas imagenes.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. AUC-ROC con IC95 bootstrap
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

print()
print("=" * 70)
print("METRICA TITULAR — AUC-ROC")
print("=" * 70)

auc_point = roc_auc_score(y_true, malignancy_scores)
print(f"  AUC-ROC punto    : {auc_point:.4f}")

# Bootstrap IC95
N_BOOT = 2000
rng = np.random.default_rng(42)
boot_aucs = []
for _ in range(N_BOOT):
    idx = rng.integers(0, len(y_true), len(y_true))
    yt  = y_true[idx]
    ys  = malignancy_scores[idx]
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue
    boot_aucs.append(roc_auc_score(yt, ys))
boot_aucs = np.array(boot_aucs)
ci_lo = float(np.percentile(boot_aucs, 2.5))
ci_hi = float(np.percentile(boot_aucs, 97.5))
print(f"  IC95 bootstrap   : [{ci_lo:.4f}, {ci_hi:.4f}]  (N_boot={N_BOOT})")
print(f"  AUC-ROC final    : {auc_point:.4f}  [{ci_lo:.4f} – {ci_hi:.4f}]")

# Average Precision
ap = average_precision_score(y_true, malignancy_scores)
print(f"\n  Average Precision (AUC-PR): {ap:.4f}")

# Curva ROC — guardar PNG
fpr, tpr, thrs = roc_curve(y_true, malignancy_scores)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, color="steelblue",
        label=f"exp08 cross-domain DDSM\nAUC = {auc_point:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]")
ax.plot([0,1],[0,1], "k--", lw=1, label="Azar (AUC=0.50)")
# Punto de operacion VinDr
thr_idx = np.searchsorted(np.sort(malignancy_scores)[::-1], VINDR_THRESHOLD)
from sklearn.metrics import confusion_matrix
cm_vindr = confusion_matrix(y_true, (malignancy_scores >= VINDR_THRESHOLD).astype(int))
tn_v, fp_v, fn_v, tp_v = cm_vindr.ravel()
sens_v = tp_v / max(tp_v + fn_v, 1)
spec_v = tn_v / max(tn_v + fp_v, 1)
ax.scatter([1-spec_v], [sens_v], color="red", zorder=5, s=80,
           label=f"Umbral VinDr={VINDR_THRESHOLD:.3f}\nSens={sens_v:.3f}  Spec={spec_v:.3f}")
ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)", fontsize=11)
ax.set_ylabel("Sensibilidad (tasa de verdaderos positivos)", fontsize=11)
ax.set_title("Curva ROC — exp08 cross-domain DDSM\n(entrenado solo en VinDr)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
roc_path = OUT_DIR / "roc_crossdomain_ddsm.png"
fig.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Curva ROC guardada: {roc_path}")

# ---------------------------------------------------------------------------
# 6. Metricas secundarias — umbral VinDr transferido
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("METRICA SECUNDARIA — UMBRAL VINDR TRANSFERIDO (0.120)")
print("CAVEAT: transferencia de punto de operacion, NO calibracion")
print("=" * 70)

print(f"\n  Umbral aplicado : {VINDR_THRESHOLD}")
print(f"  Confusion matrix:")
print(f"    TN={tn_v}  FP={fp_v}")
print(f"    FN={fn_v}  TP={tp_v}")
print(f"  Sensibilidad    : {sens_v:.4f}")
print(f"  Especificidad   : {spec_v:.4f}")
print(f"  VPP (PPV)       : {tp_v/max(tp_v+fp_v,1):.4f}")
print(f"  VPN (NPV)       : {tn_v/max(tn_v+fn_v,1):.4f}")
print(f"  Youden J        : {sens_v + spec_v - 1:.4f}")

# Distribucion del score por clase
scores_mal = malignancy_scores[y_true == 1]
scores_ben = malignancy_scores[y_true == 0]
print(f"\n  Distribucion malignancy_score por clase:")
print(f"  {'':10s}  {'BENIGN':>12}  {'MALIGNANT':>12}")
for stat_name, fn in [("media", np.mean), ("mediana", np.median),
                       ("Q1", lambda x: np.percentile(x,25)),
                       ("Q3", lambda x: np.percentile(x,75)),
                       ("min", np.min), ("max", np.max)]:
    print(f"  {stat_name:10s}  {fn(scores_ben):>12.4f}  {fn(scores_mal):>12.4f}")

# ---------------------------------------------------------------------------
# 7. Umbral optimo EN DDSM (solo diagnostico de drift, NO metrica)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("DIAGNOSTICO DE DRIFT — umbral optimo EN DDSM")
print("(Solo descriptivo. NO se usa como punto de operacion.)")
print("=" * 70)

thrs_sweep = np.linspace(0.0, 1.0, 1001)
best_j, best_thr_ddsm = -1, None
for t in thrs_sweep:
    pred = (malignancy_scores >= t).astype(int)
    if pred.sum() == 0 or pred.sum() == len(pred):
        continue
    cm = confusion_matrix(y_true, pred)
    if cm.shape != (2,2): continue
    tn_t, fp_t, fn_t, tp_t = cm.ravel()
    s = tp_t / max(tp_t+fn_t, 1)
    sp = tn_t / max(tn_t+fp_t, 1)
    j = s + sp - 1
    if j > best_j:
        best_j, best_thr_ddsm = j, t

print(f"  Umbral VinDr (operacion)   : {VINDR_THRESHOLD:.3f}")
print(f"  Umbral optimo DDSM (Youden): {best_thr_ddsm:.3f}  (solo diagnostico, no reportado)")
print(f"  Diferencia (drift de umbral): {best_thr_ddsm - VINDR_THRESHOLD:+.3f}")

# ---------------------------------------------------------------------------
# 8. AUC por volumen (prefijo A/B/C/D)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("ROBUSTEZ — AUC POR VOLUMEN (prefijo)")
print("=" * 70)

for pref in sorted(df_ok["prefix"].unique()):
    mask = df_ok["prefix"].values == pref
    yt_p = y_true[mask]
    ys_p = malignancy_scores[mask]
    n_p  = mask.sum()
    if yt_p.sum() == 0 or yt_p.sum() == n_p:
        print(f"  Prefijo {pref}: N={n_p}  MAL={yt_p.sum()}  — sin varianza, AUC indefinido")
        continue
    auc_p = roc_auc_score(yt_p, ys_p)
    print(f"  Prefijo {pref}: N={n_p}  "
          f"MAL={yt_p.sum()}  BEN={n_p-yt_p.sum()}  "
          f"AUC={auc_p:.4f}")

# ---------------------------------------------------------------------------
# Guardar resultados en CSV
# ---------------------------------------------------------------------------
results_df = df_ok.copy()
results_df["malignancy_score"] = malignancy_scores
results_df["y_true"] = y_true
results_df["pred_vindr_thr"] = (malignancy_scores >= VINDR_THRESHOLD).astype(int)
results_df.to_csv(OUT_DIR / "predictions.csv", index=False)
print(f"\n  Predicciones guardadas: {OUT_DIR}/predictions.csv")

print()
print("=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"  N total inferidos  : {len(df_ok)}  (MAL={y_true.sum()}  BEN={(1-y_true).sum()})")
print(f"  Fallos de carga    : {n_fail_load}")
print(f"  AUC-ROC            : {auc_point:.4f}  IC95=[{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Average Precision  : {ap:.4f}")
print(f"  Umbral VinDr 0.120 : Sens={sens_v:.4f}  Spec={spec_v:.4f}  "
      f"TP={tp_v}  FP={fp_v}  TN={tn_v}  FN={fn_v}")
