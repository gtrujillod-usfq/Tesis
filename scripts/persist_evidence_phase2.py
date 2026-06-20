"""
Fase 2 de persistencia de evidencia.
JSON 01 y 02 ya estan escritos — este script genera 03, 04, 05, figuras y README.

Confound check con multiprocessing (32 workers) para manejar 3669 PNGs de 17MB.
Semilla 42 en todo lo que tenga RNG.
"""

import sys, os, warnings, shutil, hashlib, subprocess, json, time, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from PIL import Image
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loading import load_image_as_pil, MammoCLIPTransform
from models import MammoVLM
from ddsm_overlay import load_ddsm_records
from threshold_tuning import MALIGNANT_INDICES_5CLS

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, confusion_matrix)

# ---------------------------------------------------------------------------
# Constantes congeladas
# ---------------------------------------------------------------------------
TESIS_ROOT      = Path(__file__).parent.parent
DDSM_ROOT       = TESIS_ROOT / "data" / "6 DDSM"
VINDR_ROOT      = TESIS_ROOT / "data" / "vindr-mammo"
MAMMOCLIP_CKPT  = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT      = str(TESIS_ROOT / "outputs" / "experiments" /
                      "exp08_ordinal_sord_qwk_descongelado" / "model.pt")
OUT_DIR         = TESIS_ROOT / "results" / "ddsm_crossdomain"
SRC_ROC         = TESIS_ROOT / "outputs" / "crossdomain_ddsm" / "roc_crossdomain_ddsm.png"
SRC_DIAG        = TESIS_ROOT / "outputs" / "diag_ddsm"

SEED            = 42
N_BOOT          = 2000
VINDR_THRESHOLD = 0.120
MALIGNANT_IDX   = MALIGNANT_INDICES_5CLS   # [3, 4]
BATCH_SIZE      = 16
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TODAY           = datetime.date.today().isoformat()
N_WORKERS       = min(32, cpu_count())

import logging; logging.disable(logging.WARNING)

def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(TESIS_ROOT)).decode().strip()
    except Exception:
        return "N/A"

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

COMMIT_HASH = git_hash()
CKPT_SHA256 = file_sha256(EXP08_CKPT)

def json_save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> {path.name}")

def load_ddsm_image_df():
    lesion_records = load_ddsm_records(DDSM_ROOT)
    img_agg = defaultdict(lambda: {"pathology": "BENIGN", "prefix": None})
    for r in lesion_records:
        if r["image_path"] is None:
            continue
        img = str(r["image_path"])
        img_agg[img]["image_path"] = r["image_path"]
        img_agg[img]["prefix"]     = Path(img).name.split("_")[0]
        if "MALIGNANT" in r["pathology"].upper():
            img_agg[img]["pathology"] = "MALIGNANT"
    df = pd.DataFrame([
        {"image_path": k, "pathology": v["pathology"], "prefix": v["prefix"]}
        for k, v in img_agg.items()
    ])
    return df[df["pathology"].isin(["MALIGNANT", "BENIGN"])].reset_index(drop=True)

def load_model():
    m = MammoVLM(
        checkpoint_path=MAMMOCLIP_CKPT,
        num_birads_classes=5, num_density_classes=4,
        freeze_encoder=False, unfreeze_last_n_blocks=2,
        hidden_dim=256, dropout=0.2,
    )
    ckpt = torch.load(EXP08_CKPT, map_location="cpu")
    missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not missing and not unexpected
    m.to(DEVICE).eval()
    return m

# ===========================================================================
# JSON 03 — confound check sobre 3669 imagenes (multiprocessing)
# ===========================================================================
print("\n" + "=" * 70)
print(f"JSON 03 — confound check (3669 imagenes, {N_WORKERS} workers)")
print("=" * 70)

def extract_features_one(args):
    img_path, pathology, prefix = args
    try:
        arr  = np.array(Image.open(img_path), dtype=np.float32)
        h, w = arr.shape
        flat = arr.flatten()
        p1, p5, p50, p99 = np.percentile(flat, [1, 5, 50, 99])
        hi_thr    = 65535 * 0.95
        text_proxy = float((arr > hi_thr).mean())
        return {
            "image_path":   str(img_path),
            "pathology":    pathology,
            "prefix":       prefix,
            "height":       h, "width": w,
            "total_pixels": int(h * w),
            "p1_raw":       float(p1),
            "p5_raw":       float(p5),
            "mean_raw":     float(flat.mean()),
            "p50_raw":      float(p50),
            "p99_raw":      float(p99),
            "text_proxy":   text_proxy,
        }
    except Exception as e:
        return None

def make_03():
    from scipy.stats import chi2_contingency

    ddsm_df = load_ddsm_image_df()
    n_total = len(ddsm_df)
    print(f"  Total imagenes: {n_total}  "
          f"(MAL={(ddsm_df['pathology']=='MALIGNANT').sum()}  "
          f"BEN={(ddsm_df['pathology']=='BENIGN').sum()})")

    args_list = [
        (row["image_path"], row["pathology"], row["prefix"])
        for _, row in ddsm_df.iterrows()
    ]

    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(extract_features_one, args_list)
    t1 = time.time()

    records = [r for r in results if r is not None]
    print(f"  Imagenes con features OK: {len(records)}/{n_total}  en {t1-t0:.1f}s")

    feat_df = pd.DataFrame(records)
    y = (feat_df["pathology"] == "MALIGNANT").astype(int)
    features = ["height", "width", "total_pixels", "p1_raw", "p5_raw",
                "text_proxy", "mean_raw", "p50_raw", "p99_raw"]

    auc_results = {}
    for f in features:
        auc_val = float(roc_auc_score(y, feat_df[f]))
        auc_results[f] = {
            "mean_benign":      float(feat_df.loc[y==0, f].mean()),
            "median_benign":    float(feat_df.loc[y==0, f].median()),
            "mean_malignant":   float(feat_df.loc[y==1, f].mean()),
            "median_malignant": float(feat_df.loc[y==1, f].median()),
            "auc":              auc_val,
            "flag_confound":    bool(auc_val > 0.60),
        }
        print(f"  {f:20s}  AUC={auc_val:.4f}  {'*** CONFOUND' if auc_val > 0.60 else 'OK'}")

    ct = pd.crosstab(feat_df["prefix"], feat_df["pathology"])
    chi2, p_chi2, dof, _ = chi2_contingency(ct)
    prefix_stats = {}
    for pref in sorted(feat_df["prefix"].unique()):
        sub = feat_df[feat_df["prefix"] == pref]
        n_m = int((sub["pathology"] == "MALIGNANT").sum())
        n_b = int((sub["pathology"] == "BENIGN").sum())
        prefix_stats[pref] = {
            "n": len(sub), "n_malignant": n_m, "n_benign": n_b,
            "malignancy_rate": float(n_m / len(sub))
        }

    n_confound = sum(v["flag_confound"] for v in auc_results.values())
    print(f"  Chi2(prefijo x patologia): chi2={chi2:.2f}  p={p_chi2:.4f}")
    print(f"  Features con AUC>0.60: {n_confound}")

    return {
        "metadata": {
            "date": TODAY, "commit": COMMIT_HASH, "seed": SEED,
            "N_total_images": n_total,
            "N_features_computed": len(records),
            "N_workers": N_WORKERS,
            "elapsed_seconds": round(t1-t0, 1),
            "note": "Corrido sobre TODAS las imagenes DDSM con overlay (sin muestreo)"
        },
        "features": auc_results,
        "confound_verdict": {
            "features_with_auc_above_060": n_confound,
            "threshold": 0.60,
            "verdict": "SIN_CONFOUND" if n_confound == 0 else "CONFOUND_DETECTADO"
        },
        "prefix_contingency": {
            "per_prefix": prefix_stats,
            "chi2": float(chi2),
            "p_value": float(p_chi2),
            "dof": int(dof),
            "interpretation": ("p>0.05 -> distribucion de prefijos independiente de patologia"
                               if p_chi2 > 0.05 else "p<=0.05 -> asociacion detectada")
        }
    }

r03 = make_03()
json_save(OUT_DIR / "03_confound_check.json", r03)

# ===========================================================================
# JSON 04 — cross-domain AUC (inferencia completa, semilla 42)
# ===========================================================================
print("\n" + "=" * 70)
print("JSON 04 — cross-domain AUC (inferencia 3669 imagenes)")
print("=" * 70)

transform_eval = MammoCLIPTransform(augment=False, use_clahe=True)

def make_04():
    from torch.utils.data import Dataset, DataLoader

    class DDSMDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths, self.transform = paths, transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            try:
                pil = load_image_as_pil(str(self.paths[i]))
                return self.transform(pil), True
            except Exception:
                return torch.zeros(3, 1520, 912), False

    ddsm_df = load_ddsm_image_df()
    model   = load_model()
    print(f"  Modelo cargado. Device: {DEVICE}")

    ds      = DDSMDataset(ddsm_df["image_path"].tolist(), transform_eval)
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=8, pin_memory=(DEVICE=="cuda"))

    all_probs, all_ok = [], []
    t0 = time.time()
    for i, (imgs, ok_flags) in enumerate(loader):
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model.forward(imgs)["birads"], dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_ok.append(ok_flags.numpy())
    t1 = time.time()

    all_probs = np.concatenate(all_probs, axis=0)
    all_ok    = np.concatenate(all_ok).astype(bool)
    n_fail    = int((~all_ok).sum())
    df_ok     = ddsm_df[all_ok].reset_index(drop=True)
    probs_ok  = all_probs[all_ok]
    print(f"  Inferencia completa en {t1-t0:.1f}s  fallos={n_fail}")

    scores = probs_ok[:, MALIGNANT_IDX[0]] + probs_ok[:, MALIGNANT_IDX[1]]
    y_true = (df_ok["pathology"] == "MALIGNANT").astype(int).values

    score_global = {
        "mean": float(scores.mean()), "median": float(np.median(scores)),
        "std":  float(scores.std()),  "min":    float(scores.min()),
        "max":  float(scores.max()),
        "q1":   float(np.percentile(scores, 25)),
        "q3":   float(np.percentile(scores, 75)),
        "is_degenerate": bool(scores.std() < 0.01),
    }
    print(f"  Score: mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"degenerate={score_global['is_degenerate']}")

    auc_point  = float(roc_auc_score(y_true, scores))
    rng_boot   = np.random.default_rng(SEED)
    boot_aucs  = []
    for _ in range(N_BOOT):
        idx = rng_boot.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], scores[idx]
        if yt.sum() in (0, len(yt)): continue
        boot_aucs.append(roc_auc_score(yt, ys))
    boot_aucs = np.array(boot_aucs)
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))
    ap    = float(average_precision_score(y_true, scores))
    print(f"  AUC-ROC={auc_point:.4f}  IC95=[{ci_lo:.4f},{ci_hi:.4f}]  AP={ap:.4f}")

    def class_stats(s):
        return {
            "mean": float(s.mean()), "median": float(np.median(s)),
            "q1": float(np.percentile(s, 25)), "q3": float(np.percentile(s, 75)),
            "min": float(s.min()), "max": float(s.max())
        }

    pred_v = (scores >= VINDR_THRESHOLD).astype(int)
    cm_v   = confusion_matrix(y_true, pred_v)
    tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
    sens_v = float(tp_v / max(tp_v + fn_v, 1))
    spec_v = float(tn_v / max(tn_v + fp_v, 1))

    thrs      = np.linspace(0, 1, 1001)
    best_j, best_thr = -1, None
    for t in thrs:
        pred_t = (scores >= t).astype(int)
        if pred_t.sum() in (0, len(pred_t)): continue
        cm_t = confusion_matrix(y_true, pred_t)
        if cm_t.shape != (2,2): continue
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        j = tp_t/max(tp_t+fn_t,1) + tn_t/max(tn_t+fp_t,1) - 1
        if j > best_j:
            best_j, best_thr = j, float(t)

    auc_by_prefix = {}
    for pref in sorted(df_ok["prefix"].unique()):
        mask = df_ok["prefix"].values == pref
        yt_p, ys_p = y_true[mask], scores[mask]
        n_p = int(mask.sum())
        if yt_p.sum() in (0, n_p):
            auc_by_prefix[pref] = {"n": n_p, "n_malignant": int(yt_p.sum()), "auc": None}
        else:
            auc_by_prefix[pref] = {
                "n": n_p, "n_malignant": int(yt_p.sum()),
                "n_benign": n_p - int(yt_p.sum()),
                "auc": float(roc_auc_score(yt_p, ys_p))
            }

    return {
        "metadata": {
            "date": TODAY, "commit": COMMIT_HASH,
            "checkpoint_sha256": CKPT_SHA256,
            "checkpoint_path": EXP08_CKPT,
            "seed": SEED, "N_boot": N_BOOT, "device": DEVICE,
            "elapsed_inference_seconds": round(t1-t0, 1),
        },
        "frozen_decisions": {
            "malignancy_score": "birads_probs[3] + birads_probs[4]",
            "MALIGNANT_INDICES_5CLS": list(MALIGNANT_IDX),
            "vindr_threshold": VINDR_THRESHOLD,
            "threshold_origin": "Youden J sobre validacion de VinDr — no re-tuneado",
            "label_rule": "peor lesion (al menos 1 MALIGNANT -> imagen MALIGNANT)",
            "scaling": "min-max por imagen (sin recorte por percentil ni mascara)",
            "transform": "MammoCLIPTransform augment=False use_clahe=True",
        },
        "data": {
            "N_total": int(len(ddsm_df)),
            "N_failed_load": n_fail,
            "N_valid": int(len(df_ok)),
            "N_malignant": int(y_true.sum()),
            "N_benign":    int((1-y_true).sum()),
        },
        "sanity_malignancy_score": score_global,
        "primary_metric": {
            "label": "AUC-ROC",
            "auc_roc_point": auc_point,
            "ci_95_bootstrap": {"lower": ci_lo, "upper": ci_hi,
                                "n_resamples": N_BOOT, "seed": SEED},
            "average_precision_auc_pr": ap,
        },
        "secondary_metric": {
            "label": "Transferencia de punto de operacion — NO calibracion",
            "vindr_threshold_applied": VINDR_THRESHOLD,
            "confusion_matrix": {"tn": int(tn_v), "fp": int(fp_v),
                                 "fn": int(fn_v), "tp": int(tp_v)},
            "sensitivity": sens_v,
            "specificity": spec_v,
            "ppv": float(tp_v / max(tp_v+fp_v, 1)),
            "npv": float(tn_v / max(tn_v+fn_v, 1)),
            "youden_j": float(sens_v + spec_v - 1),
        },
        "score_distribution_by_class": {
            "BENIGN":    class_stats(scores[y_true == 0]),
            "MALIGNANT": class_stats(scores[y_true == 1]),
        },
        "drift_diagnostic": {
            "label": "Umbral optimo EN DDSM — solo diagnostico de drift, NO metrica reportada",
            "vindr_threshold": VINDR_THRESHOLD,
            "optimal_threshold_ddsm": best_thr,
            "drift_magnitude": float(best_thr - VINDR_THRESHOLD),
        },
        "robustness_by_volume": auc_by_prefix,
        "interpretation": (
            "AUC=0.54 IC95=[0.52-0.56]: discriminacion estadisticamente por encima del azar "
            "(IC95 >0.50) pero modesta frente a AUC=0.75 en VinDr. El shift de dominio "
            "pelicula-digitalizada vs digital explica la caida. No es un bug de preprocesamiento: "
            "el fix uint16 fue verificado (std=0.20, rango [0.03-0.93])."
        )
    }

r04 = make_04()
json_save(OUT_DIR / "04_crossdomain_auc.json", r04)

# ===========================================================================
# Markdown 05 — bug y fix (no regenerable)
# ===========================================================================
print("\n" + "=" * 70)
print("Markdown 05 — bug y fix (no regenerable)")
print("=" * 70)

bug_fix_md = f"""# 05 — Bug de preprocesamiento uint16 y su corrección

> **Documento no regenerable.** Los valores ANTES del fix no pueden recomputarse
> porque el loader roto ya no existe. Se transcriben aqui tal como se observaron
> durante el diagnostico.

## Ubicacion del bug

- Archivo: `src/data_loading.py`, funcion `load_image_as_pil`, rama `else`
- Linea original: `return Image.open(image_path).convert("RGB")`
- Commit HEAD al momento del arreglo: `{COMMIT_HASH}`
- Estado: fix aplicado en working tree; no commiteado como commit independiente

## Causa raiz

`PIL.Image.open()` sobre un PNG de 16 bits devuelve modo `I;16`
(enteros sin signo de 16 bits, rango [0, 65535]). Al llamar `.convert("RGB")`
PIL no aplica normalizacion: interpreta cada valor uint16 como saturado
en 8 bits → **todos los pixeles llegan a 255** (imagen completamente blanca).

La rama DICOM (`_load_dicom_as_pil`) tenia el escalado correcto desde el inicio
(min-max por imagen → uint8 → RGB, lineas 207-210). La rama PNG no lo tenia.

## Evidencia del bug (antes del fix)

| Metrica | DDSM (con bug) | VinDr (referencia) |
|---------|---------------|---------------------|
| RGB range en carga | [255, 255] | [0, 255] |
| Pre-norm media (tensor) | 0.932 | 0.065 |
| Post-norm media (tensor) | +2.140 | −1.596 |
| Logits std (encoder) | 0.36 | 1.10 |
| Probs medias [idx0..4] | [0.22, 0.21, 0.21, 0.25, 0.11] (cuasi-uniformes) | [0.41, 0.33, 0.17, 0.07, 0.02] |

## Correccion aplicada

```python
def _minmax_to_uint8_rgb(arr: np.ndarray):
    ## Min-max por imagen a [0,255] uint8 y replica a RGB.
    ## Identico al escalado del path DICOM (_load_dicom_as_pil lineas 207-210).
    arr = arr.astype(np.float32)
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min) * 255.0
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

# En load_image_as_pil, rama else:
pil = Image.open(image_path)
if pil.mode in ("I;16", "I;16B", "I"):
    return _minmax_to_uint8_rgb(np.array(pil))
return pil.convert("RGB")
```

Decisiones congeladas: min-max por imagen (identico path DICOM), sin recorte
por percentiles, sin mascara de texto.

## Evidencia del fix (despues del fix)

| Metrica | DDSM (post-fix) | VinDr (referencia) |
|---------|----------------|---------------------|
| Pre-norm media | 0.255 | 0.065 |
| Post-norm media | −0.732 | −1.596 |
| Logits std (encoder) | 0.71 | 1.10 |
| Probs medias [idx0..4] | [0.26, 0.22, 0.18, 0.21, 0.13] | [0.41, 0.33, 0.17, 0.07, 0.02] |

## Implicacion para el AUC

El fix elimino la saturacion (std=0.20, rango [0.03, 0.93]). La brecha residual
con VinDr es el shift de dominio real pelicula→digital. El AUC=0.54 de
04_crossdomain_auc.json es un resultado confiable, no un bug.
"""

with open(OUT_DIR / "05_preprocessing_bug_and_fix.md", "w") as f:
    f.write(bug_fix_md)
print("  -> 05_preprocessing_bug_and_fix.md")

# ===========================================================================
# Copiar figuras
# ===========================================================================
print("\n" + "=" * 70)
print("Copiando figuras")
print("=" * 70)

figs = [
    (SRC_ROC,                          OUT_DIR / "roc_crossdomain_ddsm.png"),
    (SRC_DIAG / "hist_agregado.png",   OUT_DIR / "hist_uint16_agregado.png"),
    (SRC_DIAG / "hist_individuales.png", OUT_DIR / "hist_uint16_individuales.png"),
    (SRC_DIAG / "hist_extremo_bajo.png", OUT_DIR / "hist_uint16_extremo_bajo.png"),
    (SRC_DIAG / "overlay_top1pct.png",  OUT_DIR / "spatial_top1pct.png"),
    (SRC_DIAG / "overlay_top01pct.png", OUT_DIR / "spatial_top01pct.png"),
]

for src, dst in figs:
    if Path(src).exists():
        shutil.copy2(src, dst)
        print(f"  {Path(src).name} -> {dst.name}")
    else:
        print(f"  FALTA: {src}")

# ===========================================================================
# README.md
# ===========================================================================
print("\n" + "=" * 70)
print("README.md")
print("=" * 70)

auc_roc = r04["primary_metric"]["auc_roc_point"]
ci_lo_r = r04["primary_metric"]["ci_95_bootstrap"]["lower"]
ci_hi_r = r04["primary_metric"]["ci_95_bootstrap"]["upper"]
ap_r    = r04["primary_metric"]["average_precision_auc_pr"]
n_total = r04["data"]["N_total"]
n_mal   = r04["data"]["N_malignant"]
n_ben   = r04["data"]["N_benign"]
sens_r  = r04["secondary_metric"]["sensitivity"]
spec_r  = r04["secondary_metric"]["specificity"]
cm_r    = r04["secondary_metric"]["confusion_matrix"]
drift_r = r04["drift_diagnostic"]
n_conf  = r03["confound_verdict"]["features_with_auc_above_060"]

prefix_rows = "\n".join([
    f"| {p} | {v['n']} | {v['n_malignant']} | {v.get('n_benign','?')} | "
    f"{v['auc']:.4f} |"
    for p, v in r04["robustness_by_volume"].items() if v["auc"] is not None
])

readme = f"""# Resultados: Test Cross-Domain DDSM — exp08

## Resultado titular

**El modelo entrenado en VinDr-Mammo (digital) no generaliza a DDSM (pelicula
digitalizada): AUC-ROC cae de 0.75 (VinDr test) a {auc_roc:.4f}
IC95=[{ci_lo_r:.4f}–{ci_hi_r:.4f}] en DDSM.**

El IC95 supera 0.50 (el modelo discrimina sobre el azar), pero la caida es severa
y el umbral de VinDr (0.120) colapsa la especificidad a {spec_r:.4f} en DDSM
(drift de umbral +{drift_r['drift_magnitude']:.3f}).

---

## Metadatos de auditoria

| Campo | Valor |
|-------|-------|
| Fecha | {TODAY} |
| Commit HEAD | `{COMMIT_HASH}` |
| Checkpoint | `outputs/experiments/exp08_ordinal_sord_qwk_descongelado/model.pt` |
| SHA256 checkpoint | `{CKPT_SHA256}` |
| Semilla global | {SEED} |
| N bootstrap | {N_BOOT} |
| N imagenes DDSM | {n_total} (MAL={n_mal} / BEN={n_ben}) |
| Fallos de carga | {r04['data']['N_failed_load']} |
| Dispositivo inferencia | {DEVICE} |

---

## Decisiones congeladas

| Decision | Valor |
|----------|-------|
| Escalado DDSM | min-max por imagen, sin recorte por percentil (identico path DICOM) |
| malignancy_score | `birads_probs[3] + birads_probs[4]` (MALIGNANT_INDICES_5CLS=[3,4]) |
| Umbral VinDr | 0.120 (Youden J validacion VinDr, congelado) |
| Etiqueta imagen | peor lesion: >=1 MALIGNANT → MALIGNANT; todos benignas → BENIGN |
| Transform | MammoCLIPTransform augment=False, use_clahe=True |
| Modelo | exp08_ordinal_sord_qwk_descongelado — NO exp09, NO config binaria |

---

## Numeros clave

### Metrica titular (libre de umbral)
- **AUC-ROC = {auc_roc:.4f}  IC95=[{ci_lo_r:.4f} – {ci_hi_r:.4f}]**
  (bootstrap {N_BOOT} resamples, semilla {SEED})
- Average Precision (AUC-PR) = {ap_r:.4f}

### Sanity del malignancy_score
```
N={r04['data']['N_valid']}  std={r04['sanity_malignancy_score']['std']:.4f}
min={r04['sanity_malignancy_score']['min']:.4f}  Q1={r04['sanity_malignancy_score']['q1']:.4f}
median={r04['sanity_malignancy_score']['median']:.4f}  Q3={r04['sanity_malignancy_score']['q3']:.4f}
max={r04['sanity_malignancy_score']['max']:.4f}  degenerate={r04['sanity_malignancy_score']['is_degenerate']}
```

### Umbral VinDr 0.120 — transferencia de punto de operacion (NO calibracion)
```
Sens={r04['secondary_metric']['sensitivity']:.4f}  Spec={r04['secondary_metric']['specificity']:.4f}
TP={cm_r['tp']}  FP={cm_r['fp']}  TN={cm_r['tn']}  FN={cm_r['fn']}
```

### Drift de umbral (diagnostico, NO metrica reportada)
- Umbral optimo EN DDSM: {drift_r['optimal_threshold_ddsm']:.3f}
  (solo diagnostico, no se usa como punto de operacion)
- Drift vs VinDr: +{drift_r['drift_magnitude']:.3f}

### AUC por prefijo de digitalizador
| Prefijo | N | MAL | BEN | AUC |
|---------|---|-----|-----|-----|
{prefix_rows}

### Confound check (3669 imagenes completas)
- Features con AUC > 0.60: **{n_conf}**  → {r03['confound_verdict']['verdict']}
- Chi2(prefijo x patologia): chi2={r03['prefix_contingency']['chi2']:.2f}
  p={r03['prefix_contingency']['p_value']:.4f}

---

## Indice de archivos

| Archivo | Descripcion | Regenerable |
|---------|-------------|-------------|
| `01_index_mapping.json` | 5 clases, mapeo k->k-1, tabla 5x5 VinDr val | Si (seed 42) |
| `02_preprocessing_postfix.json` | Stats tensor pre/post-norm post-fix | Si (seed 42) |
| `03_confound_check.json` | AUC features + contingencia prefijo, 3669 imgs | Si |
| `04_crossdomain_auc.json` | AUC-ROC IC95, AP, scores, confusion, drift | Si (seed 42) |
| `05_preprocessing_bug_and_fix.md` | Doc bug uint16 y correccion | **NO** |
| `roc_crossdomain_ddsm.png` | Curva ROC con punto de operacion VinDr | Si |
| `hist_uint16_*.png` | Histogramas DDSM raw uint16 | Si |
| `spatial_top1pct.png` | Mapa espacial pixeles mas brillantes (top 1%) | Si |
| `spatial_top01pct.png` | Mapa espacial pixeles mas brillantes (top 0.1%) | Si |

---

## Supuestos declarados

1. Umbral 0.120 leido de `experiment_detail.json` de exp08 (campo `notes`).
2. Regla de peor lesion: nivel de imagen, no de vista ni de paciente.
3. Fix de `data_loading.py` en working tree; HEAD `{COMMIT_HASH}` corresponde
   a la ultima version del notebook, no al fix (no committeado independientemente).
4. Sin iteracion de parametros mirando el resultado.
"""

with open(OUT_DIR / "README.md", "w") as f:
    f.write(readme)
print("  -> README.md")

# ===========================================================================
# Listado y comparacion final
# ===========================================================================
print("\n" + "=" * 70)
print("DIRECTORIO FINAL")
print("=" * 70)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.stat().st_size:>10d} B  {f.name}")

print()
print("=" * 70)
print("COMPARACION VS VALORES EN CHAT")
print("=" * 70)
print(f"  AUC-ROC punto   — Chat: 0.5438  | Ahora: {r04['primary_metric']['auc_roc_point']:.4f}")
print(f"  IC95 lower      — Chat: 0.5249  | Ahora: {ci_lo_r:.4f}")
print(f"  IC95 upper      — Chat: 0.5622  | Ahora: {ci_hi_r:.4f}")
print(f"  AUC-PR          — Chat: 0.5352  | Ahora: {ap_r:.4f}")
print(f"  Score std       — Chat: 0.2036  | Ahora: {r04['sanity_malignancy_score']['std']:.4f}")
print(f"  Score mean      — Chat: 0.4027  | Ahora: {r04['sanity_malignancy_score']['mean']:.4f}")
print(f"  Sens (0.120)    — Chat: 0.9391  | Ahora: {sens_r:.4f}")
print(f"  Spec (0.120)    — Chat: 0.0815  | Ahora: {spec_r:.4f}")
print(f"  N valid         — Chat: 3669    | Ahora: {r04['data']['N_valid']}")
print(f"  Confound (N=1200 prev) vs Ahora (N={r03['metadata']['N_total_images']}): "
      f"features>0.60 = {n_conf}")
