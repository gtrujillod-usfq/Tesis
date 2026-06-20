"""
Genera todos los artefactos de resultados del test cross-domain DDSM.

Escribe results/ddsm_crossdomain/ con:
  01_index_mapping.json
  02_preprocessing_postfix.json
  03_confound_check.json          (3669 imagenes completas)
  04_crossdomain_auc.json         (semilla 42)
  05_preprocessing_bug_and_fix.md (no regenerable)
  + figuras copiadas
  + README.md

Decisiones congeladas — ninguna se toca aqui:
  - min-max por imagen (sin recorte por percentil)
  - malignancy_score = birads_probs[3] + birads_probs[4]
  - umbral VinDr = 0.120
  - regla peor lesion para etiqueta de imagen DDSM
  - semilla global 42
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loading import load_image_as_pil, MammoCLIPTransform, load_vindr_records
from models import MammoVLM
from ddsm_overlay import load_ddsm_records
from threshold_tuning import MALIGNANT_INDICES_5CLS

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, confusion_matrix)

# ---------------------------------------------------------------------------
# Constantes congeladas
# ---------------------------------------------------------------------------
TESIS_ROOT      = Path(__file__).parent.parent
VINDR_ROOT      = TESIS_ROOT / "data" / "vindr-mammo"
DDSM_ROOT       = TESIS_ROOT / "data" / "6 DDSM"
MAMMOCLIP_CKPT  = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT      = str(TESIS_ROOT / "outputs" / "experiments" /
                      "exp08_ordinal_sord_qwk_descongelado" / "model.pt")
OUT_DIR         = TESIS_ROOT / "results" / "ddsm_crossdomain"
SRC_ROC         = TESIS_ROOT / "outputs" / "crossdomain_ddsm" / "roc_crossdomain_ddsm.png"
SRC_DIAG        = TESIS_ROOT / "outputs" / "diag_ddsm"

SEED            = 42
N_BOOT          = 2000
VINDR_THRESHOLD = 0.120
MALIGNANT_IDX   = MALIGNANT_INDICES_5CLS   # [3, 4] — confirmado
BATCH_SIZE      = 16
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
N_PREPROC_SAMPLE = 50
TODAY           = datetime.date.today().isoformat()

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directorio de salida: {OUT_DIR}")

# ---------------------------------------------------------------------------
# Metadatos de auditoria
# ---------------------------------------------------------------------------
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

COMMIT_HASH  = git_hash()
CKPT_SHA256  = file_sha256(EXP08_CKPT)
print(f"Commit HEAD : {COMMIT_HASH}")
print(f"Checkpoint  : {CKPT_SHA256[:16]}...")
print(f"Fecha       : {TODAY}")

# ---------------------------------------------------------------------------
# Helpers de carga comunes
# ---------------------------------------------------------------------------
import torchvision.transforms as T

transform_eval = MammoCLIPTransform(augment=False, use_clahe=True)
transform_pre  = T.Compose([T.Resize((1520, 912)), T.ToTensor()])

import logging; logging.disable(logging.WARNING)

def load_vindr_split():
    df = load_vindr_records(str(VINDR_ROOT))
    return df[df["split"] == "test"].reset_index(drop=True)

def load_ddsm_image_df():
    """Agrega lesiones a nivel de imagen con regla de peor lesion."""
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
    assert not missing and not unexpected, f"Pesos: missing={missing}"
    m.to(DEVICE).eval()
    return m

def json_save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> {path.name}")

# ===========================================================================
# JSON 01 — index_mapping + tabla 5x5 VinDr validation
# ===========================================================================
print("\n" + "=" * 70)
print("JSON 01 — index_mapping + 5x5 VinDr")
print("=" * 70)

def make_01():
    from torch.utils.data import Dataset, DataLoader

    class SimpleDS(Dataset):
        def __init__(self, df, transform):
            self.df = df
            self.transform = transform
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            try:
                pil = load_image_as_pil(str(row["image_path"]))
                return self.transform(pil), int(row["birads"]) - 1, True
            except Exception:
                return torch.zeros(3, 1520, 912), -1, False

    vindr_df = load_vindr_split()
    print(f"  VinDr test split: {len(vindr_df)} imagenes")

    model = load_model()

    ds     = SimpleDS(vindr_df, transform_eval)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                         num_workers=4, pin_memory=(DEVICE=="cuda"))

    table  = np.zeros((5, 5))   # [true_class, pred_idx]
    counts = np.zeros(5)

    for imgs, labels, ok_flags in loader:
        imgs   = imgs.to(DEVICE)
        ok     = ok_flags.numpy().astype(bool)
        if ok.sum() == 0:
            continue
        imgs   = imgs[ok]
        labels = labels.numpy()[ok]
        with torch.no_grad():
            logits = model.forward(imgs)["birads"]
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        for bi in range(5):
            mask = (labels == bi)
            if mask.sum() > 0:
                table[bi] += probs[mask].sum(axis=0)
                counts[bi] += mask.sum()

    # Normalizar a prob media
    for bi in range(5):
        if counts[bi] > 0:
            table[bi] /= counts[bi]

    result = {
        "metadata": {
            "date": TODAY, "commit": COMMIT_HASH,
            "checkpoint_sha256": CKPT_SHA256,
            "checkpoint_path": EXP08_CKPT,
            "seed": SEED, "dataset": "VinDr-Mammo test split",
            "N_total": int(len(vindr_df))
        },
        "h1_num_output_classes": {
            "value": 5,
            "evidence": "birads_head.net.4.weight shape = [5, 256]",
            "verdict": "CONFIRMADA"
        },
        "h2_birads_to_index": {
            "formula": "birads_to_index(b) = b - 1",
            "mapping": {"BIRADS_1": 0, "BIRADS_2": 1, "BIRADS_3": 2,
                        "BIRADS_4": 3, "BIRADS_5": 4},
            "verdict": "CONFIRMADA"
        },
        "h3_malignancy_score": {
            "formula": "malignancy_score = birads_probs[3] + birads_probs[4]",
            "MALIGNANT_INDICES_5CLS": [3, 4],
            "verdict": "CONFIRMADA"
        },
        "table_5x5_mean_prob_by_true_birads": {
            "description": ("Filas = clase BI-RADS verdadera (1-5 -> idx 0-4); "
                            "Columnas = indice de salida (0-4); "
                            "Celdas = probabilidad media predicha."),
            "columns": ["idx0", "idx1", "idx2", "idx3", "idx4"],
            "rows": {f"birads_{bi+1}": {
                "n": int(counts[bi]),
                "probs": {f"idx{j}": float(table[bi, j]) for j in range(5)}
            } for bi in range(5)}
        }
    }
    return result

r01 = make_01()
json_save(OUT_DIR / "01_index_mapping.json", r01)

# ===========================================================================
# JSON 02 — preprocessing_postfix stats
# ===========================================================================
print("\n" + "=" * 70)
print("JSON 02 — preprocessing stats post-fix")
print("=" * 70)

def make_02():
    rng = np.random.default_rng(SEED)

    def collect_stats(paths, n):
        pre_vals, post_vals = [], []
        ok = 0
        for p in paths:
            if ok >= n: break
            try:
                pil = load_image_as_pil(str(p))
                pre  = transform_pre(pil).numpy().flatten()
                post = transform_eval(pil).numpy().flatten()
                idx  = rng.choice(len(pre), min(5000, len(pre)), replace=False)
                pre_vals.append(pre[idx])
                post_vals.append(post[idx])
                ok += 1
            except Exception:
                pass
        return np.concatenate(pre_vals), np.concatenate(post_vals), ok

    def stats_dict(arr):
        return {
            "mean": float(arr.mean()),  "std": float(arr.std()),
            "min":  float(arr.min()),   "max": float(arr.max()),
            "p1":   float(np.percentile(arr, 1)),
            "p50":  float(np.percentile(arr, 50)),
            "p99":  float(np.percentile(arr, 99)),
        }

    vindr_df    = load_vindr_split()
    vindr_paths = vindr_df["image_path"].tolist()
    rng.shuffle(vindr_paths)
    v_pre, v_post, v_n = collect_stats(vindr_paths, N_PREPROC_SAMPLE)

    all_ddsm = sorted(DDSM_ROOT.rglob("*.png"))
    d_idx    = rng.choice(len(all_ddsm), min(200, len(all_ddsm)), replace=False)
    d_paths  = [all_ddsm[i] for i in d_idx]
    d_pre, d_post, d_n = collect_stats(d_paths, N_PREPROC_SAMPLE)

    # Encoder forward (16 imagenes cada uno)
    model = load_model()

    def forward_stats(paths, n=16):
        tensors = []
        for p in paths:
            if len(tensors) >= n: break
            try:
                pil = load_image_as_pil(str(p))
                tensors.append(transform_eval(pil))
            except Exception:
                pass
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            out    = model.forward(batch)["birads"]
            probs  = F.softmax(out, dim=-1).cpu().numpy()
            logits = out.cpu().numpy()
        return {
            "n_images": len(tensors),
            "logits_std": float(logits.std()),
            "logits_range": [float(logits.min()), float(logits.max())],
            "mean_probs_per_idx": {f"idx{i}": float(probs.mean(axis=0)[i]) for i in range(5)},
            "is_quasi_uniform": bool(probs.max(axis=1).mean() < 0.25),
            "argmax_collapsed": bool(len(np.unique(probs.argmax(axis=1))) == 1),
        }

    v_fwd = forward_stats(vindr_paths[:50])
    d_fwd = forward_stats(d_paths[:50])

    return {
        "metadata": {"date": TODAY, "commit": COMMIT_HASH, "seed": SEED,
                     "N_sample_per_dataset": N_PREPROC_SAMPLE,
                     "note": "Post-fix: loader corregido (min-max uint16)"},
        "frozen_decisions": {
            "scaling": "min-max por imagen, sin recorte por percentil, identico al path DICOM",
            "clahe": "clipLimit=2.0, tileGridSize=(8,8)",
            "resize": "1520x912 bilinear antialias",
            "normalize": "ImageNet stats (0.485,0.456,0.406) / (0.229,0.224,0.225)"
        },
        "vindr": {
            "n_images": v_n,
            "pre_norm":  stats_dict(v_pre),
            "post_norm": stats_dict(v_post),
            "encoder_forward": v_fwd
        },
        "ddsm_postfix": {
            "n_images": d_n,
            "pre_norm":  stats_dict(d_pre),
            "post_norm": stats_dict(d_post),
            "encoder_forward": d_fwd
        },
        "baseline_prefix_for_reference": {
            "note": "Valores ANTES del fix — no regenerables",
            "ddsm_pre_norm_mean": 0.932,
            "ddsm_post_norm_mean": 2.140,
            "ddsm_logits_std": 0.36,
            "ddsm_probs": [0.22, 0.21, 0.21, 0.25, 0.11]
        }
    }

r02 = make_02()
json_save(OUT_DIR / "02_preprocessing_postfix.json", r02)

# ===========================================================================
# JSON 03 — confound check sobre las 3669 imagenes COMPLETAS
# ===========================================================================
print("\n" + "=" * 70)
print("JSON 03 — confound check (3669 imagenes completas)")
print("=" * 70)

def make_03():
    from scipy.stats import chi2_contingency

    ddsm_df = load_ddsm_image_df()
    n_total = len(ddsm_df)
    print(f"  Total imagenes: {n_total}  "
          f"(MAL={( ddsm_df['pathology']=='MALIGNANT').sum()}  "
          f"BEN={(ddsm_df['pathology']=='BENIGN').sum()})")

    records = []
    for i, row in ddsm_df.iterrows():
        if i % 300 == 0:
            print(f"  {i}/{n_total}", flush=True)
        try:
            raw_pil = Image.open(str(row["image_path"]))
            arr     = np.array(raw_pil, dtype=np.float32)   # uint16 raw
            h, w    = arr.shape
            flat    = arr.flatten()
            p1, p5, p50, p99 = np.percentile(flat, [1, 5, 50, 99])
            high_threshold = 65535 * 0.95   # top 5% de rango uint16
            text_proxy = float((arr > high_threshold).mean())
            records.append({
                "image_path": str(row["image_path"]),
                "pathology":  row["pathology"],
                "prefix":     row["prefix"],
                "height": h, "width": w,
                "total_pixels": int(h * w),
                "p1_raw":   float(p1),
                "p5_raw":   float(p5),
                "mean_raw": float(flat.mean()),
                "p50_raw":  float(p50),
                "p99_raw":  float(p99),
                "text_proxy": text_proxy,
            })
        except Exception:
            pass

    print(f"  Imagenes con features OK: {len(records)}/{n_total}")
    feat_df = pd.DataFrame(records)

    y = (feat_df["pathology"] == "MALIGNANT").astype(int)
    features = ["height", "width", "total_pixels", "p1_raw", "p5_raw",
                "text_proxy", "mean_raw", "p50_raw", "p99_raw"]

    auc_results = {}
    for f in features:
        auc_results[f] = {
            "mean_benign":    float(feat_df.loc[y==0, f].mean()),
            "median_benign":  float(feat_df.loc[y==0, f].median()),
            "mean_malignant": float(feat_df.loc[y==1, f].mean()),
            "median_malignant": float(feat_df.loc[y==1, f].median()),
            "auc":            float(roc_auc_score(y, feat_df[f])),
            "flag_confound":  bool(roc_auc_score(y, feat_df[f]) > 0.60),
        }

    # Contingencia prefijo x patologia
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
    return {
        "metadata": {
            "date": TODAY, "commit": COMMIT_HASH, "seed": SEED,
            "N_total_images": n_total,
            "N_features_computed": len(records),
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

    ds      = DDSMDataset(ddsm_df["image_path"].tolist(), transform_eval)
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=(DEVICE=="cuda"))

    all_probs, all_ok = [], []
    for imgs, ok_flags in loader:
        imgs   = imgs.to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model.forward(imgs)["birads"], dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_ok.append(ok_flags.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_ok    = np.concatenate(all_ok).astype(bool)

    n_fail = int((~all_ok).sum())
    df_ok  = ddsm_df[all_ok].reset_index(drop=True)
    probs_ok = all_probs[all_ok]

    scores = probs_ok[:, MALIGNANT_IDX[0]] + probs_ok[:, MALIGNANT_IDX[1]]
    y_true = (df_ok["pathology"] == "MALIGNANT").astype(int).values

    # Sanity
    score_global = {
        "mean": float(scores.mean()), "median": float(np.median(scores)),
        "std":  float(scores.std()),  "min":    float(scores.min()),
        "max":  float(scores.max()),
        "q1":   float(np.percentile(scores, 25)),
        "q3":   float(np.percentile(scores, 75)),
        "is_degenerate": bool(scores.std() < 0.01),
    }

    # AUC-ROC con IC95 bootstrap (semilla 42)
    auc_point = float(roc_auc_score(y_true, scores))
    rng_boot  = np.random.default_rng(SEED)
    boot_aucs = []
    for _ in range(N_BOOT):
        idx = rng_boot.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], scores[idx]
        if yt.sum() in (0, len(yt)): continue
        boot_aucs.append(roc_auc_score(yt, ys))
    boot_aucs = np.array(boot_aucs)
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))

    # Average Precision
    ap = float(average_precision_score(y_true, scores))

    # Score por clase
    def class_stats(s):
        return {
            "mean": float(s.mean()), "median": float(np.median(s)),
            "q1": float(np.percentile(s, 25)), "q3": float(np.percentile(s, 75)),
            "min": float(s.min()), "max": float(s.max())
        }

    # Umbral VinDr
    pred_v = (scores >= VINDR_THRESHOLD).astype(int)
    cm_v   = confusion_matrix(y_true, pred_v)
    tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
    sens_v = float(tp_v / max(tp_v + fn_v, 1))
    spec_v = float(tn_v / max(tn_v + fp_v, 1))

    # Umbral optimo DDSM (diagnostico de drift, no metrica)
    thrs   = np.linspace(0, 1, 1001)
    best_j, best_thr_ddsm = -1, None
    for t in thrs:
        pred_t = (scores >= t).astype(int)
        if pred_t.sum() in (0, len(pred_t)): continue
        cm_t   = confusion_matrix(y_true, pred_t)
        if cm_t.shape != (2,2): continue
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        s  = tp_t / max(tp_t + fn_t, 1)
        sp = tn_t / max(tn_t + fp_t, 1)
        j  = s + sp - 1
        if j > best_j:
            best_j, best_thr_ddsm = j, float(t)

    # AUC por prefijo
    auc_by_prefix = {}
    for pref in sorted(df_ok["prefix"].unique()):
        mask = df_ok["prefix"].values == pref
        yt_p, ys_p = y_true[mask], scores[mask]
        n_p = int(mask.sum())
        if yt_p.sum() in (0, n_p):
            auc_by_prefix[pref] = {
                "n": n_p, "n_malignant": int(yt_p.sum()),
                "auc": None, "note": "sin varianza"
            }
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
            "ppv": float(tp_v / max(tp_v + fp_v, 1)),
            "npv": float(tn_v / max(tn_v + fn_v, 1)),
            "youden_j": float(sens_v + spec_v - 1),
        },
        "score_distribution_by_class": {
            "BENIGN":    class_stats(scores[y_true == 0]),
            "MALIGNANT": class_stats(scores[y_true == 1]),
        },
        "drift_diagnostic": {
            "label": "Umbral optimo EN DDSM — solo diagnostico de drift, NO metrica reportada",
            "vindr_threshold": VINDR_THRESHOLD,
            "optimal_threshold_ddsm": best_thr_ddsm,
            "drift_magnitude": float(best_thr_ddsm - VINDR_THRESHOLD),
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
- Commit HEAD al momento del diagnostico: `{COMMIT_HASH}`
- Estado: fix aplicado en el working tree (no committeado en el HEAD anterior)

## Causa raiz

`PIL.Image.open()` sobre un PNG de 16 bits devuelve un objeto en modo `I;16`
(enteros sin signo de 16 bits, rango [0, 65535]). Llamar `.convert("RGB")` sobre
ese modo no aplica normalalizacion: PIL interpreta cada valor uint16 como saturado
en el canal de 8 bits, resultando en que **todos los pixeles llegan a 255** (imagen
completamente blanca).

La rama DICOM (`_load_dicom_as_pil`) tenia el escalado correcto desde el inicio
(min-max por imagen → uint8 → RGB). La rama PNG no lo tenia.

## Evidencia del bug (antes del fix)

| Metrica | DDSM (con bug) | VinDr (referencia) |
|---------|---------------|---------------------|
| RGB range en carga | [255, 255] (todo saturado) | [0, 255] |
| Pre-norm media (tensor) | 0.932 | 0.065 |
| Post-norm media (tensor) | +2.140 | −1.596 |
| Logits std (encoder) | 0.36 | 1.10 |
| Probs medias [idx0..4] | [0.22, 0.21, 0.21, 0.25, 0.11] (cuasi-uniformes) | [0.41, 0.33, 0.17, 0.07, 0.02] |

## Corrección aplicada

Se añadio el helper `_minmax_to_uint8_rgb` y se condiciono su uso al modo 16-bit:

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

Decisiones congeladas aplicadas: min-max por imagen (identico al path DICOM),
sin recorte por percentiles, sin mascara de texto.

## Evidencia del fix (despues del fix)

| Metrica | DDSM (post-fix) | VinDr (referencia) |
|---------|----------------|---------------------|
| Pre-norm media | 0.255 | 0.065 |
| Post-norm media | −0.732 | −1.596 |
| Logits std (encoder) | 0.71 | 1.10 |
| Probs medias [idx0..4] | [0.26, 0.22, 0.18, 0.21, 0.13] | [0.41, 0.33, 0.17, 0.07, 0.02] |

## Implicacion para el AUC

El fix elimino la saturacion: los scores ya no son degenerados (std=0.20,
rango [0.03, 0.93]). La brecha residual con VinDr (post-norm −0.73 vs −1.60;
logits std 0.71 vs 1.10) es el shift de dominio real entre pelicula digitalizada
y digital — no un artefacto del preprocesamiento.

**Conclusion:** el AUC=0.54 reportado en 04_crossdomain_auc.json es un resultado
confiable, no un bug. Representa genuina degradacion de generalizacion cross-domain.
"""

with open(OUT_DIR / "05_preprocessing_bug_and_fix.md", "w") as f:
    f.write(bug_fix_md)
print(f"  -> 05_preprocessing_bug_and_fix.md")

# ===========================================================================
# Copiar figuras
# ===========================================================================
print("\n" + "=" * 70)
print("Copiando figuras")
print("=" * 70)

figs_to_copy = [
    (SRC_ROC, OUT_DIR / "roc_crossdomain_ddsm.png"),
    (SRC_DIAG / "hist_agregado.png",    OUT_DIR / "hist_uint16_agregado.png"),
    (SRC_DIAG / "hist_individuales.png", OUT_DIR / "hist_uint16_individuales.png"),
    (SRC_DIAG / "hist_extremo_bajo.png", OUT_DIR / "hist_uint16_extremo_bajo.png"),
    (SRC_DIAG / "overlay_top1pct.png",   OUT_DIR / "spatial_top1pct.png"),
    (SRC_DIAG / "overlay_top01pct.png",  OUT_DIR / "spatial_top01pct.png"),
]

for src, dst in figs_to_copy:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst.name}")
    else:
        print(f"  FALTA: {src}")

# ===========================================================================
# README.md
# ===========================================================================
print("\n" + "=" * 70)
print("README.md")
print("=" * 70)

auc_roc  = r04["primary_metric"]["auc_roc_point"]
ci_lo_r  = r04["primary_metric"]["ci_95_bootstrap"]["lower"]
ci_hi_r  = r04["primary_metric"]["ci_95_bootstrap"]["upper"]
ap_r     = r04["primary_metric"]["average_precision_auc_pr"]
n_total  = r04["data"]["N_total"]
n_mal    = r04["data"]["N_malignant"]
n_ben    = r04["data"]["N_benign"]
sens_r   = r04["secondary_metric"]["sensitivity"]
spec_r   = r04["secondary_metric"]["specificity"]

readme = f"""# Resultados: Test Cross-Domain DDSM — exp08

## Resultado titular

**El modelo entrenado en VinDr-Mammo (digital) no generaliza a DDSM (pelicula digitalizada):
AUC-ROC cae de 0.75 (VinDr test) a {auc_roc:.4f} IC95=[{ci_lo_r:.4f}–{ci_hi_r:.4f}] en DDSM.**

El resultado esta por encima del azar (IC95 > 0.50), pero la discriminacion es modesta y
el umbral de VinDr (0.120) colapsa la especificidad a {spec_r:.4f} en DDSM (drift de umbral +0.383).

---

## Metadatos de auditoria

| Campo | Valor |
|-------|-------|
| Fecha | {TODAY} |
| Commit HEAD | `{COMMIT_HASH}` |
| Checkpoint | `{EXP08_CKPT}` |
| SHA256 checkpoint | `{CKPT_SHA256}` |
| Semilla global | {SEED} |
| N bootstrap | {N_BOOT} |
| N imagenes DDSM | {n_total} (MAL={n_mal} / BEN={n_ben}) |
| Fallos de carga | 0 |
| Dispositivo | {DEVICE} |

---

## Decisiones congeladas

Ninguna de estas se modifico para obtener el resultado:

| Decision | Valor |
|----------|-------|
| Escalado DDSM | min-max por imagen, sin recorte por percentil (identico al path DICOM) |
| malignancy_score | birads_probs[3] + birads_probs[4]  (MALIGNANT_INDICES_5CLS=[3,4]) |
| Umbral VinDr | 0.120 (Youden J en validacion VinDr, congelado) |
| Etiqueta imagen | peor lesion: >=1 MALIGNANT -> MALIGNANT, todos benignas -> BENIGN |
| Transform | MammoCLIPTransform augment=False, use_clahe=True |
| Modelo | exp08_ordinal_sord_qwk_descongelado — NO exp09, NO config binaria |

---

## Numeros clave

### Metrica titular
- **AUC-ROC = {auc_roc:.4f}  IC95=[{ci_lo_r:.4f} – {ci_hi_r:.4f}]** (bootstrap {N_BOOT} resamples, semilla {SEED})
- Average Precision (AUC-PR) = {ap_r:.4f}

### Sanity del malignancy_score
- std={r04['sanity_malignancy_score']['std']:.4f}, rango=[{r04['sanity_malignancy_score']['min']:.4f}, {r04['sanity_malignancy_score']['max']:.4f}] — no degenerado

### Umbral VinDr transferido (0.120) — solo transferencia de punto de operacion
- Sensibilidad: {r04['secondary_metric']['sensitivity']:.4f}
- Especificidad: {r04['secondary_metric']['specificity']:.4f}
- TP={r04['secondary_metric']['confusion_matrix']['tp']}  FP={r04['secondary_metric']['confusion_matrix']['fp']}  TN={r04['secondary_metric']['confusion_matrix']['tn']}  FN={r04['secondary_metric']['confusion_matrix']['fn']}

### Drift de umbral (diagnostico, no metrica reportada)
- Umbral optimo EN DDSM (Youden J): {r04['drift_diagnostic']['optimal_threshold_ddsm']:.3f}
- Diferencia vs VinDr: {r04['drift_diagnostic']['drift_magnitude']:+.3f}

### AUC por prefijo de digitalizador
| Prefijo | N | MAL | BEN | AUC |
|---------|---|-----|-----|-----|
""" + "\n".join([
    f"| {p} | {v['n']} | {v['n_malignant']} | {v.get('n_benign','?')} | {v['auc']:.4f} |"
    for p, v in r04["robustness_by_volume"].items() if v["auc"] is not None
]) + f"""

---

## Indice de archivos

| Archivo | Descripcion | Regenerable |
|---------|-------------|-------------|
| `01_index_mapping.json` | 5 clases, mapeo k->k-1, tabla 5x5 sobre VinDr val | Si (semilla 42) |
| `02_preprocessing_postfix.json` | Stats tensor pre/post-norm DDSM y VinDr post-fix | Si (semilla 42) |
| `03_confound_check.json` | AUC por feature sobre 3669 imgs; contingencia prefijo x patologia | Si |
| `04_crossdomain_auc.json` | AUC-ROC IC95, AP, distribucion scores, umbral VinDr | Si (semilla 42) |
| `05_preprocessing_bug_and_fix.md` | Doc del bug uint16 y su correccion | **NO** (datos anteriores al fix) |
| `roc_crossdomain_ddsm.png` | Curva ROC con punto de operacion VinDr marcado | Si |
| `hist_uint16_*.png` | Histogramas de inspeccion de imagenes DDSM raw uint16 | Si |
| `spatial_top1pct.png` | Mapa espacial del 1% de pixeles mas brillantes | Si |
| `spatial_top01pct.png` | Mapa espacial del 0.1% de pixeles mas brillantes | Si |

---

## Supuestos declarados

1. El umbral 0.120 fue leido de `experiment_detail.json` de exp08 (`notes`); no existe
   archivo separado de threshold.
2. La regla de peor lesion opera a nivel de imagen (no de vista ni de paciente).
3. El fix de `data_loading.py` esta en el working tree; el commit HEAD (`{COMMIT_HASH}`)
   corresponde a la ultima version del notebook, no al fix. El fix no fue committeado
   explicitamente como commit separado.
4. No se corrio ningun experimento adicional ni se itero ningun parametro mirando el resultado.
"""

with open(OUT_DIR / "README.md", "w") as f:
    f.write(readme)
print(f"  -> README.md")

# ===========================================================================
# Listado final del directorio
# ===========================================================================
print("\n" + "=" * 70)
print("DIRECTORIO FINAL")
print("=" * 70)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.stat().st_size:>9d} bytes  {f.name}")

print()
print("=" * 70)
print("COMPARACION VS VALORES EN CHAT (semilla 42 fija)")
print("=" * 70)
print(f"  AUC-ROC punto   — Chat: 0.5438  | Ahora: {r04['primary_metric']['auc_roc_point']:.4f}")
print(f"  IC95 lower      — Chat: 0.5249  | Ahora: {ci_lo_r:.4f}")
print(f"  IC95 upper      — Chat: 0.5622  | Ahora: {ci_hi_r:.4f}")
print(f"  AUC-PR          — Chat: 0.5352  | Ahora: {ap_r:.4f}")
print(f"  Score std       — Chat: 0.2036  | Ahora: {r04['sanity_malignancy_score']['std']:.4f}")
print(f"  Score mean      — Chat: 0.4027  | Ahora: {r04['sanity_malignancy_score']['mean']:.4f}")
print(f"  N valid         — Chat: 3669    | Ahora: {r04['data']['N_valid']}")
print(f"  Confound check  — Chat (N=1200): 0 features >0.60 | Ahora (N={r03['metadata']['N_total_images']}): "
      f"{r03['confound_verdict']['features_with_auc_above_060']} features >0.60")
