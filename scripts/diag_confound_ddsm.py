"""
Analisis de confound: artefactos de adquisicion DDSM vs patologia benigno/maligno.
Solo lectura. Sin modificar codigo ni modelos.

Salidas: tablas en consola + CSVs en outputs/diag_confound/
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ddsm_overlay import load_ddsm_records

TESIS_ROOT = Path(__file__).parent.parent
DDSM_ROOT  = TESIS_ROOT / "data" / "6 DDSM"
OUT_DIR    = TESIS_ROOT / "outputs" / "diag_confound"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Cargar registros a nivel de lesion y agregar a nivel de imagen
# ---------------------------------------------------------------------------
print("Cargando registros DDSM (puede tardar ~1-2 min)...")
import logging
logging.disable(logging.WARNING)  # silenciar warnings de paths no encontrados
records = load_ddsm_records(DDSM_ROOT)
logging.disable(logging.NOTSET)
print(f"  Registros de lesion cargados: {len(records)}")

# Excluir registros sin imagen
n_sin_imagen = sum(1 for r in records if r["image_path"] is None)
records_con_img = [r for r in records if r["image_path"] is not None]
print(f"  Sin imagen (excluidos): {n_sin_imagen}")
print(f"  Con imagen: {len(records_con_img)}")

# Agregar a nivel de imagen: peor lesion (MALIGNANT gana)
img_agg = defaultdict(lambda: {"pathology": "BENIGN", "paths": set(), "overlay_path": None,
                                "case_id": None, "category": None, "view": None,
                                "lesions": []})
for r in records_con_img:
    img = str(r["image_path"])
    img_agg[img]["paths"].add(img)
    img_agg[img]["overlay_path"] = r["overlay_path"]
    img_agg[img]["case_id"] = r["case_id"]
    img_agg[img]["category"] = r["category"]
    img_agg[img]["view"] = r["view"]
    img_agg[img]["lesions"].append(r["pathology"])
    # Regla de peor lesion
    if "MALIGNANT" in r["pathology"].upper():
        img_agg[img]["pathology"] = "MALIGNANT"

df_img = pd.DataFrame([
    {
        "image_path": k,
        "pathology": v["pathology"],
        "category": v["category"],
        "view": v["view"],
        "case_id": v["case_id"],
        "n_lesions": len(v["lesions"]),
    }
    for k, v in img_agg.items()
])

n_mal = (df_img["pathology"] == "MALIGNANT").sum()
n_ben = (df_img["pathology"] == "BENIGN").sum()
n_unk = (df_img["pathology"] == "UNKNOWN").sum()

print()
print("=" * 70)
print("CONTEO POR CLASE (nivel imagen, regla peor lesion)")
print("=" * 70)
print(f"  MALIGNANT : {n_mal}")
print(f"  BENIGN    : {n_ben}")
print(f"  UNKNOWN   : {n_unk}  (excluidos del analisis)")
print(f"  TOTAL con overlay + imagen: {len(df_img)}")
print(f"  Sin overlay (excluidos ya en parser): {n_sin_imagen}")

# Excluir UNKNOWN del analisis
df_img = df_img[df_img["pathology"].isin(["MALIGNANT", "BENIGN"])].reset_index(drop=True)
print(f"  Analisis final: {len(df_img)}  ({n_mal} mal + {n_ben} ben)")

# ---------------------------------------------------------------------------
# 2. Extraer prefijo de volumen del nombre de archivo
# ---------------------------------------------------------------------------
def get_prefix(path_str):
    """Extrae el prefijo de volumen (A, B, C, ...) del nombre de la imagen."""
    name = Path(path_str).name        # ej: "A_1081_1.RIGHT_CC.LJPEG.png"
    return name.split("_")[0]         # "A"

df_img["prefix"] = df_img["image_path"].apply(get_prefix)

# ---------------------------------------------------------------------------
# 3. Leer features de imagen para la muestra
#    (lee hasta N imagenes por clase para mantener balance razonable)
# ---------------------------------------------------------------------------
N_PER_CLASS = 600  # tope por clase para no exceder tiempo

rng = np.random.default_rng(42)

def sample_class(df, cls, n):
    sub = df[df["pathology"] == cls]
    if len(sub) <= n:
        return sub
    idx = rng.choice(len(sub), n, replace=False)
    return sub.iloc[sorted(idx)]

df_mal = sample_class(df_img, "MALIGNANT", N_PER_CLASS)
df_ben = sample_class(df_img, "BENIGN",    N_PER_CLASS)
df_sample = pd.concat([df_mal, df_ben]).reset_index(drop=True)

print(f"\nMuestra para analisis de features: {len(df_sample)} "
      f"({len(df_mal)} MAL + {len(df_ben)} BEN)")
print(f"Prefijos presentes: {sorted(df_sample['prefix'].unique())}")

# Funcion de extraccion de features
def extract_features(row):
    path = row["image_path"]
    try:
        pil = Image.open(path)
        arr = np.array(pil)           # uint16 sin convert
        # Sanidad: verificar que es uint16 real
        assert arr.dtype == np.uint16, f"dtype={arr.dtype}"
        assert arr.max() > 255,        f"max={arr.max()} — se leyo version de 8 bits"

        flat = arr.flatten().astype(np.float32)
        H, W = arr.shape

        # --- ADQUISICION PURA ---
        height = H
        width  = W
        total_pixels = int(H) * int(W)
        p1_raw  = float(np.percentile(flat, 1))
        p5_raw  = float(np.percentile(flat, 5))

        # Proxy de texto quemado: fraccion del top-1% local ubicada en las
        # primeras 2% de filas (fila < 0.02*H)
        thr_top1 = float(np.percentile(flat, 99))
        top1_mask = (arr >= thr_top1)
        top1_area = float(top1_mask.sum())
        top_row_cutoff = max(1, int(0.02 * H))
        top1_in_top_rows = float(top1_mask[:top_row_cutoff, :].sum())
        text_proxy = top1_in_top_rows / max(top1_area, 1)

        # --- INTENSIDAD (ambiguo) ---
        mean_raw = float(flat.mean())
        p50_raw  = float(np.percentile(flat, 50))
        p99_raw  = float(np.percentile(flat, 99))

        return {
            "height": height, "width": width, "total_pixels": total_pixels,
            "p1_raw": p1_raw, "p5_raw": p5_raw,
            "text_proxy": text_proxy,
            "mean_raw": mean_raw, "p50_raw": p50_raw, "p99_raw": p99_raw,
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:80]}

print("\nExtrayendo features de imagen...")
feats = []
for i, (_, row) in enumerate(df_sample.iterrows()):
    if i % 100 == 0:
        print(f"  {i}/{len(df_sample)}", end="\r", flush=True)
    feats.append(extract_features(row))
print(f"  {len(df_sample)}/{len(df_sample)} — listo")

df_feats = pd.DataFrame(feats)
n_errors = (~df_feats["ok"]).sum()
if n_errors > 0:
    print(f"  Errores de lectura: {n_errors}")
    if "error" in df_feats.columns:
        print(df_feats[~df_feats["ok"]]["error"].value_counts().head())

# Combinar con metadatos
df_all = pd.concat([
    df_sample.reset_index(drop=True),
    df_feats.reset_index(drop=True)
], axis=1)
df_all = df_all[df_all["ok"] == True].reset_index(drop=True)
print(f"  Imagenes con features OK: {len(df_all)}")

# ---------------------------------------------------------------------------
# 4. Calcular AUC de cada feature sola como predictor de malignidad
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score

y = (df_all["pathology"] == "MALIGNANT").astype(int).values

ACQU_FEATURES = ["height", "width", "total_pixels", "p1_raw", "p5_raw", "text_proxy"]
INTEN_FEATURES = ["mean_raw", "p50_raw", "p99_raw"]

def compute_auc(col):
    x = df_all[col].values
    if np.std(x) < 1e-10:
        return 0.5
    # Probar AUC con x y con -x, devolver el mas informativo (>0.5)
    a = roc_auc_score(y, x)
    return max(a, 1.0 - a)  # AUC siempre >= 0.5

print()
print("=" * 70)
print("AUC DE FEATURE SOLA (predictor de malignidad)")
print("AUC > 0.60 = riesgo de confound. Limite marcado con ***")
print("=" * 70)

all_results = []

def print_feature_table(feat_list, group_label):
    print(f"\n  --- {group_label} ---")
    print(f"  {'Feature':18s}  {'media_BEN':>10}  {'mediana_BEN':>11}  "
          f"{'media_MAL':>10}  {'mediana_MAL':>11}  {'AUC':>6}  {'Flag':5}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*6}  {'-'*5}")
    for feat in feat_list:
        ben_vals = df_all.loc[df_all["pathology"]=="BENIGN",  feat]
        mal_vals = df_all.loc[df_all["pathology"]=="MALIGNANT", feat]
        auc = compute_auc(feat)
        flag = "***" if auc > 0.60 else ""
        print(f"  {feat:18s}  {ben_vals.mean():10.1f}  {ben_vals.median():11.1f}  "
              f"{mal_vals.mean():10.1f}  {mal_vals.median():11.1f}  "
              f"{auc:6.4f}  {flag}")
        all_results.append({
            "feature": feat, "group": group_label,
            "media_ben": ben_vals.mean(), "mediana_ben": ben_vals.median(),
            "media_mal": mal_vals.mean(), "mediana_mal": mal_vals.median(),
            "auc": auc, "flag": flag,
        })

print_feature_table(ACQU_FEATURES, "ADQUISICION PURA")
print_feature_table(INTEN_FEATURES, "INTENSIDAD (ambiguo)")

# Guardar CSV
pd.DataFrame(all_results).to_csv(OUT_DIR / "auc_features.csv", index=False)
print(f"\n  Guardado: {OUT_DIR}/auc_features.csv")

# ---------------------------------------------------------------------------
# 5. Tabla de contingencia: volumen x patologia
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("CONTINGENCIA: PREFIJO DE VOLUMEN x PATOLOGIA")
print("Tasa global de malignidad marcada con | para comparar")
print("=" * 70)

global_malignancy_rate = (df_all["pathology"] == "MALIGNANT").mean()
print(f"\n  Tasa global de malignidad en la muestra: {global_malignancy_rate:.3f}\n")

contingency = (
    df_all.groupby("prefix")["pathology"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)
if "MALIGNANT" not in contingency.columns:
    contingency["MALIGNANT"] = 0
if "BENIGN" not in contingency.columns:
    contingency["BENIGN"] = 0
contingency["total"] = contingency["MALIGNANT"] + contingency["BENIGN"]
contingency["tasa_mal"] = contingency["MALIGNANT"] / contingency["total"].clip(lower=1)
contingency["desv_vs_global"] = contingency["tasa_mal"] - global_malignancy_rate
contingency = contingency.sort_values("prefix")

print(f"  {'Prefijo':8s}  {'BENIGN':>8}  {'MALIGNANT':>9}  {'Total':>7}  "
      f"{'Tasa_Mal':>9}  {'Desv_vs_global':>14}  {'Flag':5}")
print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*14}  {'-'*5}")

for _, row in contingency.iterrows():
    flag = "***" if abs(row["desv_vs_global"]) > 0.20 else ""
    print(f"  {str(row['prefix']):8s}  {int(row['BENIGN']):>8}  {int(row['MALIGNANT']):>9}  "
          f"{int(row['total']):>7}  {row['tasa_mal']:>9.3f}  "
          f"{row['desv_vs_global']:>+14.3f}  {flag}")

contingency.to_csv(OUT_DIR / "contingency_prefix.csv", index=False)
print(f"\n  Guardado: {OUT_DIR}/contingency_prefix.csv")

# AUC del prefijo como feature numerica (ordinal A=0, B=1, ...)
prefix_sorted = sorted(df_all["prefix"].unique())
prefix_map = {p: i for i, p in enumerate(prefix_sorted)}
df_all["prefix_num"] = df_all["prefix"].map(prefix_map)
auc_prefix = compute_auc("prefix_num")
print(f"\n  AUC del prefijo como feature ordinal ({prefix_sorted}): {auc_prefix:.4f}")

# ---------------------------------------------------------------------------
# 6. Distribucion de prefijos por clase (para detectar desequilibrio)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("DISTRIBUCION DE PREFIJOS POR CLASE")
print("=" * 70)
prefix_dist = df_all.groupby(["pathology", "prefix"]).size().unstack(fill_value=0)
print(prefix_dist.to_string())
print()

# Calcular chi2 de independencia prefijo x patologia
from scipy.stats import chi2_contingency
ct = pd.crosstab(df_all["prefix"], df_all["pathology"])
chi2, p_val, dof, expected = chi2_contingency(ct)
print(f"  Chi2(prefijo x patologia): chi2={chi2:.2f}  p={p_val:.4f}  dof={dof}")
print(f"  Interpretacion: p{'<' if p_val < 0.05 else '>'}0.05 → "
      f"{'distribucion de prefijos NO es independiente de la patologia (confound potencial)' if p_val < 0.05 else 'distribucion de prefijos es independiente de la patologia'}")

# ---------------------------------------------------------------------------
# 7. Resumen ejecutivo con numeros
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("RESUMEN EJECUTIVO")
print("=" * 70)

high_auc_acq = [r for r in all_results if r["group"]=="ADQUISICION PURA" and r["auc"]>0.60]
high_auc_int = [r for r in all_results if r["group"]=="INTENSIDAD (ambiguo)" and r["auc"]>0.60]

print(f"\n  Imagenes analizadas: {len(df_all)}  (MAL={int(y.sum())}  BEN={int((1-y).sum())})")
print(f"\n  Features de ADQUISICION PURA con AUC > 0.60 (confound inequivoco):")
if high_auc_acq:
    for r in high_auc_acq:
        print(f"    {r['feature']:18s}  AUC={r['auc']:.4f}")
else:
    print("    NINGUNA")

print(f"\n  Features de INTENSIDAD con AUC > 0.60 (puede ser biologia):")
if high_auc_int:
    for r in high_auc_int:
        print(f"    {r['feature']:18s}  AUC={r['auc']:.4f}")
else:
    print("    NINGUNA")

print(f"\n  AUC del prefijo de volumen (adquisicion pura): {auc_prefix:.4f}  "
      f"{'*** CONFOUND POTENCIAL' if auc_prefix > 0.60 else 'OK'}")
print(f"  Chi2 prefijo x patologia: p={p_val:.4f}  "
      f"{'*** DEPENDENCIA ESTADISTICA' if p_val < 0.05 else 'OK'}")

# Prefijos muy sesgados
biased = contingency[abs(contingency["desv_vs_global"]) > 0.20]
if len(biased) > 0:
    print(f"\n  Volumenes con tasa de malignidad muy desviada (>±0.20 del global):")
    for _, row in biased.iterrows():
        print(f"    prefijo={row['prefix']:3s}  tasa_mal={row['tasa_mal']:.3f}  "
              f"desv={row['desv_vs_global']:+.3f}  total={int(row['total'])}")
else:
    print(f"\n  Ningún volumen con tasa de malignidad desviada >±0.20 del global.")

print()
print("Archivos guardados:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f}")
