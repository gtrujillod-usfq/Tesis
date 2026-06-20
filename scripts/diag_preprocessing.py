"""
Diagnostico de preprocesamiento: DDSM vs VinDr con el transform de exp08.
Sin modificar codigo ni modelos.
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from PIL import Image

# -------------------------------------------------------------------
# PASO 0: importar los mismos modulos que usa exp08
# -------------------------------------------------------------------
from data_loading import (
    load_image_as_pil,
    MammoCLIPTransform,
    load_vindr_records,
)

TESIS_ROOT = Path(__file__).parent.parent
VINDR_ROOT = TESIS_ROOT / "data" / "vindr-mammo"
DDSM_ROOT  = TESIS_ROOT / "data" / "6 DDSM"
MAMMOCLIP_CKPT = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT = str(TESIS_ROOT / "outputs" / "experiments" /
                 "exp08_ordinal_sord_qwk_descongelado" / "model.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = 50

# -------------------------------------------------------------------
# PASO 1: TRANSFORM DE EVALUACION (mostrar cada etapa y stats de norm)
# -------------------------------------------------------------------
print("=" * 70)
print("PASO 1 — Transform de evaluacion de exp08")
print("=" * 70)

_t = MammoCLIPTransform(augment=False, use_clahe=True)
_t._build_transform()
print("use_clahe         :", _t.use_clahe)
print("Resize destino    :", _t.height, "x", _t.width, "  (alto x ancho)")
print("Augmentaciones    : NINGUNA  (augment=False)")
print("ToTensor          : divide por 255 → [0, 1]")
print("Normalize stats   : FIJAS — ImageNet")
print(f"  mean = {_t.IMAGENET_MEAN}")
print(f"  std  = {_t.IMAGENET_STD}")
print()
print("Pipeline torchvision construido:")
for step in _t._transform.transforms:
    print(" ", step)
print()
print(">>> H2 DESCARTADA: normalizacion es FIJA (ImageNet), NO por dataset.")
print("    Las stats POST-normalizacion SON informativas y comparables.")

# -------------------------------------------------------------------
# PASO 2: canales y profundidad de bits — codigo + valores reales
# -------------------------------------------------------------------
print()
print("=" * 70)
print("PASO 2 — Canales y profundidad de bits")
print("=" * 70)

# --- VinDr: DICOM sample ---
vindr_all = load_vindr_records(str(VINDR_ROOT))
vindr_test = vindr_all[vindr_all["split"] == "test"].reset_index(drop=True)
vindr_sample_paths = vindr_test["image_path"].iloc[:5].tolist()

print("\n-- VinDr (DICOM) --")
print("load_image_as_pil: pydicom → VOI LUT → min-max → uint8 → PIL RGB")
for p in vindr_sample_paths[:3]:
    import pydicom
    ds = pydicom.dcmread(p)
    raw = ds.pixel_array
    pil_rgb = load_image_as_pil(p)
    arr_rgb = np.array(pil_rgb)
    print(f"  {Path(p).name[:40]}  raw dtype={raw.dtype} shape={raw.shape} "
          f"raw_range=[{raw.min()},{raw.max()}]  "
          f"→ PIL mode={pil_rgb.mode} dtype=uint8 "
          f"RGB_range=[{arr_rgb.min()},{arr_rgb.max()}]")

# --- DDSM: PNG sample ---
print()
print("-- DDSM (PNG) --")
print("load_image_as_pil: Image.open().convert('RGB')  — SIN VOI LUT ni min-max")
ddsm_png_files = sorted(DDSM_ROOT.rglob("*.png"))[:100]
if not ddsm_png_files:
    ddsm_png_files = sorted(DDSM_ROOT.rglob("*.LJPEG.png"))[:100]

for p in ddsm_png_files[:3]:
    raw_pil = Image.open(p)
    raw_arr = np.array(raw_pil)  # uint16
    pil_rgb = load_image_as_pil(str(p))
    arr_rgb = np.array(pil_rgb)
    print(f"  {p.name[:40]}  raw mode={raw_pil.mode} dtype={raw_arr.dtype} "
          f"shape={raw_arr.shape} raw_range=[{raw_arr.min()},{raw_arr.max()}]  "
          f"→ PIL mode={pil_rgb.mode} dtype=uint8 "
          f"RGB_range=[{arr_rgb.min()},{arr_rgb.max()}]")

# -------------------------------------------------------------------
# PASO 3: stats por etapa (antes y despues de normalizar)
# -------------------------------------------------------------------
print()
print("=" * 70)
print("PASO 3 — Stats por etapa y dataset (N =", N_SAMPLE, "imagenes)")
print("=" * 70)

# Construir dos transforms: 1) solo resize+to_tensor, 2) completo
import torchvision.transforms as T

transform_pre = T.Compose([
    T.Resize((1520, 912)),
    T.ToTensor(),   # [0,1] sin normalizar
])
transform_full = MammoCLIPTransform(augment=False, use_clahe=True)


def collect_stats(paths, loader_fn, label, n=N_SAMPLE):
    """Retorna (pre_stats, post_stats) sobre n imagenes."""
    pre_vals = []
    post_vals = []
    errors = []
    sampled = 0
    for p in paths:
        if sampled >= n:
            break
        try:
            pil_rgb = loader_fn(str(p))
            # pre: resize + to_tensor (CLAHE incluido porque es parte del loader)
            pre_t = transform_pre(pil_rgb).numpy().flatten()
            # post: transform completo
            post_t = transform_full(pil_rgb).numpy().flatten()
            # subsample para no saturar memoria
            idx = np.random.choice(len(pre_t), min(5000, len(pre_t)), replace=False)
            pre_vals.append(pre_t[idx])
            post_vals.append(post_t[idx])
            sampled += 1
        except Exception as e:
            errors.append(str(e)[:80])
    if errors:
        print(f"  [{label}] {len(errors)} errores al cargar, primero: {errors[0]}")
    pre_all  = np.concatenate(pre_vals)  if pre_vals  else np.array([0.])
    post_all = np.concatenate(post_vals) if post_vals else np.array([0.])
    return pre_all, post_all, sampled


def print_stats(label, stage, arr):
    p = np.percentile(arr, [1, 50, 99])
    print(f"  [{label:6s}] {stage:8s}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  "
          f"p1={p[0]:.4f}  p50={p[1]:.4f}  p99={p[2]:.4f}")


np.random.seed(42)

# -- VinDr --
print("\nVinDr (test split, DICOM → uint8 RGB):")
vindr_paths = vindr_test["image_path"].tolist()[:200]
np.random.shuffle(vindr_paths)
vindr_pre, vindr_post, vindr_n = collect_stats(vindr_paths, load_image_as_pil, "VinDr")
print(f"  Imagenes procesadas: {vindr_n}")
print_stats("VinDr", "pre-norm", vindr_pre)
print_stats("VinDr", "post-norm", vindr_post)

# -- DDSM --
print(f"\nDDSM (PNG → convert('RGB')):")
all_ddsm = sorted(DDSM_ROOT.rglob("*.png"))
if len(all_ddsm) > 500:
    ddsm_sample = sorted(np.random.choice(all_ddsm, 200, replace=False).tolist(),
                         key=lambda p: str(p))
else:
    ddsm_sample = all_ddsm[:200]
ddsm_pre, ddsm_post, ddsm_n = collect_stats(
    ddsm_sample, load_image_as_pil, "DDSM"
)
print(f"  Imagenes procesadas: {ddsm_n}")
print_stats("DDSM", "pre-norm", ddsm_pre)
print_stats("DDSM", "post-norm", ddsm_post)

# -------------------------------------------------------------------
# PASO 4: artefactos de pelicula en DDSM
# -------------------------------------------------------------------
print()
print("=" * 70)
print("PASO 4 — Artefactos de pelicula en DDSM")
print("=" * 70)

n_saturated = 0
n_dark = 0
n_total = 0
max_vals = []
mean_vals = []
sample_paths_ddsm = sorted(DDSM_ROOT.rglob("*.png"))[:200]

for p in sample_paths_ddsm:
    try:
        pil_rgb = load_image_as_pil(str(p))
        arr = np.array(pil_rgb).astype(np.float32)
        mx = arr.max()
        mn_mean = arr.mean()
        max_vals.append(mx)
        mean_vals.append(mn_mean)
        if mx >= 254:
            n_saturated += 1
        if mn_mean < 5.0:
            n_dark += 1
        n_total += 1
    except Exception:
        pass

if n_total > 0:
    print(f"  Imagenes revisadas    : {n_total}")
    print(f"  Max saturado (>=254)  : {n_saturated}  ({100*n_saturated/n_total:.1f}%)")
    print(f"  Muy oscuras (mean<5)  : {n_dark}  ({100*n_dark/n_total:.1f}%)")
    print(f"  Distribucion de maximos (uint8 post-convert):")
    mv = np.array(max_vals)
    print(f"    mean={mv.mean():.1f}  std={mv.std():.1f}  "
          f"p1={np.percentile(mv,1):.1f}  p50={np.percentile(mv,50):.1f}  "
          f"p99={np.percentile(mv,99):.1f}")
    mean_v = np.array(mean_vals)
    print(f"  Distribucion de medias (uint8 post-convert):")
    print(f"    mean={mean_v.mean():.1f}  std={mean_v.std():.1f}  "
          f"p1={np.percentile(mean_v,1):.1f}  p50={np.percentile(mean_v,50):.1f}  "
          f"p99={np.percentile(mean_v,99):.1f}")

# Ejemplo de imagen especifica para ver distribucion de pixeles
print()
print("  Muestra de 5 imagenes DDSM (raw uint16 vs post-convert uint8):")
for p in sample_paths_ddsm[:5]:
    try:
        raw_pil = Image.open(p)
        raw_arr = np.array(raw_pil)
        pil_rgb = load_image_as_pil(str(p))
        arr_rgb = np.array(pil_rgb)
        ch_gray = arr_rgb[:, :, 0]  # R=G=B after grayscale→RGB
        print(f"    {p.name[:45]}")
        print(f"      raw  uint16: range=[{raw_arr.min()},{raw_arr.max()}]  "
              f"mean={raw_arr.mean():.1f}  p99={np.percentile(raw_arr,99):.0f}")
        print(f"      post uint8 : range=[{ch_gray.min()},{ch_gray.max()}]  "
              f"mean={ch_gray.mean():.1f}  p99={np.percentile(ch_gray,99):.0f}")
    except Exception as e:
        print(f"    {p.name[:45]}  ERROR: {e}")

# -------------------------------------------------------------------
# PASO 5: salidas del encoder — batch DDSM vs VinDr
# -------------------------------------------------------------------
print()
print("=" * 70)
print("PASO 5 — Salidas del encoder exp08 (forward pass)")
print("=" * 70)

from models import MammoVLM
import torch.nn.functional as F

print("Cargando modelo exp08 ...")
model = MammoVLM(
    checkpoint_path=MAMMOCLIP_CKPT,
    num_birads_classes=5,
    num_density_classes=4,
    freeze_encoder=False,
    unfreeze_last_n_blocks=2,
    hidden_dim=256,
    dropout=0.2,
)
ckpt = torch.load(EXP08_CKPT, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.to(DEVICE)
model.eval()
print("  Modelo cargado.")


def run_encoder_batch(paths, loader_fn, label, n=16):
    """Corre forward sobre n imagenes, retorna probs y logits."""
    tensors = []
    loaded = 0
    for p in paths:
        if loaded >= n:
            break
        try:
            pil_rgb = loader_fn(str(p))
            t = transform_full(pil_rgb)
            tensors.append(t)
            loaded += 1
        except Exception as e:
            pass
    if not tensors:
        print(f"  [{label}] sin imagenes cargadas")
        return

    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        outputs = model.forward(batch)
        logits = outputs["birads"]                              # [N, 5]
        probs  = F.softmax(logits, dim=-1).cpu().numpy()       # [N, 5]
        logits_np = logits.cpu().numpy()

    # Features del encoder (antes del head)
    with torch.no_grad():
        feats = model.encoder(batch).cpu().numpy()             # [N, 2048]

    print(f"\n  [{label}]  N={len(tensors)}")
    print(f"  Tensor de entrada — mean={batch.mean().item():.4f}  "
          f"std={batch.std().item():.4f}  "
          f"min={batch.min().item():.4f}  max={batch.max().item():.4f}")
    print(f"  Features encoder  — mean={feats.mean():.4f}  std={feats.std():.4f}  "
          f"min={feats.min():.4f}  max={feats.max():.4f}  "
          f"NaN={np.isnan(feats).sum()}")
    print(f"  Logits BI-RADS    — mean={logits_np.mean():.4f}  std={logits_np.std():.4f}  "
          f"min={logits_np.min():.4f}  max={logits_np.max():.4f}  "
          f"NaN={np.isnan(logits_np).sum()}")
    print(f"  Probabilidades (media sobre batch):")
    mean_probs = probs.mean(axis=0)
    for i, mp in enumerate(mean_probs):
        bar = "#" * int(mp * 40)
        print(f"    idx{i} (BR{i+1}): {mp:.4f}  {bar}")
    argmaxes = probs.argmax(axis=1)
    unique, counts = np.unique(argmaxes, return_counts=True)
    print(f"  Argmax BI-RADS predicho: " +
          ", ".join(f"idx{u}={c}" for u, c in zip(unique, counts)))
    print(f"  Colapso a una clase: {'SI - POSIBLE BUG' if len(unique)==1 else 'NO'}")
    print(f"  Probs uniformes (max<0.25): {'SI - POSIBLE BUG' if probs.max(axis=1).mean()<0.25 else 'NO'}")


# VinDr batch
np.random.shuffle(vindr_paths)
run_encoder_batch(vindr_paths[:50], load_image_as_pil, "VinDr-val", n=16)

# DDSM batch
ddsm_paths_for_enc = sorted(DDSM_ROOT.rglob("*.png"))[:100]
run_encoder_batch(ddsm_paths_for_enc, load_image_as_pil, "DDSM", n=16)

# -------------------------------------------------------------------
# VEREDICTO FINAL
# -------------------------------------------------------------------
print()
print("=" * 70)
print("VEREDICTO")
print("=" * 70)
print("""
Evaluar con los numeros arriba:
  - Si DDSM pre-norm media << VinDr pre-norm media → bug de escalado (16-bit vs 8-bit)
  - Si DDSM features col/saturadas o probs uniformes → encoder no ve señal util
  - Si DDSM logits rango muy diferente a VinDr → regimen distinto
  - Si saturacion max en DDSM >> VinDr → artefactos de pelicula inflando la señal
""")
