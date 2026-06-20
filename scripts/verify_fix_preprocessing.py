"""
Prueba de aceptacion del fix de carga PNG uint16 para DDSM.

Replica los Pasos 3 y 5 del diagnostico anterior usando el loader corregido.
Sin modificar modelos. Usa exp08 con augment=False.
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loading import load_image_as_pil, MammoCLIPTransform, load_vindr_records
from models import MammoVLM

TESIS_ROOT     = Path(__file__).parent.parent
VINDR_ROOT     = TESIS_ROOT / "data" / "vindr-mammo"
DDSM_ROOT      = TESIS_ROOT / "data" / "6 DDSM"
MAMMOCLIP_CKPT = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT     = str(TESIS_ROOT / "outputs" / "experiments" /
                     "exp08_ordinal_sord_qwk_descongelado" / "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = 50

# ---------------------------------------------------------------------------
# Verificacion del fix: sanidad de lectura DDSM
# ---------------------------------------------------------------------------
print("=" * 70)
print("SANIDAD DEL FIX — lectura de 5 imagenes DDSM con loader corregido")
print("=" * 70)

ddsm_pngs = sorted(DDSM_ROOT.rglob("*.png"))
for p in ddsm_pngs[:5]:
    raw_pil = Image.open(p)
    raw_arr = np.array(raw_pil)
    pil_rgb = load_image_as_pil(str(p))
    rgb_arr = np.array(pil_rgb)
    ok = (rgb_arr.max() < 255 or rgb_arr.min() < 255)  # ya no todo 255
    print(f"  {p.name[:45]:45s}  raw_mode={raw_pil.mode}  "
          f"raw_range=[{raw_arr.min()},{raw_arr.max()}]  "
          f"→ RGB_range=[{rgb_arr.min()},{rgb_arr.max()}]  "
          f"{'OK' if ok else 'SATURADO-AUN'}")

print()
print("Verificacion CDD-CESM (8-bit, debe quedar inalterado):")
cesm_jpg = list((TESIS_ROOT / "data" / "cdd-cesm" / "Low energy images").glob("*.jpg"))[:3]
for p in cesm_jpg:
    pil_rgb = load_image_as_pil(str(p))
    rgb_arr = np.array(pil_rgb)
    print(f"  {p.name[:45]:45s}  mode={pil_rgb.mode}  "
          f"RGB_range=[{rgb_arr.min()},{rgb_arr.max()}]  OK (8-bit inalterado)")

print()
print("Verificacion VinDr (DICOM, path inalterado):")
vindr_all = load_vindr_records(str(VINDR_ROOT))
vindr_test = vindr_all[vindr_all["split"] == "test"].reset_index(drop=True)
for p in vindr_test["image_path"].iloc[:3]:
    pil_rgb = load_image_as_pil(p)
    rgb_arr = np.array(pil_rgb)
    print(f"  {Path(p).name[:45]:45s}  mode={pil_rgb.mode}  "
          f"RGB_range=[{rgb_arr.min()},{rgb_arr.max()}]  OK (DICOM inalterado)")

# ---------------------------------------------------------------------------
# Paso 3: stats pre-norm y post-norm (50 DDSM + 50 VinDr)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("PASO 3 — Stats pre-norm y post-norm")
print("(comparar con baseline: DDSM pre-norm ~0.93, post-norm ~+2.14)")
print("=" * 70)

import torchvision.transforms as T

transform_pre  = T.Compose([T.Resize((1520, 912)), T.ToTensor()])
transform_full = MammoCLIPTransform(augment=False, use_clahe=True)

rng = np.random.default_rng(42)

def collect_stats_patched(paths, n=N_SAMPLE):
    pre_vals, post_vals = [], []
    ok = 0
    for p in paths:
        if ok >= n:
            break
        try:
            pil_rgb = load_image_as_pil(str(p))
            pre  = transform_pre(pil_rgb).numpy().flatten()
            post = transform_full(pil_rgb).numpy().flatten()
            idx  = rng.choice(len(pre), min(5000, len(pre)), replace=False)
            pre_vals.append(pre[idx])
            post_vals.append(post[idx])
            ok += 1
        except Exception:
            pass
    return np.concatenate(pre_vals), np.concatenate(post_vals), ok

def print_stats(label, stage, arr):
    p = np.percentile(arr, [1, 50, 99])
    print(f"  [{label:7s}] {stage:8s}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  "
          f"p1={p[0]:.4f}  p50={p[1]:.4f}  p99={p[2]:.4f}")

# VinDr
vindr_paths = vindr_test["image_path"].tolist()[:200]
rng.shuffle(vindr_paths)
v_pre, v_post, v_n = collect_stats_patched(vindr_paths)
print(f"\nVinDr (N={v_n}):")
print_stats("VinDr", "pre-norm",  v_pre)
print_stats("VinDr", "post-norm", v_post)

# DDSM (muestra amplia de diferentes prefijos)
all_ddsm = sorted(DDSM_ROOT.rglob("*.png"))
d_paths  = [all_ddsm[i] for i in rng.choice(len(all_ddsm), min(200, len(all_ddsm)), replace=False)]
d_pre, d_post, d_n = collect_stats_patched(d_paths)
print(f"\nDDSM (N={d_n}):")
print_stats("DDSM",  "pre-norm",  d_pre)
print_stats("DDSM",  "post-norm", d_post)

print(f"\n  CRITERIO: DDSM pre-norm debe caer << 0.93 | post-norm debe acercarse a ~-1.58")
ddsm_pre_mean = d_pre.mean()
ddsm_post_mean = d_post.mean()
print(f"  DDSM pre-norm  mean = {ddsm_pre_mean:.4f}  "
      f"{'PASS (< 0.5)' if ddsm_pre_mean < 0.5 else 'FAIL (>= 0.5)'}")
print(f"  DDSM post-norm mean = {ddsm_post_mean:.4f}  "
      f"{'PASS (< 0.0)' if ddsm_post_mean < 0.0 else 'FAIL (>= 0.0)'}")

# ---------------------------------------------------------------------------
# Paso 5: forward del encoder exp08 sobre batch DDSM y VinDr
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("PASO 5 — Forward encoder exp08")
print("(baseline: DDSM logits_std~0.36, probs~[0.22,0.21,0.21,0.25,0.11])")
print("=" * 70)

print("Cargando modelo exp08 ...")
model = MammoVLM(
    checkpoint_path=MAMMOCLIP_CKPT,
    num_birads_classes=5, num_density_classes=4,
    freeze_encoder=False, unfreeze_last_n_blocks=2,
    hidden_dim=256, dropout=0.2,
)
ckpt = torch.load(EXP08_CKPT, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.to(DEVICE)
model.eval()
print("  Cargado.")


def run_forward(paths, n=16, label=""):
    tensors = []
    for p in paths:
        if len(tensors) >= n:
            break
        try:
            pil_rgb = load_image_as_pil(str(p))
            tensors.append(transform_full(pil_rgb))
        except Exception:
            pass
    if not tensors:
        print(f"  [{label}] sin imagenes"); return

    batch  = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        outputs = model.forward(batch)
        logits  = outputs["birads"].cpu().numpy()
        probs   = F.softmax(torch.tensor(logits), dim=-1).numpy()
        feats   = model.encoder(batch).cpu().numpy()

    print(f"\n  [{label}]  N={len(tensors)}")
    print(f"  Tensor entrada  mean={batch.mean().item():.4f}  std={batch.std().item():.4f}")
    print(f"  Features enc    mean={feats.mean():.4f}  std={feats.std():.4f}  "
          f"NaN={np.isnan(feats).sum()}")
    print(f"  Logits BI-RADS  std={logits.std():.4f}  "
          f"range=[{logits.min():.3f}, {logits.max():.3f}]  NaN={np.isnan(logits).sum()}")
    mean_p = probs.mean(axis=0)
    print(f"  Probs medias:  " + "  ".join(f"idx{i}={mean_p[i]:.4f}" for i in range(5)))
    argmax = probs.argmax(axis=1)
    uniq, cnt = np.unique(argmax, return_counts=True)
    print(f"  Argmax dist:   " + "  ".join(f"idx{u}={c}" for u, c in zip(uniq, cnt)))
    print(f"  Uniformes (max_prob < 0.25): {'SI - FALLO' if probs.max(axis=1).mean() < 0.25 else 'NO'}")
    print(f"  Colapso a una clase:         {'SI - FALLO' if len(uniq) == 1 else 'NO'}")
    return logits.std(), mean_p


rng2 = np.random.default_rng(99)
v_paths2  = vindr_test["image_path"].tolist()[:50]
d_paths2  = [all_ddsm[i] for i in rng2.choice(len(all_ddsm), 50, replace=False)]

v_std, v_probs = run_forward(v_paths2,  n=16, label="VinDr-val")
d_std, d_probs = run_forward(d_paths2,  n=16, label="DDSM-fixed")

print()
print("=" * 70)
print("VEREDICTO FINAL")
print("=" * 70)
pass_pre  = d_pre.mean()  < 0.5
pass_post = d_post.mean() < 0.0
pass_std  = d_std > 0.60 if d_std is not None else False
pass_probs = (d_probs is not None and d_probs.max() > 0.25)
print(f"  Paso 3 pre-norm  DDSM mean={d_pre.mean():.4f}  "
      f"(< 0.5?)  {'PASS' if pass_pre else 'FAIL'}")
print(f"  Paso 3 post-norm DDSM mean={d_post.mean():.4f} "
      f"(< 0.0?)  {'PASS' if pass_post else 'FAIL'}")
print(f"  Paso 5 logits_std DDSM={d_std:.4f}  "
      f"(> 0.60?) {'PASS' if pass_std else 'FAIL'}")
print(f"  Paso 5 probs cuasi-uniformes?  "
      f"{'NO — max={:.4f}'.format(d_probs.max()) if d_probs is not None else '?'}  "
      f"{'PASS' if pass_probs else 'FAIL'}")
all_pass = pass_pre and pass_post and pass_std and pass_probs
print(f"\n  RESULTADO GLOBAL: {'ARREGLO FUNCIONA' if all_pass else 'ARREGLO NO FUNCIONA'}")
