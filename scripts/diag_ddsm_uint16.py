"""
Inspeccion de intensidades uint16 de DDSM.
Solo lectura. Sin cargar modelo.

Salidas guardadas en outputs/diag_ddsm/
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

TESIS_ROOT  = Path(__file__).parent.parent
DDSM_ROOT   = TESIS_ROOT / "data" / "6 DDSM"
OUT_DIR     = TESIS_ROOT / "outputs" / "diag_ddsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE = 50   # imagenes para histograma
N_SPATIAL = 8   # imagenes para analisis espacial

# ---------------------------------------------------------------------------
# Util: leer PNG como uint16 real (sin convert)
# ---------------------------------------------------------------------------
def read_uint16(path):
    pil = Image.open(path)
    arr = np.array(pil)               # uint16, shape (H, W)
    assert arr.dtype == np.uint16, f"Esperaba uint16, got {arr.dtype} en {path}"
    return arr


# ---------------------------------------------------------------------------
# Seleccionar imagenes de muestra (mezcla benign + cancer)
# ---------------------------------------------------------------------------
all_pngs = sorted(DDSM_ROOT.rglob("*.png"))
print(f"Total PNGs: {len(all_pngs)}")

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(all_pngs), min(N_SAMPLE + 20, len(all_pngs)), replace=False)
sample_paths = [all_pngs[i] for i in sorted(sample_idx)]

# Verificar que estan bien leidas (control de sanidad)
print("\n-- Verificacion de lectura uint16 (primeras 5 imagenes) --")
for p in sample_paths[:5]:
    arr = read_uint16(p)
    print(f"  {p.name[:48]:48s}  dtype={arr.dtype}  shape={arr.shape}  "
          f"range=[{arr.min()},{arr.max()}]  mean={arr.mean():.0f}")

if all(read_uint16(p).max() <= 255 for p in sample_paths[:3]):
    print("ERROR: maximos <= 255 — se esta leyendo la version de 8 bits")
    sys.exit(1)
print("  OK: valores uint16 reales confirmados.\n")


# ===========================================================================
# PASO 1: Histogramas
# ===========================================================================
print("=" * 68)
print("PASO 1 — Histogramas de valores uint16")
print("=" * 68)

# Acumular todos los pixeles (subsample para no saturar RAM)
all_pixels = []
per_image_stats = []

for p in sample_paths[:N_SAMPLE]:
    arr = read_uint16(p)
    flat = arr.flatten()
    # Subsample: maximo 20000 pixeles por imagen
    idx = rng.choice(len(flat), min(20000, len(flat)), replace=False)
    all_pixels.append(flat[idx])
    per_image_stats.append({
        "name": p.name,
        "min": int(flat.min()), "max": int(flat.max()),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "p999": float(np.percentile(flat, 99.9)),
        "p9999": float(np.percentile(flat, 99.99)),
    })

all_pixels = np.concatenate(all_pixels)

# Fracciones por extremos
p01_val    = float(np.percentile(all_pixels, 1))
p99_val    = float(np.percentile(all_pixels, 99))
p999_val   = float(np.percentile(all_pixels, 99.9))
p9999_val  = float(np.percentile(all_pixels, 99.99))

frac_above_99   = float((all_pixels > p99_val).mean())
frac_above_999  = float((all_pixels > p999_val).mean())
frac_above_9999 = float((all_pixels > p9999_val).mean())
frac_below_p01  = float((all_pixels < p01_val).mean())

print(f"N imagenes analizadas : {N_SAMPLE}")
print(f"Total pixeles muestreados: {len(all_pixels):,}")
print()
print(f"Percentiles globales:")
print(f"  p1    = {p01_val:.0f}")
print(f"  p50   = {np.percentile(all_pixels,50):.0f}")
print(f"  p99   = {p99_val:.0f}")
print(f"  p99.9 = {p999_val:.0f}")
print(f"  p99.99= {p9999_val:.0f}")
print(f"  max   = {all_pixels.max():.0f}")
print()
print(f"Fraccion de pixeles:")
print(f"  < p1      : {frac_below_p01*100:.3f}%   (fondo de pelicula)")
print(f"  > p99     : {frac_above_99*100:.3f}%")
print(f"  > p99.9   : {frac_above_999*100:.4f}%")
print(f"  > p99.99  : {frac_above_9999*100:.5f}%")

# Histograma agregado (guardado como PNG)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts, bin_edges = np.histogram(all_pixels, bins=500)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

axes[0].bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]),
            color="steelblue", alpha=0.7)
axes[0].set_title("Histograma agregado — valores uint16 DDSM (N=50 imgs)")
axes[0].set_xlabel("Valor uint16")
axes[0].set_ylabel("Frecuencia")
axes[0].axvline(p99_val, color="orange", lw=1.5, label=f"p99={p99_val:.0f}")
axes[0].axvline(p999_val, color="red", lw=1.5, label=f"p99.9={p999_val:.0f}")
axes[0].legend()

axes[1].bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]),
            color="steelblue", alpha=0.7)
axes[1].set_yscale("log")
axes[1].set_title("Histograma agregado — escala log-Y")
axes[1].set_xlabel("Valor uint16")
axes[1].set_ylabel("Frecuencia (log)")
axes[1].axvline(p99_val, color="orange", lw=1.5, label=f"p99={p99_val:.0f}")
axes[1].axvline(p999_val, color="red", lw=1.5, label=f"p99.9={p999_val:.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "hist_agregado.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\nGuardado: {OUT_DIR}/hist_agregado.png")

# Histogramas individuales de 4 imagenes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, p in zip(axes.flatten(), sample_paths[:4]):
    arr = read_uint16(p)
    flat = arr.flatten()
    ax.hist(flat, bins=300, color="steelblue", alpha=0.7, log=True)
    ax.set_title(p.name[:45], fontsize=8)
    ax.set_xlabel("Valor uint16")
    ax.set_ylabel("Freq (log)")
    pv = np.percentile(flat, [1, 99, 99.9])
    for val, col, lbl in zip(pv, ["green","orange","red"],
                              [f"p1={pv[0]:.0f}", f"p99={pv[1]:.0f}", f"p99.9={pv[2]:.0f}"]):
        ax.axvline(val, color=col, lw=1.2, label=lbl)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_individuales.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Guardado: {OUT_DIR}/hist_individuales.png")

# Stats de las primeras 10 imagenes (tabla)
print("\nStats por imagen (primeras 10):")
print(f"{'Imagen':50s}  {'min':>6}  {'max':>6}  {'p1':>6}  {'p99':>6}  {'p99.9':>7}  {'p99.99':>8}")
for s in per_image_stats[:10]:
    print(f"{s['name'][:50]:50s}  {s['min']:>6}  {s['max']:>6}  "
          f"{s['p01']:>6.0f}  {s['p99']:>6.0f}  {s['p999']:>7.0f}  {s['p9999']:>8.0f}")


# ===========================================================================
# PASO 2: Caracterizacion espacial del extremo alto
# ===========================================================================
print()
print("=" * 68)
print("PASO 2 — Caracterizacion espacial del extremo alto")
print("=" * 68)

spatial_paths = sample_paths[:N_SPATIAL]

def analyze_bright_mask(arr, threshold_percentile):
    """Componentes conexas de la mascara del top X%."""
    thresh = np.percentile(arr, threshold_percentile)
    mask = (arr >= thresh).astype(np.uint8)
    labeled, n_components = ndimage.label(mask)

    H, W = arr.shape
    comp_stats = []
    for lbl in range(1, min(n_components + 1, 200)):  # max 200 componentes
        comp_mask = (labeled == lbl)
        area = int(comp_mask.sum())
        if area == 0:
            continue
        rows, cols = np.where(comp_mask)
        cy, cx = float(rows.mean()), float(cols.mean())
        min_r, max_r = int(rows.min()), int(rows.max())
        min_c, max_c = int(cols.min()), int(cols.max())
        # Compacidad: area / (bounding box area)
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        compactness = area / max(bbox_area, 1)
        # Posicion relativa al centro de la imagen
        dist_from_center = np.sqrt(((cy/H) - 0.5)**2 + ((cx/W) - 0.5)**2)
        # En borde? (dentro de 5% del borde)
        border_margin = 0.05
        on_border = (cy/H < border_margin or cy/H > 1-border_margin or
                     cx/W < border_margin or cx/W > 1-border_margin)
        in_corner = ((cy/H < 0.15 or cy/H > 0.85) and
                     (cx/W < 0.15 or cx/W > 0.85))
        comp_stats.append({
            "area": area,
            "cy_rel": cy/H, "cx_rel": cx/W,
            "compactness": compactness,
            "dist_from_center": dist_from_center,
            "on_border": on_border,
            "in_corner": in_corner,
        })

    # Ordenar por area descendente
    comp_stats.sort(key=lambda x: x["area"], reverse=True)
    return mask, n_components, comp_stats, thresh


# Figuras overlay
fig_ov, axes_ov = plt.subplots(2, 4, figsize=(20, 10))

for img_idx, p in enumerate(spatial_paths):
    arr = read_uint16(p)
    H, W = arr.shape

    # Normalizar para visualizacion (min-max)
    arr_f = arr.astype(np.float32)
    arr_norm = (arr_f - arr_f.min()) / max(arr_f.max() - arr_f.min(), 1)

    mask_1pct,  n1,  stats_1pct,  thr1  = analyze_bright_mask(arr, 99)
    mask_01pct, n01, stats_01pct, thr01 = analyze_bright_mask(arr, 99.9)

    # Overlay: imagen + mascara top-1% en rojo
    ax = axes_ov.flatten()[img_idx]
    rgb_vis = np.stack([arr_norm]*3, axis=-1)
    rgb_vis[mask_1pct == 1, 0] = 1.0
    rgb_vis[mask_1pct == 1, 1] = 0.0
    rgb_vis[mask_1pct == 1, 2] = 0.0
    ax.imshow(rgb_vis, aspect="auto")
    ax.set_title(f"{p.name[:35]}\nthr1%={thr1:.0f}  n_comp={n1}", fontsize=7)
    ax.axis("off")

    # Stats de componentes
    total_mask_area = mask_1pct.sum()
    n_large  = sum(1 for s in stats_1pct if s["area"] > 0.001 * H * W)
    n_border = sum(1 for s in stats_1pct if s["on_border"])
    n_corner = sum(1 for s in stats_1pct if s["in_corner"])
    sizes    = [s["area"] for s in stats_1pct]
    compact  = [s["compactness"] for s in stats_1pct]
    dist_c   = [s["dist_from_center"] for s in stats_1pct]

    print(f"\n  {p.name}")
    print(f"    Shape: {H}x{W}   top-1% thr={thr1:.0f}")
    print(f"    top-1% : {n1:4d} comp | mask_area={total_mask_area} pix "
          f"({100*total_mask_area/(H*W):.2f}%)")
    if sizes:
        print(f"      tamaños: max={max(sizes)}  median={int(np.median(sizes))}  "
              f"p90={int(np.percentile(sizes,90))}")
        print(f"      compact: mean={np.mean(compact):.3f}  min={min(compact):.3f}")
        print(f"      dist_centro: mean={np.mean(dist_c):.3f}  "
              f"min={min(dist_c):.3f}  max={max(dist_c):.3f}")
        print(f"      en borde: {n_border}/{n1}  en esquina: {n_corner}/{n1}")
        # Top 5 componentes mas grandes
        print(f"      Top-5 comp (area, cy_rel, cx_rel, compact, corner):")
        for s in stats_1pct[:5]:
            print(f"        area={s['area']:6d}  cy={s['cy_rel']:.3f}  cx={s['cx_rel']:.3f}  "
                  f"comp={s['compactness']:.3f}  borde={s['on_border']}  esquina={s['in_corner']}")

    total_mask_area_01 = mask_01pct.sum()
    n_border_01 = sum(1 for s in stats_01pct if s["on_border"])
    n_corner_01 = sum(1 for s in stats_01pct if s["in_corner"])
    sizes_01 = [s["area"] for s in stats_01pct]
    print(f"    top-0.1%: {n01:4d} comp | mask_area={total_mask_area_01} pix "
          f"({100*total_mask_area_01/(H*W):.3f}%)")
    if sizes_01:
        print(f"      tamaños: max={max(sizes_01)}  median={int(np.median(sizes_01))}  "
              f"p90={int(np.percentile(sizes_01,90))}")
        print(f"      en borde: {n_border_01}/{n01}  en esquina: {n_corner_01}/{n01}")

plt.tight_layout()
plt.savefig(OUT_DIR / "overlay_top1pct.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\nGuardado: {OUT_DIR}/overlay_top1pct.png")

# Overlays del top-0.1% por separado (mas pequeno → etiquetas mas visibles)
fig_sm, axes_sm = plt.subplots(2, 4, figsize=(20, 10))
for img_idx, p in enumerate(spatial_paths):
    arr = read_uint16(p)
    arr_f = arr.astype(np.float32)
    arr_norm = (arr_f - arr_f.min()) / max(arr_f.max() - arr_f.min(), 1)
    mask_01pct, n01, stats_01pct, thr01 = analyze_bright_mask(arr, 99.9)
    ax = axes_sm.flatten()[img_idx]
    rgb_vis = np.stack([arr_norm]*3, axis=-1)
    rgb_vis[mask_01pct == 1, 0] = 1.0
    rgb_vis[mask_01pct == 1, 1] = 0.0
    rgb_vis[mask_01pct == 1, 2] = 0.0
    ax.imshow(rgb_vis, aspect="auto")
    n_corner_01 = sum(1 for s in stats_01pct if s["in_corner"])
    ax.set_title(f"{p.name[:35]}\nthr0.1%={thr01:.0f}  n_comp={n01}  corners={n_corner_01}",
                 fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "overlay_top01pct.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"Guardado: {OUT_DIR}/overlay_top01pct.png")


# ===========================================================================
# PASO 3: Extremo bajo — pico de fondo vs tejido
# ===========================================================================
print()
print("=" * 68)
print("PASO 3 — Extremo bajo: fondo de pelicula vs tejido")
print("=" * 68)

# Para 6 imagenes, histograma del rango bajo + buscar valle
low_thresholds = []
for p in sample_paths[:12]:
    arr = read_uint16(p)
    flat = arr.flatten().astype(np.float32)

    # Histograma del rango bajo (hasta p10)
    p10_val = np.percentile(flat, 10)
    low_pix = flat[flat <= p10_val * 3]  # rango de interes: hasta 3x el p10

    if len(low_pix) < 100:
        continue

    counts_low, edges_low = np.histogram(low_pix, bins=200)
    centers_low = (edges_low[:-1] + edges_low[1:]) / 2

    # Detectar valle entre fondo y tejido: suavizado + minimo local
    from scipy.signal import savgol_filter
    if len(counts_low) > 20:
        smooth = savgol_filter(counts_low.astype(float), 15, 3)
        # Buscar el primer valle descendente tras el pico de fondo (primer maximo)
        first_peak_idx = int(np.argmax(smooth[:len(smooth)//2]))
        valley_region = smooth[first_peak_idx:]
        if len(valley_region) > 5:
            valley_idx = first_peak_idx + int(np.argmin(valley_region[:len(valley_region)//2+1]))
            valley_val = centers_low[valley_idx]
            low_thresholds.append(float(valley_val))
            print(f"  {p.name[:48]:48s}  fondo_peak≈{centers_low[first_peak_idx]:.0f}  "
                  f"valle≈{valley_val:.0f}  "
                  f"p1={np.percentile(flat,1):.0f}  p5={np.percentile(flat,5):.0f}")

if low_thresholds:
    print(f"\n  Valle fondo/tejido: media={np.mean(low_thresholds):.0f}  "
          f"std={np.std(low_thresholds):.0f}  "
          f"rango=[{min(low_thresholds):.0f}, {max(low_thresholds):.0f}]")

# Histogramas de extremo bajo (primeras 4 imagenes)
fig_low, axes_low = plt.subplots(2, 2, figsize=(14, 10))
for ax, p in zip(axes_low.flatten(), sample_paths[:4]):
    arr = read_uint16(p)
    flat = arr.flatten().astype(np.float32)
    # Mostrar rango [0, p30] para ver el valle
    p30 = np.percentile(flat, 30)
    low_pix = flat[flat <= p30]
    ax.hist(low_pix, bins=200, color="teal", alpha=0.7, log=True)
    ax.set_title(f"{p.name[:40]}\nrango [0, p30={p30:.0f}]", fontsize=8)
    ax.set_xlabel("Valor uint16")
    ax.set_ylabel("Freq (log)")
    pv = np.percentile(flat, [1, 5])
    ax.axvline(pv[0], color="green", lw=1.2, label=f"p1={pv[0]:.0f}")
    ax.axvline(pv[1], color="orange", lw=1.2, label=f"p5={pv[1]:.0f}")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_extremo_bajo.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\nGuardado: {OUT_DIR}/hist_extremo_bajo.png")


# ===========================================================================
# PASO 4: Conclusion con numeros
# ===========================================================================
print()
print("=" * 68)
print("PASO 4 — Datos de soporte para la conclusion")
print("=" * 68)

# Estadisticas de los top-5 componentes en las 8 imagenes espaciales
n_large_centrales  = 0
n_small_borde      = 0
total_comps_1pct   = 0
total_imgs         = 0
area_frac_top5     = []  # fraccion del area total que acaparan las top-5 comp

for p in spatial_paths:
    arr = read_uint16(p)
    H, W = arr.shape
    _, n_comps, stats, thr = analyze_bright_mask(arr, 99)
    if not stats:
        continue
    total_imgs += 1
    total_comps_1pct += n_comps
    total_area = sum(s["area"] for s in stats)
    top5_area  = sum(s["area"] for s in stats[:5])
    if total_area > 0:
        area_frac_top5.append(top5_area / total_area)
    # Grande y central: area > 1% del total, dist_centro < 0.25
    large_cent = [s for s in stats if s["area"] > 0.01*H*W and s["dist_from_center"] < 0.30]
    small_bord = [s for s in stats if s["area"] < 500 and s["on_border"]]
    n_large_centrales += len(large_cent)
    n_small_borde     += len(small_bord)

print(f"  Imagenes analizadas: {total_imgs}")
print(f"  Componentes totales en top-1% (media/imagen): "
      f"{total_comps_1pct/max(total_imgs,1):.1f}")
print(f"  Componentes GRANDES + CENTRALES (area>1%H*W, dist<0.30): "
      f"{n_large_centrales} en {total_imgs} imgs")
print(f"  Componentes PEQUEÑAS en BORDE (area<500px, on_border): "
      f"{n_small_borde} en {total_imgs} imgs")
if area_frac_top5:
    print(f"  Fraccion de area de top-1% que acaparan top-5 comp: "
          f"media={np.mean(area_frac_top5):.3f}  min={min(area_frac_top5):.3f}  "
          f"max={max(area_frac_top5):.3f}")

print()
print(f"  Percentiles globales relevantes:")
print(f"    p1     = {p01_val:.0f}   (minimo tejido / umbral inferior)")
print(f"    p99    = {p99_val:.0f}")
print(f"    p99.9  = {p999_val:.0f}")
print(f"    p99.99 = {p9999_val:.0f}")
print(f"    max    = {all_pixels.max():.0f}")
print()
print(f"  Fraccion de pixeles > p99.9  : {frac_above_999*100:.4f}%")
print(f"  Fraccion de pixeles > p99.99 : {frac_above_9999*100:.5f}%")
print()
print("  (Ver overlays en outputs/diag_ddsm/ para verificar espacialmente)")

print()
print("Archivos guardados:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f}")
