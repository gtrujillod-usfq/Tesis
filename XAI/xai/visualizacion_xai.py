## visualizacion_xai.py
## Paso 3b: Visualizacion cualitativa de atribuciones del clasificador.
## Genera overlays IG y Grad-CAM sobre el mamograma original, con cajas GT
## y marcadores del pixel de maxima atribucion (hit/miss del Pointing Game).
##
## Funciones exportadas:
##   figura_grid_seleccion  -- grid Nx3 para la seleccion cualitativa
##   figura_dual_head       -- 2x3: cabeza birads (focal) vs densidad (difuso)

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config_xai import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    OUT_BIRADS, OUT_DENSITY, OUT_FIGURAS,
)
from carga_modelo import cargar_imagen

logger = logging.getLogger(__name__)


## =========================================================
## Utilidades internas
## =========================================================

def _a_gris(img_tensor):
    """Tensor [1, 3, H, W] -> numpy [H, W] en [0, 1] (promedio de canales RGB)."""
    arr  = img_tensor[0].cpu().numpy()   ## [3, H, W]
    gray = arr.mean(axis=0)              ## [H, W]
    lo, hi = gray.min(), gray.max()
    return (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)


def _norm_ig(attr_map):
    """
    Prepara el mapa IG para visualizacion: valor absoluto normalizado a [0, 1].
    IG tiene signo; el valor absoluto preserva la magnitud de la atribucion
    sin cancelar regiones de impacto negativo sobre el objetivo.
    """
    heat = np.abs(attr_map).astype(float)
    hi   = heat.max()
    return heat / hi if hi > 0 else heat


def _norm_gradcam(attr_map):
    """
    GradCAM ya es positivo (relu aplicado en captum). Solo normalizar a [0, 1].
    """
    heat = attr_map.astype(float).clip(0)
    hi   = heat.max()
    return heat / hi if hi > 0 else heat


def _max_pixel_rc(attr_map):
    """Retorna (row, col) del pixel de maxima atribucion (argmax del mapa original)."""
    idx = int(np.argmax(attr_map))
    W   = attr_map.shape[1]
    return idx // W, idx % W


def _es_hit(row, col, cajas_df_imagen):
    """True si (row, col) cae dentro de alguna caja de la imagen."""
    for _, caja in cajas_df_imagen.iterrows():
        if (caja['ymin_s'] <= row <= caja['ymax_s'] and
                caja['xmin_s'] <= col <= caja['xmax_s']):
            return True
    return False


def _dibujar_cajas(ax, cajas_df_imagen, color='lime'):
    """Dibuja las cajas escaladas como rectangulos sin relleno."""
    for _, caja in cajas_df_imagen.iterrows():
        rect = mpatches.Rectangle(
            (caja['xmin_s'], caja['ymin_s']),
            caja['xmax_s'] - caja['xmin_s'],
            caja['ymax_s'] - caja['ymin_s'],
            linewidth=1.5, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)


def _marcar_max(ax, attr_map):
    """Dibuja una cruz roja en el pixel de maxima atribucion."""
    row, col = _max_pixel_rc(attr_map)
    ax.plot(col, row, 'r+', markersize=10, markeredgewidth=1.5)


def _overlay_heat(ax, base, heat_norm, gamma=0.7, alpha_max=0.7,
                  cmap='inferno', contour_level=0.7, contour_color='cyan'):
    """
    Renderiza el mamograma base en gris + overlay del mapa de atribucion con
    alpha per-pixel proporcional a la magnitud local de la atribucion.

    Alpha formula: alpha[h,w] = heat_norm[h,w]**gamma * alpha_max
      - Zonas de baja atribucion -> alpha ~ 0 (mamograma intacto y legible).
      - Zonas de alta atribucion -> alpha -> alpha_max (heatmap opaco).
    gamma < 1 hace la transicion mas abrupta: las zonas medias se desvanecen
    mas rapido que con una rampa lineal.

    El colormap se aplica igual a IG y Grad-CAM para que sean comparables
    directamente entre paneles.

    Ademas dibuja un iso-contorno al nivel contour_level (fraccion del maximo)
    en color contrastante para delimitar la region de alta atribucion sin tapar
    el heatmap ni la caja GT.

    Parametros
    ----------
    ax : matplotlib.axes.Axes
    base : np.ndarray [H, W] en [0, 1], mamograma en gris.
    heat_norm : np.ndarray [H, W] en [0, 1], mapa de atribucion normalizado
        (_norm_ig o _norm_gradcam aplicado previamente).
    gamma : float, exponente de la curva de alpha (default 0.7).
    alpha_max : float, alpha maximo en las zonas de maxima atribucion (default 0.7).
    cmap : str, colormap de matplotlib (default 'inferno').
    contour_level : float, nivel de iso-contorno como fraccion del maximo [0,1].
    contour_color : str, color del contorno de iso-atribucion (default 'cyan').
    """
    ## Primero el mamograma en gris como fondo
    ax.imshow(base, cmap='gray', vmin=0, vmax=1)

    ## Construir array RGBA: colormap sobre el mapa normalizado,
    ## luego sobreescribir el canal alpha con la rampa por magnitud
    rgba          = plt.get_cmap(cmap)(heat_norm)      ## [H, W, 4], alpha=1.0
    rgba[:, :, 3] = (heat_norm ** gamma) * alpha_max   ## alpha per-pixel en [0, alpha_max]
    ax.imshow(rgba)

    ## Iso-contorno al nivel indicado si existe en el rango del mapa
    ## contour falla silenciosamente si el nivel esta fuera del rango de datos
    if heat_norm.max() > contour_level:
        ax.contour(
            heat_norm,
            levels=[contour_level],
            colors=[contour_color],
            linewidths=0.8,
        )


## =========================================================
## Grid N x 3: seleccion cualitativa
## =========================================================

def figura_grid_seleccion(
    seleccion, attr_load_func, transform, device,
    head='birads', out_attr_dir=None,
    out_figuras_dir=None, nombre='overlay_atribuciones',
):
    """
    Genera un grid N filas x 3 columnas para la seleccion cualitativa.

    Columnas por fila:
      0: mamograma original en gris con cajas GT en verde.
      1: IG sobre original (inferno, alpha per-pixel=norm**0.7 max 0.7,
         iso-contorno cyan al 70%) con caja GT y max pixel.
      2: Grad-CAM con el mismo esquema para comparabilidad directa entre metodos.
    Los titulos indican HIT o MISS segun si el max pixel cae en alguna caja.

    Parametros
    ----------
    seleccion : list de dict con claves:
        'image_id'    : str
        'image_path'  : str o None
        'cajas_imagen': pd.DataFrame (filas de cajas_df para este image_id)
        'label'       : str (etiqueta descriptiva, ej. 'sospechoso HIT 1')
    attr_load_func : callable(image_id, head, out_dir) -> dict con 'ig', 'gradcam'
    transform : MammoCLIPTransform
    device : str
    head : str, 'birads' o 'density'
    out_attr_dir : Path o None
        Directorio de atribuciones. Si None, se infiere de head.
    out_figuras_dir : Path o None
        Directorio de salida. Si None, usa OUT_FIGURAS de config_xai.
    nombre : str, nombre del PNG sin extension.

    Retorna
    -------
    Path al archivo PNG guardado.
    """
    if out_attr_dir is None:
        out_attr_dir = OUT_BIRADS if head == 'birads' else OUT_DENSITY
    if out_figuras_dir is None:
        out_figuras_dir = OUT_FIGURAS
    out_figuras_dir = Path(out_figuras_dir)
    out_figuras_dir.mkdir(parents=True, exist_ok=True)

    n = len(seleccion)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]   ## garantizar ndim==2 aunque n==1

    for i, item in enumerate(seleccion):
        img_id   = item['image_id']
        img_path = item.get('image_path')
        cajas    = item['cajas_imagen']
        label    = item.get('label', '')

        try:
            attr = attr_load_func(img_id, head, str(out_attr_dir))
        except FileNotFoundError:
            logger.warning("Atribucion no encontrada: %s; fila en blanco.", img_id)
            for j in range(3):
                axes[i, j].axis('off')
                axes[i, j].set_title(f'{img_id} -- sin datos', fontsize=7)
            continue

        ig_h = _norm_ig(attr['ig'])
        gc_h = _norm_gradcam(attr['gradcam'])

        gray = None
        if img_path:
            img_t = cargar_imagen(img_path, transform, device)
            gray  = _a_gris(img_t)
        base = gray if gray is not None else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

        ## HIT/MISS por metodo
        row_ig, col_ig = _max_pixel_rc(attr['ig'])
        row_gc, col_gc = _max_pixel_rc(attr['gradcam'])
        hit_ig = _es_hit(row_ig, col_ig, cajas)
        hit_gc = _es_hit(row_gc, col_gc, cajas)

        fb_val = cajas['finding_birads'].iloc[0] if not cajas.empty else '?'

        ## Panel 0: original + cajas
        axes[i, 0].imshow(base, cmap='gray', vmin=0, vmax=1)
        _dibujar_cajas(axes[i, 0], cajas)
        axes[i, 0].set_title(f'[{i+1}] {label}\n{img_id} | BR={fb_val}', fontsize=7)
        axes[i, 0].axis('off')

        ## Panel 1: IG overlay (alpha por magnitud + iso-contorno cyan al 70%)
        _overlay_heat(axes[i, 1], base, ig_h)
        _dibujar_cajas(axes[i, 1], cajas)
        _marcar_max(axes[i, 1], attr['ig'])
        axes[i, 1].set_title(
            f'IG ({head}) -- {"HIT" if hit_ig else "MISS"}', fontsize=7
        )
        axes[i, 1].axis('off')

        ## Panel 2: Grad-CAM overlay (mismo esquema que IG para comparabilidad)
        _overlay_heat(axes[i, 2], base, gc_h)
        _dibujar_cajas(axes[i, 2], cajas)
        _marcar_max(axes[i, 2], attr['gradcam'])
        axes[i, 2].set_title(
            f'Grad-CAM ({head}) -- {"HIT" if hit_gc else "MISS"}', fontsize=7
        )
        axes[i, 2].axis('off')

    fig.suptitle(
        f'Atribuciones XAI -- cabeza {head.upper()} -- exp08',
        fontsize=10, y=1.002,
    )
    plt.tight_layout()
    ruta = out_figuras_dir / f'{nombre}.png'
    fig.savefig(str(ruta), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Figura grid guardada: %s", ruta)
    return ruta


## =========================================================
## Figura dual-head: birads (focal) vs densidad (difuso)
## =========================================================

def figura_dual_head(
    image_id, image_path, cajas_df_imagen,
    attr_load_func, transform, device,
    out_figuras_dir=None, nombre='dual_head_contraste',
):
    """
    Genera una figura 2 x 3 que contrasta la atencion de la cabeza BI-RADS
    (focal sobre la lesion) con la de la cabeza densidad (difusa sobre el tejido).

    Layout:
      Fila 0 | birads  : [original + cajas] [IG birads + iso-contorno + max] [GradCAM birads + iso-contorno + max]
      Fila 1 | densidad: [original]         [IG density + iso-contorno + max] [GradCAM density + iso-contorno + max]

    Overlay: inferno, alpha per-pixel = norm**0.7 max 0.7, iso-contorno cyan al 70%.
    La densidad no tiene cajas GT en VinDr-Mammo; las cajas solo aparecen en la
    fila BI-RADS. La comparacion visual valida el comportamiento dual-head:
    birads debe ser focal sobre la lesion y densidad debe ser difuso sobre el tejido.

    Parametros
    ----------
    image_id : str
    image_path : str
    cajas_df_imagen : pd.DataFrame, cajas de cajas_df para este image_id.
    attr_load_func : callable(image_id, head, out_dir) -> dict
    transform : MammoCLIPTransform
    device : str
    out_figuras_dir : Path o None
    nombre : str, nombre del PNG sin extension.

    Retorna
    -------
    Path al archivo PNG guardado.
    """
    if out_figuras_dir is None:
        out_figuras_dir = OUT_FIGURAS
    out_figuras_dir = Path(out_figuras_dir)
    out_figuras_dir.mkdir(parents=True, exist_ok=True)

    img_t = cargar_imagen(image_path, transform, device)
    gray  = _a_gris(img_t)

    attr_b = attr_load_func(image_id, 'birads',  str(OUT_BIRADS))
    attr_d = attr_load_func(image_id, 'density', str(OUT_DENSITY))

    ig_b  = _norm_ig(attr_b['ig'])
    gc_b  = _norm_gradcam(attr_b['gradcam'])
    ig_d  = _norm_ig(attr_d['ig'])
    gc_d  = _norm_gradcam(attr_d['gradcam'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ## ------ Fila 0: BI-RADS ------

    ## [0,0] original + cajas GT
    axes[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=1)
    _dibujar_cajas(axes[0, 0], cajas_df_imagen)
    axes[0, 0].set_title(f'{image_id}\noriginal (cajas GT en verde)', fontsize=8)
    axes[0, 0].axis('off')

    ## [0,1] IG birads (alpha por magnitud + iso-contorno cyan al 70%)
    _overlay_heat(axes[0, 1], gray, ig_b)
    _dibujar_cajas(axes[0, 1], cajas_df_imagen)
    _marcar_max(axes[0, 1], attr_b['ig'])
    axes[0, 1].set_title('IG -- cabeza BI-RADS\n(focal sobre la lesion)', fontsize=8)
    axes[0, 1].axis('off')

    ## [0,2] Grad-CAM birads (mismo esquema que IG)
    _overlay_heat(axes[0, 2], gray, gc_b)
    _dibujar_cajas(axes[0, 2], cajas_df_imagen)
    _marcar_max(axes[0, 2], attr_b['gradcam'])
    axes[0, 2].set_title('Grad-CAM -- cabeza BI-RADS\n(focal sobre la lesion)', fontsize=8)
    axes[0, 2].axis('off')

    ## ------ Fila 1: Densidad (sin cajas) ------

    ## [1,0] original sin cajas (densidad es propiedad global, sin caja GT)
    axes[1, 0].imshow(gray, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'{image_id}\noriginal (densidad: sin cajas GT)', fontsize=8)
    axes[1, 0].axis('off')

    ## [1,1] IG density (alpha por magnitud + iso-contorno cyan al 70%)
    _overlay_heat(axes[1, 1], gray, ig_d)
    _marcar_max(axes[1, 1], attr_d['ig'])
    axes[1, 1].set_title('IG -- cabeza Densidad\n(difuso sobre tejido)', fontsize=8)
    axes[1, 1].axis('off')

    ## [1,2] Grad-CAM density (mismo esquema)
    _overlay_heat(axes[1, 2], gray, gc_d)
    _marcar_max(axes[1, 2], attr_d['gradcam'])
    axes[1, 2].set_title('Grad-CAM -- cabeza Densidad\n(difuso sobre tejido)', fontsize=8)
    axes[1, 2].axis('off')

    fig.suptitle(
        f'Contraste dual-head: BI-RADS (focal) vs Densidad (global) | {image_id}',
        fontsize=10,
    )
    plt.tight_layout()
    ruta = out_figuras_dir / f'{nombre}.png'
    fig.savefig(str(ruta), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Figura dual-head guardada: %s", ruta)
    return ruta
