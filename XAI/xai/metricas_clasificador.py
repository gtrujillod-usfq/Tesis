## metricas_clasificador.py
## Paso 3: metricas cuantitativas de fidelidad y localizacion para las atribuciones.
##
## Metricas implementadas:
##   1. Deletion AUC: mide fidelidad suprimiendo pixeles por orden de saliencia.
##      AUC menor indica que los pixeles marcados son mas relevantes para el modelo.
##   2. Pointing Game: mide localizacion comparando el pixel de maxima atribucion
##      con las cajas de hallazgos radiologicos (solo para BI-RADS).
##   3. IoU IG-GradCAM: coherencia entre los dos metodos de atribucion.
##
## Decisiones de diseno:
##   - El objetivo escalar se pasa como closure, no esta hardcodeado.
##   - torch.no_grad() se usa UNICAMENTE en la evaluacion de deletion_auc (no en
##     la computacion de mapas de atribucion, que ocurre en otro modulo).
##   - El escalado de cajas usa estiramiento directo (sin padding), igual que T.Resize.
##   - tqdm se importa con fallback a range para entornos sin la libreria.

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config_xai import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MALIGNANT_INDICES,
    TEST_CSV,
    FINDING_ANNOTATIONS_CSV,
    OUT_TABLAS,
)
from carga_modelo import baseline_imagen_negra, cargar_imagen, cargar_transform_inferencia
from atribucion_clasificador import cargar_atribucion

logger = logging.getLogger(__name__)

## Importar tqdm con fallback para entornos sin la libreria instalada
try:
    from tqdm import tqdm as _tqdm
    def _progreso(iterable, desc='', total=None):
        return _tqdm(iterable, desc=desc, total=total)
except ImportError:
    logger.warning("tqdm no disponible; usando iteracion sin barra de progreso.")
    def _progreso(iterable, desc='', total=None):
        return iterable


## =========================================================
## Closures de objetivo escalar (duplicados aqui para no
## crear dependencia circular con atribucion_clasificador)
## =========================================================

def _objetivo_birads(output_dict):
    """Objetivo BI-RADS: suma de logits BR4 y BR5 (pre-softmax, indices 3 y 4)."""
    logits = output_dict['birads']
    return logits[:, MALIGNANT_INDICES[0]] + logits[:, MALIGNANT_INDICES[1]]


def _objetivo_density(predicted_class):
    """Retorna closure para el logit de la clase de densidad indicada."""
    def objetivo(output_dict):
        return output_dict['density'][:, predicted_class]
    return objetivo


## Objetivos en probabilidad [0,1] para medir durante el borrado (Deletion AUC).
## NO se usan en IG ni GradCAM: esos siguen con logits para evitar el aplastamiento
## del gradiente de softmax sobre clases no predichas.

def _objetivo_birads_prob(output_dict):
    """Malignancy score: softmax(logits_birads)[3] + [4], en [0,1]."""
    logits = output_dict['birads']
    probs  = torch.softmax(logits, dim=-1)
    return probs[:, MALIGNANT_INDICES[0]] + probs[:, MALIGNANT_INDICES[1]]


def _objetivo_density_prob(predicted_class):
    """Probabilidad de la clase de densidad predicha via softmax, en [0,1]."""
    def objetivo(output_dict):
        logits = output_dict['density']
        probs  = torch.softmax(logits, dim=-1)
        return probs[:, predicted_class]
    return objetivo


def _get_objetivo_prob(head, predicted_class=None):
    """
    Fabrica de objetivos en probabilidad para Deletion AUC.
    Retorna valores en [0,1], garantizando que el AUC tambien caiga en [0,1].
    """
    if head == 'birads':
        return _objetivo_birads_prob
    if head == 'density':
        if predicted_class is None:
            raise ValueError(
                "predicted_class es requerido para head='density'."
            )
        return _objetivo_density_prob(predicted_class)
    raise ValueError(f"head desconocida: {head!r}. Usar 'birads' o 'density'.")


def _get_objetivo(head, predicted_class=None):
    """
    Fabrica de closures de objetivo segun la cabeza.

    Parametros
    ----------
    head : str, 'birads' o 'density'
    predicted_class : int o None (requerido para 'density')

    Retorna
    -------
    callable(output_dict) -> torch.Tensor [B]
    """
    if head == 'birads':
        return _objetivo_birads
    if head == 'density':
        if predicted_class is None:
            raise ValueError(
                "predicted_class es requerido para head='density'."
            )
        return _objetivo_density(predicted_class)
    raise ValueError(f"head desconocida: {head!r}. Usar 'birads' o 'density'.")


## =========================================================
## deletion_auc
## =========================================================

def deletion_auc(model, img_tensor, head, attr_map_2d, baseline_tensor,
                 n_steps=20, device='cpu', objetivo_func=None, predicted_class=None):
    """
    Calcula el AUC de la curva de Deletion para medir la fidelidad del mapa de atribucion.

    Procedimiento:
      1. Ordenar los pixeles de mayor a menor saliencia segun attr_map_2d.
      2. En cada fraccion f = k/n_steps (k = 0, 1, ..., n_steps):
         reemplazar los top-f pixeles con baseline_tensor y evaluar el objetivo.
      3. AUC = area bajo la curva (scores vs fracs) con np.trapz.

    Un AUC menor indica que el mapa identifica correctamente los pixeles mas
    relevantes: al suprimirlos, el score del modelo cae rapidamente.

    Parametros
    ----------
    model : MammoVLM
    img_tensor : torch.Tensor [1, 3, H, W]
    head : str
    attr_map_2d : np.ndarray [H, W]
        Mapa de atribucion (IG o GradCAM).
    baseline_tensor : torch.Tensor [1, 3, H, W]
        Representacion de imagen negra en espacio normalizado.
    n_steps : int
        Numero de pasos de supresion (20 es un balance entre precision y velocidad).
    device : str
    objetivo_func : callable o None
        Aceptado por compatibilidad pero NO se usa para medir el score durante
        el borrado. El score siempre se computa con softmax (ver score_func interno).
        Solo es relevante para la computacion de mapas de atribucion, que ocurre
        en atribucion_clasificador.py.
    predicted_class : int o None
        Requerido si head='density'.

    Retorna
    -------
    auc_score : float
        Area bajo la curva en [0, 1] (menor = mejor explicacion).
    fracs : list de float
        Fracciones de pixeles suprimidos en cada paso.
    scores : list de float
        Probabilidad softmax de malignidad (birads) o de la clase predicha
        (density) en cada paso. Rango garantizado [0, 1].
    """
    ## El score de deletion usa probabilidades softmax, no logits crudos.
    ## Los logits pre-softmax pueden ser negativos cuando el baseline los suprime,
    ## produciendo AUC negativa sin interpretacion fisica. La probabilidad esta
    ## acotada en [0,1] y produce un AUC comparable entre metodos.
    score_func = _get_objetivo_prob(head, predicted_class)

    H, W = attr_map_2d.shape
    n_pixels = H * W

    ## Aplanar el mapa y ordenar de mayor a menor saliencia
    ## El orden descendente pone primero los pixeles mas importantes
    flat_attr = attr_map_2d.flatten()                   ## [H*W]
    sorted_indices = np.argsort(flat_attr)[::-1].copy() ## indices de mayor a menor

    ## Preparar la imagen base como tensor flat [1, 3, H*W] para indexing vectorizado
    img_flat      = img_tensor.view(1, 3, -1).clone()     ## [1, 3, H*W]
    baseline_flat = baseline_tensor.to(img_tensor.device).view(1, 3, -1)

    fracs  = []
    scores = []

    for step in range(n_steps + 1):
        frac = step / n_steps
        n_mask = int(frac * n_pixels)

        ## Crear copia del tensor flat con los top-n_mask pixeles reemplazados
        img_masked = img_flat.clone()
        if n_mask > 0:
            ## indices_to_mask: los primeros n_mask pixeles mas salientesy
            indices_to_mask = torch.tensor(
                sorted_indices[:n_mask], dtype=torch.long, device=img_tensor.device
            )
            ## Reemplazar en los 3 canales simultaneamente usando indexing avanzado
            img_masked[:, :, indices_to_mask] = baseline_flat[:, :, indices_to_mask]

        ## Reconstruir a forma [1, 3, H, W] para el forward pass
        img_reconstructed = img_masked.view(1, 3, H, W)

        ## torch.no_grad() se usa SOLO en esta evaluacion de deletion:
        ## no necesitamos gradientes para medir el score del modelo enmascarado
        with torch.no_grad():
            output_dict = model(img_reconstructed)
            score_tensor = score_func(output_dict)   ## probabilidad softmax en [0,1]
            score = float(score_tensor.mean().item())

        fracs.append(frac)
        scores.append(score)

    ## AUC con la regla del trapecio (np.trapz integra scores sobre fracs)
    ## Un AUC menor indica que los pixeles suprimidos eran los mas relevantes
    auc_score = float(np.trapz(scores, fracs))

    return auc_score, fracs, scores


## =========================================================
## preparar_cajas_test
## =========================================================

def preparar_cajas_test(test_csv_path=None, findings_csv_path=None,
                        new_h=IMAGE_HEIGHT, new_w=IMAGE_WIDTH):
    """
    Prepara el DataFrame de bounding boxes escaladas al espacio de la imagen
    de entrada del modelo (1520 x 912).

    El escalado usa estiramiento directo (sin padding), identico a T.Resize((H, W)):
      escala_x = new_w / width_original    (width del DICOM original)
      escala_y = new_h / height_original   (height del DICOM original)

    Nota: escala_x usa la dimension de ancho (columnas), escala_y usa alto (filas).
    Esta asimetria es necesaria porque los DICOMs tienen dimensiones variables.

    Parametros
    ----------
    test_csv_path : str o Path o None
        Ruta al CSV del split test. Si None, usa TEST_CSV de config_xai.
    findings_csv_path : str o Path o None
        Ruta al CSV de anotaciones. Si None, usa FINDING_ANNOTATIONS_CSV.
    new_h : int, alto de la imagen de entrada del modelo (1520).
    new_w : int, ancho de la imagen de entrada del modelo (912).

    Retorna
    -------
    cajas_df : pd.DataFrame con columnas:
        image_id, image_path, xmin, ymin, xmax, ymax,
        xmin_s, ymin_s, xmax_s, ymax_s (coordenadas escaladas)
    Solo filas con al menos una caja valida (sin NaN en coordenadas).
    """
    test_csv_path     = Path(test_csv_path)     if test_csv_path     else TEST_CSV
    findings_csv_path = Path(findings_csv_path) if findings_csv_path else FINDING_ANNOTATIONS_CSV

    test_df     = pd.read_csv(str(test_csv_path))
    findings_df = pd.read_csv(str(findings_csv_path))

    ## Filtrar findings del split test con coordenadas validas
    findings_test = findings_df[findings_df['split'] == 'test'].copy()

    coord_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    ## Eliminar filas donde alguna coordenada es NaN (hallazgos sin localizacion)
    findings_test = findings_test.dropna(subset=coord_cols)

    if findings_test.empty:
        logger.warning("No se encontraron findings con coordenadas en split=test.")
        return pd.DataFrame()

    ## Join con test_df para obtener image_path (findings_csv no tiene esta columna)
    ## El join es por image_id; test_df puede tener columnas adicionales
    cols_test = ['image_id', 'image_path']
    merged = findings_test.merge(
        test_df[cols_test].drop_duplicates('image_id'),
        on='image_id',
        how='inner',
    )

    if merged.empty:
        logger.warning("El join entre findings y test_df no produjo filas.")
        return pd.DataFrame()

    ## Calcular factores de escala para estiramiento directo
    ## 'width' y 'height' en findings_csv son las dimensiones del DICOM original
    ## escala_x aplica sobre coordenadas x (columnas), escala_y sobre y (filas)
    merged['escala_x'] = new_w / merged['width']
    merged['escala_y'] = new_h / merged['height']

    ## Aplicar escala a las coordenadas de la caja
    merged['xmin_s'] = (merged['xmin'] * merged['escala_x']).round().astype(int)
    merged['ymin_s'] = (merged['ymin'] * merged['escala_y']).round().astype(int)
    merged['xmax_s'] = (merged['xmax'] * merged['escala_x']).round().astype(int)
    merged['ymax_s'] = (merged['ymax'] * merged['escala_y']).round().astype(int)

    ## Clip para asegurar que las coordenadas esten dentro de la imagen
    merged['xmin_s'] = merged['xmin_s'].clip(0, new_w - 1)
    merged['xmax_s'] = merged['xmax_s'].clip(0, new_w - 1)
    merged['ymin_s'] = merged['ymin_s'].clip(0, new_h - 1)
    merged['ymax_s'] = merged['ymax_s'].clip(0, new_h - 1)

    ## Parsear finding_birads: 'BI-RADS X' -> entero X (NaN donde no hay anotacion).
    ## El CSV tiene strings como 'BI-RADS 4'; evaluar_pointing_game_estratificado
    ## filtra con .isin([4, 5]) / .isin([1, 2, 3]), requiere tipo numerico.
    if 'finding_birads' in merged.columns:
        merged['finding_birads'] = (
            merged['finding_birads']
            .astype(str)
            .str.extract(r'(\d+)', expand=False)
            .pipe(pd.to_numeric, errors='coerce')
            .astype('Int64')
        )

    logger.info(
        "Cajas preparadas: %d hallazgos en %d imagenes del split test.",
        len(merged), merged['image_id'].nunique()
    )

    return merged.reset_index(drop=True)


## =========================================================
## pointing_game_imagen
## =========================================================

def pointing_game_imagen(attr_map_2d, cajas_df):
    """
    Evalua el Pointing Game para una imagen: verifica si el pixel de maxima
    atribucion cae dentro de alguna caja de hallazgo radiologico.

    Nota de coordenadas:
      - El pixel de maxima atribucion se indexa como (row, col) en el array numpy.
      - x corresponde a col (dimension de ancho, eje horizontal).
      - y corresponde a row (dimension de alto, eje vertical).
      - Las cajas escaladas tienen xmin_s, xmax_s para col y ymin_s, ymax_s para row.

    Parametros
    ----------
    attr_map_2d : np.ndarray [H, W]
        Mapa de atribucion (IG o GradCAM) de la imagen.
    cajas_df : pd.DataFrame
        Filas de hallazgos para esta imagen con columnas xmin_s, ymin_s, xmax_s, ymax_s.

    Retorna
    -------
    hit : bool
        True si el pixel de maxima atribucion cae en al menos una caja.
    """
    ## Encontrar la ubicacion del pixel de maxima atribucion
    flat_idx = int(np.argmax(attr_map_2d))
    W = attr_map_2d.shape[1]
    row = flat_idx // W   ## dimension vertical (y)
    col = flat_idx %  W   ## dimension horizontal (x)

    ## Verificar si (row, col) cae dentro de alguna caja
    for _, caja in cajas_df.iterrows():
        en_y = (caja['ymin_s'] <= row <= caja['ymax_s'])
        en_x = (caja['xmin_s'] <= col <= caja['xmax_s'])
        if en_y and en_x:
            return True

    return False


## =========================================================
## iou_mapas
## =========================================================

def iou_mapas(map1_2d, map2_2d, top_k=0.25):
    """
    Calcula el IoU entre los top-k% pixeles de dos mapas de atribucion.

    Binariza cada mapa con el umbral en el percentil (100 * (1 - top_k)):
      - Umbral = percentil 75 para top_k=0.25 (top 25% de los pixeles).

    IoU = |mask1 AND mask2| / |mask1 OR mask2|

    Parametros
    ----------
    map1_2d : np.ndarray [H, W]
    map2_2d : np.ndarray [H, W]
    top_k : float en (0, 1], fraccion de pixeles a considerar.

    Retorna
    -------
    iou : float en [0, 1]. Retorna 0.0 si la union es vacia.
    """
    ## Umbral como percentil sobre el mapa completo
    umbral1 = np.percentile(map1_2d, 100.0 * (1.0 - top_k))
    umbral2 = np.percentile(map2_2d, 100.0 * (1.0 - top_k))

    mask1 = map1_2d >= umbral1
    mask2 = map2_2d >= umbral2

    interseccion = np.logical_and(mask1, mask2).sum()
    union        = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(interseccion / union)


## =========================================================
## evaluar_deletion_auc
## =========================================================

def evaluar_deletion_auc(model, test_df, attr_load_func, head, device,
                          n_sample=50, out_birads_dir=None, out_density_dir=None,
                          include_random_baseline=True):
    """
    Evalua Deletion AUC sobre una muestra aleatoria del conjunto de test.

    Para cada imagen de la muestra:
      1. Carga el mapa de atribucion (IG y GradCAM) desde disco.
      2. Carga la imagen original y el baseline.
      3. Calcula deletion_auc para IG y Grad-CAM.
      4. Opcionalmente calcula un AUC con orden de pixeles aleatorio (method='random')
         para cuantificar la ganancia sobre el azar: random_auc - method_auc.

    Parametros
    ----------
    model : MammoVLM
    test_df : pd.DataFrame con columnas image_id, image_path
    attr_load_func : callable(image_id, head, out_dir) -> dict con 'ig', 'gradcam', 'meta'
        Tipicamente cargar_atribucion de atribucion_clasificador.
    head : str, 'birads' o 'density'
    device : str
    n_sample : int, numero de imagenes a evaluar.
    out_birads_dir : Path o str, directorio de atribuciones BI-RADS.
    out_density_dir : Path o str, directorio de atribuciones densidad.
    include_random_baseline : bool
        Si True (defecto), agrega una fila por imagen con method='random'
        (orden de pixeles aleatorio). Sirve como referencia nula: un metodo
        util debe tener AUC < random_auc.

    Retorna
    -------
    resultados_df : pd.DataFrame con columnas image_id, head, method, auc
        method in {'ig', 'gradcam'} + {'random'} si include_random_baseline=True.
    """
    from config_xai import OUT_BIRADS, OUT_DENSITY

    out_dir = Path(out_birads_dir) if out_birads_dir else OUT_BIRADS
    if head == 'density':
        out_dir = Path(out_density_dir) if out_density_dir else OUT_DENSITY

    ## Muestra reproducible con seed fijo
    muestra = test_df.sample(n=min(n_sample, len(test_df)), random_state=42)

    transform = cargar_transform_inferencia()
    registros = []

    ## RNG para baseline aleatorio: inicializado antes del loop para que
    ## cada imagen reciba un orden distinto de forma reproducible
    rng_random = np.random.default_rng(seed=42)

    for _, fila in _progreso(muestra.iterrows(), desc=f'Deletion AUC ({head})', total=len(muestra)):
        image_id   = fila['image_id']
        image_path = fila['image_path']

        try:
            attr = attr_load_func(image_id, head, out_dir)
        except FileNotFoundError:
            logger.warning("Atribucion no encontrada para image_id=%s; se omite.", image_id)
            continue

        ## Para density, leer la clase predicha de los metadatos del .npz.
        ## Mas preciso que leer del CSV (el CSV puede no tener la columna).
        predicted_class = None
        if head == 'density':
            predicted_class = int(attr['meta'].get('density_idx', fila.get('density_index', 0)))

        try:
            img_tensor = cargar_imagen(image_path, transform, device)
            baseline   = baseline_imagen_negra(device).to(device)
            baseline   = baseline.expand_as(img_tensor)

            ## IG y Grad-CAM
            for method in ('ig', 'gradcam'):
                attr_map = attr[method]

                auc, _, _ = deletion_auc(
                    model=model,
                    img_tensor=img_tensor,
                    head=head,
                    attr_map_2d=attr_map,
                    baseline_tensor=baseline,
                    n_steps=20,
                    device=device,
                    predicted_class=predicted_class,
                )

                registros.append({
                    'image_id': image_id,
                    'head':     head,
                    'method':   method,
                    'auc':      auc,
                })

            ## Baseline aleatorio: orden de pixeles al azar como referencia nula.
            ## AUC(random) ~ integral de la curva de un mapa sin informacion.
            ## Ganancia de cada metodo = random_auc - method_auc.
            if include_random_baseline:
                H_r, W_r = attr['ig'].shape
                random_map = rng_random.random((H_r, W_r))
                auc_rnd, _, _ = deletion_auc(
                    model=model,
                    img_tensor=img_tensor,
                    head=head,
                    attr_map_2d=random_map,
                    baseline_tensor=baseline,
                    n_steps=20,
                    device=device,
                    predicted_class=predicted_class,
                )
                registros.append({
                    'image_id': image_id,
                    'head':     head,
                    'method':   'random',
                    'auc':      auc_rnd,
                })

        except Exception as exc:
            logger.error("Error en deletion_auc para image_id=%s: %s", image_id, exc)
            continue

    return pd.DataFrame(registros)


## =========================================================
## evaluar_pointing_game
## =========================================================

def evaluar_pointing_game(attr_load_func, cajas_df, head='birads', out_birads_dir=None):
    """
    Evalua el Pointing Game sobre todas las imagenes con anotaciones de hallazgos.

    NOTA: esta metrica se aplica SOLO para head='birads'. La densidad es una
    propiedad global del tejido mamario y no tiene cajas de localizacion asociadas
    en el protocolo de anotacion de VinDr-Mammo.

    Parametros
    ----------
    attr_load_func : callable(image_id, head, out_dir) -> dict
    cajas_df : pd.DataFrame con columnas image_id, xmin_s, ymin_s, xmax_s, ymax_s
        Resultado de preparar_cajas_test().
    head : str, debe ser 'birads' (se advierte si se intenta otro valor).
    out_birads_dir : Path o None.

    Retorna
    -------
    resultados_df : pd.DataFrame con columnas image_id, method, hit
    Tambien imprime la tasa de acierto por metodo via logging.
    """
    from config_xai import OUT_BIRADS

    if head != 'birads':
        logger.warning(
            "pointing_game solo tiene sentido para head='birads'. "
            "La densidad no tiene cajas de hallazgos. head=%s sera ignorada.", head
        )
        head = 'birads'

    out_dir = Path(out_birads_dir) if out_birads_dir else OUT_BIRADS

    image_ids_unicos = cajas_df['image_id'].unique()
    registros = []

    for image_id in _progreso(image_ids_unicos, desc='Pointing Game'):
        cajas_imagen = cajas_df[cajas_df['image_id'] == image_id]

        try:
            attr = attr_load_func(image_id, head, out_dir)
        except FileNotFoundError:
            logger.warning("Atribucion no encontrada para image_id=%s; se omite.", image_id)
            continue

        for method in ('ig', 'gradcam'):
            hit = pointing_game_imagen(attr[method], cajas_imagen)
            registros.append({
                'image_id': image_id,
                'method':   method,
                'hit':      hit,
            })

    resultados_df = pd.DataFrame(registros)

    if not resultados_df.empty:
        ## Calcular y reportar la tasa de acierto por metodo
        for method in ('ig', 'gradcam'):
            subset = resultados_df[resultados_df['method'] == method]
            if not subset.empty:
                tasa = float(subset['hit'].mean())
                logger.info(
                    "Pointing Game accuracy [%s / %s]: %.4f (%d/%d imagenes)",
                    head, method, tasa, int(subset['hit'].sum()), len(subset)
                )

    return resultados_df


## =========================================================
## evaluar_iou_ig_gradcam
## =========================================================

def evaluar_iou_ig_gradcam(attr_load_func, image_ids, top_k=0.25,
                            out_attr_dir=None, head='birads'):
    """
    Calcula el IoU entre los mapas de IG y GradCAM para cada imagen.

    La coherencia entre metodos es un indicador de robustez: si dos metodos
    independientes (IG basado en gradientes y GradCAM basado en activaciones)
    senalan las mismas regiones, la explicacion es mas confiable.

    Parametros
    ----------
    attr_load_func : callable(image_id, head, out_dir) -> dict
    image_ids : list de str
    top_k : float, fraccion de pixeles para la binarizacion.
    out_attr_dir : Path o None.
        Si None, se usa OUT_BIRADS para head='birads' o OUT_DENSITY para head='density'.
    head : str, 'birads' o 'density' (por defecto 'birads').

    Retorna
    -------
    resultados_df : pd.DataFrame con columnas image_id, iou
    Tambien registra el IoU promedio via logging.
    """
    from config_xai import OUT_BIRADS, OUT_DENSITY

    if out_attr_dir:
        out_dir = Path(out_attr_dir)
    elif head == 'density':
        out_dir = OUT_DENSITY
    else:
        out_dir = OUT_BIRADS

    registros = []

    for image_id in _progreso(image_ids, desc='IoU IG vs GradCAM'):
        try:
            attr = attr_load_func(image_id, head, out_dir)
        except FileNotFoundError:
            logger.warning("Atribucion no encontrada para image_id=%s; se omite.", image_id)
            continue

        try:
            iou = iou_mapas(attr['ig'], attr['gradcam'], top_k=top_k)
            registros.append({'image_id': image_id, 'iou': iou})
        except Exception as exc:
            logger.error("Error calculando IoU para image_id=%s: %s", image_id, exc)
            continue

    resultados_df = pd.DataFrame(registros)

    if not resultados_df.empty:
        iou_promedio = float(resultados_df['iou'].mean())
        logger.info(
            "IoU promedio IG vs GradCAM (top_k=%.2f): %.4f sobre %d imagenes.",
            top_k, iou_promedio, len(resultados_df)
        )

    return resultados_df


## =========================================================
## guardar_metricas_csv
## =========================================================

def guardar_metricas_csv(df, nombre, out_tablas_dir=None):
    """
    Guarda un DataFrame de metricas como CSV en el directorio de tablas.

    Parametros
    ----------
    df : pd.DataFrame
    nombre : str
        Nombre del archivo sin extension (se agrega .csv automaticamente).
    out_tablas_dir : str, Path o None
        Si None, usa OUT_TABLAS de config_xai.
    """
    out_dir = Path(out_tablas_dir) if out_tablas_dir else OUT_TABLAS
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta = out_dir / f"{nombre}.csv"
    df.to_csv(str(ruta), index=False)

    logger.info("Metricas guardadas en %s (%d filas).", ruta, len(df))
