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
## mascara_mama / utilidades de imagen
## =========================================================

def _tensor_a_gris(img_tensor):
    """Tensor [1, 3, H, W] -> np.ndarray [H, W] en [0, 1] (promedio de canales)."""
    arr  = img_tensor[0].detach().cpu().numpy()
    gray = arr.mean(axis=0)
    lo, hi = gray.min(), gray.max()
    return (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)


def mascara_mama(img_gris):
    """
    Genera mascara binaria de la region de la mama para excluir el fondo negro
    y las etiquetas de texto (R-MLO, L-CC, etc.) de las metricas XAI.

    Algoritmo:
      1. Umbral Otsu (skimage) o 0.05 como fallback.
      2. Mayor componente conexa (scipy.ndimage.label): elimina etiquetas y bordes,
         que son al menos 10x mas pequenos que la region mamaria.
      3. binary_fill_holes: rellena huecos internos del tejido.

    La mascara se computa sobre la misma resolucion que los mapas de atribucion
    (1520x912), garantizando alineamiento pixel a pixel sin interpolacion.

    Parametros
    ----------
    img_gris : np.ndarray [H, W] en [0, 1]

    Retorna
    -------
    mascara : np.ndarray bool [H, W]
    """
    from scipy import ndimage as _ndi

    try:
        from skimage.filters import threshold_otsu
        umbral = float(threshold_otsu(img_gris))
    except ImportError:
        umbral = 0.05   ## fondo de mamogramas ~ 0.0-0.03 tras normalizacion

    binaria         = img_gris > umbral
    labeled, n_comp = _ndi.label(binaria)

    if n_comp == 0:
        return np.ones_like(img_gris, dtype=bool)

    ## Seleccionar la componente con mas pixeles: la mama
    ## Etiquetas de texto y marcadores son > 10x mas pequenos
    tamanos    = _ndi.sum(binaria, labeled, range(1, n_comp + 1))
    comp_mayor = int(np.argmax(tamanos)) + 1
    mascara    = (labeled == comp_mayor)
    mascara    = _ndi.binary_fill_holes(mascara)

    return mascara.astype(bool)


def validar_mascara_vs_cajas(cajas_df, n_sample=20, seed=42):
    """
    Sanity check: verifica que los centros de las cajas GT escaladas caigan dentro
    de la mascara. Objetivo: >= 95% de centros dentro de la mascara.
    Si < 90%, el umbral Otsu es demasiado agresivo.
    """
    transform = cargar_transform_inferencia()
    sample_ids = (
        cajas_df['image_id']
        .drop_duplicates()
        .sample(n=min(n_sample, cajas_df['image_id'].nunique()), random_state=seed)
        .tolist()
    )
    registros = []
    for img_id in sample_ids:
        filas    = cajas_df[cajas_df['image_id'] == img_id]
        img_path = filas['image_path'].iloc[0]
        img_t    = cargar_imagen(img_path, transform, 'cpu')
        gray     = _tensor_a_gris(img_t)
        mask     = mascara_mama(gray)
        n_cajas  = len(filas)
        en_mask  = 0
        for _, caja in filas.iterrows():
            cy = int((caja['ymin_s'] + caja['ymax_s']) / 2)
            cx = int((caja['xmin_s'] + caja['xmax_s']) / 2)
            cy = min(max(cy, 0), mask.shape[0] - 1)
            cx = min(max(cx, 0), mask.shape[1] - 1)
            if mask[cy, cx]:
                en_mask += 1
        registros.append({
            'image_id':         img_id,
            'n_cajas':          n_cajas,
            'cajas_en_mascara': en_mask,
            'fraccion':         en_mask / n_cajas if n_cajas > 0 else float('nan'),
        })
    resumen  = pd.DataFrame(registros)
    total_en = resumen['cajas_en_mascara'].sum()
    total    = resumen['n_cajas'].sum()
    print(f'Sanity check mascara: {total_en}/{total} centros de caja dentro de mascara '
          f'({total_en / max(total, 1):.1%}).')
    print('OK si >= 95%. Si < 90%, umbral Otsu demasiado agresivo.')
    return resumen


## =========================================================
## deletion_auc
## =========================================================

def deletion_auc(model, img_tensor, head, attr_map_2d, baseline_tensor,
                 n_steps=20, device='cpu', objetivo_func=None, predicted_class=None,
                 mascara=None):
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
    mascara : np.ndarray bool [H, W] o None
        Mascara de mama (salida de mascara_mama()). Si se proporciona, los pixeles
        fuera de la mama reciben atribucion 0 antes de ordenar: se eliminan al final
        de la curva de supresion, donde su impacto en el AUC es minimo.
        Identica para IG, Grad-CAM y random -> comparacion justa entre metodos.

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

    ## Nota sobre el efecto de la mascara en n_pixels:
    ## n_pixels = H*W incluye pixeles de fondo. Tras aplicar la mascara, esos
    ## pixeles tienen atribucion=0 y quedan al FINAL de sorted_indices (argsort
    ## desempata por posicion). Con n_steps=20 y mama ~ 10-40% del frame,
    ## solo int(p_mama * 20) = 2-8 pasos de 20 borran tejido; el resto borra
    ## fondo en el MISMO orden para IG, GradCAM y random. Esta cola identica
    ## diluye los AUC absolutos pero se cancela en la diferencia pareada
    ## (auc_metodo - auc_random) => usar comparar_deletion_auc_vs_random para
    ## evaluar la ganancia real sobre el azar.

    ## Aplanar el mapa y ordenar de mayor a menor saliencia.
    ## Mascara de mama: pone a 0 los pixeles de fondo antes del argsort.
    ## Los pixeles no-anatomicos (atribucion=0) caen al final del orden y se
    ## eliminan en los ultimos pasos de la curva, donde ya no cambia el AUC.
    ## Identica para IG, Grad-CAM y random -> comparacion justa entre metodos.
    mapa_ord  = attr_map_2d * mascara if mascara is not None else attr_map_2d
    flat_attr = mapa_ord.flatten()                       ## [H*W]
    sorted_indices = np.argsort(flat_attr)[::-1].copy()  ## indices de mayor a menor

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
## insertion_auc
## =========================================================

def insertion_auc(model, img_tensor, head, attr_map_2d, baseline_tensor,
                  n_steps=20, device='cpu', predicted_class=None,
                  mascara=None):
    """
    Calcula el AUC de la curva de Insertion para medir la fidelidad del mapa de atribucion.

    Complemento de deletion_auc: en lugar de BORRAR pixeles de la imagen original,
    REVELA progresivamente pixeles desde una imagen de partida = baseline (negro normalizado).

    Procedimiento:
      1. Imagen inicial: todos los pixeles = baseline_tensor (negro normalizado, -mean/std).
      2. Ordenar los pixeles de mayor a menor saliencia segun attr_map_2d.
      3. En cada fraccion f = k/n_steps (k = 0, 1, ..., n_steps):
         revelar (restaurar valor original) los top-f pixeles mas salientes.
      4. AUC = area bajo la curva (scores vs fracs) con np.trapz.

    Un AUC MAYOR indica que el mapa identifica correctamente los pixeles mas
    relevantes: al anadirlos primero, el score del modelo sube rapidamente.

    Simetria con deletion_auc:
      deletion(f=0) = imagen completa;  deletion(f=1) = imagen baseline.
      insertion(f=0) = imagen baseline; insertion(f=1) = imagen completa.
      score en insertion(f=1) = score en deletion(f=0): sanity check.

    Valor de inicio/tapado: baseline_imagen_negra = -mean_ImageNet/std_ImageNet
    ≈ [-2.12, -2.04, -1.80] (mismo valor que deletion usa para 'borrar').
    Ambas metricas comparten el mismo extremo de referencia.

    Nota sobre n_pixels con mascara: identica a deletion_auc. Los pixeles fuera de
    la mascara (attr=0) se revelan al final, en el mismo orden para todos los metodos.
    La comparacion pareada cancela esta cola identica.

    Parametros
    ----------
    model : MammoVLM
    img_tensor : torch.Tensor [1, 3, H, W]
    head : str, 'birads' o 'density'
    attr_map_2d : np.ndarray [H, W]
    baseline_tensor : torch.Tensor [1, 3, H, W]
        Imagen negra en espacio normalizado (baseline_imagen_negra()).
    n_steps : int
    device : str
    predicted_class : int o None (requerido si head='density')
    mascara : np.ndarray bool [H, W] o None

    Retorna
    -------
    auc_score : float
        Area bajo la curva en [0, 1] (MAYOR = mejor explicacion).
    fracs : list de float
    scores : list de float
    """
    score_func = _get_objetivo_prob(head, predicted_class)

    H, W = attr_map_2d.shape
    n_pixels = H * W

    mapa_ord       = attr_map_2d * mascara if mascara is not None else attr_map_2d
    flat_attr      = mapa_ord.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1].copy()  ## mayor a menor saliencia

    img_flat      = img_tensor.view(1, 3, -1).clone()
    baseline_flat = baseline_tensor.to(img_tensor.device).view(1, 3, -1)

    ## Imagen de partida: todos los pixeles = baseline (simetrico con deletion(f=1))
    img_start = baseline_flat.clone()

    fracs  = []
    scores = []

    for step in range(n_steps + 1):
        frac     = step / n_steps
        n_reveal = int(frac * n_pixels)

        img_revealed = img_start.clone()
        if n_reveal > 0:
            indices_reveal = torch.tensor(
                sorted_indices[:n_reveal], dtype=torch.long, device=img_tensor.device
            )
            ## Restaurar los top-n_reveal pixeles con su valor original en los 3 canales
            img_revealed[:, :, indices_reveal] = img_flat[:, :, indices_reveal]

        img_reconstructed = img_revealed.view(1, 3, H, W)

        with torch.no_grad():
            output_dict  = model(img_reconstructed)
            score_tensor = score_func(output_dict)
            score        = float(score_tensor.mean().item())

        fracs.append(frac)
        scores.append(score)

    ## Un AUC MAYOR indica que los pixeles revelados eran los mas relevantes
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

def pointing_game_imagen(attr_map_2d, cajas_df, mascara=None):
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
    mascara : np.ndarray bool [H, W] o None
        Mascara de mama. Si se proporciona, el argmax se busca solo en la region
        anatomica. Sin mascara, GradCAM puede devolver maximos en esquinas negras
        (artefacto confirmado: fila=0/1504, col=896 en imagenes de test exp08).
        Identica para IG y Grad-CAM -> comparacion justa entre metodos.

    Retorna
    -------
    hit : bool
        True si el pixel de maxima atribucion cae en al menos una caja.
    """
    ## Aplicar mascara antes del argmax: atribucion fuera de la mama es 0
    ## y nunca sera el maximo si hay cualquier atribucion positiva en el tejido
    mapa     = attr_map_2d * mascara if mascara is not None else attr_map_2d
    flat_idx = int(np.argmax(mapa))
    W        = mapa.shape[1]
    row      = flat_idx // W   ## dimension vertical (y)
    col      = flat_idx %  W   ## dimension horizontal (x)

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

def iou_mapas(map1_2d, map2_2d, top_k=0.25, mascara=None):
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
    mascara : np.ndarray bool [H, W] o None
        Mascara de mama. Si se proporciona, el percentil se calcula solo sobre
        pixeles anatomicos y el resultado queda restringido a la mama.
        Sin mascara, incluir pixeles de fondo (atribucion ~ 0) desplaza el
        percentil hacia abajo cuando > (1 - top_k) de los pixeles son fondo,
        haciendo que el top-k incluya practicamente toda la mama.

    Retorna
    -------
    iou : float en [0, 1]. Retorna 0.0 si la union es vacia.
    """
    if mascara is not None:
        ## Percentil solo sobre pixeles de tejido mamario
        mask_bool = mascara.astype(bool)
        umbral1   = np.percentile(map1_2d[mask_bool], 100.0 * (1.0 - top_k))
        umbral2   = np.percentile(map2_2d[mask_bool], 100.0 * (1.0 - top_k))
        ## El resultado se restringe a la mama: pixeles de fondo nunca en el top-k
        mask1 = (map1_2d >= umbral1) & mask_bool
        mask2 = (map2_2d >= umbral2) & mask_bool
    else:
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

            ## Mascara de mama: se computa una vez por imagen y se reutiliza para
            ## IG, Grad-CAM y random. Excluye esquinas negras y etiquetas de texto
            ## del orden de supresion, haciendo las tres curvas comparables.
            mascara = mascara_mama(_tensor_a_gris(img_tensor))

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
                    mascara=mascara,
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
                    mascara=mascara,   ## misma mascara que IG/GradCAM para comparacion justa
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
## evaluar_insertion_auc
## =========================================================

def evaluar_insertion_auc(model, test_df, attr_load_func, head, device,
                           n_sample=50, out_birads_dir=None, out_density_dir=None,
                           include_random_baseline=True):
    """
    Evalua Insertion AUC sobre una muestra aleatoria del conjunto de test.

    Complemento de evaluar_deletion_auc: mide la fidelidad revelando pixeles
    progresivamente desde una imagen baseline (negro normalizado, -mean/std).

    Un AUC MAYOR indica mejor fidelidad. Un metodo util debe tener AUC > random_auc.

    Usa la misma muestra reproducible (random_state=42) y la misma mascara de mama
    que evaluar_deletion_auc para que los pares (deletion_auc, insertion_auc) sean
    comparables por imagen.

    Parametros
    ----------
    (idem evaluar_deletion_auc; todos los parametros tienen la misma semantica)

    Retorna
    -------
    resultados_df : pd.DataFrame con columnas image_id, head, method, auc
        method in {'ig', 'gradcam'} + {'random'} si include_random_baseline=True.
    """
    from config_xai import OUT_BIRADS, OUT_DENSITY

    out_dir = Path(out_birads_dir) if out_birads_dir else OUT_BIRADS
    if head == 'density':
        out_dir = Path(out_density_dir) if out_density_dir else OUT_DENSITY

    muestra    = test_df.sample(n=min(n_sample, len(test_df)), random_state=42)
    transform  = cargar_transform_inferencia()
    registros  = []
    rng_random = np.random.default_rng(seed=42)

    for _, fila in _progreso(muestra.iterrows(), desc=f'Insertion AUC ({head})', total=len(muestra)):
        image_id   = fila['image_id']
        image_path = fila['image_path']

        try:
            attr = attr_load_func(image_id, head, out_dir)
        except FileNotFoundError:
            logger.warning("Atribucion no encontrada para image_id=%s; se omite.", image_id)
            continue

        predicted_class = None
        if head == 'density':
            predicted_class = int(attr['meta'].get('density_idx', fila.get('density_index', 0)))

        try:
            img_tensor = cargar_imagen(image_path, transform, device)
            baseline   = baseline_imagen_negra(device).to(device)
            baseline   = baseline.expand_as(img_tensor)
            mascara    = mascara_mama(_tensor_a_gris(img_tensor))

            for method in ('ig', 'gradcam'):
                auc, _, _ = insertion_auc(
                    model=model,
                    img_tensor=img_tensor,
                    head=head,
                    attr_map_2d=attr[method],
                    baseline_tensor=baseline,
                    n_steps=20,
                    device=device,
                    predicted_class=predicted_class,
                    mascara=mascara,
                )
                registros.append({'image_id': image_id, 'head': head, 'method': method, 'auc': auc})

            if include_random_baseline:
                H_r, W_r  = attr['ig'].shape
                random_map = rng_random.random((H_r, W_r))
                auc_rnd, _, _ = insertion_auc(
                    model=model,
                    img_tensor=img_tensor,
                    head=head,
                    attr_map_2d=random_map,
                    baseline_tensor=baseline,
                    n_steps=20,
                    device=device,
                    predicted_class=predicted_class,
                    mascara=mascara,
                )
                registros.append({'image_id': image_id, 'head': head, 'method': 'random', 'auc': auc_rnd})

        except Exception as exc:
            logger.error("Error en insertion_auc para image_id=%s: %s", image_id, exc)
            continue

    return pd.DataFrame(registros)


## =========================================================
## evaluar_pointing_game
## =========================================================

def evaluar_pointing_game(attr_load_func, cajas_df, head='birads', out_birads_dir=None,
                          transform=None, device='cpu'):
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
    transform : MammoCLIPTransform o None
        Si se proporciona, se carga la imagen para computar la mascara de mama
        y el argmax se restringe al tejido anatomico.
    device : str

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

        ## Mascara de mama: restringe el argmax al tejido anatomico.
        ## Sin mascara, GradCAM puede tener el maximo en esquinas negras
        ## (artefacto confirmado en exp08 para imagenes de test).
        mascara = None
        if transform is not None and 'image_path' in cajas_df.columns:
            try:
                img_path = cajas_imagen['image_path'].iloc[0]
                img_t    = cargar_imagen(img_path, transform, device)
                mascara  = mascara_mama(_tensor_a_gris(img_t))
            except Exception as exc:
                logger.warning("No se pudo computar mascara para %s: %s", image_id, exc)

        for method in ('ig', 'gradcam'):
            hit = pointing_game_imagen(attr[method], cajas_imagen, mascara=mascara)
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
                            out_attr_dir=None, head='birads', test_df=None):
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
    test_df : pd.DataFrame o None
        Si se proporciona (con columnas image_id, image_path), se carga la imagen
        para computar la mascara de mama. El percentil se calcula solo sobre
        pixeles de tejido; sin mascara, pixeles de fondo desplazarian el umbral.

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

    ## Construir mapeo image_id -> image_path si test_df esta disponible
    id_a_path = {}
    if test_df is not None:
        id_a_path = test_df.set_index('image_id')['image_path'].to_dict()
    iou_transform = cargar_transform_inferencia() if id_a_path else None

    registros = []

    for image_id in _progreso(image_ids, desc='IoU IG vs GradCAM'):
        try:
            attr = attr_load_func(image_id, head, out_dir)
        except FileNotFoundError:
            logger.warning("Atribucion no encontrada para image_id=%s; se omite.", image_id)
            continue

        ## Mascara de mama para restringir el top-k% al tejido anatomico
        mascara_iou = None
        if iou_transform is not None and image_id in id_a_path:
            try:
                img_t       = cargar_imagen(id_a_path[image_id], iou_transform, 'cpu')
                mascara_iou = mascara_mama(_tensor_a_gris(img_t))
            except Exception as exc:
                logger.warning("No se pudo computar mascara para %s: %s", image_id, exc)

        try:
            iou = iou_mapas(attr['ig'], attr['gradcam'], top_k=top_k, mascara=mascara_iou)
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
## comparar_deletion_auc_vs_random
## =========================================================

def comparar_deletion_auc_vs_random(df_dauc, n_bootstrap=2000, seed=42):
    """
    Comparacion PAREADA por imagen entre cada metodo de atribucion (IG, Grad-CAM)
    y el baseline random. La diferencia delta = auc_metodo - auc_random se calcula
    sobre las MISMAS imagenes, eliminando la variabilidad inter-imagen.

    Interpretacion:
      delta < 0 => el metodo borra pixeles mas relevantes que el azar (mejor).
      delta > 0 => el metodo es peor que el azar.
      IC95 que cruza 0 => indistinguible del azar en esta muestra.

    La comparacion pareada es mas informativa que comparar medias absolutas porque
    cancela la cola identica de pixeles de fondo que se borra en el mismo orden
    para todos los metodos (efecto de la mascara de mama sobre n_pixels).

    IC95 bootstrap percentil (n_bootstrap resamples sobre el vector de deltas).

    Parametros
    ----------
    df_dauc : pd.DataFrame con columnas image_id, head, method, auc
        Salida de evaluar_deletion_auc con include_random_baseline=True.
        Debe contener filas con method='random'.
    n_bootstrap : int, resamples bootstrap (2000 para IC95 estable).
    seed : int

    Retorna
    -------
    pd.DataFrame con columnas:
        head, metodo, n_imagenes,
        delta_medio  (auc_metodo - auc_random; negativo = mejor que random),
        ic95_lo, ic95_hi  (percentiles 2.5 y 97.5 del bootstrap),
        p_peor_random  (fraccion de resamples con delta > 0 = P(metodo peor que azar)).

    Raises
    ------
    ValueError si df_dauc no contiene filas con method='random'.
    """
    if 'random' not in df_dauc['method'].unique():
        raise ValueError(
            "df_dauc no contiene filas con method='random'. "
            "Ejecutar evaluar_deletion_auc con include_random_baseline=True."
        )

    rng = np.random.default_rng(seed)

    ## Pivote: una fila por (image_id, head), columnas por metodo
    pivot = (
        df_dauc
        .pivot_table(index=['image_id', 'head'], columns='method', values='auc')
        .reset_index()
    )

    registros = []
    for head in sorted(pivot['head'].unique()):
        bloque = pivot[pivot['head'] == head].dropna(subset=['random'])

        for metodo in ('ig', 'gradcam'):
            if metodo not in bloque.columns:
                continue
            sub    = bloque.dropna(subset=[metodo])
            n      = len(sub)
            if n == 0:
                continue

            deltas = (sub[metodo] - sub['random']).to_numpy()

            ## Bootstrap percentil: remuestrear el vector de deltas con reemplazo
            medias_boot = np.empty(n_bootstrap)
            for k in range(n_bootstrap):
                medias_boot[k] = rng.choice(deltas, size=n, replace=True).mean()

            ic_lo, ic_hi = np.percentile(medias_boot, [2.5, 97.5])
            ## Fraccion de resamples con delta > 0 (P de ser peor que random)
            p_peor = float((medias_boot > 0).mean())

            registros.append({
                'head':          head,
                'metodo':        metodo,
                'n_imagenes':    n,
                'delta_medio':   float(deltas.mean()),
                'ic95_lo':       float(ic_lo),
                'ic95_hi':       float(ic_hi),
                'p_peor_random': p_peor,
            })

    return pd.DataFrame(registros)


## =========================================================
## comparar_insertion_auc_vs_random
## =========================================================

def comparar_insertion_auc_vs_random(df_iauc, n_bootstrap=2000, seed=42):
    """
    Comparacion PAREADA por imagen entre cada metodo de atribucion (IG, Grad-CAM)
    y el baseline random para Insertion AUC.

    En insertion, un AUC MAYOR es mejor: el score sube mas rapido al revelar primero
    los pixeles del metodo. Por tanto delta = auc_metodo - auc_random, y:
      delta > 0 => el metodo revela pixeles mas relevantes que el azar (mejor).
      delta < 0 => el metodo es peor que el azar.
      IC95 que cruza 0 => indistinguible del azar en esta muestra.

    Lectura conjunta deletion + insertion (delta = metodo - random):
      Deletion delta < 0 => borrar los pixeles del metodo derrumba el score mas
                            rapido que random. Bueno.
      Insertion delta > 0 => anadir los pixeles del metodo sube el score mas rapido
                             que random. Bueno.
      Un metodo FIEL tiene deletion delta < 0 Y insertion delta > 0 (signos OPUESTOS,
      porque las dos metricas estan invertidas).

      del<0, ins>0 : coinciden -> el metodo SUPERA al random (fiel). Mejor caso.
      del>0, ins<0 : coinciden -> el metodo es PEOR que random. Disociacion real:
                     la atribucion no captura los pixeles que el modelo usa.
      del>0, ins>0 : conflicto. El baseline negro infla al random en deletion;
                     insertion es mas limpio. Si ins>0, el metodo SI localiza ->
                     el deletion era artefacto del borrado.
      del<0, ins<0 : conflicto raro (caso inverso).

    Parametros
    ----------
    df_iauc : pd.DataFrame con columnas image_id, head, method, auc
        Salida de evaluar_insertion_auc con include_random_baseline=True.
    n_bootstrap : int
    seed : int

    Retorna
    -------
    pd.DataFrame con columnas:
        head, metodo, n_imagenes,
        delta_medio  (auc_metodo - auc_random; POSITIVO = mejor que random),
        ic95_lo, ic95_hi,
        p_peor_random  (fraccion de resamples con delta < 0 = P(metodo peor)).
    """
    if 'random' not in df_iauc['method'].unique():
        raise ValueError(
            "df_iauc no contiene filas con method='random'. "
            "Ejecutar evaluar_insertion_auc con include_random_baseline=True."
        )

    rng = np.random.default_rng(seed)

    pivot = (
        df_iauc
        .pivot_table(index=['image_id', 'head'], columns='method', values='auc')
        .reset_index()
    )

    registros = []
    for head in sorted(pivot['head'].unique()):
        bloque = pivot[pivot['head'] == head].dropna(subset=['random'])

        for metodo in ('ig', 'gradcam'):
            if metodo not in bloque.columns:
                continue
            sub = bloque.dropna(subset=[metodo])
            n   = len(sub)
            if n == 0:
                continue

            deltas = (sub[metodo] - sub['random']).to_numpy()

            medias_boot = np.empty(n_bootstrap)
            for k in range(n_bootstrap):
                medias_boot[k] = rng.choice(deltas, size=n, replace=True).mean()

            ic_lo, ic_hi = np.percentile(medias_boot, [2.5, 97.5])
            ## En insertion: P(peor) = P(delta < 0) = P(metodo revela menos que random)
            p_peor = float((medias_boot < 0).mean())

            registros.append({
                'head':          head,
                'metodo':        metodo,
                'n_imagenes':    n,
                'delta_medio':   float(deltas.mean()),
                'ic95_lo':       float(ic_lo),
                'ic95_hi':       float(ic_hi),
                'p_peor_random': p_peor,
            })

    return pd.DataFrame(registros)


## =========================================================
## guardar_metricas_csv
## =========================================================

def guardar_metricas_csv(df, nombre, out_tablas_dir=None, sufijo=None):
    """
    Guarda un DataFrame de metricas como CSV en el directorio de tablas.

    Parametros
    ----------
    df : pd.DataFrame
    nombre : str
        Nombre base del archivo sin extension.
    out_tablas_dir : str, Path o None
        Si None, usa OUT_TABLAS de config_xai.
    sufijo : str o None
        Si se proporciona, el archivo se llama '{nombre}_{sufijo}.csv'.
        Si es None (defecto), se llama '{nombre}.csv' (comportamiento anterior).
        Uso: sufijo='con_mascara' para separar la corrida post-mascara del
        snapshot pre-mascara sin sobreescribirlo.
    """
    out_dir  = Path(out_tablas_dir) if out_tablas_dir else OUT_TABLAS
    out_dir.mkdir(parents=True, exist_ok=True)

    nombre_archivo = f"{nombre}_{sufijo}.csv" if sufijo else f"{nombre}.csv"
    ruta           = out_dir / nombre_archivo
    df.to_csv(str(ruta), index=False)

    logger.info("Metricas guardadas en %s (%d filas).", ruta, len(df))
