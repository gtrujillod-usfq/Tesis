## metricas_rag.py
## Paso 5: metricas de coherencia entre atribucion Shapley y grounding NLI.
##
## Metricas implementadas:
##   1. Tasa de coincidencia top-1: fraccion de casos en que el chunk de mayor
##      Shapley coincide con el chunk de mayor entailment NLI.
##   2. Correlacion de Spearman entre rankings Shapley y NLI por imagen.
##   3. Score de grounding agregado: media del NLI score del chunk mas importante
##      segun Shapley (cuanto grounding real tiene el chunk mas influyente).
##
## Ademas:
##   - Muestra estratificada por BI-RADS predicho (40 por clase, n~200).
##   - Figuras de distribucion de Shapley y NLI.
##   - CSVs de metricas por imagen y agregadas.
##
## Criterio de estratificacion del Pointing Game (Bloque A):
##   Solo para 'birads'; densidad excluida. Para el analisis de Bloque A,
##   evaluar_pointing_game_estratificado separa los casos por finding_birads
##   (sospechosos BR4-5 vs benignos) y reporta Hit Rate por grupo.

import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logging.getLogger(__name__).warning(
        "scipy no disponible; correlacion de Spearman no se calculara."
    )

try:
    from tqdm import tqdm as _tqdm
    def _progreso(iterable, desc='', total=None):
        return _tqdm(iterable, desc=desc, total=total)
except ImportError:
    def _progreso(iterable, desc='', total=None):
        return iterable

from config_xai import OUT_RAG, OUT_FIGURAS, OUT_TABLAS, TEST_CSV
from atribucion_rag import cargar_atribucion_rag

logger = logging.getLogger(__name__)


## =========================================================
## Muestra estratificada por BI-RADS predicho
## =========================================================

def crear_muestra_rag(test_df=None, n_por_clase=40, seed=42):
    """
    Construye una muestra estratificada SIN REEMPLAZO del test set para el analisis RAG.

    Estrategia:
      - Por cada BI-RADS de referencia (col 'birads', indice 0-4):
        tomar min(n_por_clase, disponibles) filas sin reemplazo.
        No se sobremuestrea: el n real por clase puede ser menor que n_por_clase.
      - Los image_id son unicos dentro de cada clase (no hay duplicados).

    Razon: el Shapley sobre Qwen2.5 es costoso (8 forward passes por imagen).
    Trabajar con la distribucion natural de clases evita artefactos de muestreo
    y permite reportar la tasa de coincidencia POR CLASE sin sesgos.

    Parametros
    ----------
    test_df : pd.DataFrame o None
        Si None, carga desde TEST_CSV. Debe tener columna 'birads' (0-4).
    n_por_clase : int
        Maximo de filas por clase; el n real es min(n_por_clase, disponibles).
    seed : int
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    pd.DataFrame con columnas del test_df original mas 'birads_clase' (0-4).
    Informe el n real por clase via logging.
    """
    if test_df is None:
        test_df = pd.read_csv(str(TEST_CSV))

    rng = np.random.default_rng(seed)
    partes = []

    col_birads = 'birads'
    if col_birads not in test_df.columns:
        raise ValueError(
            f"Columna '{col_birads}' no encontrada en test_df. "
            f"Columnas disponibles: {list(test_df.columns)}"
        )

    ## Garantizar image_id unicos antes de estratificar
    col_id = 'image_id'
    if col_id in test_df.columns:
        test_df = test_df.drop_duplicates(subset=col_id)

    for clase in range(5):
        subset = test_df[test_df[col_birads] == clase].copy()
        n_disp = len(subset)

        if n_disp == 0:
            logger.warning("Clase BI-RADS %d: 0 casos disponibles en el test set.", clase)
            continue

        ## Submuestreo SIN reemplazo; n real = min(n_por_clase, disponibles)
        n_sel = min(n_por_clase, n_disp)
        idx   = rng.choice(n_disp, size=n_sel, replace=False)
        seleccion = subset.iloc[idx].copy()
        seleccion['birads_clase'] = clase

        logger.info("Clase BI-RADS %d: %d disponibles, seleccionados %d.", clase, n_disp, n_sel)
        partes.append(seleccion)

    if not partes:
        raise RuntimeError("No se encontraron casos en ninguna clase BI-RADS.")

    muestra = pd.concat(partes, ignore_index=True)
    logger.info("Muestra RAG total: %d casos unicos.", len(muestra))
    return muestra


## =========================================================
## Tasa de coincidencia top-1
## =========================================================

def coincidencia_top1(shapley_dict, nli_dict):
    """
    Indica si el chunk con mayor valor Shapley coincide con el de mayor NLI.

    Parametros
    ----------
    shapley_dict : dict {int: float}
        Valores de Shapley por indice de chunk.
    nli_dict : dict {int: float}
        Scores NLI (entailment medio) por indice de chunk.

    Retorna
    -------
    bool : True si el top-1 coincide.
    """
    top_shapley = max(shapley_dict, key=shapley_dict.get)
    top_nli     = max(nli_dict,     key=nli_dict.get)
    return top_shapley == top_nli


## =========================================================
## Evaluacion de la muestra RAG
## =========================================================

def evaluar_muestra_rag(image_ids, out_rag_dir=None):
    """
    Evalua las metricas RAG sobre una lista de image_ids previamente procesados
    por calcular_atribuciones_rag() y guardados con guardar_atribucion_rag().

    Parametros
    ----------
    image_ids : list de str
        Identificadores de imagen. Debe existir {image_id}_rag.json en out_rag_dir.
    out_rag_dir : Path o None
        Directorio donde se guardaron los JSON. Por defecto OUT_RAG.

    Retorna
    -------
    pd.DataFrame con una fila por imagen y columnas:
        image_id, coincidencia_top1, spearman_rho, spearman_pval,
        shapley_max_chunk, nli_max_chunk, shapley_max_score, nli_max_score,
        nli_score_shapley_top (entailment del chunk mas importante segun Shapley)
    """
    if out_rag_dir is None:
        out_rag_dir = OUT_RAG

    filas = []

    for image_id in _progreso(image_ids, desc='Evaluando metricas RAG'):
        try:
            dato = cargar_atribucion_rag(image_id, out_dir=out_rag_dir)
        except FileNotFoundError:
            logger.warning("image_id %s no encontrado; se omite.", image_id)
            continue

        shapley_dict = dato['shapley_values']
        nli_dict     = dato['nli_scores']

        top_shapley = max(shapley_dict, key=shapley_dict.get)
        top_nli     = max(nli_dict,     key=nli_dict.get)
        coincide    = int(top_shapley == top_nli)

        ## Correlacion de Spearman entre el ranking Shapley y el ranking NLI
        ## (solo si scipy esta disponible y hay al menos 3 chunks)
        if _HAS_SCIPY and len(shapley_dict) >= 3:
            indices_comunes = sorted(shapley_dict.keys())
            sha_vec = [shapley_dict[i] for i in indices_comunes]
            nli_vec = [nli_dict.get(i, 0.0) for i in indices_comunes]
            rho, pval = spearmanr(sha_vec, nli_vec)
        else:
            rho, pval = float('nan'), float('nan')

        ## Score de grounding del chunk mas influyente segun Shapley
        nli_score_del_top_shapley = nli_dict.get(top_shapley, 0.0)

        birads_clase = dato['prediction'].get('birads_pred', -1)

        filas.append({
            'image_id':             image_id,
            'birads_clase':         birads_clase,
            'coincidencia_top1':    coincide,
            'spearman_rho':         rho,
            'spearman_pval':        pval,
            'shapley_max_chunk':    top_shapley,
            'nli_max_chunk':        top_nli,
            'shapley_max_score':    shapley_dict[top_shapley],
            'nli_max_score':        nli_dict[top_nli],
            'nli_score_shapley_top': nli_score_del_top_shapley,
        })

    if not filas:
        logger.warning("No se encontraron resultados RAG para evaluar.")
        return pd.DataFrame()

    return pd.DataFrame(filas)


## =========================================================
## Resumen agregado
## =========================================================

def calcular_resumen_rag(df_metricas):
    """
    Calcula estadisticas de metricas RAG POR CLASE BI-RADS y una fila de totales.

    IMPORTANTE: la tasa de coincidencia global NO se reporta como simple promedio
    pooled porque la muestra esta estratificada (la distribucion de clases puede
    no reflejar la prevalencia real). Se reportan las tasas POR CLASE; el usuario
    puede calcular un promedio ponderado por prevalencia si lo desea.

    Parametros
    ----------
    df_metricas : pd.DataFrame
        Salida de evaluar_muestra_rag().

    Retorna
    -------
    Tuple de dos DataFrames:
        df_por_clase : una fila por clase BI-RADS (0-4), con columnas:
            birads_clase, n, tasa_coincidencia_top1,
            mean_spearman_rho, mean_nli_score_shapley_top
        df_total : una sola fila con promedios macro (sin ponderar por clase),
            n_total, macro_tasa_coincidencia, macro_spearman_rho,
            macro_nli_score_shapley_top
    """
    if df_metricas.empty:
        return pd.DataFrame(), pd.DataFrame()

    ## Por clase
    df_por_clase = (
        df_metricas
        .groupby('birads_clase')
        .agg(
            n                        = ('coincidencia_top1', 'count'),
            tasa_coincidencia_top1   = ('coincidencia_top1', 'mean'),
            mean_spearman_rho        = ('spearman_rho', 'mean'),
            mean_nli_score_shapley_top = ('nli_score_shapley_top', 'mean'),
        )
        .reset_index()
    )

    ## Total: promedio macro (media de las tasas por clase, no pooled)
    ## Esto es el promedio no ponderado entre clases (igual peso a cada clase)
    df_total = pd.DataFrame([{
        'n_total':                    len(df_metricas),
        'macro_tasa_coincidencia':    df_por_clase['tasa_coincidencia_top1'].mean(),
        'macro_spearman_rho':         df_por_clase['mean_spearman_rho'].mean(),
        'macro_nli_score_shapley_top': df_por_clase['mean_nli_score_shapley_top'].mean(),
        'n_clases':                   len(df_por_clase),
    }])

    return df_por_clase, df_total


## =========================================================
## Pointing Game estratificado por finding_birads (Bloque A)
## =========================================================

def evaluar_pointing_game_estratificado(attr_load_func, cajas_df):
    """
    Evalua el Pointing Game de la cabeza BI-RADS estratificado por finding_birads,
    para IG y Grad-CAM por separado.

    La estratificacion separa los hallazgos en:
      - 'sospechoso': finding_birads in {4, 5} (BR4 y BR5)
      - 'benigno':    finding_birads in {1, 2, 3}

    Nota sobre solapamiento: el numero total de imagenes en ambos estratos puede
    superar el numero de imagenes unicas en cajas_df porque una misma imagen puede
    tener hallazgos de ambas categorias (p.ej. un hallazgo BR4 y otro BR2). Esas
    imagenes aparecen en 'sospechoso' Y en 'benigno', lo cual es correcto por diseno:
    se evalua si la atencion del modelo cae en el hallazgo relevante de cada grupo.

    Solo 'birads'; densidad excluida (hallazgos morfologicos no tienen caja de densidad).

    Parametros
    ----------
    attr_load_func : callable(image_id, head, out_dir) -> dict
        Funcion que carga las atribuciones (cargar_atribucion de atribucion_clasificador).
    cajas_df : pd.DataFrame
        Salida de metricas_clasificador.preparar_cajas_test().
        Debe tener columnas: image_id, finding_birads, xmin_s, ymin_s, xmax_s, ymax_s.

    Retorna
    -------
    pd.DataFrame con columnas: metodo, grupo, hit_rate, n_imagenes
        metodo in {'ig', 'gradcam'}; grupo in {'sospechoso', 'benigno'}.
    """
    from config_xai import IMAGE_HEIGHT, IMAGE_WIDTH, OUT_BIRADS

    H, W = IMAGE_HEIGHT, IMAGE_WIDTH

    def hit_para_imagen(image_id, group_df, method):
        try:
            data = attr_load_func(image_id, 'birads', str(OUT_BIRADS))
        except FileNotFoundError:
            return None
        attr_map = data[method]   ## 'ig' o 'gradcam'
        flat_idx = int(np.argmax(attr_map.flatten()))
        row = flat_idx // W
        col = flat_idx  % W

        for _, caja in group_df.iterrows():
            if (caja['ymin_s'] <= row <= caja['ymax_s'] and
                    caja['xmin_s'] <= col <= caja['xmax_s']):
                return True
        return False

    ## Diagnostico de entrada: columnas, distribucion finding_birads, solapamiento .npz
    print(f'cajas_df columnas: {cajas_df.columns.tolist()}')
    if 'finding_birads' in cajas_df.columns:
        fb = cajas_df['finding_birads']
        print(f'finding_birads  dtype={fb.dtype}  unique={sorted(fb.dropna().unique())}  NaN={fb.isna().sum()}')
    npz_ids = {p.stem.replace('_birads', '') for p in OUT_BIRADS.glob('*_birads.npz')}
    n_overlap = len(set(cajas_df['image_id'].unique()) & npz_ids)
    print(f'image_ids en cajas_df: {cajas_df["image_id"].nunique()}  con .npz en OUT_BIRADS: {n_overlap}')

    ## Clasificar hallazgos en grupos
    if 'finding_birads' not in cajas_df.columns:
        raise ValueError("cajas_df debe tener la columna 'finding_birads'.")

    ## finding_birads puede ser string "BI-RADS N" (si no fue parseado en preparar_cajas_test)
    ## o ya un entero Int64. Aplicar extraccion regex de forma robusta en ambos casos.
    birads_numerico = (
        cajas_df['finding_birads']
        .astype(str)
        .str.extract(r'(\d+)', expand=False)
        .astype('Int64')
    )

    registros = []
    for grupo, label in [('sospechoso', [4, 5]), ('benigno', [1, 2, 3])]:
        subset = cajas_df[birads_numerico.isin(label)]
        image_ids_grupo = subset['image_id'].unique()
        print(f'Grupo {grupo!r}: {len(image_ids_grupo)} imagenes unicas')

        for method in ('ig', 'gradcam'):
            hits = []
            for img_id in _progreso(image_ids_grupo, desc=f'Pointing {grupo}/{method}'):
                img_df = subset[subset['image_id'] == img_id]
                resultado = hit_para_imagen(img_id, img_df, method)
                if resultado is not None:
                    hits.append(int(resultado))

            hit_rate = float(np.mean(hits)) if hits else float('nan')
            registros.append({
                'metodo':      method,
                'grupo':       grupo,
                'hit_rate':    hit_rate,
                'n_imagenes':  len(hits),
            })

    return pd.DataFrame(registros)


## =========================================================
## Persistencia: CSVs y figuras
## =========================================================

def guardar_metricas_rag_csv(df, nombre, out_tablas_dir=None):
    """
    Guarda un DataFrame de metricas RAG como CSV en out_tablas_dir.

    Parametros
    ----------
    df : pd.DataFrame
    nombre : str
        Nombre del archivo sin extension (ej. 'metricas_rag_por_imagen').
    out_tablas_dir : Path o None
        Por defecto OUT_TABLAS de config_xai.
    """
    if out_tablas_dir is None:
        out_tablas_dir = OUT_TABLAS
    out_tablas_dir = Path(out_tablas_dir)
    out_tablas_dir.mkdir(parents=True, exist_ok=True)

    ruta = out_tablas_dir / f"{nombre}.csv"
    df.to_csv(str(ruta), index=False, encoding='utf-8')
    logger.info("CSV guardado: %s", ruta)


def figura_distribucion_shapley(df_metricas, out_figuras_dir=None):
    """
    Genera una figura de caja (boxplot) con los valores de Shapley por chunk
    y la distribucion de coincidencias top-1 por clase BI-RADS.

    La figura se guarda como 'shapley_distribucion.png'.

    Parametros
    ----------
    df_metricas : pd.DataFrame
        Salida de evaluar_muestra_rag(). Debe tener columnas:
        'shapley_max_score', 'nli_max_score', 'coincidencia_top1', 'birads_clase'.
    out_figuras_dir : Path o None
        Por defecto OUT_FIGURAS de config_xai.
    """
    if out_figuras_dir is None:
        out_figuras_dir = OUT_FIGURAS
    out_figuras_dir = Path(out_figuras_dir)
    out_figuras_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')  ## backend sin pantalla para entorno de servidor
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib no disponible; figura no generada.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ## Panel izquierdo: distribucion de NLI score del chunk top Shapley
    axes[0].hist(
        df_metricas['nli_score_shapley_top'].dropna(),
        bins=20,
        edgecolor='black',
        color='steelblue',
    )
    axes[0].set_title('NLI entailment del chunk top-Shapley')
    axes[0].set_xlabel('Score de entailment NLI')
    axes[0].set_ylabel('Frecuencia')
    axes[0].axvline(df_metricas['nli_score_shapley_top'].mean(),
                    color='red', linestyle='--', label='Media')
    axes[0].legend()

    ## Panel derecho: tasa de coincidencia top-1 por clase BI-RADS predicha
    coincidencia_por_clase = (
        df_metricas
        .groupby('birads_clase')['coincidencia_top1']
        .mean()
        .reset_index()
    )
    axes[1].bar(
        coincidencia_por_clase['birads_clase'].astype(str),
        coincidencia_por_clase['coincidencia_top1'],
        color='steelblue',
        edgecolor='black',
    )
    axes[1].set_title('Tasa de coincidencia Shapley-NLI por BI-RADS')
    axes[1].set_xlabel('BI-RADS predicho (indice 0-4)')
    axes[1].set_ylabel('Tasa de coincidencia top-1')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(df_metricas['coincidencia_top1'].mean(),
                    color='red', linestyle='--', label='Media global')
    axes[1].legend()

    plt.tight_layout()
    ruta = out_figuras_dir / 'shapley_distribucion.png'
    fig.savefig(str(ruta), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Figura guardada: %s", ruta)
