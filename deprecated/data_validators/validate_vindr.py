## Validador de VinDr-Mammo v1.0.0
## Referencia: https://www.physionet.org/content/vindr-mammo/1.0.0/
## Paper: Nguyen et al., Scientific Data, Vol 10, Art.277 (2023)
##
## Nombre real del archivo confirmado en validacion previa:
##   breast-level_annotations.csv  (guion entre 'breast' y 'level')

import re
from pathlib import Path

import pandas as pd

from ._base import resultado_vacio, calcular_estado


## Constantes de la especificacion oficial
_N_ESTUDIOS_ESPERADOS   = 5000
_N_DICOMS_ESPERADOS     = 20000
_N_TRAIN                = 4000
_N_TEST                 = 1000
_DENSIDADES_VALIDAS     = {"DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"}
_VISTAS_VALIDAS         = {"CC", "MLO"}
_LADOS_VALIDOS          = {"L", "R"}
_BIRADS_VALIDOS         = {1, 2, 3, 4, 5}
_N_DUPLICADOS_ESPERADOS = 1

## Nombre real del CSV de nivel de mama (con guion, no guion bajo)
_BREAST_CSV_NAMES = [
    "breast-level_annotations.csv",   ## nombre oficial PhysioNet
    "breast_level_annotations.csv",   ## variante alternativa por si acaso
]
_FINDING_CSV     = "finding_annotations.csv"


def _find_breast_csv(root: Path):
    """Busca el CSV de anotaciones a nivel de mama con nombre exacto o variante."""
    for name in _BREAST_CSV_NAMES:
        p = root / name
        if p.exists():
            return p
    ## Busqueda flexible como ultimo recurso
    for p in root.glob("breast*.csv"):
        if "level" in p.name.lower() and "annotation" in p.name.lower():
            return p
    return None


def validate(config: dict) -> dict:
    """
    Valida la estructura e integridad de VinDr-Mammo.

    Parametros
    ----------
    config : dict
        Debe contener la clave 'vindr_mammo' con la ruta al dataset.
    """
    res  = resultado_vacio("VinDr-Mammo")
    root = Path(config.get("vindr_mammo", ""))

    if not root.exists():
        res["issues"].append(f"Directorio no encontrado: {root}")
        res["estado"] = "ERROR"
        return res

    ## Verificar archivos requeridos con nombres flexibles
    breast_csv_path = _find_breast_csv(root)
    if breast_csv_path is None:
        res["issues"].append(
            "Archivo requerido no encontrado: breast-level_annotations.csv "
            "(o variante). Confirmar descarga desde physionet.org/content/vindr-mammo"
        )

    finding_csv_path = root / _FINDING_CSV
    if not finding_csv_path.exists():
        res["issues"].append(f"Archivo requerido no encontrado: {_FINDING_CSV}")

    ## metadata.csv: opcional en VinDr-Mammo (no siempre incluido en descarga)
    meta_csv_path = root / "metadata.csv"
    if not meta_csv_path.exists():
        res["warnings"].append(
            "metadata.csv no encontrado (archivo opcional en VinDr-Mammo)"
        )

    if res["issues"]:
        res["estado"] = calcular_estado(res)
        return res

    ## Cargar CSVs
    breast_df  = pd.read_csv(str(breast_csv_path))
    finding_df = pd.read_csv(str(finding_csv_path))

    ## Normalizar typo oficial del dataset en nombre de columna de vista
    if "view_positition" in breast_df.columns:
        breast_df = breast_df.rename(
            columns={"view_positition": "view_position"}
        )

    res["info"]["breast_csv_nombre"] = breast_csv_path.name

    n_estudios = breast_df["study_id"].nunique()
    n_filas    = len(breast_df)

    res["info"]["n_estudios"]      = n_estudios
    res["info"]["n_filas_breast"]  = n_filas
    res["info"]["n_filas_finding"] = len(finding_df)

    ## Contar DICOMs en disco
    images_dir = root / "images"
    n_dicoms   = 0
    if images_dir.is_dir():
        n_dicoms = sum(1 for _ in images_dir.rglob("*.dicom"))
        res["info"]["n_dicoms_disco"] = n_dicoms

    ## Metricas escalares
    res["metricas"]["n_imagenes"]  = n_filas
    res["metricas"]["n_casos"]     = n_filas
    res["metricas"]["n_pacientes"] = n_estudios

    ## Densidad mamaria: valores como "DENSITY A", "DENSITY B", etc.
    if "breast_density" in breast_df.columns:
        d_vals = breast_df["breast_density"].dropna().astype(str)
        res["distribuciones"]["densidad"] = d_vals.value_counts().to_dict()
        invalidos = (~d_vals.isin(_DENSIDADES_VALIDAS)).sum()
        if invalidos > 0:
            res["issues"].append(
                f"breast_density: {invalidos} valores fuera de DENSITY A/B/C/D"
            )

    ## BI-RADS
    if "breast_birads" in breast_df.columns:
        b_vals = pd.to_numeric(breast_df["breast_birads"], errors="coerce")
        res["distribuciones"]["birads"] = (
            b_vals.dropna().astype(int).value_counts().sort_index().to_dict()
        )

    ## Vista
    if "view_position" in breast_df.columns:
        v_vals = breast_df["view_position"].dropna().astype(str).str.upper()
        res["distribuciones"]["vista"] = v_vals.value_counts().to_dict()

    ## Lado
    if "laterality" in breast_df.columns:
        l_vals = breast_df["laterality"].dropna().astype(str).str.upper()
        res["distribuciones"]["lado"] = l_vals.value_counts().to_dict()

    ## Duplicados
    dupes = breast_df.duplicated(subset=["study_id", "image_id"]).sum()
    if dupes > _N_DUPLICADOS_ESPERADOS:
        res["warnings"].append(
            f"breast CSV: {dupes} filas duplicadas (esperado <= {_N_DUPLICADOS_ESPERADOS})"
        )

    ## Bboxes con coordenadas negativas en finding
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        if col in finding_df.columns:
            n_neg = (pd.to_numeric(finding_df[col], errors="coerce") < 0).sum()
            if n_neg > 0:
                res["warnings"].append(
                    f"finding_annotations: {n_neg} bboxes con {col} negativo"
                )

    ## Split train/test
    if "split" in breast_df.columns:
        split_counts = breast_df["split"].value_counts().to_dict()
        res["info"]["split_counts"] = split_counts

    ## VinDr-Mammo no tiene columna de patologia explicita
    ## (la clasificacion se infiere de breast_birads y hallazgos)
    ## distribuciones["patologia"] queda vacio intencionalmente

    res["estado"] = calcular_estado(res)
    return res