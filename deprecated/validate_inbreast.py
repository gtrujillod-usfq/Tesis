## Validador de INbreast v1.0
## Referencia: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset
## Paper: Moreira et al., Academic Radiology, Vol 19, No 2 (2012)
##
## Notas de la validacion previa:
##   - CSV separado por punto y coma (sep=";")
##   - Columna BI-RADS: "Bi-Rads" (con guion, no "birads" ni "BIRADS")
##   - Patient ID = "removed" por anonimizacion de Kaggle (no es error)
##   - Vista 'FB': 1 caso valido (no es error)

from pathlib import Path
import pandas as pd

from ._base import resultado_vacio, calcular_estado


_N_DICOMS_ESPERADOS = 410
_ACR_VALIDOS        = {1, 2, 3, 4}
_VISTAS_VALIDAS     = {"CC", "MLO", "FB"}
_LADOS_VALIDOS      = {"L", "R"}
_BIRADS_VALIDOS     = {1, 2, 3, 4, 5, 6}


def _find_birads_col(columns) -> str:
    """
    Busca la columna de BI-RADS con tolerancia a variantes de nombre.
    Necesario porque INbreast usa 'Bi-Rads' (con guion) y la busqueda
    simple de subcadena 'birads' falla al no encontrar 'bi-rads'.
    """
    for c in columns:
        c_norm = c.lower().replace("-", "").replace("_", "").replace(" ", "")
        if "birads" in c_norm:
            return c
    return None


def validate(config: dict) -> dict:
    """
    Valida la estructura e integridad de INbreast.

    Parametros
    ----------
    config : dict
        Debe contener la clave 'inbreast' con la ruta al dataset.
    """
    res  = resultado_vacio("INbreast")
    root = Path(config.get("inbreast", ""))

    if not root.exists():
        res["issues"].append(f"Directorio no encontrado: {root}")
        res["estado"] = "ERROR"
        return res

    ## Buscar CSV principal con variantes de nombre
    csv_path = None
    for candidate in ["INbreast.csv", "inbreast.csv", "INBreast.csv"]:
        if (root / candidate).exists():
            csv_path = root / candidate
            break
    if csv_path is None:
        for p in root.rglob("*.csv"):
            if "inbreast" in p.name.lower():
                csv_path = p
                break

    if csv_path is None:
        res["issues"].append("CSV de metadatos no encontrado (INbreast.csv)")
        res["estado"] = calcular_estado(res)
        return res

    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    n_filas = len(df)

    res["info"]["csv_nombre"] = csv_path.name
    res["info"]["n_filas"]    = n_filas
    res["metricas"]["n_casos"] = n_filas

    ## ACR (densidad mamaria): valores numericos 1-4
    if "ACR" in df.columns:
        acr_vals = pd.to_numeric(df["ACR"], errors="coerce")
        res["distribuciones"]["densidad"] = (
            acr_vals.dropna().astype(int).value_counts().sort_index().to_dict()
        )
        invalidos = (~acr_vals.dropna().isin(_ACR_VALIDOS)).sum()
        if invalidos > 0:
            res["issues"].append(f"ACR: {invalidos} valores fuera de [1-4]")

    ## BI-RADS: buscar con tolerancia a variantes de nombre (Bi-Rads, BIRADS, etc.)
    birads_col = _find_birads_col(df.columns)
    if birads_col:
        b_vals = pd.to_numeric(df[birads_col], errors="coerce")
        res["distribuciones"]["birads"] = (
            b_vals.dropna().astype(int).value_counts().sort_index().to_dict()
        )
        res["info"]["birads_col_encontrada"] = birads_col
        invalidos = (~b_vals.dropna().isin(_BIRADS_VALIDOS)).sum()
        if invalidos > 0:
            res["issues"].append(
                f"{birads_col}: {invalidos} valores fuera de [1-6]"
            )
    else:
        res["warnings"].append(
            "Columna BI-RADS no encontrada. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    ## Vista
    view_col = next(
        (c for c in df.columns if c.strip().lower() == "view"), None
    )
    if view_col:
        v_vals = df[view_col].dropna().astype(str).str.strip().str.upper()
        res["distribuciones"]["vista"] = v_vals.value_counts().to_dict()
        invalidos = (~v_vals.isin(_VISTAS_VALIDAS)).sum()
        if invalidos > 0:
            res["issues"].append(
                f"View: {invalidos} valores fuera de {_VISTAS_VALIDAS}"
            )

    ## Lateralidad
    lat_col = next(
        (c for c in df.columns if "lateral" in c.lower()), None
    )
    if lat_col:
        l_vals = df[lat_col].dropna().astype(str).str.strip().str.upper()
        res["distribuciones"]["lado"] = l_vals.value_counts().to_dict()

    ## DICOMs en disco
    n_dicoms = sum(1 for _ in root.rglob("*.dcm"))
    res["info"]["n_dicoms_disco"]   = n_dicoms
    res["metricas"]["n_imagenes"]   = n_dicoms
    res["metricas"]["n_pacientes"]  = n_dicoms

    if abs(n_dicoms - _N_DICOMS_ESPERADOS) > 20:
        res["warnings"].append(
            f"DICOMs en disco: {n_dicoms} (esperado ~{_N_DICOMS_ESPERADOS})"
        )

    ## XMLs de anotacion (solo BI-RADS >= 3, comportamiento esperado)
    n_xml = sum(1 for _ in root.rglob("*.xml"))
    res["info"]["n_xmls_disco"] = n_xml

    ## INbreast no tiene columna de patologia explicita en el CSV principal
    ## (la clasificacion se basa en BI-RADS y hallazgos en XML)
    ## distribuciones["patologia"] queda vacio intencionalmente

    res["estado"] = calcular_estado(res)
    return res