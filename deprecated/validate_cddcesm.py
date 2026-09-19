## Validador de CDD-CESM v1
## Referencia: https://www.cancerimagingarchive.net/collection/cdd-cesm/
## Paper: Khaled et al., Scientific Data, Vol 9, Art.122 (2022)

from pathlib import Path
import pandas as pd

from ._base import resultado_vacio, calcular_estado


_N_PACIENTES_ESPERADOS = 326
_N_IMAGENES_ESPERADAS  = 2006
_BIRADS_VALIDOS        = {1, 2, 3, 4, 5, 6}
_TIPOS_VALIDOS         = {"DM", "CESM"}
_COL_PATOLOGIA         = "Pathology Classification/ Follow up"
_COL_BIRADS            = "BIRADS"
_COL_DENSIDAD          = "Breast density (ACR)"


def validate(config: dict) -> dict:
    res  = resultado_vacio("CDD-CESM")
    root = Path(config.get("cdd_cesm", ""))

    if not root.exists():
        res["issues"].append(f"Directorio no encontrado: {root}")
        res["estado"] = "ERROR"
        return res

    ## Buscar el XLSX de anotaciones
    xlsx_path = None
    for p in root.rglob("*.xlsx"):
        if "annotation" in p.name.lower() or "cesm" in p.name.lower():
            xlsx_path = p
            break
    if xlsx_path is None:
        for p in root.rglob("*.xlsx"):
            xlsx_path = p
            break

    if xlsx_path is None:
        res["issues"].append("XLSX de anotaciones no encontrado")
        res["estado"] = calcular_estado(res)
        return res

    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]
    n_filas = len(df)

    res["info"]["xlsx"] = xlsx_path.name
    res["info"]["n_filas"] = n_filas
    res["metricas"]["n_casos"] = n_filas

    ## Pacientes unicos
    if "Patient_ID" in df.columns:
        n_pac = df["Patient_ID"].nunique()
        res["metricas"]["n_pacientes"] = n_pac
        res["info"]["n_pacientes"]     = n_pac
        if abs(n_pac - _N_PACIENTES_ESPERADOS) > 5:
            res["warnings"].append(
                f"Patient_ID unicos: {n_pac} (esperado ~{_N_PACIENTES_ESPERADOS})"
            )

    ## BI-RADS
    if _COL_BIRADS in df.columns:
        b_vals = pd.to_numeric(df[_COL_BIRADS], errors="coerce")
        res["distribuciones"]["birads"] = (
            b_vals.dropna().astype(int).value_counts().sort_index().to_dict()
        )
        invalidos = (~b_vals.dropna().isin(_BIRADS_VALIDOS)).sum()
        if invalidos > 0:
            res["issues"].append(
                f"{_COL_BIRADS}: {invalidos} valores fuera de [1-6]"
            )

    ## Densidad ACR
    if _COL_DENSIDAD in df.columns:
        d_vals = df[_COL_DENSIDAD].dropna().astype(str).str.strip()
        res["distribuciones"]["densidad"] = d_vals.value_counts().to_dict()

    ## Patologia
    if _COL_PATOLOGIA in df.columns:
        p_vals = df[_COL_PATOLOGIA].dropna().astype(str).str.strip()
        res["distribuciones"]["patologia"] = p_vals.value_counts().to_dict()

    ## Tipo DM vs CESM
    if "Type" in df.columns:
        t_vals = df["Type"].dropna().astype(str).str.strip().str.upper()
        res["info"]["tipo_dist"] = t_vals.value_counts().to_dict()

    ## Contar imagenes JPEG en disco
    n_imagenes = sum(1 for _ in root.rglob("*.jpg")) + sum(1 for _ in root.rglob("*.jpeg"))
    res["metricas"]["n_imagenes"] = n_imagenes
    res["info"]["n_imagenes_disco"] = n_imagenes

    delta = abs(n_imagenes - _N_IMAGENES_ESPERADAS)
    if delta > 50:
        res["warnings"].append(
            f"Imagenes en disco: {n_imagenes} (esperado ~{_N_IMAGENES_ESPERADAS})"
        )

    res["estado"] = calcular_estado(res)
    return res