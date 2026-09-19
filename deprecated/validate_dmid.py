## Validador de DMID v1.0
## Referencia: https://figshare.com/articles/dataset/24522883
## Paper: Oza et al., Data in Brief, Vol 45, Art.108669 (2022)
##
## Notas de la validacion previa:
##   - 510 DICOMs, 511 TIFFs (1 extra sin DICOM correspondiente)
##   - DICOMs usan JPEG Lossless -> usar TIFFs para entrenamiento
##   - Clase "not_defined" es categoria valida del dataset (36 casos)
##   - 137 imagenes con multiples lesiones -> 647 filas en CSV limpio
##   - CSV limpio generado: DMID_metadata_clean.csv

from pathlib import Path
import pandas as pd

from ._base import resultado_vacio, calcular_estado


_N_TIFFS_ESPERADOS  = 511
_N_DICOMS_ESPERADOS = 510
## "not_defined" es una clase valida en DMID (36 casos confirmados)
## Se aceptan tanto abreviaturas (B/M/N) como palabras completas
## (benign/malignant/normal), ya que el CSV puede venir en cualquiera
## de los dos formatos segun la version del dataset
_CLASES_VALIDAS     = {"B", "M", "N", "not_defined"}
_MAPA_CLASE_NORM = {
    ## Abreviaturas
    "B": "B", "M": "M", "N": "N",
    ## Palabras completas (minuscula)
    "BENIGN": "B", "MALIGNANT": "M", "NORMAL": "N",
    ## Variante valida del dataset
    "NOT_DEFINED": "not_defined",
}
_TEJIDOS_VALIDOS    = {"F", "G", "D"}
_VISTAS_VALIDAS     = {"CCRT", "CCLT", "MLORT", "MLOLT"}


def _find_dmid_csv(root: Path):
    """
    Busca el CSV limpio de DMID en orden de preferencia:
      1. DMID_metadata_clean.csv junto al dataset
      2. DMID_metadata_clean.csv en el directorio de trabajo
      3. Cualquier CSV con 'dmid' o 'metadata' en el nombre dentro del root
    """
    candidates = [
        root / "DMID_metadata_clean.csv",
        Path("DMID_metadata_clean.csv"),
        Path("..") / "DMID_metadata_clean.csv",
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)

    for p in root.rglob("*.csv"):
        name_lower = p.name.lower()
        if "dmid" in name_lower or ("metadata" in name_lower and "clean" in name_lower):
            return p

    return None


def validate(config: dict) -> dict:
    """
    Valida la estructura e integridad de DMID.

    Parametros
    ----------
    config : dict
        Debe contener la clave 'dmid' con la ruta al dataset.
    """
    res  = resultado_vacio("DMID")
    root = Path(config.get("dmid", ""))

    if not root.exists():
        res["issues"].append(f"Directorio no encontrado: {root}")
        res["estado"] = "ERROR"
        return res

    ## Buscar CSV limpio
    csv_path = _find_dmid_csv(root)
    if csv_path is None:
        res["warnings"].append(
            "DMID_metadata_clean.csv no encontrado. "
            "Copiar el archivo generado en la validacion inicial al directorio del dataset."
        )
    else:
        df = pd.read_csv(csv_path)
        n_filas = len(df)
        res["info"]["csv_nombre"] = csv_path.name
        res["info"]["n_filas"]    = n_filas
        res["metricas"]["n_casos"] = n_filas

        ## Clase (patologia): acepta abreviaturas (B/M/N) o palabras completas
        ## (benign/malignant/normal). Se normaliza a un formato canonico.
        if "class" in df.columns:
            cls_raw = df["class"].dropna().astype(str).str.strip()

            ## Normalizar: mapear cada valor a su forma canonica
            ## La clave de normalizacion usa upper() salvo para not_defined
            def _normalizar_clase(valor):
                v = valor.strip()
                if v.lower() == "not_defined":
                    return "not_defined"
                return _MAPA_CLASE_NORM.get(v.upper(), v)

            cls_norm = cls_raw.apply(_normalizar_clase)

            ## Distribucion con nombres legibles para el grafico
            nombre_map = {
                "B":           "BENIGN",
                "M":           "MALIGNANT",
                "N":           "NORMAL",
                "not_defined": "not_defined",
            }
            cls_dist = cls_norm.value_counts().to_dict()
            res["distribuciones"]["patologia"] = {
                nombre_map.get(k, k): v for k, v in cls_dist.items()
            }

            ## Validar contra clases canonicas (ya normalizadas)
            invalidos = (~cls_norm.isin(_CLASES_VALIDAS)).sum()
            if invalidos > 0:
                res["issues"].append(
                    f"class: {invalidos} valores inesperados fuera de "
                    f"B/M/N/not_defined (o benign/malignant/normal): "
                    f"{cls_norm[~cls_norm.isin(_CLASES_VALIDAS)].unique().tolist()}"
                )

        ## Tejido mamario
        if "tissue" in df.columns:
            t_vals = df["tissue"].dropna().astype(str).str.strip().str.upper()
            res["distribuciones"]["densidad"] = t_vals.value_counts().to_dict()

        ## Vista
        if "view" in df.columns:
            v_vals = df["view"].dropna().astype(str).str.strip().str.upper()
            res["distribuciones"]["vista"] = v_vals.value_counts().to_dict()

    ## Contar TIFFs en disco (extensiones .tif y .tiff, case-insensitive via rglob)
    n_tiffs = (
        sum(1 for _ in root.rglob("*.tif"))
        + sum(1 for _ in root.rglob("*.tiff"))
        + sum(1 for _ in root.rglob("*.TIF"))
        + sum(1 for _ in root.rglob("*.TIFF"))
    )
    n_dicoms = sum(1 for _ in root.rglob("*.dcm"))

    res["info"]["n_tiffs_disco"]  = n_tiffs
    res["info"]["n_dicoms_disco"] = n_dicoms
    res["metricas"]["n_imagenes"] = n_tiffs if n_tiffs > 0 else n_dicoms

    if n_tiffs > 0 and n_tiffs < _N_TIFFS_ESPERADOS - 5:
        res["issues"].append(
            f"TIFFs en disco: {n_tiffs} (esperado ~{_N_TIFFS_ESPERADOS})"
        )
    elif n_tiffs == 0 and n_dicoms == 0:
        res["issues"].append(
            "No se encontraron TIFFs ni DICOMs en el directorio del dataset"
        )

    res["estado"] = calcular_estado(res)
    return res