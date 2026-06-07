## label_normalization.py
## Normalizacion de etiquetas BI-RADS al estandar de clasificacion por imagen
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Fundamento clinico:
##   El sistema ACR BI-RADS define categorias 0-6, pero BI-RADS 6 significa
##   "malignidad conocida comprobada por biopsia". Esta categoria depende de
##   informacion histopatologica previa (una biopsia ya realizada), NO de los
##   hallazgos radiologicos de la imagen. Por tanto, un modelo que clasifica
##   mamografias debe predecir 0-5, no 6.
##
##   BI-RADS 6 se remapea a BI-RADS 5 (altamente sugestivo de malignidad)
##   porque ambos representan el extremo maligno y es clinicamente coherente
##   agruparlos para la tarea de clasificacion por imagen.

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

## Rango valido de BI-RADS para clasificacion por imagen
BIRADS_MIN = 0
BIRADS_MAX = 5

## Mapeo de remapeo: BI-RADS 6 -> 5
BIRADS_REMAP = {6: 5}


def normalize_birads(value: int) -> int:
    ##
    ## Normaliza un valor BI-RADS al rango estandar 0-5
    ##
    ## Aplica:
    ##   - Remapeo 6 -> 5 (malignidad confirmada se agrupa con altamente sugestivo)
    ##   - Clipping al rango [0, 5]
    ##
    ## Parametros:
    ##   value: nivel BI-RADS original (puede ser 0-6)
    ##
    ## Retorna: nivel BI-RADS normalizado (0-5)
    ##
    value = int(value)

    ## Aplicar remapeo si corresponde
    if value in BIRADS_REMAP:
        value = BIRADS_REMAP[value]

    ## Clipping de seguridad al rango valido
    return max(BIRADS_MIN, min(BIRADS_MAX, value))


def normalize_birads_list(values: List[int]) -> Tuple[List[int], Dict]:
    ##
    ## Normaliza una lista de valores BI-RADS y reporta los cambios
    ##
    ## Parametros:
    ##   values: lista de niveles BI-RADS originales
    ##
    ## Retorna: (lista_normalizada, reporte_de_cambios)
    ##   reporte_de_cambios contiene cuantos valores se remapearon
    ##
    normalized = []
    remap_counts = {}

    for v in values:
        v_int = int(v)
        v_norm = normalize_birads(v_int)
        normalized.append(v_norm)

        ## Contar remapeos
        if v_int != v_norm:
            key = f"{v_int}_to_{v_norm}"
            remap_counts[key] = remap_counts.get(key, 0) + 1

    report = {
        "total_values": len(values),
        "remapped_count": sum(remap_counts.values()),
        "remap_details": remap_counts,
    }

    return normalized, report


def generate_remap_summary(report: Dict, dataset_name: str = "") -> str:
    ##
    ## Genera un resumen textual del remapeo para mostrar en el notebook
    ##
    lines = []
    prefix = f"[{dataset_name}] " if dataset_name else ""

    total = report["total_values"]
    remapped = report["remapped_count"]

    if remapped == 0:
        lines.append(f"{prefix}Sin remapeos necesarios ({total} valores, todos en rango 0-5)")
    else:
        lines.append(f"{prefix}{remapped} de {total} valores remapeados:")
        for key, count in report["remap_details"].items():
            origen, destino = key.split("_to_")
            lines.append(f"{prefix}  BI-RADS {origen} -> BI-RADS {destino}: {count} casos")

    return "\n".join(lines)


def get_clinical_explanation() -> str:
    ##
    ## Retorna la explicacion clinica del remapeo para documentacion de tesis
    ##
    return (
        "ESTANDARIZACION BI-RADS (0-5):\n"
        "El sistema ACR BI-RADS define categorias 0-6. La categoria 6 indica "
        "'malignidad conocida comprobada por biopsia', que depende de un "
        "diagnostico histopatologico previo y no de los hallazgos radiologicos "
        "de la imagen. Dado que el modelo clasifica a partir de la imagen, "
        "BI-RADS 6 se remapea a BI-RADS 5 (altamente sugestivo de malignidad), "
        "agrupando ambos en el extremo maligno. El modelo predice el rango "
        "estandar 0-5 conforme a la practica de clasificacion por imagen."
    )
