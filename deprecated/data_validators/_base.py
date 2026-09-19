## Utilidades compartidas por todos los validadores
## Define la estructura estandar del diccionario de retorno

from dataclasses import dataclass, field
from typing import Any


ESTADO_OK      = "OK"
ESTADO_WARNING = "WARNING"
ESTADO_ERROR   = "ERROR"


def resultado_vacio(nombre: str) -> dict:
    """
    Retorna la estructura base que debe completar cada validador.
    Todos los validadores deben retornar un dict con estas claves.
    """
    return {
        "dataset":       nombre,
        "estado":        ESTADO_OK,
        "issues":        [],
        "warnings":      [],
        "info":          {},
        ## Metricas escalares para el panel de resumen
        "metricas": {
            "n_imagenes":     0,
            "n_casos":        0,
            "n_pacientes":    0,
            "cobertura_pct":  100.0,
        },
        ## Distribuciones para graficos (todas opcionales segun dataset)
        "distribuciones": {
            "patologia":  {},   ## {"MALIGNANT": N, "BENIGN": M, ...}
            "birads":     {},   ## {0: N, 1: M, 2: K, ...}
            "densidad":   {},   ## {1: N, 2: M, 3: K, 4: L}
            "vista":      {},   ## {"CC": N, "MLO": M}
            "lado":       {},   ## {"LEFT": N, "RIGHT": M}
        },
    }


def calcular_estado(resultado: dict) -> str:
    """Recalcula el estado global en funcion de issues y warnings."""
    if resultado["issues"]:
        return ESTADO_ERROR
    if resultado["warnings"]:
        return ESTADO_WARNING
    return ESTADO_OK