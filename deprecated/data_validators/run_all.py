## Orquestador de validadores de datasets
## Llama a cada validador individual y consolida los resultados
## en una estructura apta para visualizacion en el notebook principal.

import time
from typing import Optional

from . import (
    validate_vindr,
    validate_inbreast,
    validate_dmid,
    validate_cddcesm,
    validate_cbisddsm,
)


## Orden de presentacion en el dashboard
_VALIDATORS = [
    ("VinDr-Mammo", validate_vindr.validate,    "vindr_mammo"),
    ("INbreast",    validate_inbreast.validate,  "inbreast"),
    ("DMID",        validate_dmid.validate,      "dmid"),
    ("CDD-CESM",    validate_cddcesm.validate,   "cdd_cesm"),
    ("CBIS-DDSM",   validate_cbisddsm.validate,  "cbis_ddsm"),
]


def run_all(
    dataset_roots: dict,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta todos los validadores y retorna el resultado consolidado.

    Parametros
    ----------
    dataset_roots : dict
        Diccionario con las rutas de cada dataset, por ejemplo:
            {
                "vindr_mammo": "../data/vindr-mammo",
                "inbreast":    "../data/inbreast",
                "dmid":        "../data/dmid",
                "cdd_cesm":    "../data/cdd-cesm",
                "cbis_ddsm":   "../data/cbis-ddsm",
            }
    verbose : bool
        Si True, imprime el estado de cada validador al ejecutarse.

    Retorna
    -------
    dict con claves:
        "resultados" : dict[str, dict]  <- resultado por dataset
        "resumen"    : dict             <- conteos globales de OK/WARNING/ERROR
        "tiempo_total_s" : float
    """
    t0 = time.time()
    resultados = {}

    for nombre, fn_validate, config_key in _VALIDATORS:
        if verbose:
            print(f"  Validando {nombre}...", end=" ", flush=True)
        t_inicio = time.time()
        try:
            resultado = fn_validate(dataset_roots)
        except Exception as exc:
            resultado = {
                "dataset":        nombre,
                "estado":         "ERROR",
                "issues":         [f"Excepcion inesperada: {exc}"],
                "warnings":       [],
                "info":           {},
                "metricas":       {"n_imagenes": 0, "n_casos": 0,
                                   "n_pacientes": 0, "cobertura_pct": 0.0},
                "distribuciones": {"patologia": {}, "birads": {},
                                   "densidad": {}, "vista": {}, "lado": {}},
            }
        elapsed = time.time() - t_inicio
        resultado["tiempo_s"] = round(elapsed, 2)

        if verbose:
            estado = resultado["estado"]
            n_iss  = len(resultado["issues"])
            n_wrn  = len(resultado["warnings"])
            print(f"{estado}  ({n_iss} errores, {n_wrn} warnings, {elapsed:.1f}s)")

        resultados[nombre] = resultado

    ## Resumen global
    conteos = {"OK": 0, "WARNING": 0, "ERROR": 0}
    for r in resultados.values():
        conteos[r["estado"]] = conteos.get(r["estado"], 0) + 1

    total_imagenes = sum(
        r["metricas"].get("n_imagenes", 0) for r in resultados.values()
    )
    total_casos = sum(
        r["metricas"].get("n_casos", 0) for r in resultados.values()
    )

    return {
        "resultados":    resultados,
        "resumen": {
            **conteos,
            "total_datasets":  len(resultados),
            "total_imagenes":  total_imagenes,
            "total_casos":     total_casos,
        },
        "tiempo_total_s": round(time.time() - t0, 2),
    }