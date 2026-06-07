## threshold_tuning.py
## Ajuste de umbral para la deteccion binaria benigno/maligno
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## La decision benigno/maligno por defecto usa argmax sobre las 5 clases BI-RADS,
## lo que tiende a subdetectar malignos cuando el modelo esta sesgado a las clases
## mayoritarias. Este modulo calcula un score de malignidad continuo (suma de las
## probabilidades de las clases malignas) y busca el umbral que maximiza el indice
## de Youden (J = sensibilidad + especificidad - 1).
##
## IMPORTANTE (metodologia): el umbral optimo debe elegirse en el conjunto de
## VALIDACION y luego aplicarse fijo al de TEST. Elegir el umbral sobre el mismo
## test donde se reportan los resultados infla las metricas (optimizacion sobre
## el test). Las funciones de este modulo separan ambos pasos explicitamente.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

## Indices de las clases malignas en la numeracion de 5 clases (BI-RADS 1-5)
## indice = BI-RADS - 1, por lo que maligno (BI-RADS 4, 5) = indices 3, 4
MALIGNANT_INDICES_5CLS = [3, 4]


def malignancy_score(y_probs: np.ndarray, malignant_indices=None) -> np.ndarray:
    ##
    ## Calcula el score continuo de malignidad por muestra
    ##
    ## Es la suma de las probabilidades de las clases malignas. Un score alto
    ## significa que el modelo cree que el caso es maligno (BI-RADS 4-5).
    ##
    ## Parametros:
    ##   y_probs: [n, num_classes] probabilidades por clase (softmax)
    ##   malignant_indices: indices de las clases malignas (default: [3, 4])
    ##
    ## Retorna: [n] score de malignidad en [0, 1]
    ##
    if malignant_indices is None:
        malignant_indices = MALIGNANT_INDICES_5CLS
    y_probs = np.asarray(y_probs)
    return y_probs[:, malignant_indices].sum(axis=1)


def binary_labels(y_true_idx: np.ndarray, malignant_indices=None) -> np.ndarray:
    ##
    ## Convierte las etiquetas BI-RADS (indices) a binario: 1=maligno, 0=benigno
    ##
    if malignant_indices is None:
        malignant_indices = MALIGNANT_INDICES_5CLS
    return np.isin(np.asarray(y_true_idx), malignant_indices).astype(int)


def metrics_at_threshold(
    scores: np.ndarray, bin_true: np.ndarray, threshold: float
) -> Dict:
    ##
    ## Calcula sensibilidad, especificidad y Youden J a un umbral dado
    ##
    ## Una muestra se clasifica como maligna si su score >= threshold.
    ##
    bin_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))
    tn = int(np.sum((bin_true == 0) & (bin_pred == 0)))
    fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))
    fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    youden_j = sensitivity + specificity - 1
    balanced_acc = (sensitivity + specificity) / 2
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)
          if (ppv + sensitivity) > 0 else 0.0)

    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "youden_j": float(youden_j),
        "balanced_accuracy": float(balanced_acc),
        "f1_malignant": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def find_optimal_threshold(
    y_probs: np.ndarray,
    y_true_idx: np.ndarray,
    malignant_indices=None,
    n_steps: int = 200,
) -> Tuple[float, Dict, List[Dict]]:
    ##
    ## Encuentra el umbral que maximiza el indice de Youden
    ##
    ## DEBE ejecutarse sobre el conjunto de VALIDACION (no test).
    ##
    ## Parametros:
    ##   y_probs: [n, num_classes] probabilidades del modelo en validacion
    ##   y_true_idx: [n] etiquetas verdaderas (indices BI-RADS)
    ##   malignant_indices: indices de clases malignas (default [3,4])
    ##   n_steps: numero de umbrales a probar entre 0 y 1
    ##
    ## Retorna: (umbral_optimo, metricas_en_optimo, barrido_completo)
    ##
    scores = malignancy_score(y_probs, malignant_indices)
    bin_true = binary_labels(y_true_idx, malignant_indices)

    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    sweep = [metrics_at_threshold(scores, bin_true, t) for t in thresholds]

    ## Elegir el umbral con mayor Youden J
    best = max(sweep, key=lambda m: m["youden_j"])
    logger.info(
        "Umbral optimo (validacion): %.4f | sensibilidad=%.4f especificidad=%.4f J=%.4f",
        best["threshold"], best["sensitivity"], best["specificity"], best["youden_j"],
    )
    return best["threshold"], best, sweep


def apply_threshold(
    y_probs: np.ndarray,
    y_true_idx: np.ndarray,
    threshold: float,
    malignant_indices=None,
) -> Dict:
    ##
    ## Aplica un umbral fijo (elegido en validacion) al conjunto de TEST
    ##
    ## Esta es la evaluacion honesta: el umbral viene de validacion, se aplica
    ## sin modificar al test, y se reportan las metricas resultantes.
    ##
    scores = malignancy_score(y_probs, malignant_indices)
    bin_true = binary_labels(y_true_idx, malignant_indices)
    result = metrics_at_threshold(scores, bin_true, threshold)
    logger.info(
        "Umbral aplicado a test: %.4f | sensibilidad=%.4f especificidad=%.4f",
        threshold, result["sensitivity"], result["specificity"],
    )
    return result


def compare_argmax_vs_threshold(
    y_probs: np.ndarray,
    y_true_idx: np.ndarray,
    threshold: float,
    malignant_indices=None,
) -> Dict:
    ##
    ## Compara la deteccion binaria por argmax (default) vs por umbral ajustado
    ##
    ## Util para cuantificar cuanto mejora la sensibilidad el threshold tuning.
    ##
    if malignant_indices is None:
        malignant_indices = MALIGNANT_INDICES_5CLS

    y_probs = np.asarray(y_probs)
    bin_true = binary_labels(y_true_idx, malignant_indices)

    ## Metodo argmax: la prediccion es maligna si la clase argmax es maligna
    argmax_pred = np.argmax(y_probs, axis=1)
    bin_pred_argmax = np.isin(argmax_pred, malignant_indices).astype(int)
    tp = int(np.sum((bin_true == 1) & (bin_pred_argmax == 1)))
    tn = int(np.sum((bin_true == 0) & (bin_pred_argmax == 0)))
    fp = int(np.sum((bin_true == 0) & (bin_pred_argmax == 1)))
    fn = int(np.sum((bin_true == 1) & (bin_pred_argmax == 0)))
    argmax_metrics = {
        "method": "argmax",
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "fn": fn, "fp": fp,
    }

    ## Metodo umbral
    threshold_metrics = apply_threshold(y_probs, y_true_idx, threshold, malignant_indices)
    threshold_metrics["method"] = "threshold"

    return {"argmax": argmax_metrics, "threshold": threshold_metrics}
