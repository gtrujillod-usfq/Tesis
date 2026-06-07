## medical_metrics.py
## Area 3: Metricas del Dominio Medico
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Implementa metricas clinicas para evaluar el modelo en terminos medicos:
##   1. Clasificacion multiclase BI-RADS 0-5 (matriz confusion, F1, MCC, AUC)
##   2. Metricas por severidad clinica (benigno 0-3 vs maligno 4-5)
##   3. Reproducibilidad inter-evaluador (Cohen Kappa, Weighted Kappa)
##   4. Analisis de errores clinicamente relevantes
##
## Corte clinico estandar: BI-RADS 0-3 = benigno/seguimiento, 4-5 = biopsia

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

## Corte clinico estandar para agrupacion binaria
##
## Numeracion de 6 clases (indices 0-5 = BI-RADS 0-5), usada en exp01-05:
BENIGN_LEVELS = [0, 1, 2, 3]
MALIGNANT_LEVELS = [4, 5]

## Numeracion de 5 clases (indices 0-4 = BI-RADS 1-5), usada desde exp06:
##   indice = BI-RADS - 1, por lo tanto:
##   benigno  = BI-RADS 1, 2, 3 = indices 0, 1, 2
##   maligno  = BI-RADS 4, 5     = indices 3, 4
BENIGN_LEVELS_5CLS = [0, 1, 2]
MALIGNANT_LEVELS_5CLS = [3, 4]


class BIRADSClassificationMetrics:
    ## Metricas de clasificacion multiclase para BI-RADS 0-5

    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
    ) -> Dict:
        ##
        ## Calcula todas las metricas de clasificacion multiclase
        ##
        ## Parametros:
        ##   y_true: [n] etiquetas verdaderas (0-5)
        ##   y_pred: [n] predicciones (0-5)
        ##   y_probs: [n, 6] probabilidades por clase (opcional, para AUC)
        ##
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        results = {
            "accuracy": self._accuracy(y_true, y_pred),
            "confusion_matrix": self._confusion_matrix(y_true, y_pred).tolist(),
            "per_class": self._per_class_metrics(y_true, y_pred),
            "macro_f1": self._macro_f1(y_true, y_pred),
            "weighted_f1": self._weighted_f1(y_true, y_pred),
            "mcc": self._matthews_corrcoef(y_true, y_pred),
            "quadratic_kappa": self._quadratic_weighted_kappa(y_true, y_pred),
        }

        if y_probs is not None:
            results["auc_ovr"] = self._auc_one_vs_rest(y_true, np.asarray(y_probs))

        return results

    def _accuracy(self, y_true, y_pred) -> float:
        return float(np.mean(y_true == y_pred))

    def _confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        ## Matriz de confusion [num_classes, num_classes]
        ## Filas = verdad, columnas = prediccion
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    def _per_class_metrics(self, y_true, y_pred) -> Dict:
        ##
        ## Precision, recall y F1 por cada nivel BI-RADS
        ##
        per_class = {}
        for cls in range(self.num_classes):
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            support = int(np.sum(y_true == cls))

            per_class[f"birads_{cls}"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        return per_class

    def _macro_f1(self, y_true, y_pred) -> float:
        ## F1 macro: promedio simple de F1 por clase
        f1_scores = []
        for cls in range(self.num_classes):
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            f1_scores.append(f1)
        return float(np.mean(f1_scores))

    def _weighted_f1(self, y_true, y_pred) -> float:
        ## F1 weighted: promedio ponderado por soporte de cada clase
        f1_scores = []
        weights = []
        for cls in range(self.num_classes):
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            support = np.sum(y_true == cls)
            f1_scores.append(f1)
            weights.append(support)

        weights = np.array(weights)
        if weights.sum() == 0:
            return 0.0
        return float(np.average(f1_scores, weights=weights))

    def _matthews_corrcoef(self, y_true, y_pred) -> float:
        ##
        ## Matthews Correlation Coefficient multiclase
        ## Robusto ante desbalance de clases (importante en BI-RADS)
        ##
        cm = self._confusion_matrix(y_true, y_pred).astype(float)
        n = cm.sum()
        if n == 0:
            return 0.0

        ## Sumas por fila y columna
        t = cm.sum(axis=1)  ## verdad por clase
        p = cm.sum(axis=0)  ## prediccion por clase
        c = np.trace(cm)    ## aciertos totales

        ## Formula MCC multiclase
        cov_ytyp = c * n - np.dot(t, p)
        cov_ypyp = n * n - np.dot(p, p)
        cov_ytyt = n * n - np.dot(t, t)

        denom = np.sqrt(cov_ypyp * cov_ytyt)
        if denom == 0:
            return 0.0
        return float(cov_ytyp / denom)

    def _quadratic_weighted_kappa(self, y_true, y_pred) -> float:
        ##
        ## Quadratic Weighted Kappa
        ## Penaliza mas los errores entre niveles BI-RADS distantes
        ## (confundir BI-RADS 1 con 5 es peor que 1 con 2)
        ## Es la metrica estandar para escalas ordinales como BI-RADS
        ##
        cm = self._confusion_matrix(y_true, y_pred).astype(float)
        n = self.num_classes

        ## Matriz de pesos cuadraticos
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

        ## Matriz de acuerdo esperado por azar
        row_marginals = cm.sum(axis=1)
        col_marginals = cm.sum(axis=0)
        total = cm.sum()
        if total == 0:
            return 0.0

        expected = np.outer(row_marginals, col_marginals) / total

        ## Kappa ponderado
        observed_disagreement = np.sum(weights * cm)
        expected_disagreement = np.sum(weights * expected)

        if expected_disagreement == 0:
            return 0.0
        return float(1.0 - observed_disagreement / expected_disagreement)

    def _auc_one_vs_rest(self, y_true, y_probs) -> Dict:
        ##
        ## AUC-ROC one-vs-rest por cada clase
        ## Calculado manualmente para no depender de sklearn
        ##
        auc_per_class = {}
        aucs = []

        for cls in range(self.num_classes):
            ## Etiquetas binarias: 1 si es la clase, 0 si no
            binary_true = (y_true == cls).astype(int)
            scores = y_probs[:, cls]

            auc = self._binary_auc(binary_true, scores)
            auc_per_class[f"birads_{cls}"] = float(auc)
            if not np.isnan(auc):
                aucs.append(auc)

        auc_per_class["macro_avg"] = float(np.mean(aucs)) if aucs else 0.0
        return auc_per_class

    def _binary_auc(self, y_true, scores) -> float:
        ##
        ## AUC binario via la formula de rangos (equivalente a Mann-Whitney U)
        ##
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return float("nan")

        ## Ordenar por score y asignar rangos
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)

        ## Manejar empates promediando rangos
        sorted_scores = scores[order]
        i = 0
        while i < len(sorted_scores):
            j = i
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            if j - i > 1:
                avg_rank = np.mean(ranks[order[i:j]])
                ranks[order[i:j]] = avg_rank
            i = j

        sum_ranks_pos = np.sum(ranks[y_true == 1])
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)


class ClinicalSeverityMetrics:
    ## Metricas por severidad clinica (benigno 0-3 vs maligno 4-5)
    ##
    ## Esta es la decision clinica mas critica: distinguir casos que
    ## necesitan biopsia (4-5) de los que solo requieren seguimiento (0-3)

    def __init__(self, benign_levels=None, malignant_levels=None):
        self.benign_levels = benign_levels or BENIGN_LEVELS
        self.malignant_levels = malignant_levels or MALIGNANT_LEVELS

    def to_binary(self, birads: np.ndarray) -> np.ndarray:
        ## Convierte BI-RADS 0-5 a binario: 0=benigno, 1=maligno/sospechoso
        birads = np.asarray(birads)
        binary = np.isin(birads, self.malignant_levels).astype(int)
        return binary

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        ##
        ## Calcula metricas clinicas binarias
        ##
        bin_true = self.to_binary(y_true)
        bin_pred = self.to_binary(y_pred)

        ## Matriz de confusion binaria
        tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))
        tn = int(np.sum((bin_true == 0) & (bin_pred == 0)))
        fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))
        fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))

        ## Metricas clinicas
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  ## recall, detectar maligno
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  ## evitar falsa alarma
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          ## valor predictivo positivo
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          ## valor predictivo negativo
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        ## F1 para clase maligna (la critica)
        f1 = (2 * ppv * sensitivity / (ppv + sensitivity)
              if (ppv + sensitivity) > 0 else 0.0)

        ## Balanced accuracy
        balanced_acc = (sensitivity + specificity) / 2

        ## Youden's J (indice de calidad del punto de corte)
        youden_j = sensitivity + specificity - 1

        return {
            "confusion_binary": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "f1_malignant": float(f1),
            "youden_j": float(youden_j),
            "false_negatives": fn,
            "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0,
        }


class InterRaterReliability:
    ## Reproducibilidad inter-evaluador (modelo vs radiologo)
    ## Cohen Kappa y Weighted Kappa, estandar de oro en radiologia

    def compute_cohen_kappa(self, rater1: np.ndarray, rater2: np.ndarray) -> float:
        ##
        ## Cohen's Kappa: acuerdo entre dos evaluadores corregido por azar
        ##
        rater1 = np.asarray(rater1)
        rater2 = np.asarray(rater2)
        n = len(rater1)
        if n == 0:
            return 0.0

        ## Acuerdo observado
        observed = np.mean(rater1 == rater2)

        ## Acuerdo esperado por azar
        classes = np.unique(np.concatenate([rater1, rater2]))
        expected = 0.0
        for cls in classes:
            p1 = np.mean(rater1 == cls)
            p2 = np.mean(rater2 == cls)
            expected += p1 * p2

        if expected == 1.0:
            return 1.0
        return float((observed - expected) / (1 - expected))

    def compute_weighted_kappa(
        self, rater1: np.ndarray, rater2: np.ndarray, num_classes: int = 6
    ) -> float:
        ##
        ## Weighted Kappa (lineal) para escalas ordinales
        ## Penaliza segun la distancia entre categorias
        ##
        rater1 = np.asarray(rater1)
        rater2 = np.asarray(rater2)
        n = len(rater1)
        if n == 0:
            return 0.0

        ## Matriz de confusion entre evaluadores
        cm = np.zeros((num_classes, num_classes))
        for a, b in zip(rater1, rater2):
            cm[int(a), int(b)] += 1

        ## Pesos lineales
        weights = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = abs(i - j) / (num_classes - 1)

        ## Marginales
        row_marg = cm.sum(axis=1)
        col_marg = cm.sum(axis=0)
        expected = np.outer(row_marg, col_marg) / n

        observed_disagreement = np.sum(weights * cm)
        expected_disagreement = np.sum(weights * expected)

        if expected_disagreement == 0:
            return 0.0
        return float(1.0 - observed_disagreement / expected_disagreement)

    def interpret_kappa(self, kappa: float) -> str:
        ## Interpretacion estandar de Kappa (Landis & Koch)
        if kappa < 0:
            return "sin acuerdo"
        elif kappa < 0.20:
            return "acuerdo leve"
        elif kappa < 0.40:
            return "acuerdo regular"
        elif kappa < 0.60:
            return "acuerdo moderado"
        elif kappa < 0.80:
            return "acuerdo sustancial"
        else:
            return "acuerdo casi perfecto"


class ClinicalErrorAnalysis:
    ## Analisis de errores clinicamente relevantes
    ## Distingue errores leves de errores graves

    def __init__(self, num_classes: int = 6, benign_levels=None, malignant_levels=None):
        self.num_classes = num_classes
        ## Niveles de corte benigno/maligno (parametrizables segun la numeracion)
        self.benign_levels = benign_levels if benign_levels is not None else BENIGN_LEVELS
        self.malignant_levels = malignant_levels if malignant_levels is not None else MALIGNANT_LEVELS

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        ##
        ## Clasifica los errores por gravedad clinica
        ##
        ## - Error leve: diferencia de 1 nivel BI-RADS
        ## - Error moderado: diferencia de 2 niveles
        ## - Error grave: diferencia de 3+ niveles, o cruzar el umbral
        ##   benigno/maligno (falsos negativos peligrosos)
        ##
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        errors = y_true != y_pred
        n_errors = int(np.sum(errors))
        n_total = len(y_true)

        diffs = np.abs(y_true - y_pred)

        mild = int(np.sum((diffs == 1)))
        moderate = int(np.sum((diffs == 2)))
        severe = int(np.sum((diffs >= 3)))

        ## Falsos negativos peligrosos: maligno predicho como benigno
        dangerous_fn = int(np.sum(
            np.isin(y_true, self.malignant_levels) & np.isin(y_pred, self.benign_levels)
        ))

        ## Falsos positivos: benigno predicho como maligno
        false_positives = int(np.sum(
            np.isin(y_true, self.benign_levels) & np.isin(y_pred, self.malignant_levels)
        ))

        return {
            "total_samples": n_total,
            "total_errors": n_errors,
            "error_rate": float(n_errors / n_total) if n_total > 0 else 0.0,
            "mild_errors": mild,
            "moderate_errors": moderate,
            "severe_errors": severe,
            "dangerous_false_negatives": dangerous_fn,
            "false_positives_biopsy": false_positives,
            "mean_birads_distance": float(np.mean(diffs)),
        }


class MedicalMetricsReport:
    ## Reporte consolidado de todas las metricas del dominio medico (Area 3)

    def __init__(self, num_classes: int = 6, benign_levels=None, malignant_levels=None):
        ##
        ## Parametros:
        ##   num_classes: numero de clases BI-RADS (6 para exp01-05, 5 desde exp06)
        ##   benign_levels, malignant_levels: indices que definen el corte
        ##     binario benigno/maligno. Si no se especifican, se infieren del
        ##     num_classes (5 clases -> numeracion BI-RADS 1-5; otro -> 0-5).
        ##
        self.num_classes = num_classes

        ## Inferir el corte correcto segun la numeracion si no se especifica
        if benign_levels is None or malignant_levels is None:
            if num_classes == 5:
                ## exp06+: indices 0-4 = BI-RADS 1-5
                benign_levels = BENIGN_LEVELS_5CLS
                malignant_levels = MALIGNANT_LEVELS_5CLS
            else:
                ## exp01-05: indices 0-5 = BI-RADS 0-5
                benign_levels = BENIGN_LEVELS
                malignant_levels = MALIGNANT_LEVELS

        self.benign_levels = benign_levels
        self.malignant_levels = malignant_levels

        self.classification = BIRADSClassificationMetrics(num_classes)
        self.severity = ClinicalSeverityMetrics(
            benign_levels=benign_levels, malignant_levels=malignant_levels
        )
        self.inter_rater = InterRaterReliability()
        self.error_analysis = ClinicalErrorAnalysis(
            num_classes, benign_levels=benign_levels, malignant_levels=malignant_levels
        )

    def compute_full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
    ) -> Dict:
        ##
        ## Genera el reporte completo de Area 3
        ##
        report = {
            "area_3_medical_metrics": {
                "multiclass_classification": self.classification.compute_all(
                    y_true, y_pred, y_probs
                ),
                "clinical_severity": self.severity.compute_all(y_true, y_pred),
                "inter_rater_reliability": {
                    "cohen_kappa": self.inter_rater.compute_cohen_kappa(y_true, y_pred),
                    "weighted_kappa": self.inter_rater.compute_weighted_kappa(
                        y_true, y_pred, self.num_classes
                    ),
                },
                "error_analysis": self.error_analysis.analyze(y_true, y_pred),
            }
        }

        ## Agregar interpretacion de kappa
        wk = report["area_3_medical_metrics"]["inter_rater_reliability"]["weighted_kappa"]
        report["area_3_medical_metrics"]["inter_rater_reliability"]["interpretation"] = (
            self.inter_rater.interpret_kappa(wk)
        )

        return report

    def generate_summary(self, report: Dict) -> str:
        ##
        ## Genera resumen textual del reporte de Area 3
        ##
        m = report["area_3_medical_metrics"]
        cls = m["multiclass_classification"]
        sev = m["clinical_severity"]
        irr = m["inter_rater_reliability"]
        err = m["error_analysis"]

        lines = []
        lines.append("=" * 70)
        lines.append("AREA 3: METRICAS DEL DOMINIO MEDICO")
        lines.append("=" * 70)
        lines.append("")

        ## Encabezado segun la numeracion (5 clases = BI-RADS 1-5, otro = 0-5)
        rango_birads = "1-5" if self.num_classes == 5 else "0-5"
        lines.append(f"CLASIFICACION MULTICLASE BI-RADS ({rango_birads})")
        lines.append("-" * 70)
        lines.append(f"  Accuracy:               {cls['accuracy']:.4f}")
        lines.append(f"  Macro F1:               {cls['macro_f1']:.4f}")
        lines.append(f"  Weighted F1:            {cls['weighted_f1']:.4f}")
        lines.append(f"  MCC:                    {cls['mcc']:.4f}")
        lines.append(f"  Quadratic Kappa:        {cls['quadratic_kappa']:.4f}")
        if "auc_ovr" in cls:
            lines.append(f"  AUC-ROC (macro):        {cls['auc_ovr']['macro_avg']:.4f}")
        lines.append("")

        ## Describir el corte benigno/maligno segun los niveles reales (en BI-RADS)
        if self.num_classes == 5:
            corte_desc = "benigno BR1-3 vs maligno BR4-5"
        else:
            corte_desc = "benigno 0-3 vs maligno 4-5"
        lines.append(f"SEVERIDAD CLINICA ({corte_desc})")
        lines.append("-" * 70)
        lines.append(f"  Sensibilidad:           {sev['sensitivity']:.4f}")
        lines.append(f"  Especificidad:          {sev['specificity']:.4f}")
        lines.append(f"  VPP (precision):        {sev['ppv']:.4f}")
        lines.append(f"  VPN:                    {sev['npv']:.4f}")
        lines.append(f"  Balanced Accuracy:      {sev['balanced_accuracy']:.4f}")
        lines.append(f"  F1 (maligno):           {sev['f1_malignant']:.4f}")
        lines.append(f"  Youden J:               {sev['youden_j']:.4f}")
        lines.append("")

        lines.append("REPRODUCIBILIDAD INTER-EVALUADOR (modelo vs radiologo)")
        lines.append("-" * 70)
        lines.append(f"  Cohen Kappa:            {irr['cohen_kappa']:.4f}")
        lines.append(f"  Weighted Kappa:         {irr['weighted_kappa']:.4f}")
        lines.append(f"  Interpretacion:         {irr['interpretation']}")
        lines.append("")

        lines.append("ANALISIS DE ERRORES CLINICOS")
        lines.append("-" * 70)
        lines.append(f"  Tasa de error:          {err['error_rate']:.4f}")
        lines.append(f"  Errores leves (1 nivel): {err['mild_errors']}")
        lines.append(f"  Errores moderados (2):   {err['moderate_errors']}")
        lines.append(f"  Errores graves (3+):     {err['severe_errors']}")
        lines.append(f"  Falsos negativos graves: {err['dangerous_false_negatives']}")
        lines.append(f"    (maligno clasificado como benigno)")
        lines.append(f"  Falsos positivos:        {err['false_positives_biopsy']}")
        lines.append(f"    (benigno enviado a biopsia)")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
