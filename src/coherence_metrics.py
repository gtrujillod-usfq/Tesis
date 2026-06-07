## coherence_metrics.py
## Area 1: Metricas de alineacion LLM-Vision
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Este modulo implementa metricas para validar que el VLM genera
## explicaciones coherentes con sus predicciones, mantiene consistencia
## entre ejecuciones y calibra correctamente su confianza.

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    ## Registro de una prediccion del modelo para analizar coherencia
    image_id: str
    birads_pred: int
    birads_confidence: float
    findings_pred: List[str]
    explanation_text: str
    ground_truth_birads: int
    ground_truth_findings: Optional[List[str]] = None


class CoherenceValidator:
    ## Valida la coherencia entre prediccion y explicacion generada
    ## Integra el lexico oficial BI-RADS para validacion terminologica
    ##
    ## Metricas calculadas:
    ## 1. birads_mention_coherence: explicacion menciona keywords del nivel predicho
    ## 2. findings_support_coherence: hallazgos predichos aparecen en explicacion
    ## 3. explanation_completeness: longitud adecuada de la explicacion
    ## 4. terminology_adherence: uso de terminologia oficial BI-RADS
    ## 5. recommendation_coherence: recomendacion clinica apropiada al nivel
    
    def __init__(self, use_medical_vocabulary: bool = True):
        ##
        ## Inicializa el validador con lexico BI-RADS oficial
        ## Si use_medical_vocabulary=True, usa BIRADSLexicon para
        ## validacion terminologica enriquecida
        ##
        self.use_medical_vocabulary = use_medical_vocabulary
        self.lexicon = None
        
        if use_medical_vocabulary:
            try:
                from medical_vocabulary import BIRADSLexicon
                self.lexicon = BIRADSLexicon()
            except ImportError:
                logger.warning("medical_vocabulary no disponible, usando keywords basicas")
                self.use_medical_vocabulary = False
        
        ## Keywords basicas como fallback
        self.birads_keywords = {
            0: ["incompleto", "evaluacion adicional", "requiere", "incomplete", "additional"],
            1: ["negativo", "sin hallazgos", "normal", "negative", "no findings"],
            2: ["benigno", "no maligno", "benignidad", "benign", "non-malignant"],
            3: ["probablemente benigno", "seguimiento", "probable", "probably benign", "follow-up"],
            4: ["sospechoso", "biopsia", "puede ser maligno", "suspicious", "biopsy"],
            5: ["altamente sugestivo", "malignidad", "muy sospechoso", "highly suggestive", "malignancy"],
        }
    
    def validate_record(self, record: PredictionRecord) -> Dict[str, float]:
        ##
        ## Evalua la coherencia de un registro de prediccion
        ## Retorna diccionario con metricas de coherencia incluyendo
        ## metricas de adherencia terminologica si el lexico esta disponible
        ##
        
        results = {
            "birads_mention_coherence": self._check_birads_mention(record),
            "findings_support_coherence": self._check_findings_support(record),
            "explanation_completeness": self._check_explanation_completeness(record),
            "terminology_adherence": self._check_terminology_adherence(record),
            "recommendation_coherence": self._check_recommendation_coherence(record),
            "overall_coherence": 0.0,
        }
        
        ## Coherencia general es el promedio ponderado de todas las metricas
        weights = {
            "birads_mention_coherence": 0.25,
            "findings_support_coherence": 0.20,
            "explanation_completeness": 0.15,
            "terminology_adherence": 0.20,
            "recommendation_coherence": 0.20,
        }
        
        weighted_sum = sum(
            results[key] * weight
            for key, weight in weights.items()
        )
        results["overall_coherence"] = weighted_sum
        
        return results
    
    def _check_birads_mention(self, record: PredictionRecord) -> float:
        ##
        ## Valida que la explicacion mencione palabras clave
        ## correspondientes al nivel BI-RADS predicho
        ##
        ## Logica: un informe correcto suele estar en un solo idioma y no
        ## necesita repetir todos los sinonimos. Por eso evaluamos por idioma
        ## de forma independiente y tomamos el mejor resultado. Mencionar al
        ## menos 2 keywords relevantes del nivel se considera plena coherencia.
        ##
        explanation_lower = record.explanation_text.lower()

        ## Obtener keywords del lexico oficial o del fallback
        if self.lexicon:
            keywords_es = self.lexicon.get_keywords_for_birads(record.birads_pred, "es")
            keywords_en = self.lexicon.get_keywords_for_birads(record.birads_pred, "en")
        else:
            keywords_es = self.birads_keywords.get(record.birads_pred, [])
            keywords_en = []

        ## Contar coincidencias por idioma
        found_es = sum(1 for kw in keywords_es if kw.lower() in explanation_lower)
        found_en = sum(1 for kw in keywords_en if kw.lower() in explanation_lower)

        ## Tomar el idioma con mas coincidencias (el idioma del informe)
        best_found = max(found_es, found_en)

        if best_found == 0:
            ## Sin keywords del nivel: revisar coincidencia parcial
            ## (terminos individuales dentro de keywords compuestas)
            all_keywords = keywords_es + keywords_en
            partial = 0
            for kw in all_keywords:
                words = kw.lower().split()
                if any(w in explanation_lower for w in words if len(w) > 4):
                    partial += 1
                    break
            return 0.25 if partial > 0 else 0.0

        ## Mencionar 2 o mas keywords relevantes = plena coherencia
        ## Mencionar 1 keyword = coherencia parcial alta
        if best_found >= 2:
            return 1.0
        else:
            return 0.6

    def _check_findings_support(self, record: PredictionRecord) -> float:
        ##
        ## Valida que los hallazgos predichos se mencionen en la explicacion
        ## Usa matching flexible: un hallazgo cuenta como mencionado si su
        ## frase completa aparece, o si aparecen sus palabras significativas
        ## (esto cubre variaciones morfologicas como spiculated/espiculado)
        ##
        if not record.findings_pred or len(record.findings_pred) == 0:
            return 1.0

        explanation_lower = record.explanation_text.lower()
        mentioned = 0

        for finding in record.findings_pred:
            finding_lower = finding.lower()

            ## Coincidencia exacta de la frase
            if finding_lower in explanation_lower:
                mentioned += 1
                continue

            ## Coincidencia por palabras significativas (>4 caracteres)
            ## Se considera mencionado si aparece al menos una palabra clave
            ## o su raiz (primeros 6 caracteres para variaciones de idioma)
            words = [w for w in finding_lower.split() if len(w) > 4]
            if words:
                matches = 0
                for w in words:
                    root = w[:6]
                    if w in explanation_lower or root in explanation_lower:
                        matches += 1
                if matches > 0:
                    mentioned += 1

        return mentioned / len(record.findings_pred)

    def _check_explanation_completeness(self, record: PredictionRecord) -> float:
        ##
        ## Valida que la explicacion tenga longitud adecuada
        ## Usa heuristica: minimo 50 caracteres, maximo 2000
        ##
        text_length = len(record.explanation_text)
        
        if text_length < 50:
            return 0.0
        elif text_length < 100:
            return 0.5
        elif text_length > 2000:
            return 0.7
        else:
            return 1.0
    
    def _check_terminology_adherence(self, record: PredictionRecord) -> float:
        ##
        ## Valida que la explicacion use terminologia oficial del lexico BI-RADS
        ## Busca descriptores de masa, calcificaciones, densidad, etc.
        ## Requiere el modulo medical_vocabulary
        ##
        if not self.lexicon:
            return 0.5
        
        return self.lexicon.compute_terminology_adherence(record.explanation_text)
    
    def _check_recommendation_coherence(self, record: PredictionRecord) -> float:
        ##
        ## Valida que la recomendacion clinica en el texto sea coherente
        ## con el nivel BI-RADS predicho
        ## Ejemplo: BI-RADS 5 deberia recomendar biopsia, no seguimiento rutinario
        ##
        if not self.lexicon:
            return 0.5
        
        return self.lexicon.validate_recommendation_coherence(
            record.birads_pred, record.explanation_text
        )


class CalibrationAnalyzer:
    ## Analiza si las confianzas predichas por el modelo se alinean
    ## con su desempeño real (calibracion)
    ##
    ## Modelo bien calibrado: confianza 80% deberia acertar ~80% de las veces
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        ground_truth: np.ndarray,
        num_classes: int = 6,
    ) -> Dict[str, float]:
        ##
        ## Calcula metricas de calibracion
        ##
        ## Parametros:
        ## predictions: [n_samples] predicciones del modelo (clase)
        ## confidences: [n_samples] confianza [0, 1]
        ## ground_truth: [n_samples] etiquetas verdaderas
        ## num_classes: numero de clases BI-RADS (0-5 = 6 clases)
        ##
        ## Nota: el Brier Score y NLL multiclase se calculan manualmente
        ## para evitar incompatibilidades entre versiones de scikit-learn
        ## (brier_score_loss solo acepta arrays 1D en versiones recientes)
        ##

        predictions = np.asarray(predictions)
        confidences = np.asarray(confidences)
        ground_truth = np.asarray(ground_truth)
        n = len(predictions)

        assert n == len(confidences) == len(ground_truth)

        ## Exactitud
        accuracy = np.mean(predictions == ground_truth)

        ## Construir matrices one-hot de probabilidad
        ## one_hot_preds: probabilidad asignada a la clase predicha = confidence
        ## y el resto de la masa se reparte uniformemente entre las otras clases
        prob_matrix = np.full((n, num_classes), 0.0)
        for i in range(n):
            remaining = (1.0 - confidences[i]) / (num_classes - 1)
            prob_matrix[i, :] = remaining
            prob_matrix[i, predictions[i]] = confidences[i]

        ## Matriz one-hot de verdad terreno
        one_hot_truth = np.zeros((n, num_classes))
        one_hot_truth[np.arange(n), ground_truth] = 1.0

        ## Brier Score multiclase: media de la suma de diferencias cuadradas
        ## sobre todas las clases (rango [0, 2], menor es mejor)
        brier = np.mean(np.sum((prob_matrix - one_hot_truth) ** 2, axis=1))

        ## Negative Log Likelihood multiclase
        ## Se toma la probabilidad asignada a la clase verdadera
        eps = 1e-12
        true_class_probs = prob_matrix[np.arange(n), ground_truth]
        true_class_probs = np.clip(true_class_probs, eps, 1.0)
        nll = -np.mean(np.log(true_class_probs))

        ## Expected Calibration Error (ECE)
        ece = self._compute_ece(predictions, confidences, ground_truth)

        ## Maximum Calibration Error (MCE)
        mce = self._compute_mce(predictions, confidences, ground_truth)

        return {
            "accuracy": float(accuracy),
            "brier_score": float(brier),
            "ece": float(ece),
            "mce": float(mce),
            "nll": float(nll),
        }
    
    def _compute_ece(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        ##
        ## Expected Calibration Error
        ## Mide el error promedio entre confianza promedio y exactitud en cada bin
        ##
        
        ## Crear bins de confianza
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        ece = 0.0
        for bin_idx in range(self.n_bins):
            mask = bin_indices == bin_idx
            if not mask.any():
                continue
            
            bin_confidences = confidences[mask]
            bin_predictions = predictions[mask]
            bin_truth = ground_truth[mask]
            
            bin_accuracy = np.mean(bin_predictions == bin_truth)
            bin_confidence = np.mean(bin_confidences)
            
            bin_weight = np.sum(mask) / len(predictions)
            ece += bin_weight * np.abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def _compute_mce(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        ##
        ## Maximum Calibration Error
        ## Toma el error maximo en cualquier bin
        ##
        
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        max_error = 0.0
        for bin_idx in range(self.n_bins):
            mask = bin_indices == bin_idx
            if not mask.any():
                continue
            
            bin_accuracy = np.mean(predictions[mask] == ground_truth[mask])
            bin_confidence = np.mean(confidences[mask])
            
            error = np.abs(bin_confidence - bin_accuracy)
            max_error = max(max_error, error)
        
        return max_error
    
    def plot_calibration_curve(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        ground_truth: np.ndarray,
        save_path: Optional[Path] = None,
    ):
        ##
        ## Grafica la curva de calibracion: confianza vs exactitud por bin
        ##
        
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        bin_confidences = []
        bin_accuracies = []
        
        for bin_idx in range(self.n_bins):
            mask = bin_indices == bin_idx
            if not mask.any():
                continue
            
            bin_confidences.append(np.mean(confidences[mask]))
            bin_accuracies.append(np.mean(predictions[mask] == ground_truth[mask]))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ## Diagonal perfecta
        ax.plot([0, 1], [0, 1], "k--", label="Calibracion perfecta", lw=2)
        
        ## Curva real
        ax.scatter(bin_confidences, bin_accuracies, s=100, alpha=0.6, label="Modelo")
        ax.plot(bin_confidences, bin_accuracies, "b-", alpha=0.4)
        
        ax.set_xlabel("Confianza promedio", fontsize=12)
        ax.set_ylabel("Exactitud", fontsize=12)
        ax.set_title("Curva de Calibracion del Modelo", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Calibration curve guardada en: %s", save_path)
        
        plt.close()


class ConsistencyAnalyzer:
    ## Analiza la consistencia: misma imagen predicha multiples veces
    ## deberia dar el mismo resultado (o muy similar con pequeña variacion)
    
    def __init__(self):
        self.predictions_history = defaultdict(list)
    
    def record_prediction(self, image_id: str, birads_pred: int, confidence: float):
        ##
        ## Registra una prediccion para posterior analisis de consistencia
        ##
        self.predictions_history[image_id].append({
            "birads": birads_pred,
            "confidence": confidence,
        })
    
    def compute_consistency_metrics(self) -> Dict[str, float]:
        ##
        ## Calcula metricas de consistencia a nivel de imagen
        ##
        
        if len(self.predictions_history) == 0:
            logger.warning("No hay predicciones registradas para consistencia")
            return {}
        
        ## Solo considerar imagenes con multiples predicciones
        multi_pred_images = {
            img_id: preds
            for img_id, preds in self.predictions_history.items()
            if len(preds) > 1
        }
        
        if len(multi_pred_images) == 0:
            return {"consistency_available": False}
        
        results = {
            "consistency_available": True,
            "num_images_evaluated": len(multi_pred_images),
            "birads_agreement_rate": 0.0,
            "confidence_std_mean": 0.0,
            "confidence_std_max": 0.0,
        }
        
        birads_agreements = []
        confidence_stds = []
        
        for img_id, preds in multi_pred_images.items():
            birads_values = np.array([p["birads"] for p in preds])
            confidence_values = np.array([p["confidence"] for p in preds])
            
            ## Acuerdo en BI-RADS: 1 si todos iguales, 0 si hay discrepancia
            birads_agreement = 1.0 if len(np.unique(birads_values)) == 1 else 0.0
            birads_agreements.append(birads_agreement)
            
            ## Desviacion estandar de confianzas
            confidence_std = np.std(confidence_values)
            confidence_stds.append(confidence_std)
        
        results["birads_agreement_rate"] = float(np.mean(birads_agreements))
        results["confidence_std_mean"] = float(np.mean(confidence_stds))
        results["confidence_std_max"] = float(np.max(confidence_stds))
        
        return results


class AlignmentMetricsCollector:
    ## Colector principal que agrupa todas las metricas de Area 1
    
    def __init__(self, n_calibration_bins: int = 10):
        self.coherence_validator = CoherenceValidator()
        self.calibration_analyzer = CalibrationAnalyzer(n_bins=n_calibration_bins)
        self.consistency_analyzer = ConsistencyAnalyzer()
        
        self.coherence_scores = []
        self.all_metrics = {}
    
    def add_prediction(self, record: PredictionRecord):
        ##
        ## Agrega una prediccion para analizar
        ##
        coherence_metrics = self.coherence_validator.validate_record(record)
        self.coherence_scores.append(coherence_metrics["overall_coherence"])
        
        self.consistency_analyzer.record_prediction(
            record.image_id,
            record.birads_pred,
            record.birads_confidence
        )
        
        self.all_metrics[record.image_id] = {
            "coherence": coherence_metrics,
            "record": record,
        }
    
    def compute_final_report(self) -> Dict:
        ##
        ## Genera reporte final con todas las metricas de Area 1
        ##
        
        report = {
            "area_1_alignment_metrics": {
                "coherence": {
                    "mean_coherence": float(np.mean(self.coherence_scores)) if self.coherence_scores else 0.0,
                    "std_coherence": float(np.std(self.coherence_scores)) if self.coherence_scores else 0.0,
                    "min_coherence": float(np.min(self.coherence_scores)) if self.coherence_scores else 0.0,
                    "max_coherence": float(np.max(self.coherence_scores)) if self.coherence_scores else 0.0,
                    "coherence_above_threshold_75": float(
                        np.mean(np.array(self.coherence_scores) >= 0.75)
                    ) if self.coherence_scores else 0.0,
                },
                "consistency": self.consistency_analyzer.compute_consistency_metrics(),
            }
        }
        
        return report
    
    def generate_summary(self) -> str:
        ##
        ## Genera resumen textual de metricas de coherencia y calibracion
        ##
        report = self.compute_final_report()
        coherence_metrics = report["area_1_alignment_metrics"]["coherence"]
        consistency_metrics = report["area_1_alignment_metrics"]["consistency"]
        
        summary = []
        summary.append("=" * 70)
        summary.append("AREA 1: METRICAS DE ALINEACION LLM-VISION")
        summary.append("=" * 70)
        summary.append("")
        
        summary.append("COHERENCIA EXPLICACION-PREDICCION")
        summary.append("-" * 70)
        summary.append(f"  Coherencia promedio:              {coherence_metrics['mean_coherence']:.4f}")
        summary.append(f"  Desviacion estandar:              {coherence_metrics['std_coherence']:.4f}")
        summary.append(f"  Coherencia minima:                {coherence_metrics['min_coherence']:.4f}")
        summary.append(f"  Coherencia maxima:                {coherence_metrics['max_coherence']:.4f}")
        summary.append(f"  % con coherencia >= 0.75:         {coherence_metrics['coherence_above_threshold_75']*100:.1f}%")
        summary.append("")
        
        summary.append("CONSISTENCIA DE PREDICCIONES")
        summary.append("-" * 70)
        if consistency_metrics.get("consistency_available"):
            summary.append(f"  Imagenes evaluadas:               {consistency_metrics['num_images_evaluated']}")
            summary.append(f"  Acuerdo BI-RADS:                  {consistency_metrics['birads_agreement_rate']*100:.1f}%")
            summary.append(f"  Desv. estandar confianza (media): {consistency_metrics['confidence_std_mean']:.4f}")
            summary.append(f"  Desv. estandar confianza (max):   {consistency_metrics['confidence_std_max']:.4f}")
        else:
            summary.append("  (Requiere multiples predicciones por imagen)")
        summary.append("")
        
        summary.append("=" * 70)
        
        return "\n".join(summary)
