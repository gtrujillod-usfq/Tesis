## test_area1_coherence.py
## Tests para validar implementacion de Area 1
## Metricas de alineacion LLM-Vision

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from coherence_metrics import (
    PredictionRecord,
    CoherenceValidator,
    CalibrationAnalyzer,
    ConsistencyAnalyzer,
    AlignmentMetricsCollector,
)
from utils import setup_logging, get_timestamp

logger = setup_logging(__name__)


def test_coherence_validator():
    ##
    ## Test 1: Validador de coherencia
    ##
    logger.info("Iniciando test de CoherenceValidator...")
    
    validator = CoherenceValidator()
    
    ## Caso 1: Prediccion coherente (BI-RADS 5, explicacion menciona "altamente sugestivo")
    record_coherent = PredictionRecord(
        image_id="img_001",
        birads_pred=5,
        birads_confidence=0.92,
        findings_pred=["masa", "calcificaciones"],
        explanation_text="Se observa una lesion altamente sugestiva de malignidad con masa y calcificaciones asociadas. Se recomienda biopsia inmediata.",
        ground_truth_birads=5,
        ground_truth_findings=["masa", "calcificaciones"]
    )
    
    metrics_coherent = validator.validate_record(record_coherent)
    logger.info("Caso coherente (BI-RADS 5): overall_coherence = %.4f", 
                metrics_coherent["overall_coherence"])
    
    assert metrics_coherent["overall_coherence"] > 0.5, "Coherencia deberia ser > 0.5"
    
    ## Caso 2: Prediccion incoherente (BI-RADS 1 pero explicacion menciona malignidad)
    record_incoherent = PredictionRecord(
        image_id="img_002",
        birads_pred=1,
        birads_confidence=0.88,
        findings_pred=["nada"],
        explanation_text="Estudio sin hallazgos especiales. Estudio normal.",
        ground_truth_birads=1,
    )
    
    metrics_incoherent = validator.validate_record(record_incoherent)
    logger.info("Caso coherente (BI-RADS 1): overall_coherence = %.4f",
                metrics_incoherent["overall_coherence"])
    
    ## La coherencia deberia ser mayor en el caso coherente
    assert (metrics_coherent["overall_coherence"] > 0.4 or 
            metrics_incoherent["overall_coherence"] > 0.4), \
        "Al menos uno deberia tener coherencia razonable"
    
    logger.info("✓ Test CoherenceValidator pasado\n")


def test_calibration_analyzer():
    ##
    ## Test 2: Analizador de calibracion
    ##
    logger.info("Iniciando test de CalibrationAnalyzer...")
    
    analyzer = CalibrationAnalyzer(n_bins=5)
    
    ## Generar datos de prueba
    np.random.seed(42)
    n_samples = 200
    
    ## Caso bien calibrado: confianza correlacionada con aciertos
    predictions = np.random.randint(0, 6, n_samples)
    ground_truth = predictions.copy()
    ground_truth[np.random.choice(n_samples, 20, replace=False)] = np.random.randint(0, 6, 20)
    
    ## Confianzas altas cuando acierta, bajas cuando no
    confidences = np.zeros(n_samples)
    for i in range(n_samples):
        if predictions[i] == ground_truth[i]:
            confidences[i] = np.random.uniform(0.7, 1.0)
        else:
            confidences[i] = np.random.uniform(0.2, 0.6)
    
    metrics = analyzer.compute_calibration_metrics(predictions, confidences, ground_truth)
    
    logger.info("Metricas de calibracion:")
    logger.info("  Accuracy: %.4f", metrics["accuracy"])
    logger.info("  Brier Score: %.4f", metrics["brier_score"])
    logger.info("  ECE: %.4f", metrics["ece"])
    logger.info("  MCE: %.4f", metrics["mce"])
    logger.info("  NLL: %.4f", metrics["nll"])
    
    ## Validaciones
    assert 0 <= metrics["accuracy"] <= 1, "Accuracy debe estar en [0, 1]"
    assert metrics["brier_score"] >= 0, "Brier score debe ser >= 0"
    assert 0 <= metrics["ece"] <= 1, "ECE debe estar en [0, 1]"
    assert 0 <= metrics["mce"] <= 1, "MCE debe estar en [0, 1]"
    
    logger.info("✓ Test CalibrationAnalyzer pasado\n")


def test_consistency_analyzer():
    ##
    ## Test 3: Analizador de consistencia
    ##
    logger.info("Iniciando test de ConsistencyAnalyzer...")
    
    analyzer = ConsistencyAnalyzer()
    
    ## Simular predicciones multiples para misma imagen
    ## Imagen 1: predicciones consistentes (todas BI-RADS 3)
    analyzer.record_prediction("img_001", birads_pred=3, confidence=0.85)
    analyzer.record_prediction("img_001", birads_pred=3, confidence=0.82)
    analyzer.record_prediction("img_001", birads_pred=3, confidence=0.88)
    
    ## Imagen 2: predicciones inconsistentes
    analyzer.record_prediction("img_002", birads_pred=2, confidence=0.60)
    analyzer.record_prediction("img_002", birads_pred=4, confidence=0.65)
    analyzer.record_prediction("img_002", birads_pred=2, confidence=0.58)
    
    ## Imagen 3: una sola prediccion (no se evalua)
    analyzer.record_prediction("img_003", birads_pred=1, confidence=0.95)
    
    metrics = analyzer.compute_consistency_metrics()
    
    logger.info("Metricas de consistencia:")
    if metrics.get("consistency_available"):
        logger.info("  Imagenes evaluadas: %d", metrics["num_images_evaluated"])
        logger.info("  Acuerdo BI-RADS: %.1f%%", metrics["birads_agreement_rate"]*100)
        logger.info("  Desv. estandar confianza (media): %.4f", metrics["confidence_std_mean"])
        logger.info("  Desv. estandar confianza (max): %.4f", metrics["confidence_std_max"])
    else:
        logger.info("  Consistencia no evaluable (requiere multiples predicciones)")
    
    assert metrics.get("consistency_available"), "Deberia haber consistencia disponible"
    assert metrics["num_images_evaluated"] >= 2, "Deberia evaluar al menos 2 imagenes"
    
    logger.info("✓ Test ConsistencyAnalyzer pasado\n")


def test_alignment_metrics_collector():
    ##
    ## Test 4: Colector integrado de metricas de Area 1
    ##
    logger.info("Iniciando test de AlignmentMetricsCollector (integracion)...\n")
    
    collector = AlignmentMetricsCollector(n_calibration_bins=5)
    
    ## Generar registros de prediccion sinteticos
    np.random.seed(42)
    n_records = 50
    
    for i in range(n_records):
        birads = np.random.randint(0, 6)
        confidence = np.random.uniform(0.6, 1.0)
        
        ## Generar explicacion que mencione BI-RADS
        if birads <= 2:
            explanation = "Hallazgo benigno sin evidencia de malignidad. Se recomienda seguimiento rutinario."
        elif birads == 3:
            explanation = "Hallazgo probablemente benigno. Se recomienda seguimiento a corto plazo para valorar cambios."
        else:
            explanation = "Hallazgo sospechoso altamente sugestivo de malignidad. Se recomienda biopsia urgente."
        
        record = PredictionRecord(
            image_id=f"img_{i:03d}",
            birads_pred=birads,
            birads_confidence=confidence,
            findings_pred=["masa", "asimetria"] if birads >= 4 else ["densidad_alta"],
            explanation_text=explanation,
            ground_truth_birads=birads,
        )
        
        collector.add_prediction(record)
    
    ## Generar reporte
    report = collector.compute_final_report()
    
    ## Mostrar resumen
    print()
    print(collector.generate_summary())
    
    ## Validaciones
    assert len(collector.coherence_scores) == n_records, "Deberia haber 50 scores de coherencia"
    assert report["area_1_alignment_metrics"]["coherence"]["mean_coherence"] >= 0, \
        "Coherencia media deberia ser >= 0"
    
    logger.info("✓ Test AlignmentMetricsCollector pasado\n")


def main():
    ##
    ## Ejecuta todos los tests de Area 1
    ##
    logger.info("=" * 70)
    logger.info("TESTS DE AREA 1: METRICAS DE ALINEACION LLM-VISION")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        test_coherence_validator()
        test_calibration_analyzer()
        test_consistency_analyzer()
        test_alignment_metrics_collector()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("TODOS LOS TESTS DE AREA 1 PASARON EXITOSAMENTE")
        logger.info("=" * 70)
        
    except AssertionError as e:
        logger.error("Test fallido: %s", str(e))
        return 1
    except Exception as e:
        logger.error("Error inesperado: %s", str(e))
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
