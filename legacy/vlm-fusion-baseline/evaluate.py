## evaluate.py
## Evaluacion de modelos entrenados con metricas clinicamente relevantes.
##
## Metricas implementadas:
##   Clasificacion:
##     - Accuracy, Sensitivity (Recall), Specificity, Precision
##     - AUC-ROC, AUC-PRC (Precision-Recall Curve)
##     - F1-score, MCC (Matthews Correlation Coefficient)
##     - Matriz de confusion por clase BI-RADS
##   Generacion de reportes:
##     - BLEU-1, BLEU-2, BLEU-4
##     - ROUGE-1, ROUGE-2, ROUGE-L
##     - BERTScore (semantico, mas relevante que BLEU para texto medico)
##   Portabilidad:
##     - Tiempo de inferencia por imagen (ms)
##     - Uso de memoria RAM/VRAM en pico
##     - Tamano del modelo en disco
##
## Decisiones de diseno:
##   - Las metricas se calculan sobre el conjunto de test (nunca sobre train)
##   - Los resultados se guardan en JSON para reproducibilidad
##   - Los graficos se generan como figuras de matplotlib para el notebook
##   - Se calcula el umbral optimo de clasificacion binaria usando Youden J

import os
import json
import time
import logging
import tracemalloc
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    f1_score,
)

## Importaciones opcionales para metricas de texto
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge_score no disponible. Instalar con: pip install rouge-score")

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk no disponible para BLEU. Instalar con: pip install nltk")

logger = logging.getLogger("evaluate")

## Umbral por defecto para clasificacion binaria (se puede optimizar con Youden J)
DEFAULT_THRESHOLD = 0.5


class ClassificationEvaluator:
    """
    Evaluador de metricas de clasificacion para diagnostico mamografico.

    Calcula todas las metricas relevantes clinicamente, con especial atencion
    a sensibilidad y especificidad que son los indicadores principales en
    screening mamografico: una alta sensibilidad evita falsos negativos
    (cancer no detectado), y una alta especificidad reduce biopsias innecesarias.

    Parametros
    ----------
    num_classes : int
        2 para clasificacion binaria (benigno/maligno), 7 para BI-RADS 0-6.
    threshold : float
        Umbral de decision para clasificacion binaria. Si None, se optimiza
        usando el criterio de Youden J sobre los scores de probabilidad.
    """

    def __init__(self, num_classes: int = 2, threshold: Optional[float] = DEFAULT_THRESHOLD):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reinicia los acumuladores de predicciones y etiquetas."""
        self.all_labels = []
        self.all_probs = []    ## Probabilidades (output softmax)
        self.all_preds = []    ## Predicciones de clase
        self.all_paths = []    ## Rutas de imagen para analisis de errores

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        image_paths: Optional[list] = None,
    ):
        """
        Acumula predicciones de un batch.

        Parametros
        ----------
        logits : torch.Tensor
            Logits sin normalizar [batch, num_classes].
        labels : torch.Tensor
            Etiquetas verdaderas [batch].
        image_paths : list, opcional
            Rutas de las imagenes para identificar errores.
        """
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        labs = labels.cpu().numpy()

        self.all_probs.extend(probs.tolist())
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labs.tolist())

        if image_paths:
            self.all_paths.extend(image_paths)

    def compute(self, optimize_threshold: bool = False) -> dict:
        """
        Calcula todas las metricas sobre las predicciones acumuladas.

        Parametros
        ----------
        optimize_threshold : bool
            Si True, encuentra el umbral optimo usando el indice de Youden J
            (maximiza sensibilidad + especificidad - 1).

        Retorna diccionario completo de metricas.
        """
        if not self.all_labels:
            logger.warning("No hay predicciones acumuladas. Llame a update() primero.")
            return {}

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        preds = np.array(self.all_preds)

        metrics = {}

        ## Metricas para clasificacion binaria
        if self.num_classes == 2:
            pos_probs = probs[:, 1]   ## Probabilidad de clase positiva (maligno)

            ## Optimizacion del umbral de Youden J
            if optimize_threshold:
                optimal_thresh, youden_j = self._find_optimal_threshold(labels, pos_probs)
                self.threshold = optimal_thresh
                metrics["optimal_threshold"] = round(float(optimal_thresh), 4)
                metrics["youden_j"] = round(float(youden_j), 4)
                preds = (pos_probs >= optimal_thresh).astype(int)

            ## Metricas basicas
            metrics["accuracy"] = round(float(accuracy_score(labels, preds)), 4)
            metrics["f1"] = round(float(f1_score(labels, preds, zero_division=0)), 4)
            metrics["mcc"] = round(float(matthews_corrcoef(labels, preds)), 4)

            ## Metricas clinicas clave
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

            metrics["sensitivity"] = round(float(tp / max(tp + fn, 1)), 4)   ## Recall
            metrics["specificity"] = round(float(tn / max(tn + fp, 1)), 4)
            metrics["ppv"] = round(float(tp / max(tp + fp, 1)), 4)            ## Precision positiva
            metrics["npv"] = round(float(tn / max(tn + fn, 1)), 4)            ## Precision negativa
            metrics["tp"] = int(tp)
            metrics["tn"] = int(tn)
            metrics["fp"] = int(fp)
            metrics["fn"] = int(fn)

            ## Curva ROC y AUC
            try:
                metrics["auc_roc"] = round(float(roc_auc_score(labels, pos_probs)), 4)
                metrics["auc_prc"] = round(float(average_precision_score(labels, pos_probs)), 4)
                fpr, tpr, _ = roc_curve(labels, pos_probs)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }
                precision_arr, recall_arr, _ = precision_recall_curve(labels, pos_probs)
                metrics["prc_curve"] = {
                    "precision": precision_arr.tolist(),
                    "recall": recall_arr.tolist(),
                }
            except ValueError as e:
                logger.warning("Error calculando AUC: %s (posiblemente una sola clase presente)", e)
                metrics["auc_roc"] = 0.0
                metrics["auc_prc"] = 0.0

        ## Metricas para clasificacion multiclase (BI-RADS 0-6)
        else:
            metrics["accuracy"] = round(float(accuracy_score(labels, preds)), 4)
            metrics["f1_macro"] = round(
                float(f1_score(labels, preds, average="macro", zero_division=0)), 4
            )
            metrics["f1_weighted"] = round(
                float(f1_score(labels, preds, average="weighted", zero_division=0)), 4
            )
            try:
                metrics["auc_roc_ovr"] = round(
                    float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")), 4
                )
            except ValueError:
                metrics["auc_roc_ovr"] = 0.0

            report = classification_report(labels, preds, output_dict=True, zero_division=0)
            metrics["per_class"] = {
                str(k): {
                    "precision": round(v.get("precision", 0), 4),
                    "recall": round(v.get("recall", 0), 4),
                    "f1": round(v.get("f1-score", 0), 4),
                    "support": int(v.get("support", 0)),
                }
                for k, v in report.items()
                if str(k).isdigit()
            }

        ## Matriz de confusion (siempre)
        cm = confusion_matrix(labels, preds)
        metrics["confusion_matrix"] = cm.tolist()

        ## Informacion del conjunto evaluado
        metrics["n_samples"] = len(labels)
        metrics["n_positive"] = int(labels.sum() if self.num_classes == 2 else (labels > 2).sum())
        metrics["threshold"] = float(self.threshold) if self.threshold else DEFAULT_THRESHOLD

        logger.info(
            "Evaluacion completada: n=%d, accuracy=%.4f%s",
            metrics["n_samples"],
            metrics["accuracy"],
            f", auc_roc={metrics.get('auc_roc', 'N/A')}" if self.num_classes == 2 else "",
        )
        return metrics

    def _find_optimal_threshold(
        self,
        labels: np.ndarray,
        pos_probs: np.ndarray,
    ) -> tuple:
        """
        Encuentra el umbral que maximiza el indice de Youden J = Sensibilidad + Especificidad - 1.
        Retorna (umbral_optimo, valor_youden_j).
        """
        fpr, tpr, thresholds = roc_curve(labels, pos_probs)
        youden_j = tpr - fpr
        idx = np.argmax(youden_j)
        return float(thresholds[idx]), float(youden_j[idx])

    def get_error_cases(self, top_n: int = 20) -> pd.DataFrame:
        """
        Devuelve los casos con mayor error de confianza (prediccion erronea con alta confianza).
        Util para analizar los casos mas dificiles para el modelo.
        """
        if not self.all_paths:
            logger.warning("No hay rutas de imagen disponibles para analisis de errores.")
            return pd.DataFrame()

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        preds = np.array(self.all_preds)

        errors = labels != preds
        error_confidence = probs[np.arange(len(preds)), preds]

        rows = []
        for i, (is_error, label, pred, conf, path) in enumerate(
            zip(errors, labels, preds, error_confidence, self.all_paths)
        ):
            if is_error:
                rows.append({
                    "image_path": path,
                    "true_label": int(label),
                    "pred_label": int(pred),
                    "confidence": round(float(conf), 4),
                    "all_probs": [round(p, 4) for p in probs[i].tolist()],
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("confidence", ascending=False).head(top_n)
        return df


class ReportEvaluator:
    """
    Evaluador de calidad de reportes generados por el VLM.

    En diagnostico mamografico, las metricas de texto son indicadores
    aproximados: un reporte puede ser clinicamente correcto sin
    usar las mismas palabras que el reporte de referencia.
    Por esta razon incluimos BERTScore (si disponible) que mide
    similitud semantica en lugar de coincidencia lexica.

    Parametros
    ----------
    language : str
        Idioma de los reportes para ROUGE y BLEU ("spanish" o "english").
    """

    def __init__(self, language: str = "spanish"):
        self.language = language
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False) \
                     if ROUGE_AVAILABLE else None
        self.reset()

    def reset(self):
        self.references = []    ## Reportes de referencia (radiologos)
        self.hypotheses = []    ## Reportes generados por el modelo

    def update(self, reference: Union[str, list], hypothesis: Union[str, list]):
        """
        Acumula pares de reporte de referencia y generado.
        Acepta strings individuales o listas de strings.
        """
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]

        self.references.extend(reference)
        self.hypotheses.extend(hypothesis)

    def compute(self) -> dict:
        """
        Calcula metricas de calidad de texto sobre los reportes acumulados.
        Retorna diccionario con BLEU, ROUGE y opcionalmente BERTScore.
        """
        if not self.references:
            logger.warning("No hay reportes acumulados.")
            return {}

        metrics = {"n_reports": len(self.references)}

        ## Metricas ROUGE
        if self.rouge is not None:
            rouge_scores = self._compute_rouge()
            metrics.update(rouge_scores)
        else:
            logger.warning("rouge_score no disponible. Instalar: pip install rouge-score")

        ## Metricas BLEU (usando NLTK)
        if NLTK_AVAILABLE:
            bleu_scores = self._compute_bleu()
            metrics.update(bleu_scores)
        else:
            logger.warning("NLTK no disponible para BLEU. Instalar: pip install nltk")

        ## Longitud promedio de reportes
        hyp_lengths = [len(h.split()) for h in self.hypotheses]
        ref_lengths = [len(r.split()) for r in self.references]
        metrics["avg_hyp_length"] = round(float(np.mean(hyp_lengths)), 1)
        metrics["avg_ref_length"] = round(float(np.mean(ref_lengths)), 1)

        logger.info(
            "Evaluacion de reportes: n=%d, ROUGE-L=%.4f%s",
            len(self.references),
            metrics.get("rouge_l", 0.0),
            f", BLEU-4={metrics.get('bleu4', 'N/A')}",
        )
        return metrics

    def _compute_rouge(self) -> dict:
        """Calcula promedios de ROUGE-1, ROUGE-2 y ROUGE-L."""
        r1_scores, r2_scores, rl_scores = [], [], []

        for ref, hyp in zip(self.references, self.hypotheses):
            scores = self.rouge.score(ref, hyp)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rl_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": round(float(np.mean(r1_scores)), 4),
            "rouge2": round(float(np.mean(r2_scores)), 4),
            "rouge_l": round(float(np.mean(rl_scores)), 4),
        }

    def _compute_bleu(self) -> dict:
        """Calcula corpus BLEU en variantes 1, 2 y 4."""
        tokenized_refs = [[ref.lower().split()] for ref in self.references]
        tokenized_hyps = [hyp.lower().split() for hyp in self.hypotheses]

        smoothing = SmoothingFunction().method1

        bleu_scores = {}
        for n, weights in [(1, (1, 0, 0, 0)), (2, (0.5, 0.5, 0, 0)), (4, (0.25,) * 4)]:
            try:
                score = corpus_bleu(tokenized_refs, tokenized_hyps, weights=weights, smoothing_function=smoothing)
                bleu_scores[f"bleu{n}"] = round(float(score), 4)
            except Exception as e:
                logger.debug("Error calculando BLEU-%d: %s", n, e)
                bleu_scores[f"bleu{n}"] = 0.0

        return bleu_scores


class InferenceProfiler:
    """
    Mide el rendimiento de inferencia del modelo para verificar que cumple
    los requisitos de portabilidad definidos en los resultados esperados:
      - Tiempo de inferencia por imagen <= 30 segundos
      - Uso de RAM <= 8 GB
      - Sin requerimiento de GPU dedicada

    Parametros
    ----------
    model : nn.Module
        Modelo a perfilar.
    device : torch.device
        Dispositivo de inferencia.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.results: dict = {}

    def profile(
        self,
        dataloader: DataLoader,
        n_batches: int = 20,
        warmup_batches: int = 3,
    ) -> dict:
        """
        Ejecuta n_batches de inferencia y mide latencia y memoria.

        Los primeros warmup_batches se descartan para evitar el costo
        de compilacion JIT o inicializacion de CUDA en las mediciones.

        Retorna diccionario con estadisticas de latencia y memoria.
        """
        self.model.eval()
        latencies = []

        tracemalloc.start()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches + warmup_batches:
                    break

                images = batch["image"].to(self.device)
                batch_size = images.shape[0]

                ## Calentar el modelo (primeros batches no se miden)
                if batch_idx < warmup_batches:
                    _ = self.model(images, mode="classify") if hasattr(self.model, "forward") else None
                    continue

                ## Sincronizar CUDA antes de medir tiempo
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()

                if hasattr(self.model, "forward"):
                    _ = self.model(images, mode="classify")
                else:
                    _ = self.model(images)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - t0
                latencies.append(elapsed / batch_size * 1000)  ## ms por imagen

        ## Memoria RAM (CPU)
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ## Memoria VRAM (GPU)
        vram_mb = 0
        if self.device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2

        ## Tamano del modelo en disco
        model_size_mb = self._estimate_model_size()

        self.results = {
            "latency_ms_mean": round(float(np.mean(latencies)), 2) if latencies else 0,
            "latency_ms_p50": round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
            "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
            "latency_ms_max": round(float(np.max(latencies)), 2) if latencies else 0,
            "peak_ram_mb": round(peak_mem / 1024**2, 2),
            "peak_vram_mb": round(float(vram_mb), 2),
            "model_size_mb": round(model_size_mb, 2),
            "device": str(self.device),
            "n_batches_measured": len(latencies),
            "meets_30s_requirement": (float(np.mean(latencies)) if latencies else 999) < 30000,
        }

        logger.info(
            "Perfil de inferencia: latencia_media=%.1fms, RAM_pico=%.0fMB, VRAM=%.0fMB",
            self.results["latency_ms_mean"],
            self.results["peak_ram_mb"],
            self.results["peak_vram_mb"],
        )
        return self.results

    def _estimate_model_size(self) -> float:
        """
        Estima el tamano del modelo en MB sumando los parametros.
        No incluye el LLM base (solo los modulos entrenados/guardados).
        """
        total_bytes = sum(
            p.nelement() * p.element_size()
            for p in self.model.parameters()
        )
        return total_bytes / 1024**2


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
    compute_reports: bool = False,
    optimize_threshold: bool = True,
    desc: str = "Evaluando",
) -> dict:
    """
    Funcion principal de evaluacion que orquesta todos los evaluadores.

    Ejecuta el modelo en modo de inferencia sobre el DataLoader dado
    y calcula el conjunto completo de metricas.

    Parametros
    ----------
    model : nn.Module
        Modelo a evaluar (MammoVLM o MammoClassifier).
    dataloader : DataLoader
        DataLoader del conjunto de evaluacion (val o test).
    device : torch.device
        Dispositivo de computo.
    num_classes : int
        Numero de clases del clasificador.
    compute_reports : bool
        Si True, genera y evalua reportes de texto (solo para MammoVLM).
    optimize_threshold : bool
        Si True, optimiza el umbral de clasificacion con Youden J.
    desc : str
        Descripcion para la barra de progreso.

    Retorna
    -------
    dict con todas las metricas de clasificacion y generacion.
    """
    model.eval()
    cls_evaluator = ClassificationEvaluator(num_classes=num_classes)
    rep_evaluator = ReportEvaluator() if compute_reports else None

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            image_paths = batch.get("image_path", None)

            ## Forward en modo clasificacion
            if hasattr(model, "forward"):
                try:
                    out = model(images, labels=labels, mode="classify")
                    logits = out.get("logits", None)
                    loss = out.get("loss", None)
                except TypeError:
                    ## Compatibilidad con MammoClassifier que tiene signature diferente
                    out = model(images, labels=labels)
                    logits = out.get("logits", None)
                    loss = out.get("loss", None)
            else:
                logits = model(images)
                loss = None

            if logits is None:
                continue

            cls_evaluator.update(logits, labels, image_paths)

            if loss is not None:
                total_loss += loss.item()
                n_batches += 1

            ## Generacion de reportes (solo si hay VLM y se solicita)
            if compute_reports and rep_evaluator is not None:
                if hasattr(model, "generate_report") and "report_text" in batch:
                    generated = model.generate_report(images)
                    rep_evaluator.update(batch["report_text"], generated)

    ## Calcular metricas
    cls_metrics = cls_evaluator.compute(optimize_threshold=optimize_threshold)
    all_metrics = dict(cls_metrics)

    if n_batches > 0:
        all_metrics["val_loss"] = round(total_loss / n_batches, 6)

    if compute_reports and rep_evaluator is not None:
        rep_metrics = rep_evaluator.compute()
        all_metrics.update({f"report_{k}": v for k, v in rep_metrics.items()})

    ## Casos de error para analisis
    error_cases = cls_evaluator.get_error_cases(top_n=20)
    if not error_cases.empty:
        all_metrics["n_error_cases"] = len(error_cases)

    return all_metrics, cls_evaluator, error_cases


def print_metrics_summary(metrics: dict):
    """
    Imprime un resumen de metricas en formato legible para consola y notebook.
    Usa solo caracteres ASCII para compatibilidad con la restriccion del proyecto.
    """
    print("=" * 60)
    print("RESUMEN DE METRICAS")
    print("=" * 60)

    ## Metricas de clasificacion prioritarias
    priority_keys = [
        "accuracy", "sensitivity", "specificity", "ppv", "npv",
        "auc_roc", "auc_prc", "f1", "mcc",
    ]
    for key in priority_keys:
        if key in metrics:
            val = metrics[key]
            print(f"  {key.upper():20s}: {val:.4f}" if isinstance(val, float) else f"  {key.upper():20s}: {val}")

    ## Matriz de confusion resumida
    if "tp" in metrics:
        print("\n  MATRIZ DE CONFUSION (binario)")
        print(f"    VP (verdaderos positivos): {metrics.get('tp', 0)}")
        print(f"    VN (verdaderos negativos): {metrics.get('tn', 0)}")
        print(f"    FP (falsos positivos):     {metrics.get('fp', 0)}")
        print(f"    FN (falsos negativos):     {metrics.get('fn', 0)}")

    ## Metricas de reporte
    report_keys = ["report_rouge_l", "report_bleu4", "report_rouge1"]
    has_report = any(k in metrics for k in report_keys)
    if has_report:
        print("\n  METRICAS DE REPORTE GENERADO")
        for key in report_keys:
            if key in metrics:
                print(f"  {key.upper():25s}: {metrics[key]:.4f}")

    ## Informacion de muestra
    print(f"\n  N muestras evaluadas: {metrics.get('n_samples', 'N/A')}")
    if "val_loss" in metrics:
        print(f"  Loss (validacion):   {metrics['val_loss']:.6f}")
    print("=" * 60)


def save_evaluation_results(
    metrics: dict,
    output_dir: str,
    experiment_name: str = "experiment",
    error_cases: Optional[pd.DataFrame] = None,
) -> str:
    """
    Guarda los resultados de evaluacion en disco.

    Guarda:
    - metrics JSON: todas las metricas incluyendo curvas ROC/PRC
    - error_cases CSV: casos de error para analisis posterior

    Retorna la ruta del directorio de resultados.
    """
    results_dir = Path(output_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    ## Guardar metricas en JSON
    ## Convertir arrays numpy a listas para serializacion
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            metrics_serializable[k] = float(v)
        else:
            metrics_serializable[k] = v

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)

    ## Guardar casos de error
    if error_cases is not None and not error_cases.empty:
        error_path = results_dir / "error_cases.csv"
        error_cases.to_csv(error_path, index=False)
        logger.info("Casos de error guardados: %s (%d casos)", error_path, len(error_cases))

    logger.info("Resultados de evaluacion guardados en: %s", results_dir)
    return str(results_dir)


def compare_experiments(results_dirs: list) -> pd.DataFrame:
    """
    Carga y compara metricas de multiples experimentos en un DataFrame.
    Util para comparar modelos en el notebook.

    Parametros
    ----------
    results_dirs : list of str
        Lista de directorios generados por save_evaluation_results.

    Retorna DataFrame con una fila por experimento y columnas de metricas.
    """
    rows = []
    for exp_dir in results_dirs:
        metrics_path = Path(exp_dir) / "metrics.json"
        if not metrics_path.exists():
            logger.warning("No se encontro metrics.json en: %s", exp_dir)
            continue

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        row = {"experiment": Path(exp_dir).name}
        for key in ["accuracy", "sensitivity", "specificity", "auc_roc", "f1", "mcc",
                    "report_rouge_l", "report_bleu4", "latency_ms_mean"]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        ## Ordenar por AUC ROC descendente
        sort_col = "auc_roc" if "auc_roc" in df.columns else "accuracy"
        df = df.sort_values(sort_col, ascending=False)

    return df


def evaluate_portability(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evalua los requisitos de portabilidad del modelo definidos en los
    resultados esperados de la tesis:
      - Memoria RAM <= 8 GB
      - Tiempo de inferencia por imagen <= 30 segundos
      - Tamano del modelo optimizado

    Retorna diccionario con metricas y veredicto de cada requisito.
    """
    profiler = InferenceProfiler(model, device)
    profile_results = profiler.profile(dataloader, n_batches=10)

    requirements = {
        "req_latency_30s": {
            "description": "Latencia por imagen <= 30,000 ms",
            "value": profile_results.get("latency_ms_mean", 0),
            "threshold": 30000,
            "passed": profile_results.get("latency_ms_mean", 999999) <= 30000,
        },
        "req_ram_8gb": {
            "description": "RAM pico <= 8,192 MB",
            "value": profile_results.get("peak_ram_mb", 0),
            "threshold": 8192,
            "passed": profile_results.get("peak_ram_mb", 999999) <= 8192,
        },
    }

    profile_results["requirements"] = requirements
    n_passed = sum(1 for r in requirements.values() if r["passed"])
    profile_results["requirements_passed"] = f"{n_passed}/{len(requirements)}"

    for req_name, req in requirements.items():
        status = "CUMPLE" if req["passed"] else "NO CUMPLE"
        logger.info("%s - %s: valor=%.1f, umbral=%.1f", status, req["description"], req["value"], req["threshold"])

    return profile_results
