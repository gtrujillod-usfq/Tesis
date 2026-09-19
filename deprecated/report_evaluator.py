## report_evaluator.py
## Area 4: Evaluacion de Hallazgos
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Compara los hallazgos predichos por el modelo contra los hallazgos
## reales descritos por el radiologo en los reportes de CDD-CESM.
##
## Flujo:
##   1. ReportFindingsParser: extrae hallazgos del texto libre del reporte
##      (campo "Findings" de CDD-CESM) usando el lexico BI-RADS
##   2. FindingsComparator: compara hallazgos predichos vs reales por slot
##   3. FindingsEvaluationReport: consolida metricas (F1, precision, recall)
##
## El parser mapea texto libre a slots estructurados usando el vocabulario
## controlado, garantizando coherencia con el esquema de Area 2.

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ReportFindingsParser:
    ## Extrae hallazgos estructurados del texto libre de un reporte
    ## radiologico usando el lexico BI-RADS y el esquema de slots
    ##
    ## Convierte texto como "spiculated mass with pleomorphic calcifications"
    ## en slots: {mass_margin: spiculated, calc_morphology: fine pleomorphic}

    def __init__(self, lexicon=None, schema=None):
        ##
        ## Parametros:
        ##   lexicon: BIRADSLexicon con descriptores y sinonimos
        ##   schema: FindingsSchema con los slots a llenar
        ##
        if lexicon is None:
            from medical_vocabulary import birads_lexicon
            lexicon = birads_lexicon
        if schema is None:
            from findings_schema import findings_schema
            schema = findings_schema

        self.lexicon = lexicon
        self.schema = schema

        ## Construir mapa de descriptor -> slot para busqueda rapida
        self._build_descriptor_map()

    def _build_descriptor_map(self):
        ##
        ## Construye un mapa que asocia cada descriptor del lexico
        ## (canonical + sinonimos) al slot del esquema correspondiente
        ##
        ## La asociacion se hace por categoria BI-RADS:
        ## un descriptor de categoria "mass_margin" llena el slot "mass_margin"
        ##
        self.category_to_slot = {}
        for slot in self.schema.slots:
            self.category_to_slot[slot.birads_category] = slot.name

        ## Mapa: texto a buscar -> (slot_name, canonical_value)
        self.search_terms = []
        for descriptor in self.lexicon.get_all_descriptors():
            slot_name = self.category_to_slot.get(descriptor.category)
            if slot_name is None:
                continue

            ## Verificar que el valor canonical existe en el slot
            slot = self.schema.get_slot(slot_name)
            if slot is None:
                continue

            ## El canonical del lexico debe mapear a un valor del slot
            canonical_value = self._map_to_slot_value(descriptor.canonical, slot)
            if canonical_value is None:
                continue

            ## Agregar canonical y todos los sinonimos como terminos de busqueda
            all_terms = [descriptor.canonical] + descriptor.synonyms
            for term in all_terms:
                self.search_terms.append((term.lower(), slot_name, canonical_value))

        ## Ordenar por longitud descendente para priorizar matches mas especificos
        ## (ej: "fine pleomorphic" antes que "fine")
        self.search_terms.sort(key=lambda x: len(x[0]), reverse=True)

    def _map_to_slot_value(self, descriptor_canonical: str, slot) -> Optional[str]:
        ##
        ## Mapea un descriptor del lexico a un valor valido del slot
        ## Retorna el valor del slot que mejor coincide, o None
        ##
        desc_lower = descriptor_canonical.lower()
        for value in slot.values:
            if value.lower() == desc_lower:
                return value
        ## Match parcial: el descriptor contiene o esta contenido en el valor
        for value in slot.values:
            if value == "ausente":
                continue
            if desc_lower in value.lower() or value.lower() in desc_lower:
                return value
        return None

    def parse(self, report_text: str) -> Dict[str, str]:
        ##
        ## Extrae hallazgos del texto libre del reporte
        ##
        ## Parametros:
        ##   report_text: texto del reporte (campo Findings de CDD-CESM)
        ##
        ## Retorna: dict slot_name -> valor extraido (solo slots detectados)
        ##
        if not report_text or not isinstance(report_text, str):
            return {}

        text_lower = report_text.lower()
        extracted = {}

        ## Buscar cada termino del lexico en el texto
        ## Solo se llena cada slot una vez (con el primer match, que es el mas largo)
        for term, slot_name, canonical_value in self.search_terms:
            if slot_name in extracted:
                continue  ## slot ya lleno con match mas especifico

            ## Buscar el termino como palabra/frase completa
            if self._term_in_text(term, text_lower):
                extracted[slot_name] = canonical_value

        return extracted

    def _term_in_text(self, term: str, text: str) -> bool:
        ##
        ## Verifica si un termino aparece en el texto como palabra completa
        ## Usa word boundaries para evitar matches parciales erroneos
        ##
        pattern = r"\b" + re.escape(term) + r"\b"
        return bool(re.search(pattern, text))

    def parse_birads(self, birads_raw) -> int:
        ##
        ## Extrae y normaliza el nivel BI-RADS del campo ACR_BIRADS
        ##
        if birads_raw is None:
            return 1
        try:
            cleaned = str(birads_raw).replace("BIRADS", "").replace("BI-RADS", "").strip()
            value = int(float(cleaned))
            return max(0, min(5, value))
        except (ValueError, TypeError):
            return 1

    def parse_density(self, density_raw) -> str:
        ##
        ## Normaliza la densidad mamaria a valor del slot breast_density
        ##
        if density_raw is None:
            return "scattered fibroglandular"

        raw = str(density_raw).strip().upper()
        density_map = {
            "A": "almost entirely fatty",
            "B": "scattered fibroglandular",
            "C": "heterogeneously dense",
            "D": "extremely dense",
            "1": "almost entirely fatty",
            "2": "scattered fibroglandular",
            "3": "heterogeneously dense",
            "4": "extremely dense",
        }
        return density_map.get(raw, "scattered fibroglandular")


class FindingsComparator:
    ## Compara hallazgos predichos por el modelo contra hallazgos reales
    ## extraidos de los reportes, slot por slot

    def __init__(self, schema=None):
        if schema is None:
            from findings_schema import findings_schema
            schema = findings_schema
        self.schema = schema

    def compare_single(
        self,
        predicted: Dict[str, str],
        ground_truth: Dict[str, str],
    ) -> Dict[str, Dict]:
        ##
        ## Compara un par (prediccion, verdad) slot por slot
        ##
        ## Retorna por cada slot:
        ##   - match: True si coinciden exactamente
        ##   - pred_present: True si el modelo predijo un hallazgo (no ausente)
        ##   - gt_present: True si el reporte tiene el hallazgo
        ##
        results = {}

        for slot in self.schema.slots:
            slot_name = slot.name

            pred_val = predicted.get(slot_name, "ausente")
            gt_val = ground_truth.get(slot_name, "ausente")

            ## Para breast_density, ausente no aplica (siempre presente)
            if slot_name == "breast_density":
                pred_present = pred_val != "ausente"
                gt_present = gt_val != "ausente"
            else:
                pred_present = pred_val != "ausente"
                gt_present = gt_val != "ausente"

            results[slot_name] = {
                "predicted": pred_val,
                "ground_truth": gt_val,
                "match": pred_val == gt_val,
                "pred_present": pred_present,
                "gt_present": gt_present,
            }

        return results


class FindingsEvaluationReport:
    ## Consolida la evaluacion de hallazgos sobre un conjunto de casos
    ## Calcula F1, precision y recall por slot y globales

    def __init__(self, schema=None):
        if schema is None:
            from findings_schema import findings_schema
            schema = findings_schema
        self.schema = schema
        self.comparator = FindingsComparator(schema)

    def evaluate_batch(
        self,
        predictions: List[Dict[str, str]],
        ground_truths: List[Dict[str, str]],
    ) -> Dict:
        ##
        ## Evalua un lote de predicciones contra sus reportes reales
        ##
        ## Parametros:
        ##   predictions: lista de dicts slot->valor predichos por el modelo
        ##   ground_truths: lista de dicts slot->valor extraidos de reportes
        ##
        ## Retorna: reporte completo de Area 4
        ##
        assert len(predictions) == len(ground_truths)
        n = len(predictions)

        ## Acumuladores por slot
        slot_stats = {slot.name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "exact_match": 0}
                      for slot in self.schema.slots}

        ## Comparar cada caso
        for pred, gt in zip(predictions, ground_truths):
            comparison = self.comparator.compare_single(pred, gt)

            for slot_name, result in comparison.items():
                stats = slot_stats[slot_name]

                ## Para deteccion de presencia (presente vs ausente)
                if result["pred_present"] and result["gt_present"]:
                    stats["tp"] += 1
                elif result["pred_present"] and not result["gt_present"]:
                    stats["fp"] += 1
                elif not result["pred_present"] and result["gt_present"]:
                    stats["fn"] += 1
                else:
                    stats["tn"] += 1

                ## Exact match del valor (no solo presencia)
                if result["match"]:
                    stats["exact_match"] += 1

        ## Calcular metricas por slot
        per_slot_metrics = {}
        for slot_name, stats in slot_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            exact_acc = stats["exact_match"] / n if n > 0 else 0.0

            per_slot_metrics[slot_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "exact_match_accuracy": float(exact_acc),
                "tp": tp, "fp": fp, "fn": fn, "tn": stats["tn"],
            }

        ## Metricas globales (macro y micro)
        all_f1 = [m["f1"] for m in per_slot_metrics.values()]
        all_precision = [m["precision"] for m in per_slot_metrics.values()]
        all_recall = [m["recall"] for m in per_slot_metrics.values()]
        all_exact = [m["exact_match_accuracy"] for m in per_slot_metrics.values()]

        ## Micro: agregando todos los tp/fp/fn
        total_tp = sum(s["tp"] for s in slot_stats.values())
        total_fp = sum(s["fp"] for s in slot_stats.values())
        total_fn = sum(s["fn"] for s in slot_stats.values())
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if (micro_precision + micro_recall) > 0 else 0.0)

        report = {
            "area_4_findings_evaluation": {
                "num_cases": n,
                "per_slot": per_slot_metrics,
                "macro_f1": float(np.mean(all_f1)),
                "macro_precision": float(np.mean(all_precision)),
                "macro_recall": float(np.mean(all_recall)),
                "mean_exact_match": float(np.mean(all_exact)),
                "micro_f1": float(micro_f1),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
            }
        }

        return report

    def generate_summary(self, report: Dict) -> str:
        ##
        ## Genera resumen textual del reporte de Area 4
        ##
        m = report["area_4_findings_evaluation"]

        lines = []
        lines.append("=" * 70)
        lines.append("AREA 4: EVALUACION DE HALLAZGOS")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Casos evaluados: {m['num_cases']}")
        lines.append("")

        lines.append("METRICAS GLOBALES")
        lines.append("-" * 70)
        lines.append(f"  Macro F1:               {m['macro_f1']:.4f}")
        lines.append(f"  Macro Precision:        {m['macro_precision']:.4f}")
        lines.append(f"  Macro Recall:           {m['macro_recall']:.4f}")
        lines.append(f"  Micro F1:               {m['micro_f1']:.4f}")
        lines.append(f"  Exact Match (promedio): {m['mean_exact_match']:.4f}")
        lines.append("")

        lines.append("METRICAS POR SLOT DE HALLAZGO")
        lines.append("-" * 70)
        lines.append(f"  {'Slot':28s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Exact':>6s}")
        lines.append("  " + "-" * 56)
        for slot_name, sm in m["per_slot"].items():
            lines.append(
                f"  {slot_name:28s} {sm['f1']:6.3f} {sm['precision']:6.3f} "
                f"{sm['recall']:6.3f} {sm['exact_match_accuracy']:6.3f}"
            )
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def parse_cddcesm_row(
    row: Dict,
    parser: ReportFindingsParser,
    findings_col: str = "Findings",
    birads_col: str = "ACR_BIRADS",
    density_col: str = "Breast_Density",
) -> Dict:
    ##
    ## Procesa una fila del CSV/XLSX de CDD-CESM y extrae los hallazgos
    ## estructurados (ground truth) para comparar con el modelo
    ##
    ## Parametros:
    ##   row: fila del dataframe (dict o pandas Series)
    ##   parser: ReportFindingsParser configurado
    ##   findings_col, birads_col, density_col: nombres de columnas
    ##
    ## Retorna: dict con birads, density y hallazgos estructurados
    ##
    ## Extraer texto de hallazgos
    findings_text = str(row.get(findings_col, "") or "")

    ## Parsear hallazgos del texto libre
    findings = parser.parse(findings_text)

    ## Agregar densidad (siempre presente)
    density = parser.parse_density(row.get(density_col))
    findings["breast_density"] = density

    ## BI-RADS
    birads = parser.parse_birads(row.get(birads_col))

    return {
        "birads": birads,
        "findings": findings,
        "findings_text": findings_text,
    }
