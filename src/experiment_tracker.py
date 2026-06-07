## experiment_tracker.py
## Sistema de registro de experimentos del MammoVLM V2
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Registra cada experimento de entrenamiento de forma estructurada para
## construir un historial comparable a lo largo de la tesis. Enfoque hibrido:
##   - Automatico: configuracion, metricas, fecha, duracion
##   - Manual: hipotesis y descripcion de que se cambio (lo llena el investigador)
##
## Cada experimento se guarda en su propia carpeta sin sobreescribir. Un
## registro maestro (experiments_registry.json) acumula una fila por experimento
## para comparar la evolucion del modelo.

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ExperimentTracker:
    ## Gestiona el registro y comparacion de experimentos

    def __init__(self, experiments_dir):
        ##
        ## Parametros:
        ##   experiments_dir: directorio raiz donde se guardan los experimentos
        ##
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.experiments_dir / "experiments_registry.json"

    def _load_registry(self) -> List[Dict]:
        ## Carga el registro maestro de experimentos
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_registry(self, registry: List[Dict]):
        ## Guarda el registro maestro
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def register_experiment(
        self,
        experiment_id: str,
        hypothesis: str,
        config_dict: Dict,
        test_metrics: Dict,
        training_history: Dict,
        model_path: Optional[str] = None,
        notes: str = "",
        training_duration_s: Optional[float] = None,
        training_duration_str: str = "",
    ) -> Path:
        ##
        ## Registra un experimento completo
        ##
        ## Parametros:
        ##   experiment_id: identificador unico (ej. "exp01_baseline")
        ##   hypothesis: descripcion manual de la hipotesis o que se cambio
        ##   config_dict: configuracion de entrenamiento usada
        ##   test_metrics: reporte de metricas sobre el test set (Area 3)
        ##   training_history: historial de loss/acc por epoca
        ##   model_path: ruta al modelo entrenado (se copia al experimento)
        ##   notes: notas adicionales del investigador
        ##   training_duration_s: duracion del entrenamiento en segundos
        ##   training_duration_str: duracion formateada (ej. "5h 30m 12s")
        ##
        ## Retorna: ruta de la carpeta del experimento
        ##
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ## Extraer metricas clave del reporte de Area 3 para el resumen
        key_metrics = self._extract_key_metrics(test_metrics)

        ## Guardar el detalle completo del experimento
        experiment_data = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "hypothesis": hypothesis,
            "notes": notes,
            "training_duration_s": training_duration_s,
            "training_duration_str": training_duration_str,
            "config": config_dict,
            "key_metrics": key_metrics,
            "test_metrics_full": test_metrics,
            "training_history": training_history,
        }

        detail_path = exp_dir / "experiment_detail.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

        ## Copiar el modelo si se proporciono
        if model_path and Path(model_path).exists():
            dest_model = exp_dir / "model.pt"
            shutil.copy2(model_path, dest_model)

        ## Actualizar el registro maestro con una fila resumida
        registry = self._load_registry()

        ## Si ya existe un experimento con ese ID, reemplazarlo
        registry = [r for r in registry if r["experiment_id"] != experiment_id]

        registry_entry = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "hypothesis": hypothesis,
            "training_duration_str": training_duration_str,
            **key_metrics,
            "config_summary": self._summarize_config(config_dict),
        }
        registry.append(registry_entry)

        ## Ordenar por timestamp
        registry.sort(key=lambda r: r["timestamp"])
        self._save_registry(registry)

        return exp_dir

    def _extract_key_metrics(self, test_metrics: Dict) -> Dict:
        ##
        ## Extrae las metricas clave del reporte de Area 3 para el resumen
        ##
        try:
            m = test_metrics["area_3_medical_metrics"]
            cls = m["multiclass_classification"]
            sev = m["clinical_severity"]
            irr = m["inter_rater_reliability"]
            err = m["error_analysis"]

            return {
                "accuracy": round(cls["accuracy"], 4),
                "macro_f1": round(cls["macro_f1"], 4),
                "weighted_f1": round(cls["weighted_f1"], 4),
                "mcc": round(cls["mcc"], 4),
                "quadratic_kappa": round(cls["quadratic_kappa"], 4),
                "auc_macro": round(cls.get("auc_ovr", {}).get("macro_avg", 0.0), 4),
                "sensitivity": round(sev["sensitivity"], 4),
                "specificity": round(sev["specificity"], 4),
                "balanced_accuracy": round(sev["balanced_accuracy"], 4),
                "cohen_kappa": round(irr["cohen_kappa"], 4),
                "dangerous_false_negatives": err["dangerous_false_negatives"],
                "error_rate": round(err["error_rate"], 4),
            }
        except (KeyError, TypeError):
            return {}

    def _summarize_config(self, config_dict: Dict) -> str:
        ##
        ## Genera un resumen textual breve de la configuracion
        ##
        parts = []
        if "freeze_encoder" in config_dict:
            parts.append("encoder congelado" if config_dict["freeze_encoder"] else "encoder fine-tuned")
        if "stage1_epochs" in config_dict:
            parts.append(f"s1:{config_dict['stage1_epochs']}ep")
        if "stage2_epochs" in config_dict:
            parts.append(f"s2:{config_dict['stage2_epochs']}ep")
        if "stage1_lr" in config_dict:
            parts.append(f"lr:{config_dict['stage1_lr']}")
        return ", ".join(parts)

    def get_comparison_table(self) -> List[Dict]:
        ##
        ## Retorna el registro completo para mostrar como tabla comparativa
        ##
        return self._load_registry()

    def print_comparison(self):
        ##
        ## Imprime una tabla comparativa de todos los experimentos
        ##
        registry = self._load_registry()
        if not registry:
            print("No hay experimentos registrados todavia.")
            return

        print("=" * 100)
        print("HISTORIAL DE EXPERIMENTOS - MammoVLM V2")
        print("=" * 100)
        print()

        ## Encabezado de la tabla
        header = f"{'ID':<28} {'Acc':>7} {'MacF1':>7} {'QKap':>7} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'CohK':>7}"
        print(header)
        print("-" * 100)

        for exp in registry:
            row = (
                f"{exp['experiment_id']:<28} "
                f"{exp.get('accuracy', 0):>7.4f} "
                f"{exp.get('macro_f1', 0):>7.4f} "
                f"{exp.get('quadratic_kappa', 0):>7.4f} "
                f"{exp.get('auc_macro', 0):>7.4f} "
                f"{exp.get('sensitivity', 0):>7.4f} "
                f"{exp.get('specificity', 0):>7.4f} "
                f"{exp.get('cohen_kappa', 0):>7.4f}"
            )
            print(row)

        print()
        print("Leyenda: Acc=Accuracy, MacF1=Macro F1, QKap=Quadratic Kappa,")
        print("         AUC=AUC-ROC macro, Sens=Sensibilidad, Spec=Especificidad,")
        print("         CohK=Cohen Kappa")
        print()

        ## Mostrar hipotesis de cada experimento
        print("HIPOTESIS DE CADA EXPERIMENTO:")
        print("-" * 100)
        for exp in registry:
            print(f"  {exp['experiment_id']}:")
            print(f"    {exp.get('hypothesis', 'sin descripcion')}")
            print(f"    Config: {exp.get('config_summary', '')}")
            duracion = exp.get('training_duration_str', '')
            if duracion:
                print(f"    Duracion de entrenamiento: {duracion}")
            print()

    def get_best_experiment(self, metric: str = "quadratic_kappa") -> Optional[Dict]:
        ##
        ## Retorna el experimento con mejor valor en la metrica indicada
        ##
        registry = self._load_registry()
        if not registry:
            return None
        valid = [r for r in registry if metric in r]
        if not valid:
            return None
        return max(valid, key=lambda r: r[metric])
