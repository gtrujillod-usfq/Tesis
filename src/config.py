## config.py
## Configuracion centralizada para MammoVLM V2
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Esta configuracion define todos los hiperparametros, rutas y opciones
## de entrenamiento/evaluacion del VLM final.

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class DatasetConfig:
    ## Rutas de datos
    root_dir: str = "/data"
    vindr_path: str = "vindr-mammo"
    rsna_path: str = "rsna-breast-cancer"
    cdd_cesm_path: str = "cdd-cesm"
    cbis_path: str = "cbis-ddsm"
    
    ## Configuracion de datos
    image_size: int = 512
    max_report_length: int = 512
    batch_size: int = 32
    num_workers: int = 8
    
    ## Splits
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42
    
    ## Datasets a usar
    use_vindr: bool = True
    use_rsna: bool = True
    use_cdd_cesm: bool = True
    use_cbis: bool = True


@dataclass
class ModelConfig:
    ## Arquitectura visual
    visual_encoder: str = "dinov2"
    image_embed_dim: int = 1024
    freeze_encoder_layers: int = -1
    use_moe: bool = True
    
    ## Arquitectura LLM
    llm_name: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    
    ## Proyeccion visual-textual
    projection_layers: int = 4
    projection_dropout: float = 0.1
    
    ## Clasificacion BI-RADS (multiclase 0-5)
    num_birads_classes: int = 6
    
    ## Head de hallazgos (Area 2)
    num_findings_classes: int = 8
    findings_names: List[str] = field(default_factory=lambda: [
        "calcificaciones",
        "densidad_alta",
        "asimetria",
        "distorsion",
        "masa",
        "lesion_sospechosa",
        "cambios_vasculares",
        "otros"
    ])
    
    ## LoRA para LLM
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    ## Compilacion
    compile_model: bool = True
    device: str = "auto"


@dataclass
class TrainingConfig:
    ## Optimizador y scheduler
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    num_epochs: int = 50
    
    ## Scheduler
    scheduler_type: str = "cosine"
    scheduler_T_max: int = 50
    
    ## Loss weights
    cls_loss_weight: float = 1.0
    findings_loss_weight: float = 0.5
    density_loss_weight: float = 0.1
    coherence_loss_weight: float = 0.2
    
    ## Regularizacion
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    
    ## Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n_checkpoints: int = 3


@dataclass
class EvaluationConfig:
    ## Metricas de coherencia (Area 1)
    compute_coherence: bool = True
    coherence_threshold: float = 0.75
    
    ## Calibracion
    compute_calibration: bool = True
    n_bins_calibration: int = 10
    
    ## Reproducibilidad inter-evaluador (Area 3)
    compute_inter_rater: bool = True
    
    ## Hallazgos (Area 4)
    compute_findings_metrics: bool = True
    findings_f1_threshold: float = 0.5
    
    ## Portabilidad
    compute_portability: bool = True
    max_latency_ms: float = 30000
    max_ram_mb: float = 8000


class MammoVLMConfig:
    ## Configuracion global que agrupa todas las subconfiguraciones
    
    def __init__(self):
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        
        ## Rutas globales
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.outputs_dir = self.project_root / "outputs"
        self.results_dir = self.project_root / "results"
        self.checkpoints_dir = self.project_root / "checkpoints"
        
        ## Crear directorios si no existen
        for d in [self.data_dir, self.outputs_dir, self.results_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        ## Convertir configuracion a diccionario para logging
        return {
            "dataset": self.dataset.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
        }


## Instancia global de configuracion
config = MammoVLMConfig()
