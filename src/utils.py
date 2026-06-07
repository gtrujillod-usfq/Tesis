## utils.py
## Utilidades generales para MammoVLM V2
## Funciones auxiliares para logging, manejo de archivos, normalizacion, etc.

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from datetime import datetime


def setup_logging(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    ##
    ## Configura logger con formato estandar
    ##
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    ## Formato con timestamp
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ## Handler a consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    ## Handler a archivo si se especifica
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_json(data: Dict[str, Any], path: Path, indent: int = 2):
    ##
    ## Guarda diccionario como JSON con manejo de tipos especiales
    ##
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def json_serializer(obj):
        ## Manejo de tipos numpy y torch
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=json_serializer)


def load_json(path: Path) -> Dict[str, Any]:
    ##
    ## Carga diccionario desde JSON
    ##
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    ##
    ## Retorna timestamp formateado YYYY-MM-DD_HH-MM-SS
    ##
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    ##
    ## Cuenta parametros totales y entrenables del modelo
    ##
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percentage": 100 * trainable / max(total, 1),
    }


def get_device(device_str: str = "auto") -> torch.device:
    ##
    ## Obtiene dispositivo disponible (CUDA, MPS, CPU)
    ##
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    return device


def normalize_birads(value: int) -> int:
    ##
    ## Normaliza valor BI-RADS al rango estandar [0, 5]
    ## Incluye remapeo de BI-RADS 6 (malignidad confirmada por biopsia) a 5
    ## La logica canonica vive en label_normalization.py; esta funcion delega
    ## en ella para mantener una unica fuente de verdad
    ##
    from label_normalization import normalize_birads as _norm
    return _norm(value)


def birads_to_risk_category(birads: int) -> str:
    ##
    ## Convierte numero BI-RADS a categoria de riesgo
    ##
    risk_map = {
        0: "incompleto",
        1: "benigno",
        2: "benigno",
        3: "probablemente_benigno",
        4: "sospechoso",
        5: "altamente_maligno",
    }
    return risk_map.get(birads, "desconocido")


class AverageMeter:
    ##
    ## Calcula y almacena el promedio de metricas durante entrenamiento
    ##
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.values = []
        self.sum = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1):
        self.values.append(value)
        self.sum += value * n
        self.count += n
    
    def get_average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count
    
    def get_stats(self) -> Dict[str, float]:
        if len(self.values) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        values = np.array(self.values)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }


def print_separator(char: str = "=", width: int = 70):
    ## Imprime linea separadora limpia sin caracteres especiales
    print(char * width)


def print_section(title: str, width: int = 70):
    ## Imprime seccion con titulo
    print_separator("=", width)
    print(f"{title:^{width}}")
    print_separator("=", width)
