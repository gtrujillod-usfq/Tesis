## xai/__init__.py
## Paquete de interpretabilidad (XAI) para el experimento canonico exp08
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Modulos:
##   config_xai          : unica fuente de verdad de rutas y constantes
##   carga_modelo        : instanciacion del modelo exp08 y del generador RAG
##   atribucion_clasificador : Integrated Gradients y Grad-CAM por cabeza
##   metricas_clasificador   : Deletion AUC, Pointing Game, IoU IG-GradCAM
##   atribucion_rag          : Shapley exacto y grounding NLI sobre chunks
##   metricas_rag            : tasas de coincidencia de fuente, tablas y figuras

## Los modulos internos del paquete usan imports planos (from config_xai import,
## from carga_modelo import) que requieren que xai/ este en sys.path.
## Se agrega aqui para que funcione tanto al ejecutar como script como al
## importar como paquete desde cualquier directorio.
import sys as _sys
from pathlib import Path as _Path
_XAI_PKG_DIR = str(_Path(__file__).resolve().parent)
if _XAI_PKG_DIR not in _sys.path:
    _sys.path.insert(0, _XAI_PKG_DIR)
