## config_xai.py
## Unica fuente de verdad de rutas y constantes para el pipeline XAI
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## REGLA DE ORO:
##   Toda ruta de LECTURA apunta a recursos dentro de Tesis/ (fuera de XAI/).
##   Toda ruta de ESCRITURA apunta a Tesis/XAI/outputs/.
##   Ninguna ruta usa EXPERIMENT_NAME ni EXPERIMENT_ID (apuntan a exp09).
##
## Rutas de lectura confirmadas en el Paso 0b del reconocimiento:
##   - model.pt     : outputs/experiments/exp08.../model.pt  (121 MB, solo-lectura)
##   - mammo_clip   : models/mammo_clip_b5.tar               (1.6 GB, solo-lectura)
##   - test_csv     : outputs/test_sets/test_set_vindr.csv    (4000 filas)
##   - findings_csv : data/vindr-mammo/finding_annotations.csv
##   - rag_index    : data/rag_index/
##   - literatura   : Libros/
##
## Constantes del encoder (confirmadas en Paso 0b):
##   - Entrada: (1520, 912) alto x ancho, estiramiento directo (sin padding)
##   - Mapa espacial en _conv_head: [batch, 2048, 48, 29] para entrada (1,3,1520,912)
##   - Capa objetivo Grad-CAM: encoder.backbone._conv_head
##   - Cabeza BI-RADS: 5 logits (indices 0-4 = BI-RADS 1-5)
##     objetivo XAI:  malignancy_score = probs[3] + probs[4]
##   - Cabeza densidad: 4 logits (indices 0-3 = DENSITY A-D)
##     objetivo XAI:  prob de la clase de densidad predicha

from pathlib import Path

## =========================================================
## Guard de seguridad: verificar que el experimento canonico
## es exp08. Falla de forma ruidosa si se cambia por error.
## =========================================================

EXPERIMENT_FINAL = 'exp08_ordinal_sord_qwk_descongelado'

assert EXPERIMENT_FINAL == 'exp08_ordinal_sord_qwk_descongelado', (
    f"Experimento incorrecto: se esperaba exp08_ordinal_sord_qwk_descongelado, "
    f"obtenido {EXPERIMENT_FINAL!r}. No usar EXPERIMENT_NAME ni EXPERIMENT_ID."
)

## =========================================================
## Raiz del proyecto (dos niveles arriba de este archivo:
## Tesis/XAI/xai/config_xai.py -> Tesis/)
## =========================================================

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent   ## Tesis/
XAI_ROOT     = _THIS_FILE.parent.parent          ## Tesis/XAI/

## =========================================================
## Rutas de LECTURA (solo-lectura; dentro de Tesis/, fuera de XAI/)
## =========================================================

## Checkpoint del modelo exp08 (solo-lectura)
## Confirmado: outputs/experiments/exp08.../model.pt, clave 'model_state_dict'
EXP08_MODEL_PT = (
    PROJECT_ROOT / 'outputs' / 'experiments' / EXPERIMENT_FINAL / 'model.pt'
)

## Checkpoint preentrenado de Mammo-CLIP (EfficientNet-B5, 1.6 GB)
## Libreria: efficientnet_pytorch (lukemelas), NO timm
## Prefijo en el state_dict: 'image_encoder.'
MAMMOCLIP_CHECKPOINT = PROJECT_ROOT / 'models' / 'mammo_clip_b5.tar'

## CSV del split de TEST de exp08 (4000 filas, columnas sin bounding boxes)
## Columnas: image_path, birads, density_index, study_id, image_id, ...
TEST_CSV = PROJECT_ROOT / 'outputs' / 'test_sets' / 'test_set_vindr.csv'

## CSV de anotaciones de hallazgos (bounding boxes y categorias)
## Columnas: image_id, xmin, ymin, xmax, ymax, height, width, finding_categories, ...
## Join con TEST_CSV por image_id
FINDING_ANNOTATIONS_CSV = (
    PROJECT_ROOT / 'data' / 'vindr-mammo' / 'finding_annotations.csv'
)

## Indice FAISS del RAG (construido con PubMedBERT, chunks ~400 tokens)
RAG_INDEX_DIR = PROJECT_ROOT / 'data' / 'rag_index'

## Directorio de literatura medica (PDFs fuente del RAG)
LITERATURE_DIR = PROJECT_ROOT / 'Libros'

## =========================================================
## Constantes de arquitectura exp08 (confirmadas en Paso 0b)
## =========================================================

## Resolucion de entrada del encoder (alto x ancho)
IMAGE_HEIGHT = 1520
IMAGE_WIDTH  = 912

## Numero de bloques descongelados del encoder en exp08
UNFREEZE_LAST_N_BLOCKS = 2

## Nombres de las cabezas del modelo (claves en el dict de salida)
HEAD_BIRADS   = 'birads'
HEAD_DENSITY  = 'density'

## Numero de clases por cabeza
NUM_BIRADS_CLASSES   = 5   ## indices 0-4 = BI-RADS 1-5
NUM_DENSITY_CLASSES  = 4   ## indices 0-3 = DENSITY A-D

## Capa objetivo para Grad-CAM: ultima conv antes del GAP
## En efficientnet_pytorch, encoder.backbone._conv_head
## Forma del mapa espacial confirmada: [batch, 2048, 48, 29] para (1,3,1520,912)
GRADCAM_TARGET_LAYER = 'encoder.backbone._conv_head'
FEATURE_MAP_H = 48
FEATURE_MAP_W = 29

## Indices de las clases malignas para el objetivo escalar de BI-RADS
## malignancy_score = probs[MALIGNANT_INDICES[0]] + probs[MALIGNANT_INDICES[1]]
MALIGNANT_INDICES = [3, 4]   ## BR4 y BR5 (indices 3 y 4)

## =========================================================
## Configuracion del RAG (confirmada en Paso 0b)
## =========================================================

RAG_TOP_K = 3          ## chunks recuperados por consulta en exp08
NUM_RAG_SUBSETS = 8    ## 2^3 subconjuntos para Shapley exacto
QWEN_MODEL_ID   = 'Qwen/Qwen2.5-7B-Instruct'
QWEN_DTYPE      = 'bfloat16'

## =========================================================
## Rutas de ESCRITURA (exclusivamente bajo Tesis/XAI/outputs/)
## =========================================================

OUTPUTS_ROOT = XAI_ROOT / 'outputs'

## Atribuciones y mapas del clasificador (por cabeza)
OUT_BIRADS   = OUTPUTS_ROOT / 'clasificador' / 'birads'
OUT_DENSITY  = OUTPUTS_ROOT / 'clasificador' / 'densidad'

## Resultados de atribucion del RAG (Shapley y NLI)
OUT_RAG      = OUTPUTS_ROOT / 'rag'

## Figuras y tablas de metricas
OUT_FIGURAS  = OUTPUTS_ROOT / 'figuras'
OUT_TABLAS   = OUTPUTS_ROOT / 'tablas'

## =========================================================
## Verificaciones de existencia en tiempo de importacion
## (advierten si un recurso de lectura no existe en disco)
## =========================================================

def _advertir_si_falta(path: Path, nombre: str) -> None:
    ## Emite un aviso si el recurso de solo-lectura no existe en disco.
    ## No lanza excepcion para no bloquear la importacion del modulo.
    if not path.exists():
        import warnings
        warnings.warn(
            f"[config_xai] Recurso no encontrado: {nombre} -> {path}",
            stacklevel=2,
        )

_advertir_si_falta(EXP08_MODEL_PT,          'EXP08_MODEL_PT')
_advertir_si_falta(MAMMOCLIP_CHECKPOINT,     'MAMMOCLIP_CHECKPOINT')
_advertir_si_falta(TEST_CSV,                 'TEST_CSV')
_advertir_si_falta(FINDING_ANNOTATIONS_CSV,  'FINDING_ANNOTATIONS_CSV')
