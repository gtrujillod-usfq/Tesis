## carga_modelo.py
## Paso 1a: instanciacion del clasificador exp08 (MammoVLM) y utilidades de carga.
## El RAG/LLM se carga en un modulo separado (carga_rag.py, Paso 1b).
##
## Decisiones de diseno:
##   - sys.path se modifica dinamicamente para no requerir instalacion del paquete src/.
##   - torch.no_grad() NUNCA se llama aqui: los gradientes deben fluir para IG y GradCAM.
##   - .float() fuerza float32 independientemente del dtype del checkpoint.
##   - El baseline de imagen negra se calcula como (-mean/std) canal a canal,
##     que corresponde al tensor normalizado de un array de ceros en RGB.

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

## Agrega src/ al path de busqueda de modulos para importar MammoVLM y utilidades.
## Se hace en tiempo de importacion del modulo para que todas las funciones lo hereden.
_XAI_DIR     = Path(__file__).resolve().parent          ## Tesis/XAI/xai/
_TESIS_ROOT  = _XAI_DIR.parent.parent                   ## Tesis/
_SRC_DIR     = _TESIS_ROOT / 'src'

## xai/ se agrega para que los imports internos (from config_xai import, from carga_modelo
## import) funcionen tanto al ejecutar como script como al importar como paquete.
if str(_XAI_DIR) not in sys.path:
    sys.path.insert(0, str(_XAI_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

## Importaciones del paquete src/ de la tesis
from models import MammoVLM
from data_loading import MammoCLIPTransform, load_image_as_pil

from config_xai import (
    EXP08_MODEL_PT,
    MAMMOCLIP_CHECKPOINT,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    UNFREEZE_LAST_N_BLOCKS,
)

logger = logging.getLogger(__name__)


## =========================================================
## Constantes de arquitectura de exp08
## =========================================================

## Parametros de MammoVLM usados en exp08
_EFFICIENTNET_NAME    = 'efficientnet-b5'
_NUM_BIRADS_CLASSES   = 5
_NUM_DENSITY_CLASSES  = 4
_FREEZE_ENCODER       = True      ## se descongela parcialmente con unfreeze_last_n_blocks
_HIDDEN_DIM           = 256
_DROPOUT              = 0.2

## Estadisticas de normalizacion ImageNet usadas por MammoCLIPTransform
## Fuente: mean y std del preentrenamiento de EfficientNet en ImageNet-1k
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


## =========================================================
## cargar_modelo_exp08
## =========================================================

def cargar_modelo_exp08(device='auto'):
    """
    Instancia MammoVLM con la configuracion de exp08, carga los pesos entrenados
    y prepara el modelo para inferencia XAI (modo eval, gradientes habilitados).

    Parametros
    ----------
    device : str
        'auto' selecciona CUDA si esta disponible, de lo contrario CPU.
        Tambien acepta 'cuda', 'cuda:0', 'cpu', etc.

    Retorna
    -------
    model : MammoVLM
        Modelo en modo eval con pesos de exp08, en float32.
    device_str : str
        Cadena del dispositivo efectivo (ej. 'cuda:0' o 'cpu').

    Notas
    -----
    - torch.no_grad() NO se aplica aqui. Los gradientes deben fluir para
      Integrated Gradients y GradCAM en pasos posteriores.
    - .float() garantiza float32 incluso si el checkpoint tiene tensores en
      otro dtype, evitando errores de precision mixta en captum.
    - MammoVLM.__init__ llama internamente self.encoder.load_backbone()
      con MAMMOCLIP_CHECKPOINT, por lo que el encoder ya esta cargado.
    """
    ## Resolver dispositivo efectivo
    if device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device

    logger.info("Dispositivo seleccionado: %s", device_str)

    ## Instanciar MammoVLM con la configuracion exacta de exp08.
    ## freeze_encoder=True con unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS
    ## replica el estado de entrenamiento de exp08.
    ## __init__ llama load_backbone() internamente, lo que carga mammo_clip_b5.tar.
    logger.info("Instanciando MammoVLM (efficientnet-b5, checkpoint: %s)...", MAMMOCLIP_CHECKPOINT)
    model = MammoVLM(
        checkpoint_path=str(MAMMOCLIP_CHECKPOINT),
        efficientnet_name=_EFFICIENTNET_NAME,
        num_birads_classes=_NUM_BIRADS_CLASSES,
        num_density_classes=_NUM_DENSITY_CLASSES,
        freeze_encoder=_FREEZE_ENCODER,
        hidden_dim=_HIDDEN_DIM,
        dropout=_DROPOUT,
        unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
    )

    ## Cargar los pesos del experimento exp08 desde el checkpoint
    logger.info("Cargando state_dict desde %s...", EXP08_MODEL_PT)
    checkpoint = torch.load(str(EXP08_MODEL_PT), map_location='cpu')

    ## La clave 'model_state_dict' es la convencion usada en el entrenamiento de exp08
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    logger.info("State dict cargado correctamente (%d claves).", len(state_dict))

    ## Convertir a float32 por seguridad (captum requiere float32 para IG)
    model = model.float()

    ## Mover al dispositivo antes de poner en eval para que los buffers
    ## internos (batch norm) queden en el dispositivo correcto
    model = model.to(device_str)

    ## Modo eval: desactiva dropout y usa estadisticas de batch norm almacenadas.
    ## NO llama torch.no_grad(); los gradientes fluyen para atribucion.
    model.eval()

    logger.info("MammoVLM listo en %s (modo eval, gradientes habilitados).", device_str)
    return model, device_str


## =========================================================
## cargar_transform_inferencia
## =========================================================

def cargar_transform_inferencia():
    """
    Retorna el transform de inferencia identico al usado durante el entrenamiento
    de exp08: resize directo a (1520, 912) con CLAHE, sin augmentacion.

    Retorna
    -------
    transform : MammoCLIPTransform
        Pipeline de preprocesamiento listo para aplicar a imagenes PIL.
    """
    transform = MammoCLIPTransform(
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        augment=False,
        use_clahe=True,
    )
    return transform


## =========================================================
## cargar_imagen
## =========================================================

def cargar_imagen(image_path, transform, device):
    """
    Carga una imagen mamografica desde disco, aplica el transform de inferencia
    y la convierte en un tensor listo para el modelo.

    Parametros
    ----------
    image_path : str o Path
        Ruta al archivo de imagen (PNG/JPG tipicamente).
    transform : callable
        Transform devuelto por cargar_transform_inferencia().
    device : str
        Dispositivo destino ('cpu', 'cuda', etc.).

    Retorna
    -------
    img_tensor : torch.Tensor
        Tensor de forma [1, 3, 1520, 912] en float32 en el dispositivo indicado.
    """
    ## load_image_as_pil es la funcion de src/data_loading.py que maneja
    ## la apertura correcta de imagenes mamograficas (incluyendo DICOM si aplica)
    pil_img = load_image_as_pil(str(image_path))

    ## El transform devuelve un tensor [C, H, W]; unsqueeze agrega dimension de batch
    img_tensor = transform(pil_img)         ## [3, 1520, 912]
    img_tensor = img_tensor.unsqueeze(0)    ## [1, 3, 1520, 912]
    img_tensor = img_tensor.to(device).float()

    return img_tensor


## =========================================================
## baseline_imagen_negra
## =========================================================

def baseline_imagen_negra(device):
    """
    Construye el tensor de baseline para Integrated Gradients correspondiente
    a una imagen con todos los pixeles en cero (negro absoluto).

    El baseline correcto para IG en imagenes normalizadas con ImageNet NO es
    un tensor de ceros [0, 0, 0] en el espacio normalizado, sino el tensor
    que resulta de aplicar la normalizacion a un array de ceros en RGB:

        baseline_c = (0.0 - mean_c) / std_c  = -mean_c / std_c

    Esto garantiza que el baseline representa "ausencia de informacion visual"
    en el espacio de entrada del modelo, lo que es el punto de partida semantico
    correcto para IG (Sundararajan et al., 2017).

    Parametros
    ----------
    device : str
        Dispositivo donde crear el tensor.

    Retorna
    -------
    baseline : torch.Tensor
        Tensor de forma [1, 3, 1520, 912] con los valores de imagen negra
        en el espacio normalizado, en float32.
    """
    ## Calcular el valor normalizado de un pixel negro (0.0) por canal
    ## Formula: normalized = (pixel - mean) / std => para pixel=0: -mean/std
    baseline_vals = [
        -_IMAGENET_MEAN[c] / _IMAGENET_STD[c]
        for c in range(3)
    ]

    ## Crear tensor de forma [1, 3, 1, 1] y expandir a [1, 3, H, W]
    baseline = torch.tensor(baseline_vals, dtype=torch.float32, device=device)
    baseline = baseline.view(1, 3, 1, 1)
    baseline = baseline.expand(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    ## .clone() para que sea un tensor contiguo independiente (no un view expandido)
    ## Esto evita errores en operaciones in-place dentro de captum
    baseline = baseline.clone()

    return baseline


## =========================================================
## obtener_prediccion_base
## =========================================================

def obtener_prediccion_base(model, img_tensor):
    """
    Ejecuta el forward pass del modelo y retorna logits, probabilidades e indices
    de prediccion para ambas cabezas.

    IMPORTANTE: esta funcion NO usa torch.no_grad(). El contexto de gradientes
    se controla desde el codigo que llama esta funcion segun el uso:
      - En exploracion/visualizacion: el llamador puede usar torch.no_grad().
      - En atribucion (IG, GradCAM): el llamador NO debe usar torch.no_grad().

    Parametros
    ----------
    model : MammoVLM
        Modelo en modo eval.
    img_tensor : torch.Tensor
        Tensor [1, 3, 1520, 912] float32 en el dispositivo del modelo.

    Retorna
    -------
    resultado : dict con claves:
        'birads_logits'  : torch.Tensor [1, 5] pre-softmax
        'density_logits' : torch.Tensor [1, 4] pre-softmax
        'birads_idx'     : int, indice de la clase BI-RADS predicha
        'density_idx'    : int, indice de la clase de densidad predicha
        'birads_probs'   : torch.Tensor [1, 5] probabilidades
        'density_probs'  : torch.Tensor [1, 4] probabilidades
    """
    ## forward() de MammoVLM retorna {"birads": [B,5], "density": [B,4]} pre-softmax
    salida = model(img_tensor)

    birads_logits  = salida['birads']    ## [1, 5]
    density_logits = salida['density']   ## [1, 4]

    ## softmax sobre la dimension de clases para obtener probabilidades
    birads_probs  = torch.softmax(birads_logits,  dim=1)
    density_probs = torch.softmax(density_logits, dim=1)

    ## argmax para la clase predicha
    birads_idx  = int(birads_probs.argmax(dim=1).item())
    density_idx = int(density_probs.argmax(dim=1).item())

    return {
        'birads_logits':  birads_logits,
        'density_logits': density_logits,
        'birads_idx':     birads_idx,
        'density_idx':    density_idx,
        'birads_probs':   birads_probs,
        'density_probs':  density_probs,
    }
