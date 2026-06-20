## atribucion_clasificador.py
## Paso 2: Integrated Gradients y Grad-CAM para las cabezas BI-RADS y densidad.
##
## Decisiones de diseno:
##   - El objetivo escalar para BI-RADS es logits[3] + logits[4] (BR4+BR5 pre-softmax).
##     Usar pre-softmax es correcto para IG porque softmax suprime los gradientes
##     de clases dominantes y diluye las atribuciones de clases no predichas.
##   - Para densidad, el objetivo es el logit de la clase predicha (predicted_class).
##   - IG se aplica con captum y baseline de imagen negra (-mean/std).
##   - GradCAM usa LayerGradCam de captum sobre encoder.backbone._conv_head.
##   - El upsample de GradCAM es bilineal para preservar continuidad espacial.
##   - Las atribuciones se guardan como .npz con metadatos JSON para trazabilidad.

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from captum.attr import IntegratedGradients, LayerGradCam

from config_xai import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MALIGNANT_INDICES,
)
from carga_modelo import baseline_imagen_negra

logger = logging.getLogger(__name__)


## =========================================================
## _construir_objetivo_birads
## =========================================================

def _construir_objetivo_birads():
    """
    Retorna una funcion closure que calcula el objetivo escalar de malignidad
    para la cabeza BI-RADS a partir de los logits del modelo.

    El objetivo es la suma de los logits pre-softmax de BR4 y BR5 (indices 3 y 4).
    Se usa pre-softmax porque softmax introduce dependencias entre clases que
    distorsionan la atribucion: el gradiente de softmax(i) respecto a logit(j)
    es negativo para i != j, lo que puede cancelar atribuciones reales.

    Retorna
    -------
    func : callable(output_dict) -> torch.Tensor escalar
    """
    idx0, idx1 = MALIGNANT_INDICES[0], MALIGNANT_INDICES[1]

    def objetivo(output_dict):
        ## output_dict tiene claves 'birads' y 'density' segun MammoVLM.forward()
        logits = output_dict['birads']  ## [B, 5]
        return logits[:, idx0] + logits[:, idx1]

    return objetivo


def _construir_objetivo_density(predicted_class):
    """
    Retorna un closure para el objetivo de la cabeza de densidad.

    Parametros
    ----------
    predicted_class : int
        Indice de la clase de densidad (0-3) cuyo logit se maximiza.

    Retorna
    -------
    func : callable(output_dict) -> torch.Tensor escalar
    """
    def objetivo(output_dict):
        logits = output_dict['density']  ## [B, 4]
        return logits[:, predicted_class]

    return objetivo


## =========================================================
## _wrapper_forward_para_captum
## =========================================================

def _hacer_wrapper_forward(model, objetivo_func):
    """
    Crea un wrapper que acepta un tensor de entrada y retorna el escalar objetivo.
    Captum requiere una funcion f(input) -> scalar para IntegratedGradients
    y para LayerGradCam cuando se usa el argumento forward_func.

    Parametros
    ----------
    model : MammoVLM
    objetivo_func : callable(output_dict) -> torch.Tensor [B]

    Retorna
    -------
    wrapper : callable(input_tensor) -> torch.Tensor [B]
    """
    def wrapper(input_tensor):
        output_dict = model(input_tensor)
        return objetivo_func(output_dict)

    return wrapper


## =========================================================
## calcular_ig
## =========================================================

def calcular_ig(model, img_tensor, head, predicted_class=None, n_steps=50,
                device='cpu', internal_batch_size=8):
    """
    Calcula Integrated Gradients para una imagen y una cabeza del modelo.

    IG mide cuanto contribuye cada pixel (relativo al baseline) al escalar objetivo.
    Formalmente: IG_i(x) = (x_i - x'_i) * integral_0^1 [dF/dx_i](x' + a*(x-x')) da
    donde x es la imagen, x' es el baseline, F es el objetivo escalar.
    La integral se aproxima con n_steps pasos de Riemann.

    Parametros
    ----------
    model : MammoVLM
        Modelo en modo eval con gradientes habilitados.
    img_tensor : torch.Tensor
        Tensor [1, 3, H, W] float32.
    head : str
        'birads' o 'density'.
    predicted_class : int o None
        Requerido si head='density'. Ignorado para head='birads'.
    n_steps : int
        Numero de pasos de la aproximacion de Riemann (mayor = mas preciso pero lento).
    device : str
        Dispositivo donde crear el baseline.
    internal_batch_size : int
        Numero de steps de IG procesados en paralelo por captum. Con internal_batch_size=None
        captum apila los n_steps en un solo batch; a 1520x912 el tensor intermedio del
        depthwise conv de EfficientNet supera 2^31 elementos y dispara
        canUse32BitIndexMath en CUDA. Valor 8 mantiene el pico de memoria acotado
        sin reducir n_steps ni degradar la precision de la atribucion.

    Retorna
    -------
    attr_map_2d : np.ndarray de forma [H, W]
        Mapa de atribucion 2D obtenido como norma L2 sobre los 3 canales.
        Valores positivos indican contribucion hacia el objetivo (malignidad o clase dada).

    Raises
    ------
    ValueError
        Si head='density' y predicted_class es None.
    """
    if head == 'density' and predicted_class is None:
        raise ValueError(
            "predicted_class es requerido cuando head='density'. "
            "Pasa el indice de la clase predicha (0-3)."
        )

    ## Seleccionar el objetivo segun la cabeza
    if head == 'birads':
        objetivo_func = _construir_objetivo_birads()
    else:
        objetivo_func = _construir_objetivo_density(predicted_class)

    ## Wrapper compatible con captum: f(tensor) -> scalar
    forward_func = _hacer_wrapper_forward(model, objetivo_func)

    ## Baseline: imagen negra en espacio normalizado, mismo dispositivo que img_tensor
    baseline = baseline_imagen_negra(device)
    baseline = baseline.to(img_tensor.device)

    ## Expandir baseline a la forma exacta de img_tensor (por si H o W difieren)
    baseline = baseline.expand_as(img_tensor)

    ## IntegratedGradients requiere que el modelo NO este envuelto en torch.no_grad()
    ## porque internamente computa gradientes del output respecto a la entrada
    ig = IntegratedGradients(forward_func)

    ## .attribute() retorna tensor [1, 3, H, W] con las atribuciones por pixel y canal
    ## internal_batch_size=8: captum procesa los n_steps en bloques de 8 en lugar de
    ## apilarlos todos en un solo batch. Evita que el tensor intermedio del depthwise
    ## conv de EfficientNet supere 2^31 elementos a 1520x912 (canUse32BitIndexMath).
    attributions = ig.attribute(
        inputs=img_tensor,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',   ## integracion de Gauss-Legendre, mas estable que riemann
        return_convergence_delta=False,
        internal_batch_size=internal_batch_size,
    )

    ## Agregar sobre los 3 canales usando norma L2:
    ## attr_2d[h,w] = sqrt(sum_c attr[0,c,h,w]^2)
    ## La norma L2 es preferible a la suma porque preserva la magnitud total
    ## independientemente del signo de cada canal.
    attr_map_2d = attributions[0].norm(dim=0).detach().cpu().numpy()  ## [H, W]

    return attr_map_2d


## =========================================================
## calcular_gradcam
## =========================================================

def calcular_gradcam(model, img_tensor, head, predicted_class=None, device='cpu'):
    """
    Calcula Grad-CAM para una imagen usando la capa encoder.backbone._conv_head.

    GradCAM localiza las regiones del mapa de activacion cuya respuesta tiene
    mayor gradiente respecto al objetivo escalar. El mapa espacial de [48, 29]
    se upsamplea a [1520, 912] con interpolacion bilineal.

    Parametros
    ----------
    model : MammoVLM
    img_tensor : torch.Tensor [1, 3, H, W]
    head : str, 'birads' o 'density'
    predicted_class : int o None (requerido para head='density')
    device : str (no se usa directamente; el tensor ya esta en su dispositivo)

    Retorna
    -------
    cam_map_2d : np.ndarray de forma [H, W] = [1520, 912]
        Mapa de Grad-CAM upsampled, con valores >= 0 (relu aplicado).

    Raises
    ------
    ValueError
        Si head='density' y predicted_class es None.
    """
    if head == 'density' and predicted_class is None:
        raise ValueError(
            "predicted_class es requerido cuando head='density'."
        )

    ## Seleccionar objetivo
    if head == 'birads':
        objetivo_func = _construir_objetivo_birads()
    else:
        objetivo_func = _construir_objetivo_density(predicted_class)

    forward_func = _hacer_wrapper_forward(model, objetivo_func)

    ## La capa objetivo es encoder.backbone._conv_head (ultima conv del backbone,
    ## forma de salida [B, 2048, 48, 29] para entrada (1,3,1520,912))
    target_layer = model.encoder.backbone._conv_head

    layer_gc = LayerGradCam(forward_func, layer=target_layer)

    ## relu_attributions=True aplica ReLU al CAM antes de devolver,
    ## eliminando contribuciones negativas (regiones que inhiben el objetivo)
    ## Retorna tensor de forma [1, 1, 48, 29]
    attributions = layer_gc.attribute(
        inputs=img_tensor,
        relu_attributions=True,
    )

    ## Agregar sobre la dimension de canales (mean): [1, 1, 48, 29]
    cam_aggregated = attributions.mean(dim=1, keepdim=True)  ## [1, 1, 48, 29]

    ## Upsamplear a la resolucion de entrada (1520, 912) con interpolacion bilineal
    ## align_corners=False es la convencion estandar para CAMs
    cam_upsampled = F.interpolate(
        cam_aggregated,
        size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode='bilinear',
        align_corners=False,
    )  ## [1, 1, 1520, 912]

    ## Extraer array 2D [1520, 912]
    cam_map_2d = cam_upsampled[0, 0].detach().cpu().numpy()

    return cam_map_2d


## =========================================================
## calcular_atribuciones_imagen
## =========================================================

def calcular_atribuciones_imagen(model, img_tensor, head, predicted_class=None,
                                  n_steps=50, device='cpu'):
    """
    Calcula IG y Grad-CAM para una imagen y retorna ambos mapas en un dict.

    Parametros
    ----------
    model : MammoVLM
    img_tensor : torch.Tensor [1, 3, H, W]
    head : str, 'birads' o 'density'
    predicted_class : int o None
    n_steps : int, pasos de IG
    device : str

    Retorna
    -------
    dict con claves:
        'ig'      : np.ndarray [H, W]
        'gradcam' : np.ndarray [H, W]
    """
    logger.info("Calculando IG para head=%s ...", head)
    ig_map = calcular_ig(
        model=model,
        img_tensor=img_tensor,
        head=head,
        predicted_class=predicted_class,
        n_steps=n_steps,
        device=device,
    )

    logger.info("Calculando GradCAM para head=%s ...", head)
    gradcam_map = calcular_gradcam(
        model=model,
        img_tensor=img_tensor,
        head=head,
        predicted_class=predicted_class,
        device=device,
    )

    return {
        'ig':      ig_map,
        'gradcam': gradcam_map,
    }


## =========================================================
## guardar_atribucion
## =========================================================

def guardar_atribucion(attr_dict, image_id, head, out_dir, meta=None):
    """
    Guarda los mapas de atribucion como archivo .npz con metadatos opcionales.

    Parametros
    ----------
    attr_dict : dict con claves 'ig' y 'gradcam', ambos np.ndarray [H, W]
    image_id : str
        Identificador unico de la imagen (columna image_id del CSV).
    head : str, 'birads' o 'density'
    out_dir : str o Path
        Directorio de salida (se crea si no existe).
    meta : dict o None
        Metadatos adicionales (prediccion, probabilities, etc.).
        Se serializa como JSON string dentro del .npz.

    Nombre del archivo: {image_id}_{head}.npz
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nombre = f"{image_id}_{head}.npz"
    ruta   = out_dir / nombre

    ## Serializar metadatos a JSON string para almacenamiento en npz
    meta_str = json.dumps(meta if meta is not None else {})

    np.savez_compressed(
        str(ruta),
        ig=attr_dict['ig'],
        gradcam=attr_dict['gradcam'],
        meta=np.array(meta_str),     ## npz requiere array; string se guarda como 0-d array
    )

    logger.info("Atribucion guardada en %s", ruta)


## =========================================================
## cargar_atribucion
## =========================================================

def cargar_atribucion(image_id, head, out_dir):
    """
    Carga un archivo .npz de atribucion previamente guardado con guardar_atribucion().

    Parametros
    ----------
    image_id : str
    head : str, 'birads' o 'density'
    out_dir : str o Path

    Retorna
    -------
    dict con claves:
        'ig'      : np.ndarray [H, W]
        'gradcam' : np.ndarray [H, W]
        'meta'    : dict (decodificado desde JSON)

    Raises
    ------
    FileNotFoundError si el archivo no existe.
    """
    ruta = Path(out_dir) / f"{image_id}_{head}.npz"

    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo de atribucion no encontrado: {ruta}. "
            f"Ejecuta calcular_atribuciones_imagen() primero."
        )

    data = np.load(str(ruta), allow_pickle=True)

    ## Decodificar metadatos desde JSON string (0-d array de numpy)
    meta_str  = str(data['meta'])
    meta_dict = json.loads(meta_str)

    return {
        'ig':      data['ig'],
        'gradcam': data['gradcam'],
        'meta':    meta_dict,
    }
