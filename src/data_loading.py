## data_loading.py
## Carga de datos de VinDr-Mammo para entrenamiento del MammoVLM V2 (exp06)
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Esta version esta enfocada exclusivamente en VinDr-Mammo y en el encoder
## Mammo-CLIP (EfficientNet-B5, alta resolucion). Cambios respecto a la version
## multi-dataset anterior:
##   - Solo VinDr-Mammo (se eliminaron los loaders de CBIS, CDD-CESM, DMID, INbreast)
##   - Alta resolucion nativa de Mammo-CLIP (1520x912), sin tiling 3x3
##   - Preprocesamiento alineado con Mammo-CLIP (normalizacion ImageNet)
##   - Solo dos tareas: BI-RADS (1-5) y densidad (DENSITY A/B/C/D)
##   - Sin masking multi-dataset (todas las imagenes de VinDr tienen ambas etiquetas)
##
## Referencia del encoder: Ghosh et al., "Mammo-CLIP", MICCAI 2024.
## Referencia del dataset: Nguyen et al., Scientific Data 10:277 (2023).

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

## Resolver el import de apply_voi_lut una sola vez (a nivel de modulo) con
## fallback: la ruta pydicom.pixels es la nueva (pydicom >= 3.0); la ruta
## pydicom.pixel_data_handlers.util es la antigua (se elimina en pydicom 4.0).
## Hacerlo aqui (y no dentro de la funcion) evita el warning repetido por imagen.
try:
    from pydicom.pixels import apply_voi_lut as _apply_voi_lut
except ImportError:
    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut as _apply_voi_lut
    except ImportError:
        _apply_voi_lut = None


## ============================================================
## Preprocesamiento de imagenes alineado con Mammo-CLIP
## ============================================================
##
## Mammo-CLIP fue entrenado con imagenes redimensionadas a 1520x912 y
## normalizacion ImageNet estandar. Replicamos ese preprocesamiento para
## que el encoder reciba imagenes con la misma distribucion que vio en su
## preentrenamiento, maximizando el beneficio del transfer learning.

class MammoCLIPTransform:
    ## Preprocesamiento de una sola imagen de alta resolucion para Mammo-CLIP
    ##
    ## A diferencia de la version multi-escala anterior (vista global + 9 parches),
    ## aqui se produce UNA sola imagen de alta resolucion. El encoder EfficientNet-B5
    ## de Mammo-CLIP preserva el detalle fino (microcalcificaciones) gracias a la
    ## resolucion alta nativa, sin necesidad de tiling.

    ## Normalizacion ImageNet (la que usa Mammo-CLIP / efficientnet_pytorch)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    ## Resolucion nativa de Mammo-CLIP: alto x ancho = 1520 x 912
    DEFAULT_HEIGHT = 1520
    DEFAULT_WIDTH = 912

    def __init__(
        self,
        height: int = 1520,
        width: int = 912,
        augment: bool = False,
        use_clahe: bool = True,
    ):
        ##
        ## Parametros:
        ##   height, width: dimensiones de salida (default 1520x912, nativo Mammo-CLIP)
        ##   augment: si True, aplica augmentations suaves para fine-tuning
        ##   use_clahe: si True, aplica CLAHE para realzar el contraste local
        ##
        self.height = height
        self.width = width
        self.augment = augment
        self.use_clahe = use_clahe
        self._transform = None

    def _build_transform(self):
        ## Construye el pipeline de torchvision (resize + augmentations + normalizacion)
        import torchvision.transforms as T

        ## Redimensionar a la resolucion de Mammo-CLIP (alto, ancho)
        ops = [T.Resize((self.height, self.width))]

        ## Augmentations suaves para fine-tuning (mas conservadoras que el
        ## preentrenamiento de Mammo-CLIP, que usaba rotaciones de 20 grados).
        ## En mamografia hay que cuidar no distorsionar la lateralidad ni la
        ## orientacion anatomica, por eso se usan transformaciones leves.
        if self.augment:
            ops.append(T.RandomHorizontalFlip(p=0.5))
            ops.append(T.RandomRotation(degrees=10))
            ops.append(T.RandomAffine(degrees=0, translate=(0.05, 0.05)))

        ops.append(T.ToTensor())
        ops.append(T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD))
        self._transform = T.Compose(ops)

    def __call__(self, pil_image):
        ##
        ## Retorna un tensor [3, height, width] preprocesado para Mammo-CLIP
        ##
        ## Aplicar CLAHE para realzar contraste local si esta habilitado
        if self.use_clahe:
            pil_image = apply_clahe(pil_image)

        if self._transform is None:
            self._build_transform()
        return self._transform(pil_image)


def apply_clahe(pil_image):
    ##
    ## Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
    ## para mejorar el contraste local del tejido mamario
    ##
    ## CLAHE realza diferencias sutiles de densidad sin saturar, lo que
    ## mejora la visibilidad de microcalcificaciones y bordes de lesiones.
    ##
    from PIL import Image

    try:
        import cv2
    except ImportError:
        ## Si OpenCV no esta disponible, retornar la imagen sin cambios
        return pil_image

    ## Convertir a array en escala de grises
    arr = np.array(pil_image.convert("L"))

    ## Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(arr)

    ## Volver a RGB (replicar el canal gris 3 veces)
    enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
    return Image.fromarray(enhanced_rgb)


## ============================================================
## Carga de imagenes DICOM de VinDr
## ============================================================

def load_image_as_pil(image_path: str):
    ##
    ## Carga una imagen DICOM de VinDr y la retorna como PIL Image RGB de 8 bits
    ##
    ## VinDr-Mammo distribuye las imagenes en formato DICOM de 16 bits. La
    ## conversion a 8 bits aplica el VOI LUT (windowing) del DICOM si esta
    ## disponible, que es la forma clinicamente correcta de mapear el rango
    ## dinamico (preserva el contraste diagnostico mejor que un min-max simple).
    ##
    from PIL import Image

    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix in (".dcm", ".dicom"):
        return _load_dicom_as_pil(image_path)
    else:
        ## Por si en algun momento se usan PNG preprocesados
        return Image.open(image_path).convert("RGB")


def _load_dicom_as_pil(image_path: str):
    ##
    ## Carga un DICOM y lo convierte a PIL RGB de 8 bits
    ##
    ## Pasos:
    ##   1. Leer pixel_array
    ##   2. Aplicar VOI LUT (windowing) si el DICOM lo define; esto mapea el
    ##      rango dinamico de 16 bits al rango de visualizacion de forma
    ##      clinicamente correcta. Si no esta disponible, cae a min-max.
    ##   3. Manejar inversion MONOCHROME1 (algunos DICOM invierten la escala)
    ##   4. Normalizar a [0, 255] y convertir a RGB
    ##
    from PIL import Image
    import pydicom

    ds = pydicom.dcmread(image_path)
    arr = ds.pixel_array

    ## Intentar aplicar VOI LUT (windowing) usando pydicom
    ## Esto respeta los parametros de ventana definidos por el equipo
    ## (_apply_voi_lut se resolvio a nivel de modulo, con fallback de version)
    if _apply_voi_lut is not None:
        try:
            arr = _apply_voi_lut(arr, ds)
        except Exception:
            ## Si el VOI LUT falla para esta imagen, se usa el array crudo (min-max abajo)
            pass

    arr = arr.astype(np.float32)

    ## Manejar inversion MONOCHROME1 antes de normalizar
    ## (en MONOCHROME1 los valores altos son oscuros; lo invertimos para que
    ## el tejido denso quede claro, como en MONOCHROME2)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    ## Normalizacion min-max a [0, 255]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    arr = arr.astype(np.uint8)

    ## Convertir a RGB (replicar canal gris 3 veces)
    return Image.fromarray(arr).convert("RGB")


## ============================================================
## Mapeo de densidad y BI-RADS de VinDr
## ============================================================

## Mapeo de densidad VinDr (DENSITY A/B/C/D) a indice de clase 0-3
## Las categorias siguen la escala ACR de densidad mamaria:
##   A = casi enteramente grasa, B = densidad fibroglandular dispersa,
##   C = heterogeneamente densa, D = extremadamente densa
DENSITY_TO_INDEX = {
    "DENSITY A": 0,
    "DENSITY B": 1,
    "DENSITY C": 2,
    "DENSITY D": 3,
}
DENSITY_NAMES = [
    "almost entirely fatty",
    "scattered fibroglandular",
    "heterogeneously dense",
    "extremely dense",
]
NUM_DENSITY_CLASSES = 4

## BI-RADS de VinDr: 1-5. Se mapean a indices 0-4 para la clasificacion.
##   BI-RADS 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4
NUM_BIRADS_CLASSES = 5


def birads_to_index(birads: int) -> int:
    ##
    ## Convierte el BI-RADS de VinDr (1-5) a indice de clase (0-4)
    ##
    return int(birads) - 1


def index_to_birads(index: int) -> int:
    ##
    ## Convierte el indice de clase (0-4) de vuelta a BI-RADS (1-5)
    ##
    return int(index) + 1


def density_to_index(density_raw: str) -> Optional[int]:
    ##
    ## Convierte la densidad de VinDr (ej. "DENSITY A") a indice de clase (0-3)
    ## Retorna None si el valor no es reconocido
    ##
    if density_raw is None:
        return None
    key = str(density_raw).strip().upper()
    return DENSITY_TO_INDEX.get(key, None)


## ============================================================
## Loader de VinDr-Mammo
## ============================================================

def load_vindr_records(root: str) -> pd.DataFrame:
    ##
    ## Carga los registros de VinDr-Mammo a nivel de mama
    ##
    ## Retorna un DataFrame con columnas:
    ##   image_path: ruta al DICOM
    ##   birads: BI-RADS 1-5 (entero)
    ##   density_index: indice de densidad 0-3 (o NaN si no reconocido)
    ##   study_id, image_id, laterality, view_position, split: metadatos
    ##
    root = Path(root)

    ## Localizar el CSV de anotaciones a nivel de mama (nombre con guion)
    breast_csv = None
    for name in ["breast-level_annotations.csv", "breast_level_annotations.csv"]:
        if (root / name).exists():
            breast_csv = root / name
            break
    if breast_csv is None:
        logger.warning("VinDr: CSV de anotaciones no encontrado en %s", root)
        return pd.DataFrame()

    df = pd.read_csv(breast_csv)

    ## Normalizar el typo oficial del nombre de columna de vista
    if "view_positition" in df.columns:
        df = df.rename(columns={"view_positition": "view_position"})

    records = []
    n_sin_imagen = 0
    n_sin_birads = 0
    for _, row in df.iterrows():
        study_id = str(row.get("study_id", ""))
        image_id = str(row.get("image_id", ""))
        img_path = root / "images" / study_id / f"{image_id}.dicom"

        if not img_path.exists():
            n_sin_imagen += 1
            continue

        ## BI-RADS: debe ser 1-5
        birads_raw = row.get("breast_birads")
        birads = _parse_birads_number(birads_raw)
        if birads is None or birads < 1 or birads > 5:
            n_sin_birads += 1
            continue

        ## Densidad (indice 0-3, puede ser None si el valor es invalido)
        density_idx = density_to_index(row.get("breast_density"))

        records.append({
            "image_path": str(img_path),
            "birads": birads,
            "density_index": density_idx if density_idx is not None else np.nan,
            "study_id": study_id,
            "image_id": image_id,
            "laterality": str(row.get("laterality", "")),
            "view_position": str(row.get("view_position", "")),
            "split": str(row.get("split", "")),
        })

    if n_sin_imagen > 0:
        logger.warning("VinDr: %d filas sin DICOM en disco (omitidas)", n_sin_imagen)
    if n_sin_birads > 0:
        logger.warning("VinDr: %d filas sin BI-RADS valido (omitidas)", n_sin_birads)

    result = pd.DataFrame(records)
    logger.info("VinDr: %d registros cargados", len(result))
    return result


def _parse_birads_number(raw) -> Optional[int]:
    ##
    ## Extrae el numero BI-RADS de un valor que puede venir como
    ## "BI-RADS 3", "3", o 3. Retorna None si no se puede parsear.
    ##
    import re
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    match = re.search(r"(\d)", str(raw))
    return int(match.group(1)) if match else None


## ============================================================
## Dataset PyTorch para VinDr
## ============================================================

class MammoDataset:
    ## Dataset de VinDr-Mammo para clasificacion BI-RADS + densidad
    ##
    ## Cada item retorna:
    ##   image: tensor [3, H, W] preprocesado para Mammo-CLIP (alta resolucion)
    ##   birads: tensor escalar (indice 0-4, correspondiente a BI-RADS 1-5)
    ##   density: tensor escalar (indice 0-3, o IGNORE_INDEX si falta)
    ##
    ## A diferencia de la version multi-dataset, aqui no hay masking complejo:
    ## todas las imagenes de VinDr tienen BI-RADS. La densidad puede faltar en
    ## casos raros, y en ese caso se marca con IGNORE_INDEX (-100).

    IGNORE_INDEX = -100

    def __init__(self, records: pd.DataFrame, transform, augment: bool = False):
        import torch
        self.torch = torch
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        ## Cargar y preprocesar la imagen (una sola imagen de alta resolucion)
        try:
            pil_img = load_image_as_pil(row["image_path"])
            image = self.transform(pil_img)
        except Exception as e:
            logger.warning("Error cargando %s: %s", row.get("image_path"), e)
            ## Fallback: imagen negra con las dimensiones esperadas
            h = getattr(self.transform, "height", 1520)
            w = getattr(self.transform, "width", 912)
            image = self.torch.zeros(3, h, w)

        ## BI-RADS: indice 0-4 (de BI-RADS 1-5)
        birads_index = birads_to_index(int(row["birads"]))

        ## Densidad: indice 0-3, o IGNORE_INDEX si falta
        density_val = row.get("density_index", np.nan)
        if density_val is None or (isinstance(density_val, float) and np.isnan(density_val)):
            density_index = self.IGNORE_INDEX
        else:
            density_index = int(density_val)

        return {
            "image": image,
            "birads": self.torch.tensor(birads_index, dtype=self.torch.long),
            "density": self.torch.tensor(density_index, dtype=self.torch.long),
            "dataset": "vindr",
        }


def build_vindr_records(dataset_root: str) -> pd.DataFrame:
    ##
    ## Funcion de conveniencia: carga los registros de VinDr desde su raiz
    ##
    ## Parametros:
    ##   dataset_root: ruta a la carpeta de VinDr-Mammo
    ##
    return load_vindr_records(dataset_root)
