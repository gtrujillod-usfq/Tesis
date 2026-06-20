## ddsm_overlay.py
## Carga y procesamiento del dataset DDSM con anotaciones OVERLAY
## para el proyecto MammoVLM (Tesis de maestria)
##
## Conversion del codigo MATLAB Read Overlay (LastV1) a Python.
## Funciones principales:
##   - parse_overlay: parsea un archivo .OVERLAY y reconstruye los contornos
##     de las lesiones desde chain code de Freeman (8-conectividad)
##   - find_image_for_overlay: mapea una ruta .OVERLAY a su imagen .LJPEG.png
##   - draw_lesion_contour: dibuja el contorno sobre la imagen PIL
##   - crop_lesion_roi: recorta la region de interes de la lesion
##   - load_ddsm_records: carga todos los pares (imagen, overlay) del dataset
##
## Estructura del dataset en disco:
##   data/6 DDSM/{benign|cancer} cases/{benign|cancer}_XX/caseXXXX/*.LJPEG.png
##   data/6 DDSM/{benign|cancer} cases/{benigns|cancers}/{benign|cancer}_XX/caseXXXX/*.OVERLAY
##
## Referencia: DDSM - Digital Database for Screening Mammography
## http://www.eng.usf.edu/cvprg/Mammography/Database.html

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

## Tabla de desplazamiento para Freeman chain code 8-conectividad (codigos 0-7)
## xC[i] = desplazamiento en X (columna) para el codigo i
## yC[i] = desplazamiento en Y (fila, aumenta hacia abajo) para el codigo i
##
## Codigo:  0   1   2   3   4   5   6   7
## Grafico: ^  ↗   →  ↘   ↓  ↙   ←  ↖
_CHAIN_DX = [0,  1,  1,  1,  0, -1, -1, -1]
_CHAIN_DY = [-1, -1, 0,  1,  1,  1,  0, -1]


## ============================================================
## Estructuras de datos
## ============================================================

@dataclass
class LesionContour:
    ## Contorno reconstruido de una lesion individual
    abnormality_id: int          ## Indice de la anomalia (1-based) dentro del overlay
    outline_id: int              ## Indice del outline dentro de la anomalia (1-based)
    x: np.ndarray                ## Coordenadas X del contorno (columna en la imagen)
    y: np.ndarray                ## Coordenadas Y del contorno (fila en la imagen)

    @property
    def points(self) -> np.ndarray:
        ## Retorna array [[x0,y0], [x1,y1], ...] shape (N, 2)
        return np.column_stack([self.x, self.y])

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        ## Retorna (x_min, y_min, x_max, y_max) del bounding box del contorno
        return int(self.x.min()), int(self.y.min()), int(self.x.max()), int(self.y.max())


@dataclass
class LesionAnnotation:
    ## Anotacion clinica completa de una anomalia en un overlay
    abnormality_id: int
    pathology: str               ## "MALIGNANT" o "BENIGN" (o "BENIGN_WITHOUT_CALLBACK")
    assessment: int              ## BI-RADS 1-5 (puede ser 0 si no se pudo parsear)
    subtlety: int                ## Escala 1-5 de dificultad de deteccion
    lesion_type: str             ## "MASS" o "CALCIFICATION"
    ## Atributos de masa
    mass_shape: str = ""         ## IRREGULAR, OVAL, ROUND, LOBULATED, etc.
    mass_margins: str = ""       ## SPICULATED, CIRCUMSCRIBED, ILL_DEFINED, etc.
    ## Atributos de calcificacion
    calc_type: str = ""          ## PLEOMORPHIC, AMORPHOUS, etc.
    calc_distribution: str = ""  ## CLUSTERED, LINEAR, SEGMENTAL, etc.
    contours: List[LesionContour] = field(default_factory=list)

    @property
    def is_malignant(self) -> bool:
        return "MALIGNANT" in self.pathology.upper()

    @property
    def label(self) -> int:
        ## 1 = maligno, 0 = benigno
        return 1 if self.is_malignant else 0


@dataclass
class OverlayData:
    ## Datos completos de un archivo .OVERLAY
    overlay_path: Path
    image_path: Optional[Path]       ## Ruta a la imagen .LJPEG.png correspondiente
    total_abnormalities: int
    lesions: List[LesionAnnotation]  ## Una entrada por anomalia

    @property
    def all_contours(self) -> List[LesionContour]:
        ## Retorna todos los contornos de todas las lesiones
        return [c for les in self.lesions for c in les.contours]


## ============================================================
## Parseo del archivo OVERLAY
## ============================================================

def parse_overlay(overlay_path) -> OverlayData:
    ##
    ## Parsea un archivo .OVERLAY del dataset DDSM y reconstruye los
    ## contornos de las lesiones desde la representacion en chain code.
    ##
    ## Retorna un OverlayData con todas las lesiones y sus contornos.
    ##
    overlay_path = Path(overlay_path)
    with open(overlay_path, "r") as f:
        text = f.read()

    tokens = text.split()
    image_path = find_image_for_overlay(overlay_path)

    total_abnormalities = _get_token_after(tokens, "TOTAL_ABNORMALITIES", default=0, cast=int)

    lesions = []
    for ab_idx in range(1, total_abnormalities + 1):
        lesion = _parse_abnormality(tokens, ab_idx, overlay_path)
        lesions.append(lesion)

    return OverlayData(
        overlay_path=overlay_path,
        image_path=image_path,
        total_abnormalities=total_abnormalities,
        lesions=lesions,
    )


def _parse_abnormality(tokens: List[str], ab_idx: int, overlay_path: Path) -> LesionAnnotation:
    ##
    ## Extrae la anotacion de la anomalia ab_idx desde la lista de tokens.
    ## Los campos ASSESSMENT, SUBTLETY, PATHOLOGY aparecen en orden dentro
    ## de cada bloque ABNORMALITY.
    ##
    ## Busca el ab_idx-esimo token "ABNORMALITY" y extrae los campos
    ## hasta el siguiente bloque "ABNORMALITY" o fin del archivo.
    ##
    ab_positions = [i for i, t in enumerate(tokens) if t == "ABNORMALITY"]

    if ab_idx - 1 >= len(ab_positions):
        logger.warning("%s: anomalia %d no encontrada", overlay_path.name, ab_idx)
        return LesionAnnotation(abnormality_id=ab_idx, pathology="UNKNOWN",
                                assessment=0, subtlety=0, lesion_type="UNKNOWN")

    start = ab_positions[ab_idx - 1]
    end = ab_positions[ab_idx] if ab_idx < len(ab_positions) else len(tokens)
    block = tokens[start:end]

    ## Extrae campos del bloque
    pathology = _get_token_after(block, "PATHOLOGY", default="UNKNOWN")
    assessment = _get_token_after(block, "ASSESSMENT", default=0, cast=int)
    subtlety = _get_token_after(block, "SUBTLETY", default=0, cast=int)

    ## Tipo de lesion y atributos especificos
    lesion_type, mass_shape, mass_margins, calc_type, calc_dist = _parse_lesion_type(block)

    ## Reconstruir contornos desde chain codes para esta anomalia
    contours = _reconstruct_contours(tokens, ab_idx)

    return LesionAnnotation(
        abnormality_id=ab_idx,
        pathology=pathology,
        assessment=assessment,
        subtlety=subtlety,
        lesion_type=lesion_type,
        mass_shape=mass_shape,
        mass_margins=mass_margins,
        calc_type=calc_type,
        calc_distribution=calc_dist,
        contours=contours,
    )


def _parse_lesion_type(block: List[str]) -> Tuple[str, str, str, str, str]:
    ##
    ## Extrae tipo de lesion y sus atributos del bloque de tokens de una anomalia.
    ## El formato en el OVERLAY es:
    ##   LESION_TYPE MASS SHAPE <shape> MARGINS <margins>
    ## o bien:
    ##   LESION_TYPE CALCIFICATION TYPE <type> DISTRIBUTION <distribution>
    ##
    lesion_type = mass_shape = mass_margins = calc_type = calc_dist = ""
    try:
        lt_idx = block.index("LESION_TYPE")
    except ValueError:
        return lesion_type, mass_shape, mass_margins, calc_type, calc_dist

    if lt_idx + 1 < len(block):
        lesion_type = block[lt_idx + 1]

    if lesion_type == "MASS":
        mass_shape = _get_token_after(block, "SHAPE", default="")
        mass_margins = _get_token_after(block, "MARGINS", default="")
    elif lesion_type == "CALCIFICATION":
        calc_type = _get_token_after(block, "TYPE", default="")
        calc_dist = _get_token_after(block, "DISTRIBUTION", default="")

    return lesion_type, mass_shape, mass_margins, calc_type, calc_dist


def _reconstruct_contours(tokens: List[str], ab_idx: int) -> List[LesionContour]:
    ##
    ## Reconstruye los contornos de la anomalia ab_idx desde los chain codes
    ## del archivo OVERLAY completo.
    ##
    ## Estructura en el OVERLAY:
    ##   TOTAL_OUTLINES <N>         <- para la anomalia ab_idx
    ##   BOUNDARY                   <- inicio del contorno (puede repetirse N veces)
    ##   <x0> <y0> <c1> <c2> ... #  <- punto inicial + chain codes + terminador
    ##
    ## La logica de busqueda sigue el mismo orden que edge_reconstruction.m:
    ## los bloques BOUNDARY/CORE aparecen en orden para cada anomalia.
    ##
    ## Encontrar todos los TOTAL_OUTLINES y BOUNDARY del archivo
    total_outlines_positions = [i for i, t in enumerate(tokens) if t == "TOTAL_OUTLINES"]
    boundary_positions = [i for i, t in enumerate(tokens) if t in ("BOUNDARY", "CORE")]
    hash_positions = [i for i, t in enumerate(tokens) if t == "#"]

    if ab_idx - 1 >= len(total_outlines_positions):
        return []

    n_outlines_pos = total_outlines_positions[ab_idx - 1]
    try:
        n_outlines = int(tokens[n_outlines_pos + 1])
    except (IndexError, ValueError):
        return []

    ## Acumular el offset de outlines de anomalias anteriores
    outline_offset = 0
    for prev_idx in range(ab_idx - 1):
        if prev_idx < len(total_outlines_positions):
            pos = total_outlines_positions[prev_idx]
            try:
                outline_offset += int(tokens[pos + 1])
            except (IndexError, ValueError):
                pass

    contours = []
    for out_i in range(n_outlines):
        global_out_idx = outline_offset + out_i
        if global_out_idx >= len(boundary_positions):
            break
        bp = boundary_positions[global_out_idx]
        if global_out_idx >= len(hash_positions):
            break
        hp = hash_positions[global_out_idx]

        ## Tokens del contorno: x0, y0, c1, c2, ..., (hasta #)
        contour_tokens = tokens[bp + 1: hp]
        if len(contour_tokens) < 2:
            continue

        try:
            x0 = int(contour_tokens[0])
            y0 = int(contour_tokens[1])
            chain_codes = [int(c) for c in contour_tokens[2:] if c.lstrip("-").isdigit()]
        except (ValueError, IndexError):
            logger.warning("Error parseando contorno %d de anomalia %d", out_i + 1, ab_idx)
            continue

        ## Reconstruir coordenadas desde chain code (equivalente a extract_curve en MATLAB)
        xs = [x0]
        ys = [y0]
        cx, cy = x0, y0
        for code in chain_codes:
            if 0 <= code <= 7:
                cx += _CHAIN_DX[code]
                cy += _CHAIN_DY[code]
                xs.append(cx)
                ys.append(cy)

        contours.append(LesionContour(
            abnormality_id=ab_idx,
            outline_id=out_i + 1,
            x=np.array(xs, dtype=np.int32),
            y=np.array(ys, dtype=np.int32),
        ))

    return contours


## ============================================================
## Utilidades de paths
## ============================================================

def find_image_for_overlay(overlay_path) -> Optional[Path]:
    ##
    ## Dado un path de .OVERLAY, retorna el path de la imagen .LJPEG.png
    ## correspondiente.
    ##
    ## Regla de mapeo en el dataset 6 DDSM:
    ##   OVERLAY: .../cancer cases/cancers/cancer_04/case1081/A_X.VIEW.OVERLAY
    ##   Imagen:  .../cancer cases/cancer_04/case1081/A_X.VIEW.LJPEG.png
    ##
    ## Es decir: se sube un nivel (eliminar el directorio {benigns|cancers})
    ## y se cambia la extension de .OVERLAY a .LJPEG.png
    ##
    overlay_path = Path(overlay_path)
    ## El directorio {benigns|cancers} esta 3 niveles arriba del archivo.
    ## Estructura del OVERLAY: .../{benign|cancer} cases/{benigns|cancers}/{category}/{case}/file.OVERLAY
    ## Estructura de la imagen: .../{benign|cancer} cases/{category}/{case}/file.LJPEG.png
    stem = overlay_path.stem  ## Ej: "A_1081_1.RIGHT_CC"
    case_dir = overlay_path.parent           ## .../cancers/cancer_04/case1081
    ## Subir 3 niveles para llegar al folder "{benign|cancer} cases"
    top_category_dir = case_dir.parent.parent.parent
    image_path = top_category_dir / case_dir.parent.name / case_dir.name / f"{stem}.LJPEG.png"
    if image_path.exists():
        return image_path

    ## Fallback: la imagen puede estar junto al OVERLAY (como en el folder Read Overlay)
    for ext in (".LJPEG.png", ".jpg", ".png"):
        alt = overlay_path.with_suffix("").with_suffix(ext)
        if alt.exists():
            return alt
        alt2 = overlay_path.parent / f"{stem}{ext}"
        if alt2.exists():
            return alt2

    return None


## ============================================================
## Visualizacion
## ============================================================

def draw_lesion_contour(pil_image, overlay_data: OverlayData,
                        color=(255, 0, 0), width: int = 2):
    ##
    ## Dibuja los contornos de todas las lesiones del overlay sobre la imagen PIL.
    ##
    ## Retorna una copia de la imagen con los contornos superpuestos.
    ## El color indica patologia: rojo por defecto (maligno), verde para benigno.
    ##
    from PIL import ImageDraw

    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for lesion in overlay_data.lesions:
        ## Color por patologia si no se especifico uno
        lesion_color = (255, 0, 0) if lesion.is_malignant else (0, 200, 0)
        lesion_color = color if color != (255, 0, 0) else lesion_color

        for contour in lesion.contours:
            if len(contour.x) < 2:
                continue
            ## Crear lista de puntos para PIL: [(x0,y0), (x1,y1), ...]
            pts = list(zip(contour.x.tolist(), contour.y.tolist()))
            ## Cerrar el contorno conectando el ultimo punto con el primero
            pts.append(pts[0])
            draw.line(pts, fill=lesion_color, width=width)

    return img


def crop_lesion_roi(pil_image, contour: LesionContour,
                    padding: int = 20) -> Tuple:
    ##
    ## Recorta la region de interes (ROI) alrededor del bounding box del contorno.
    ##
    ## Retorna (roi_image, (x_min, y_min, x_max, y_max)) donde las coordenadas
    ## son las del recorte en la imagen original (con padding incluido).
    ##
    from PIL import Image

    img_w, img_h = pil_image.size
    x_min, y_min, x_max, y_max = contour.bbox

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img_w, x_max + padding)
    y_max = min(img_h, y_max + padding)

    roi = pil_image.crop((x_min, y_min, x_max, y_max))
    return roi, (x_min, y_min, x_max, y_max)


def create_lesion_mask(pil_image, contour: LesionContour) -> np.ndarray:
    ##
    ## Crea una mascara binaria (0/1) del tamano de la imagen original con
    ## la region interior al contorno de la lesion marcada con 1.
    ##
    from PIL import Image, ImageDraw

    img_w, img_h = pil_image.size
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)

    pts = list(zip(contour.x.tolist(), contour.y.tolist()))
    if len(pts) >= 3:
        draw.polygon(pts, fill=1)
    elif len(pts) >= 2:
        draw.line(pts, fill=1, width=3)

    return np.array(mask_img, dtype=np.uint8)


## ============================================================
## Carga del dataset completo
## ============================================================

def load_ddsm_records(ddsm_root) -> List[dict]:
    ##
    ## Escanea el dataset 6 DDSM y carga todos los pares (imagen, overlay)
    ## con sus metadatos clinicos.
    ##
    ## Retorna una lista de diccionarios con los campos:
    ##   image_path:    Path a la imagen .LJPEG.png
    ##   overlay_path:  Path al archivo .OVERLAY
    ##   case_id:       Identificador del caso (ej. "case1081")
    ##   category:      "benign" o "cancer" (segun carpeta padre)
    ##   pathology:     "MALIGNANT" o "BENIGN" (del overlay)
    ##   assessment:    BI-RADS 1-5 (del overlay)
    ##   subtlety:      Subtlety 1-5 (del overlay)
    ##   lesion_type:   "MASS" o "CALCIFICATION"
    ##   mass_shape:    Forma de la masa (si aplica)
    ##   mass_margins:  Margenes de la masa (si aplica)
    ##   view:          Vista mamografica (LEFT_CC, LEFT_MLO, etc.)
    ##   n_abnormalities: Numero de anomalias en el overlay
    ##   overlay_data:  Objeto OverlayData completo (contiene contornos)
    ##
    ddsm_root = Path(ddsm_root)
    records = []

    overlay_files = list(ddsm_root.rglob("*.OVERLAY"))
    logger.info("DDSM: encontrados %d archivos OVERLAY en %s", len(overlay_files), ddsm_root)

    for ov_path in overlay_files:
        ## Determinar categoria (benign/cancer) desde el path
        parts = [p.lower() for p in ov_path.parts]
        if any("cancer" in p for p in parts):
            category = "cancer"
        else:
            category = "benign"

        try:
            ov_data = parse_overlay(ov_path)
        except Exception as e:
            logger.warning("Error parseando %s: %s", ov_path, e)
            continue

        if ov_data.image_path is None:
            logger.debug("Sin imagen para %s", ov_path.name)

        ## Extraer la vista desde el nombre del archivo
        ## Formato: A_XXXX_X.VIEW.OVERLAY  →  VIEW = LEFT_CC, etc.
        name_parts = ov_path.stem.split(".")
        view = name_parts[1] if len(name_parts) >= 2 else ""

        case_id = ov_path.parent.name

        ## Crear un registro por lesion (anomalia)
        for lesion in ov_data.lesions:
            records.append({
                "image_path": ov_data.image_path,
                "overlay_path": ov_path,
                "case_id": case_id,
                "category": category,
                "pathology": lesion.pathology,
                "assessment": lesion.assessment,
                "subtlety": lesion.subtlety,
                "lesion_type": lesion.lesion_type,
                "mass_shape": lesion.mass_shape,
                "mass_margins": lesion.mass_margins,
                "calc_type": lesion.calc_type,
                "calc_distribution": lesion.calc_distribution,
                "view": view,
                "n_abnormalities": ov_data.total_abnormalities,
                "abnormality_id": lesion.abnormality_id,
                "n_contours": len(lesion.contours),
                "label": lesion.label,
                "overlay_data": ov_data,
            })

    logger.info("DDSM: %d registros de lesiones cargados", len(records))
    return records


## ============================================================
## Dataset PyTorch para DDSM con regiones de lesiones
## ============================================================

class DDSMDataset:
    ## Dataset de DDSM para MammoVLM con contornos de lesiones
    ##
    ## Cada item retorna:
    ##   image:         tensor [3, H, W] preprocesado para Mammo-CLIP
    ##   roi:           tensor [3, H_roi, W_roi] del recorte de la lesion (opcional)
    ##   mask:          tensor [1, H, W] mascara binaria de la lesion
    ##   label:         tensor escalar (0=benigno, 1=maligno)
    ##   assessment:    tensor escalar BI-RADS (0-4, de 1-5)
    ##   meta:          dict con metadatos (case_id, view, lesion_type, etc.)

    def __init__(self, records: List[dict], transform=None, roi_size: int = 224,
                 return_roi: bool = True, return_mask: bool = True):
        ##
        ## Parametros:
        ##   records:     lista retornada por load_ddsm_records()
        ##   transform:   transformacion para la imagen completa (MammoCLIPTransform)
        ##   roi_size:    tamano al que se redimensiona el ROI de la lesion
        ##   return_roi:  si True, incluye el ROI recortado en cada item
        ##   return_mask: si True, incluye la mascara de la lesion en cada item
        ##
        ## Filtra registros sin imagen en disco
        self.records = [r for r in records if r["image_path"] is not None
                        and Path(r["image_path"]).exists()]
        self.transform = transform
        self.roi_size = roi_size
        self.return_roi = return_roi
        self.return_mask = return_mask
        n_skipped = len(records) - len(self.records)
        if n_skipped > 0:
            logger.warning("DDSMDataset: %d registros sin imagen en disco omitidos", n_skipped)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import torch
        from PIL import Image

        rec = self.records[idx]
        img_path = rec["image_path"]
        ov_data: OverlayData = rec["overlay_data"]
        ab_id = rec["abnormality_id"]

        ## Cargar imagen
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Error cargando %s: %s", img_path, e)
            h = 1520 if self.transform is None else getattr(self.transform, "height", 1520)
            w = 912  if self.transform is None else getattr(self.transform, "width", 912)
            return self._empty_item(h, w, rec)

        img_w, img_h = pil_img.size

        ## Obtener los contornos de la lesion correspondiente
        lesion = next((l for l in ov_data.lesions if l.abnormality_id == ab_id), None)
        contours = lesion.contours if lesion else []

        ## Imagen completa con transformacion
        if self.transform is not None:
            image_tensor = self.transform(pil_img)
        else:
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(pil_img)

        item = {
            "image": image_tensor,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "assessment": torch.tensor(max(0, rec["assessment"] - 1), dtype=torch.long),
            "dataset": "ddsm",
            "meta": {
                "case_id": rec["case_id"],
                "view": rec["view"],
                "lesion_type": rec["lesion_type"],
                "pathology": rec["pathology"],
                "mass_shape": rec["mass_shape"],
                "mass_margins": rec["mass_margins"],
                "image_path": str(img_path),
            },
        }

        ## ROI de la lesion (recorte del bounding box)
        if self.return_roi and contours:
            roi = self._build_roi(pil_img, contours[0])
            item["roi"] = roi

        ## Mascara binaria de la lesion
        if self.return_mask and contours:
            mask = self._build_mask(pil_img, contours, img_h, img_w)
            item["mask"] = mask

        return item

    def _build_roi(self, pil_img, contour: LesionContour):
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        roi, _ = crop_lesion_roi(pil_img, contour, padding=20)
        roi = roi.resize((self.roi_size, self.roi_size), Image.BILINEAR)
        if self.transform is not None:
            return self.transform(roi)
        return TF.to_tensor(roi)

    def _build_mask(self, pil_img, contours: List[LesionContour], img_h: int, img_w: int):
        import torch

        combined = np.zeros((img_h, img_w), dtype=np.uint8)
        for contour in contours:
            combined = np.maximum(combined, create_lesion_mask(pil_img, contour))
        mask_tensor = torch.from_numpy(combined).unsqueeze(0).float()
        return mask_tensor

    def _empty_item(self, h: int, w: int, rec: dict):
        import torch
        return {
            "image": torch.zeros(3, h, w),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "assessment": torch.tensor(0, dtype=torch.long),
            "dataset": "ddsm",
            "meta": {"case_id": rec.get("case_id", ""), "view": rec.get("view", ""),
                     "lesion_type": rec.get("lesion_type", ""), "pathology": rec.get("pathology", ""),
                     "image_path": str(rec.get("image_path", ""))},
        }


## ============================================================
## Utilidades auxiliares
## ============================================================

def _get_token_after(tokens: List[str], keyword: str, default=None, cast=None):
    ##
    ## Retorna el token inmediatamente despues de la primera aparicion de keyword.
    ##
    try:
        idx = tokens.index(keyword)
        val = tokens[idx + 1]
        return cast(val) if cast is not None else val
    except (ValueError, IndexError):
        return default
    except Exception:
        return default


def summarize_dataset(ddsm_root) -> dict:
    ##
    ## Imprime un resumen estadistico del dataset DDSM.
    ## Util para entender la distribucion de clases antes de entrenar.
    ##
    records = load_ddsm_records(ddsm_root)
    if not records:
        return {}

    n_total = len(records)
    n_malignant = sum(1 for r in records if r["label"] == 1)
    n_benign = n_total - n_malignant
    n_with_image = sum(1 for r in records if r["image_path"] is not None
                       and Path(r["image_path"]).exists())
    n_mass = sum(1 for r in records if r["lesion_type"] == "MASS")
    n_calc = sum(1 for r in records if r["lesion_type"] == "CALCIFICATION")
    views = {}
    for r in records:
        views[r["view"]] = views.get(r["view"], 0) + 1

    summary = {
        "total_lesions": n_total,
        "malignant": n_malignant,
        "benign": n_benign,
        "with_image_on_disk": n_with_image,
        "mass": n_mass,
        "calcification": n_calc,
        "views": views,
    }
    print(f"DDSM Dataset Summary ({ddsm_root})")
    print(f"  Total lesiones:     {n_total}")
    print(f"  Malignas:           {n_malignant} ({100*n_malignant/n_total:.1f}%)")
    print(f"  Benignas:           {n_benign} ({100*n_benign/n_total:.1f}%)")
    print(f"  Con imagen en disco:{n_with_image}")
    print(f"  Masas:              {n_mass}")
    print(f"  Calcificaciones:    {n_calc}")
    print(f"  Por vista:          {views}")
    return summary
