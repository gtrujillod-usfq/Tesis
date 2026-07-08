## generate_lesion_input_figures.py
## Genera 10 imagenes de mamografia de VinDr-Mammo con la caja GT de la lesion
## dibujada encima, para usar como bloque de entrada de un diagrama del pipeline
## No entrena ni modifica nada: solo lee datos reales del disco y dibuja
## Usa exactamente el mismo preprocesamiento de visualizacion que las figuras
## XAI existentes (XAI/xai/carga_modelo.py, XAI/xai/visualizacion_xai.py):
## load_image_as_pil + MammoCLIPTransform(1520x912, CLAHE) + escalado de cajas

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TESIS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TESIS_ROOT / "src"))

from data_loading import load_image_as_pil, MammoCLIPTransform

## ============================================================
## Constantes (identicas a XAI/xai/config_xai.py, unica fuente de verdad)
## ============================================================

FINDING_ANNOTATIONS_CSV = TESIS_ROOT / "data" / "vindr-mammo" / "finding_annotations.csv"
VINDR_IMAGES_DIR = TESIS_ROOT / "data" / "vindr-mammo" / "images"
OUT_DIR = TESIS_ROOT / "figures" / "lesiones_input"

## Resolucion de entrada del encoder, misma que usan las figuras XAI existentes
IMAGE_HEIGHT = 1520
IMAGE_WIDTH = 912

SEED = 42
N_CASOS = 10
## Color y grosor de la caja GT, identico a _dibujar_cajas en visualizacion_xai.py
BOX_COLOR = "lime"
BOX_LINEWIDTH = 1.8

## Tope superior de area relativa de la caja: evita elegir cajas que cubren
## casi toda la mama (que no se verian como "una lesion" en el diagrama)
MAX_BOX_AREA_REL = 0.15

## Estudios excluidos tras inspeccion visual: el DICOM original tiene texto
## de lateralidad/vista quemado en el pixel (ej. "R-MLO", "R"), anterior a
## cualquier dibujo de este script. Se excluyen para cumplir el requisito de
## no tener texto sobre la imagen
EXCLUDE_STUDY_IDS = {
    "da6ce640b22100d327694fac20224427",
    "e11cb06c657324160b15e72be89f6c28",
}


def _a_gris(img_tensor):
    ## Tensor [3, H, W] (normalizado ImageNet + CLAHE) -> numpy [H, W] en [0, 1]
    ## Identico a _a_gris() en XAI/xai/visualizacion_xai.py: promedio de canales
    ## RGB y min-max por imagen, para visualizacion consistente con las figuras
    ## XAI ya existentes
    arr = img_tensor.cpu().numpy()
    gray = arr.mean(axis=0)
    lo, hi = gray.min(), gray.max()
    return (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)


def seleccionar_casos(findings_df):
    ##
    ## Selecciona 10 casos distintos (un image_id por study_id) con lesion
    ## claramente visible, priorizando BI-RADS 4/5 y alternando entre masas
    ## y calcificaciones sospechosas para diversidad
    ##
    coord_cols = ["xmin", "ymin", "xmax", "ymax"]
    df = findings_df.dropna(subset=coord_cols).copy()

    ## Solo hallazgos de una unica categoria (evita ambiguedad visual/etiqueta
    ## multiple sobre el mismo bounding box en el diagrama)
    df = df[df["finding_categories"].isin(["['Mass']", "['Suspicious Calcification']"])]

    ## Priorizar categorias BI-RADS sospechosas
    df = df[df["finding_birads"].isin(["BI-RADS 5", "BI-RADS 4"])]

    ## Area de la caja relativa al tamano de la imagen original (proxy de
    ## "lesion claramente visible": ni un punto minusculo ni casi toda la mama)
    box_area = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
    img_area = df["width"] * df["height"]
    df["box_area_rel"] = box_area / img_area
    df = df[df["box_area_rel"] <= MAX_BOX_AREA_REL]

    ## Verificar que el DICOM exista en disco (VinDr puede tener descargas parciales)
    def _existe(row):
        return (VINDR_IMAGES_DIR / row["study_id"] / f"{row['image_id']}.dicom").exists()

    df["file_exists"] = df.apply(_existe, axis=1)
    df = df[df["file_exists"]]

    ## Excluir estudios con texto quemado en el pixel (ver EXCLUDE_STUDY_IDS)
    df = df[~df["study_id"].isin(EXCLUDE_STUDY_IDS)]

    ## Orden de prioridad: BI-RADS 5 antes que 4, y dentro de cada uno, caja
    ## mas grande primero (mas clara y visible en la figura)
    df["birads_rank"] = df["finding_birads"].map({"BI-RADS 5": 0, "BI-RADS 4": 1})
    df = df.sort_values(["birads_rank", "box_area_rel"], ascending=[True, False])

    ## Alternar entre Mass y Suspicious Calcification para asegurar diversidad
    ## de tipo de hallazgo, tomando como maximo un caso por study_id (10 casos
    ## distintos, no 10 vistas del mismo paciente/estudio)
    seleccionados = []
    studies_usados = set()
    colas = {
        "['Mass']": df[df["finding_categories"] == "['Mass']"].itertuples(),
        "['Suspicious Calcification']": df[
            df["finding_categories"] == "['Suspicious Calcification']"
        ].itertuples(),
    }
    categorias_ciclo = ["['Mass']", "['Suspicious Calcification']"]
    idx_ciclo = 0
    agotadas = set()

    while len(seleccionados) < N_CASOS and len(agotadas) < len(categorias_ciclo):
        cat = categorias_ciclo[idx_ciclo % len(categorias_ciclo)]
        idx_ciclo += 1
        if cat in agotadas:
            continue
        avanzo = False
        for fila in colas[cat]:
            if fila.study_id in studies_usados:
                continue
            seleccionados.append(fila)
            studies_usados.add(fila.study_id)
            avanzo = True
            break
        if not avanzo:
            agotadas.add(cat)

    return seleccionados[:N_CASOS]


def main():
    findings_df = pd.read_csv(FINDING_ANNOTATIONS_CSV)
    print(f"Anotaciones de hallazgos cargadas: {len(findings_df)} filas ({FINDING_ANNOTATIONS_CSV})")

    casos = seleccionar_casos(findings_df)
    print(f"Casos seleccionados: {len(casos)}")
    assert len(casos) == N_CASOS, f"Se esperaban {N_CASOS} casos, se obtuvieron {len(casos)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = MammoCLIPTransform(
        height=IMAGE_HEIGHT, width=IMAGE_WIDTH, augment=False, use_clahe=True
    )

    trazabilidad = []
    print()
    print("=" * 70)
    print("CASOS SELECCIONADOS")
    print("=" * 70)

    for i, fila in enumerate(casos, start=1):
        image_path = VINDR_IMAGES_DIR / fila.study_id / f"{fila.image_id}.dicom"

        ## Cargar y preprocesar la imagen (identico al pipeline de inferencia
        ## y a las figuras XAI existentes)
        pil_img = load_image_as_pil(str(image_path))
        img_tensor = transform(pil_img)
        gray = _a_gris(img_tensor)

        ## Escalar la caja GT al espacio de la imagen de entrada (1520x912),
        ## con estiramiento directo identico a T.Resize((H, W)) -- misma formula
        ## que preparar_cajas_test() en XAI/xai/metricas_clasificador.py
        escala_x = IMAGE_WIDTH / fila.width
        escala_y = IMAGE_HEIGHT / fila.height
        xmin_s = int(round(fila.xmin * escala_x))
        xmax_s = int(round(fila.xmax * escala_x))
        ymin_s = int(round(fila.ymin * escala_y))
        ymax_s = int(round(fila.ymax * escala_y))
        xmin_s = max(0, min(xmin_s, IMAGE_WIDTH - 1))
        xmax_s = max(0, min(xmax_s, IMAGE_WIDTH - 1))
        ymin_s = max(0, min(ymin_s, IMAGE_HEIGHT - 1))
        ymax_s = max(0, min(ymax_s, IMAGE_HEIGHT - 1))

        ## Dibujar: mamografia en gris + caja GT en verde, sin texto ni ejes
        fig, ax = plt.subplots(figsize=(IMAGE_WIDTH / 150, IMAGE_HEIGHT / 150), dpi=150)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        rect = mpatches.Rectangle(
            (xmin_s, ymin_s), xmax_s - xmin_s, ymax_s - ymin_s,
            linewidth=BOX_LINEWIDTH, edgecolor=BOX_COLOR, facecolor="none",
        )
        ax.add_patch(rect)
        ax.axis("off")
        plt.tight_layout(pad=0)

        out_name = f"lesion_{i:02d}.png"
        out_path = OUT_DIR / out_name
        fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        registro = {
            "archivo": out_name,
            "study_id": fila.study_id,
            "image_id": fila.image_id,
            "laterality": fila.laterality,
            "view_position": fila.view_position,
            "breast_birads": fila.breast_birads,
            "finding_birads": fila.finding_birads,
            "finding_categories": fila.finding_categories,
            "box_original_xmin": float(fila.xmin),
            "box_original_ymin": float(fila.ymin),
            "box_original_xmax": float(fila.xmax),
            "box_original_ymax": float(fila.ymax),
            "box_escalada_xmin": xmin_s,
            "box_escalada_ymin": ymin_s,
            "box_escalada_xmax": xmax_s,
            "box_escalada_ymax": ymax_s,
            "resolucion_imagen_guardada": [IMAGE_HEIGHT, IMAGE_WIDTH],
        }
        trazabilidad.append(registro)

        print(f"  {out_name}: {fila.finding_birads}  {fila.finding_categories}  "
              f"study_id={fila.study_id}  image_id={fila.image_id}")

    ## Guardar JSON de trazabilidad
    traza_path = OUT_DIR / "trazabilidad.json"
    with open(traza_path, "w") as f:
        json.dump(trazabilidad, f, indent=2, default=str)

    print()
    print(f"Imagenes guardadas en: {OUT_DIR}")
    print(f"Trazabilidad guardada en: {traza_path}")


if __name__ == "__main__":
    main()
