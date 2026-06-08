# MammoVLM: Diagnóstico Mamográfico Asistido por IA

Sistema de análisis mamográfico que integra clasificación BI-RADS/densidad con generación
automática de informes diagnósticos en español. Desarrollado como tesis de maestría en la
Universidad San Francisco de Quito.

---

## Descripción y arquitectura

MammoVLM combina un clasificador visual de alta resolución con un módulo generador de
informes basado en recuperación de literatura médica:

**Clasificador (BI-RADS 1–5 y densidad ACR A–D)**

- **Encoder visual**: Mammo-CLIP (EfficientNet-B5 preentrenado en mamografías VinDr),
  con fine-tuning parcial de los últimos 2 bloques convolucionales.
- **Dual-head de clasificación**: cabezas independientes para BI-RADS y densidad con
  proyección lineal y dropout.
- **Loss ordinal híbrida**: SORD (Soft Ordinal Labels) + QWK (Quadratic Weighted Kappa)
  con penalización asimétrica para subclasificaciones clínicamente peligrosas
  (β = 2.0 para subgrading de BI-RADS 4/5).
- **Threshold tuning clínico**: optimización del umbral de decisión sobre el conjunto de
  validación para maximizar la sensibilidad a casos malignos (BI-RADS ≥ 4).

**Generador de informes (RAG + LLM)**

- **LLM**: Qwen2.5-7B-Instruct (descargado por separado desde Hugging Face).
- **RAG**: índice FAISS sobre literatura médica (ACR BI-RADS Atlas + artículos de referencia)
  con embeddings de `NeuML/pubmedbert-base-embeddings` (cargado vía HuggingFace `transformers`).
- **Diseño de seguridad clínica**: el informe se construye exclusivamente a partir de las
  predicciones del modelo (BI-RADS + densidad + recomendación). No se generan hallazgos
  morfológicos específicos (forma de masa, morfología de calcificaciones) porque el modelo
  no los predice; un informe que no inventa hallazgos es preferible a uno que los alucina.

---

## Resultados principales

### Clasificación (exp08 — experimento definitivo)

| Métrica | Valor |
|---|---|
| Accuracy | 0.6945 |
| Macro F1 | 0.3944 |
| Cohen Kappa | 0.2567 |
| Quadratic Weighted Kappa | 0.4755 |
| AUC-ROC | 0.7500 |

### Threshold tuning (umbral validado sobre conjunto de validación)

| Configuración | Sensibilidad | Especificidad |
|---|---|---|
| Argmax (baseline) | 0.3737 | — |
| Umbral optimizado | **0.7576** | 0.7007 |

### Generador de informes (evaluación factual, n=33 informes)

| Dimensión | Score |
|---|---|
| Fidelidad factual | 0.967 |
| Sin fuga de idioma | 1.000 |
| Sin alucinaciones | 1.000 |
| Completitud estructural | 1.000 |
| Coherencia de recomendación | 0.730 |
| Adherencia terminológica ajustada | 0.910 |

---

## Reproducibilidad

### Datos

El dataset **VinDr-Mammo** no está incluido en este repositorio por licencia restringida.
Para obtenerlo:

1. Registrarse en [PhysioNet](https://physionet.org/content/vindr-mammo/) y aceptar el
   acuerdo de uso de datos.
2. Descargar el dataset y ubicarlo en `data/vindr-mammo/` dentro del directorio del proyecto.

El dataset VinDr-Mammo contiene mamografías digitales con anotaciones BI-RADS y densidad
ACR realizadas por radiólogos expertos del Hospital 108 y el Hospital K (Vietnam).

### Modelos preentrenados

- **Mammo-CLIP** (EfficientNet-B5): descargar desde
  [shawn24/Mammo-CLIP](https://huggingface.co/shawn24/Mammo-CLIP) y ubicar el checkpoint
  en `models/mammo_clip_b5.tar`.
- **Qwen2.5-7B-Instruct**: se descarga automáticamente desde Hugging Face al ejecutar el
  módulo de generación de informes (`Qwen/Qwen2.5-7B-Instruct`).

### Instalación

```bash
git clone https://github.com/gtrujillod-usfq/Tesis.git
cd Tesis
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# PyTorch con CUDA 12.4 (ajustar según GPU disponible):
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### Entrenamiento

Desde el notebook:
```
main.ipynb → Sección 9 (entrenamiento)
```

O en background con el script:
```bash
# 1. Revisar/ajustar train_runner_config.json
# 2. Lanzar:
nohup python train_runner.py > /dev/null 2>&1 &
```

### Evaluación

```
main.ipynb → Sección 9.7  (métricas en test set)
main.ipynb → Sección 11   (pipeline VLM end-to-end)
main.ipynb → Sección 12   (evaluación del generador de informes)
```

### Hardware

El modelo fue entrenado en **NVIDIA H200** (80 GB VRAM). Para reproducir el entrenamiento
se recomienda una GPU con al menos 24 GB de VRAM. La inferencia y evaluación pueden
ejecutarse con menos memoria reduciendo `batch_size` en `train_runner_config.json`.

---

## Cita

Si utilizas este trabajo en tu investigación, por favor cítalo como:

> Trujillo Delgado, G. (2026). *MammoVLM: Diagnóstico Mamográfico Asistido por IA*.
> Tesis de Maestría, Universidad San Francisco de Quito. Director: Noel Pérez-Pérez, PhD.

```bibtex
@mastersthesis{trujillo2026mammovlm,
  author    = {Trujillo Delgado, Geovanny},
  title     = {{MammoVLM: Diagnóstico Mamográfico Asistido por IA}},
  school    = {Universidad San Francisco de Quito},
  year      = {2026},
  type      = {Tesis de Maestría},
  note      = {Director: Noel Pérez-Pérez, PhD}
}
```

---

## Atribuciones y referencias

### Datasets

- **VinDr-Mammo.** Nguyen, H.T., Nguyen, H.Q., Pham, H.H., Lam, K., Le, L.T., Dao, M., & Vu, V.
  (2023). VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in
  full-field digital mammography. *Scientific Data*, 10, 277.
  https://doi.org/10.1038/s41597-023-02100-7

### Modelos preentrenados

- **Mammo-CLIP.** Ghosh, S., Poynton, C.B., Visweswaran, S., & Batmanghelich, K. (2024).
  Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness
  in Mammography. En *MICCAI 2024*, LNCS vol. 15012. Springer.
  https://doi.org/10.1007/978-3-031-72390-2_59
- **Qwen2.5.** Qwen Team, Yang, A., et al. (2025). Qwen2.5 Technical Report. arXiv:2412.15115.
  https://doi.org/10.48550/arXiv.2412.15115
- **EfficientNet.** Tan, M., & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks. En *ICML 2019*, pp. 6105–6114. arXiv:1905.11946

### Métodos y estándares clínicos

- **SORD.** Diaz, R., & Marathe, A. (2019). Soft Labels for Ordinal Regression. En *CVPR 2019*,
  pp. 4738–4747. DOI: 10.1109/CVPR.2019.00487
- **ACR BI-RADS Atlas.** American College of Radiology. (2013). *ACR BI-RADS Atlas: Breast
  Imaging Reporting and Data System* (5.ª ed.). Reston, VA: American College of Radiology.

### Herramientas y librerías principales

- **PyTorch** — framework de aprendizaje profundo.
- **Transformers (Hugging Face)** — carga y uso de Qwen2.5 y del modelo de embeddings RAG.
- **FAISS (Facebook AI Research)** — índice de recuperación vectorial del RAG.
- **pydicom** — lectura y decodificación de imágenes DICOM.
- **efficientnet_pytorch** — backbone EfficientNet-B5 compatible con Mammo-CLIP.

---

## Licencia

Este código se distribuye bajo la licencia [MIT](LICENSE).

**Nota**: el dataset VinDr-Mammo y los pesos de Mammo-CLIP tienen sus propias licencias
independientes; consulta las fuentes originales antes de redistribuirlos.
