# VLM de fusion multimodal (baseline exploratorio)

## Procedencia
Codigo recuperado del material exploratorio de la tesis de maestria. NO estaba
en el control de versiones del proyecto ni en el arbol de trabajo del servidor;
se conservaba solo como adjunto de trabajo. Se incorpora aqui para preservarlo
en git y para servir de punto de partida al trabajo de titulacion. Verificar
contra cualquier version mas reciente antes de construir sobre el.

## Que es esto
Un Vision Language Model multimodal real, anterior a la version publicada de la
tesis. NO es el sistema reportado en C8 (la cascada, que vive en `src/`). Es el
punto de reentrada para el trabajo de titulacion: la fusion imagen-texto que C8
no pudo entrenar y que ahora si es posible con los datos del SIME.

## Por que no vive en deprecated/
No es codigo deprecado. Es un baseline a resucitar. La carpeta `deprecated/`
contiene validadores de datasets y scripts de evaluacion obsoletos, sin relacion
con esta arquitectura.

## Arquitectura
- `VisualEncoder` (models.py): encoder de imagen conmutable. Soporta CLIP
  ViT-L/14 y DINOv2-Large via transformers, y ConvNeXt-Base o ResNet50 via timm.
- `ProjectionMLP` (models.py): MLP de 4 capas que alinea el espacio visual con el
  espacio de embeddings del LLM. Es la pieza central de la fusion.
- LLM Qwen2.5-7B-Instruct con fine-tuning LoRA (via peft).
- `MammoVLM.generate(...)` (models.py): alimenta el LLM con `inputs_embeds`, es
  decir, los rasgos visuales proyectados SI entran al LLM.
- `VisualMoE` (models.py): selector opcional de encoder segun densidad mamaria.
- `train_vlm()` (train.py): loop de fine-tuning multimodal.
- `MammoClassifier` (models.py) y `train_classifier()` (train.py): la rama de
  clasificacion, incluida para referencia.

## Diferencia clave con C8 (src/)
En C8 el generador de reportes recibe SOLO escalares (BI-RADS, densidad,
confianza) y nunca ve la imagen. Aqui el LLM recibe rasgos visuales proyectados.
Esa fusion es exactamente lo que C8 perdio y lo que hay que recuperar.

## Por que se abandono
VinDr-Mammo no tiene texto de reporte pareado con la imagen. Sin pares
imagen-reporte, la `ProjectionMLP` y el LLM no tenian senal de entrenamiento
multimodal, asi que el VLM era inentrenable y se reemplazo por la cascada RAG
como compromiso documentado. No fue un descuido.

## Como retomarlo con el SIME
1. Curar pares imagen-reporte del SIME, con protocolo y criterios de exclusion.
2. Preprocesar la imagen segun el encoder elegido.
3. Entrenar `ProjectionMLP` + LoRA con `train_vlm` sobre los pares.
4. Validar en un hold-out del propio SIME; no asumir transferencia de dominio.
5. Auditar con XAI (Integrated Gradients, Grad-CAM, Insertion AUC) que la fusion
   realmente ancla en la lesion, no solo que el texto suena bien.

## Inventario
- `models.py`: ProjectionMLP, VisualEncoder, VisualMoE, MammoVLM, MammoClassifier.
- `train.py`: train_vlm, train_classifier, y utilidades de entrenamiento.
- `evaluate.py`: evaluacion y metricas.
- `requirements.txt`: dependencias derivadas de los imports reales.

## Advertencias
- Codigo pensado para H200 (BF16, Flash Attention 2 opcional). No es codigo de
  produccion.
- Los encoders de este baseline (CLIP, DINOv2, ConvNeXt) NO son el EfficientNet-B5
  de Mammo-CLIP que usa C8. Decidir conscientemente que encoder usar.
- Este baseline usa semilla unica y no tiene hold-out end-to-end. Corregir ambos
  como parte del trabajo de titulacion.
