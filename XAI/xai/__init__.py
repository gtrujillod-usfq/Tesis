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
