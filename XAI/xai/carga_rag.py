## carga_rag.py
## Paso 1b: carga del pipeline RAG (indice FAISS + PubMedBERT) y del LLM Qwen2.5-7B.
## El clasificador MammoVLM se carga por separado en carga_modelo.py (Paso 1a).
##
## Decisiones de diseno:
##   - CorpusIndexer.build_index() carga el cache en disco sin reconstruir
##     el indice (usa _load_cached_index() internamente).
##   - load_llm_for_generation() de report_generator.py carga Qwen2.5-7B en bfloat16.
##   - ReportGenerator se configura con rag_top_k=RAG_TOP_K (3, igual que exp08).
##   - cargar_nli() carga mDeBERTa-v3-base-mnli-xnli para grounding de sentencias.
##   - El indice NLI de la clase 'entailment' se resuelve desde model.config.id2label.

import sys
import logging
from pathlib import Path

import torch

_XAI_DIR    = Path(__file__).resolve().parent
_TESIS_ROOT = _XAI_DIR.parent.parent
_SRC_DIR    = _TESIS_ROOT / 'src'

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rag import EmbeddingModel, CorpusIndexer, ReportRetriever
from report_generator import ReportGenerator, load_llm_for_generation

from config_xai import (
    RAG_INDEX_DIR,
    LITERATURE_DIR,
    RAG_TOP_K,
    QWEN_MODEL_ID,
    QWEN_DTYPE,
)

logger = logging.getLogger(__name__)

## Identificador del modelo NLI para grounding multilingue
NLI_MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'


## =========================================================
## cargar_pipeline_rag
## =========================================================

def cargar_pipeline_rag(device='auto'):
    """
    Carga el indice FAISS desde disco y construye el pipeline RAG con Qwen2.5-7B.

    El indice se lee desde RAG_INDEX_DIR (solo-lectura). El modelo LLM se carga
    en bfloat16 usando load_llm_for_generation() de src/report_generator.py.

    Parametros
    ----------
    device : str
        'auto', 'cuda', 'cpu'. Aplica al LLM y al modelo de embeddings.

    Retorna
    -------
    generator : ReportGenerator
        Pipeline completo listo para generate().
    retriever : ReportRetriever
        Recuperador FAISS con rag_top_k=3 (igual que exp08).
    indexer : CorpusIndexer
        Indexador con el indice cargado (acceso a indexer.chunks).
    llm : AutoModelForCausalLM
        Modelo Qwen2.5-7B en bfloat16.
    tokenizer : AutoTokenizer
        Tokenizador de Qwen2.5-7B.
    """
    if device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device

    ## Cargar modelo de embeddings PubMedBERT (768-dim)
    logger.info("Cargando modelo de embeddings PubMedBERT...")
    emb_model = EmbeddingModel(device=device_str)

    ## Cargar indice FAISS desde cache (no reconstruye)
    ## build_index() llama _load_cached_index() internamente
    logger.info("Cargando indice FAISS desde %s...", RAG_INDEX_DIR)
    indexer = CorpusIndexer(
        embedding_model=emb_model,
        index_dir=RAG_INDEX_DIR,
    )
    n = indexer.build_index(str(LITERATURE_DIR), force_rebuild=False)
    logger.info("Indice cargado: %d chunks.", n)

    ## Construir el recuperador con top_k=3 (config de exp08)
    retriever = ReportRetriever(
        corpus_indexer=indexer,
        embedding_model=emb_model,
        top_k=RAG_TOP_K,
    )

    ## Cargar LLM Qwen2.5-7B en bfloat16
    logger.info("Cargando LLM %s...", QWEN_MODEL_ID)
    llm, tokenizer = load_llm_for_generation(
        model_name=QWEN_MODEL_ID,
        device=device_str,
        dtype=QWEN_DTYPE,
    )

    ## Construir ReportGenerator con todos los componentes
    generator = ReportGenerator(
        retriever=retriever,
        llm=llm,
        tokenizer=tokenizer,
        language='es',
        use_rag=True,
        rag_top_k=RAG_TOP_K,
    )

    logger.info("Pipeline RAG listo (dispositivo: %s).", device_str)
    return generator, retriever, indexer, llm, tokenizer


## =========================================================
## cargar_nli
## =========================================================

def cargar_nli(device='auto'):
    """
    Carga el modelo NLI mDeBERTa-v3-base-mnli-xnli para grounding de sentencias.

    El modelo es un clasificador de tres clases: entailment / neutral / contradiction.
    El indice de la clase 'entailment' se extrae de model.config.id2label para
    ser robusto a distintas versiones del modelo.

    Parametros
    ----------
    device : str
        'auto', 'cuda', 'cpu'.

    Retorna
    -------
    nli_model : AutoModelForSequenceClassification
    nli_tokenizer : AutoTokenizer
    entailment_idx : int
        Indice de la clase 'entailment' en los logits de salida.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device

    logger.info("Cargando modelo NLI: %s...", NLI_MODEL_ID)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    nli_model.to(device_str)
    nli_model.eval()

    ## Resolver el indice de 'entailment' desde el config del modelo
    id2label = nli_model.config.id2label          ## dict {int: str}
    entailment_idx = None
    for idx, label in id2label.items():
        if 'entailment' in label.lower():
            entailment_idx = int(idx)
            break

    if entailment_idx is None:
        ## Fallback: indice 0 es el convenio mas comun para modelos MNLI
        logger.warning(
            "No se encontro la clase 'entailment' en id2label %s. "
            "Usando indice 0 como fallback.", id2label
        )
        entailment_idx = 0

    logger.info(
        "NLI cargado en %s. Indice entailment: %d (label: %s).",
        device_str, entailment_idx, id2label.get(entailment_idx, '?')
    )
    return nli_model, nli_tokenizer, entailment_idx


## =========================================================
## prediccion_a_dict_generador
## =========================================================

def prediccion_a_dict_generador(prediccion_base, malignancy_score=None):
    """
    Convierte la salida de obtener_prediccion_base() al formato que espera
    ReportGenerator.generate().

    Parametros
    ----------
    prediccion_base : dict
        Salida de carga_modelo.obtener_prediccion_base(): claves
        'birads_idx', 'density_idx', 'birads_probs', 'density_probs', etc.
    malignancy_score : float o None
        Score de malignidad (prob BR4+BR5). Si es None, se computa desde probs.

    Retorna
    -------
    dict con claves:
        'birads_pred'        : int (indice 0-4, ReportGenerator suma +1 internamente)
        'birads_is_level'    : False (indicamos que es indice, no nivel 1-5)
        'birads_confidence'  : float (prob de la clase predicha)
        'density_pred'       : int (indice 0-3)
        'malignancy_score'   : float o None
    """
    birads_idx  = prediccion_base['birads_idx']
    density_idx = prediccion_base['density_idx']

    ## Confianza = probabilidad de la clase BI-RADS predicha
    birads_probs = prediccion_base['birads_probs']
    birads_conf  = float(birads_probs[0, birads_idx].item())

    ## Score de malignidad: P(BR4) + P(BR5) si no se proporciona
    if malignancy_score is None:
        probs = birads_probs[0]
        malignancy_score = float((probs[3] + probs[4]).item())

    return {
        'birads_pred':       birads_idx,
        'birads_is_level':   False,      ## indice 0-4, no nivel 1-5
        'birads_confidence': birads_conf,
        'density_pred':      density_idx,
        'malignancy_score':  malignancy_score,
    }
