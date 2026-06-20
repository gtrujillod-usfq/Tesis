## atribucion_rag.py
## Paso 4: atribucion RAG mediante Shapley exacto (8 subconjuntos) y grounding NLI.
##
## Decisiones de diseno:
##   - El informe objetivo se genera UNA VEZ por imagen con decodificacion greedy
##     (do_sample=False, determinista). Los 8 subconjuntos se evaluan con teacher
##     forcing sobre ese informe fijo. Razon: no queremos que el Shapley compare
##     textos distintos; el objetivo debe ser el mismo para todos los subconjuntos.
##   - Shapley exacto (no SHAP kernel): con k=3 hay 8 subconjuntos, la enumeracion
##     directa es exacta y rapida. La libreria shap no se usa.
##   - Teacher forcing: se tokeniza [prefix + informe], se hace un forward pass,
##     y se extraen los log-probs de los tokens del informe dado el prefix.
##   - NLI (grounding): modelo mDeBERTa-v3, premise=chunk, hypothesis=oracion del
##     informe. La puntuacion de grounding de cada chunk es la media de entailment
##     sobre todas las oraciones del informe.
##   - El filtro CJK y el refuerzo del system prompt de ReportGenerator se aplican
##     en la generacion greedy (que usa _run_generation internamente). El Shapley
##     usa teacher forcing directo sin regeneracion, lo cual es correcto: el objetivo
##     ya esta fijado y no puede contener texto no latino.

import json
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config_xai import RAG_TOP_K, NUM_RAG_SUBSETS, OUT_RAG
from rag import augment_prompt

logger = logging.getLogger(__name__)


## =========================================================
## Constantes de Shapley para k=3
## =========================================================

## Con k=3 chunks hay 8 subconjuntos (2^3).
## Los pesos de Shapley w(s, k) = s! * (k-s-1)! / k! para s = |S|, k = 3:
##   w(0, 3) = 0! * 2! / 6 = 2/6 = 1/3
##   w(1, 3) = 1! * 1! / 6 = 1/6
##   w(2, 3) = 2! * 0! / 6 = 2/6 = 1/3
_K_CHUNKS = RAG_TOP_K   ## 3

def _peso_shapley(s: int, k: int = _K_CHUNKS) -> float:
    """Peso de Shapley para un subconjunto de tamano s con k jugadores."""
    return math.factorial(s) * math.factorial(k - s - 1) / math.factorial(k)


## =========================================================
## Construccion del prompt por subconjunto
## =========================================================

def _construir_prompt_base(generator, prediction_dict):
    """
    Construye el prompt sin chunks RAG a partir de los componentes de ReportGenerator.

    El prompt sigue el formato de chat de Qwen:
        <|im_start|>system\\n{system}\\<|im_end|>\\n
        <|im_start|>user\\n{findings}\\n\\n{instruction}\\<|im_end|>\\n
        <|im_start|>assistant\\n

    Parametros
    ----------
    generator : ReportGenerator
    prediction_dict : dict
        Salida de carga_rag.prediccion_a_dict_generador().

    Retorna
    -------
    str : prompt base sin contexto RAG.
    """
    from report_generator import DENSITY_DESCRIPTIONS_ES, DENSITY_CLINICAL_NOTE_ES

    pb = generator.prompt_builder

    ## Normalizar BI-RADS a nivel 1-5 (ReportGenerator suma +1 si no es nivel)
    birads_raw = int(prediction_dict['birads_pred'])
    if prediction_dict.get('birads_is_level', False):
        birads_level = birads_raw
    else:
        birads_level = birads_raw + 1

    birads_conf  = float(prediction_dict.get('birads_confidence', 0.0))
    density_idx  = prediction_dict.get('density_pred', 0)
    mal_score    = prediction_dict.get('malignancy_score', None)

    density_desc = DENSITY_DESCRIPTIONS_ES.get(density_idx, 'no especificada')
    density_note = DENSITY_CLINICAL_NOTE_ES.get(density_idx, '')

    ## Recomendacion estandar segun BI-RADS level
    if generator.lexicon:
        recommendation = generator.lexicon.get_recommendation_for_birads(birads_level)
    else:
        recommendation = 'Consultar guia clinica'

    system_prompt  = pb.build_system_prompt()
    findings_block = pb.build_findings_block(
        birads_level, birads_conf, density_desc, density_note,
        recommendation, mal_score
    )
    instruction = pb.build_instruction()

    user_content = f"{findings_block}\n\n{instruction}"
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def construir_prompt_con_subconjunto(generator, prediction_dict, chunks_retrieved,
                                      chunk_indices):
    """
    Construye el prompt aumentado con el subconjunto indicado de chunks RAG.

    Parametros
    ----------
    generator : ReportGenerator
    prediction_dict : dict
    chunks_retrieved : list de dict
        Lista de los RAG_TOP_K chunks recuperados (como devuelve retriever.retrieve()).
    chunk_indices : list de int
        Indices (dentro de chunks_retrieved) del subconjunto a incluir.
        Lista vacia = prompt sin contexto RAG.

    Retorna
    -------
    str : prompt aumentado listo para tokenizar.
    """
    base_prompt = _construir_prompt_base(generator, prediction_dict)

    if not chunk_indices:
        ## Subconjunto vacio: no se aumenta el prompt
        return base_prompt

    subset_chunks = [chunks_retrieved[i] for i in chunk_indices]
    augmented = augment_prompt(base_prompt, subset_chunks, language='es')
    return augmented


## =========================================================
## Generacion greedy determinista
## =========================================================

def generar_informe_greedy(generator, prediction_dict, retriever):
    """
    Genera el informe de forma determinista (greedy, do_sample=False).

    Esta generacion se hace UNA VEZ por imagen. El texto resultante se usa como
    objetivo fijo para el teacher forcing del Shapley.

    El filtro CJK y el refuerzo del system prompt de ReportGenerator._run_generation
    aplican con normalidad aqui (regeneracion si hay texto no latino).

    Parametros
    ----------
    generator : ReportGenerator
    prediction_dict : dict
        Salida de carga_rag.prediccion_a_dict_generador().
    retriever : ReportRetriever

    Retorna
    -------
    informe_texto : str
        Informe generado de forma determinista.
    chunks_recuperados : list de dict
        Los RAG_TOP_K chunks usados (mismos que en exp08).
    """
    from report_generator import DENSITY_DESCRIPTIONS_ES, DENSITY_CLINICAL_NOTE_ES

    pb = generator.prompt_builder

    birads_raw = int(prediction_dict['birads_pred'])
    birads_level = birads_raw if prediction_dict.get('birads_is_level', False) else birads_raw + 1
    birads_conf  = float(prediction_dict.get('birads_confidence', 0.0))
    density_idx  = prediction_dict.get('density_pred', 0)
    mal_score    = prediction_dict.get('malignancy_score', None)

    density_desc = DENSITY_DESCRIPTIONS_ES.get(density_idx, 'no especificada')
    density_note = DENSITY_CLINICAL_NOTE_ES.get(density_idx, '')

    if generator.lexicon:
        recommendation = generator.lexicon.get_recommendation_for_birads(birads_level)
    else:
        recommendation = 'Consultar guia clinica'

    ## Recuperar los chunks (misma logica que ReportGenerator.generate)
    query = retriever.build_query_from_findings(
        birads_pred=birads_level,
        findings=[],
        density=density_desc,
    )
    chunks_recuperados = retriever.retrieve(query, top_k=RAG_TOP_K)

    ## Construir prompt aumentado con los 3 chunks
    system_prompt  = pb.build_system_prompt()
    findings_block = pb.build_findings_block(
        birads_level, birads_conf, density_desc, density_note,
        recommendation, mal_score
    )
    instruction = pb.build_instruction()
    user_content = f"{findings_block}\n\n{instruction}"

    base_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_prompt = augment_prompt(base_prompt, chunks_recuperados, language='es')

    ## Tokenizar y generar de forma determinista
    inputs = generator.tokenizer(full_prompt, return_tensors='pt').to(generator.llm.device)

    ## Llamar a _run_generation con do_sample=False (greedy, determinista)
    informe_texto = generator._run_generation(
        inputs, max_new_tokens=500, temperature=None, do_sample=False
    )

    ## Aplicar filtro CJK (igual que en generate())
    if generator._contains_non_latin(informe_texto):
        logger.warning("Fuga de idioma en generacion greedy; truncando texto CJK.")
        informe_texto = generator._truncate_at_non_latin(informe_texto)

    return informe_texto.strip(), chunks_recuperados


## =========================================================
## Teacher forcing: log-prob del informe dado un prompt
## =========================================================

def log_prob_teacher_forcing(llm, tokenizer, prompt_prefix, target_text):
    """
    Computa la log-probabilidad de target_text dado prompt_prefix.

    Procedimiento:
      1. Tokenizar prompt_prefix por separado para obtener prefix_len.
      2. Tokenizar prompt_prefix + target_text como un unico string.
      3. Hacer un forward pass sobre la secuencia completa.
      4. Extraer los log-probs en las posiciones [prefix_len-1, full_len-1)
         (logit en t predice token en t+1).

    Parametros
    ----------
    llm : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    prompt_prefix : str
        Prompt que incluye system + findings + instruccion + (opcionalmente) chunks RAG.
        Termina con '<|im_start|>assistant\\n'.
    target_text : str
        Informe objetivo (generado con greedy una vez).

    Retorna
    -------
    total_log_prob : float
        Suma de log-probs de los tokens del target_text. Valores mas altos indican
        que el modelo es mas probable de generar ese informe dado el prompt.
    """
    device = llm.device

    ## Tokenizar prefix y target por SEPARADO, luego concatenar los IDs.
    ## Razon: tokenizer.encode(prefix + target) puede producir tokens distintos
    ## en la frontera si el tokenizador BPE fusiona el ultimo token del prefix
    ## con el primero del target (fusion BPE). Concatenar IDs evita este problema.
    ## add_special_tokens=False en ambos: el prefix ya contiene los tokens de chat;
    ## el target no debe recibir un BOS adicional.
    prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text,   add_special_tokens=False)
    prefix_len = len(prefix_ids)
    target_len = len(target_ids)

    if target_len == 0:
        logger.warning("El target_text esta vacio despues de tokenizar. Retornando 0.0.")
        return 0.0

    ## Concatenar IDs para el forward pass
    full_ids = prefix_ids + target_ids
    full_len = len(full_ids)

    ## Forward pass: los gradientes no son necesarios aqui
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = llm(input_ids=input_tensor)
    logits = outputs.logits  ## [1, full_len, vocab_size]

    ## Logits relevantes: posiciones [prefix_len-1, full_len-1)
    ## El logit en la posicion t predice el token en la posicion t+1.
    ## Logit[prefix_len-1] predice el primer token del target (target_ids[0]).
    relevant_logits = logits[0, prefix_len - 1 : full_len - 1, :]  ## [target_len, vocab_size]

    ## Log-probs via log_softmax para estabilidad numerica
    log_probs = F.log_softmax(relevant_logits, dim=-1)   ## [target_len, vocab_size]

    ## Seleccionar los log-probs de los tokens del target
    target_token_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    token_log_probs     = log_probs[torch.arange(target_len), target_token_tensor]  ## [target_len]

    total_log_prob = token_log_probs.sum().item()
    return total_log_prob


## =========================================================
## Shapley exacto sobre k=3 chunks
## =========================================================

def calcular_shapley_rag(generator, llm, tokenizer, prediction_dict,
                          chunks_recuperados, informe_objetivo):
    """
    Calcula el valor de Shapley exacto para cada uno de los 3 chunks RAG.

    Para k=3 jugadores hay 8 subconjuntos (2^3). Se evalua v(S) = teacher-forced
    log-prob del informe_objetivo dado el prompt construido con el subconjunto S.

    Formula de Shapley para el jugador j (chunk j):
        phi_j = sum_{S subset de {0,1,2} \\ {j}}
                  w(|S|, 3) * [v(S union {j}) - v(S)]
        donde w(s, 3) = s! * (2-s)! / 6

    Parametros
    ----------
    generator : ReportGenerator
    llm : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    prediction_dict : dict
    chunks_recuperados : list de dict
        Lista de exactamente RAG_TOP_K=3 chunks.
    informe_objetivo : str
        Informe generado de forma greedy (objetivo fijo para teacher forcing).

    Retorna
    -------
    dict con claves:
        'shapley_values'  : dict {chunk_idx: phi_j} para j in {0,1,2}
        'v_subsets'       : dict {frozenset: log_prob} para los 8 subconjuntos
        'informe_objetivo': str (el texto objetivo usado)
    """
    k = len(chunks_recuperados)
    if k != _K_CHUNKS:
        raise ValueError(
            f"Se esperaban {_K_CHUNKS} chunks recuperados, se recibieron {k}. "
            "Verifica que retriever.top_k == RAG_TOP_K."
        )

    ## Evaluar v(S) para los 2^k subconjuntos
    ## Representamos S como frozenset de indices
    all_indices = list(range(k))
    v_subsets = {}

    logger.info("Evaluando %d subconjuntos para Shapley...", NUM_RAG_SUBSETS)

    for bits in range(2 ** k):
        ## Decodificar el subset desde la representacion binaria
        subset = frozenset(j for j in range(k) if (bits >> j) & 1)

        ## Construir el prompt con este subconjunto de chunks
        prompt_subset = construir_prompt_con_subconjunto(
            generator, prediction_dict, chunks_recuperados, list(subset)
        )

        ## Teacher forcing: log-prob del informe objetivo dado este prompt
        lp = log_prob_teacher_forcing(llm, tokenizer, prompt_subset, informe_objetivo)
        v_subsets[subset] = lp

        subset_str = '{' + ','.join(str(j) for j in sorted(subset)) + '}'
        logger.debug("v(%s) = %.4f", subset_str, lp)

    ## Calcular valores de Shapley para cada chunk j
    shapley_values = {}
    for j in range(k):
        phi_j = 0.0
        ## Iterar sobre todos los subconjuntos S de all_indices que NO contienen j
        other = [m for m in all_indices if m != j]
        for s_size in range(len(other) + 1):
            for S_tuple in combinations(other, s_size):
                S = frozenset(S_tuple)
                S_union_j = S | {j}
                marginal = v_subsets[S_union_j] - v_subsets[S]
                weight   = _peso_shapley(s_size, k)
                phi_j   += weight * marginal

        shapley_values[j] = phi_j
        logger.info("phi_%d = %.4f  (fuente: %s)",
                    j, phi_j, chunks_recuperados[j].get('source', '?'))

    ## Verificar propiedad de eficiencia: sum(phi_j) == v(full) - v(empty)
    ## Esta propiedad se cumple exactamente (hasta precision flotante) para
    ## la formula de Shapley. Si falla, hay un bug en la acumulacion.
    v_full  = v_subsets[frozenset({0, 1, 2})]
    v_empty = v_subsets[frozenset()]
    sum_phi = sum(shapley_values.values())
    eficiencia_error = abs(sum_phi - (v_full - v_empty))
    if eficiencia_error > 1e-6:
        raise AssertionError(
            f"Propiedad de eficiencia de Shapley violada: "
            f"sum(phi)={sum_phi:.6f}, v(full)-v(empty)={v_full - v_empty:.6f}, "
            f"error={eficiencia_error:.2e}. Revisar acumulacion de subconjuntos."
        )
    logger.info(
        "Eficiencia Shapley OK: sum(phi)=%.4f, v(full)-v(empty)=%.4f, error=%.2e",
        sum_phi, v_full - v_empty, eficiencia_error
    )

    return {
        'shapley_values':   shapley_values,
        'v_subsets':        {str(sorted(s)): v for s, v in v_subsets.items()},
        'informe_objetivo': informe_objetivo,
        'eficiencia_error': eficiencia_error,
    }


## =========================================================
## Grounding NLI: entailment por chunk sobre oraciones del informe
## =========================================================

def _segmentar_oraciones(texto):
    """
    Segmenta un informe en oraciones usando puntuacion basica.

    Separadores: '. ', '.\n', '? ', '! '. Las lineas vacias se ignoran.
    Retorna una lista de strings no vacios.
    """
    import re
    ## Reemplazar saltos de linea por espacio antes de separar
    texto_limpio = texto.replace('\n', ' ')
    ## Separar por punto seguido de espacio o fin de cadena
    oraciones = re.split(r'(?<=[.!?])\s+', texto_limpio)
    return [o.strip() for o in oraciones if o.strip()]


def calcular_grounding_nli(informe_texto, chunks_recuperados, nli_model,
                            nli_tokenizer, entailment_idx):
    """
    Calcula el score de grounding NLI de cada chunk sobre el informe.

    Para cada chunk j y para cada oracion s del informe:
        score(j, s) = P(entailment | premise=chunk_j, hypothesis=s)

    El score de grounding del chunk j es la media sobre todas las oraciones.
    El 'chunk de soporte' es el de mayor score medio.

    Parametros
    ----------
    informe_texto : str
    chunks_recuperados : list de dict
        Cada dict tiene al menos la clave 'text'.
    nli_model : AutoModelForSequenceClassification
    nli_tokenizer : AutoTokenizer
    entailment_idx : int
        Indice de la clase 'entailment' en los logits de salida del modelo NLI.

    Retorna
    -------
    dict con claves:
        'scores_por_chunk'   : dict {chunk_idx: mean_entailment_float}
        'scores_por_oracion' : dict {chunk_idx: list[float]} una por oracion
        'chunk_soporte'      : int, indice del chunk con mayor entailment medio
        'n_oraciones'        : int
    """
    device = nli_model.device
    oraciones = _segmentar_oraciones(informe_texto)
    n_oraciones = len(oraciones)

    if n_oraciones == 0:
        logger.warning("No se encontraron oraciones en el informe. Retornando scores 0.")
        k = len(chunks_recuperados)
        return {
            'scores_por_chunk':   {j: 0.0 for j in range(k)},
            'scores_por_oracion': {j: []  for j in range(k)},
            'chunk_soporte':      0,
            'n_oraciones':        0,
        }

    scores_por_oracion = {}
    scores_por_chunk   = {}

    for j, chunk in enumerate(chunks_recuperados):
        chunk_text = chunk.get('text', '')
        oracion_scores = []

        for oracion in oraciones:
            ## NLI: premise = chunk_text, hypothesis = oracion del informe
            ## Si la oracion esta respaldada por el chunk, se clasifica como entailment
            encoding = nli_tokenizer(
                chunk_text,
                oracion,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)

            with torch.no_grad():
                logits = nli_model(**encoding).logits  ## [1, 3]

            probs = torch.softmax(logits, dim=-1)[0]   ## [3]
            entailment_prob = float(probs[entailment_idx].item())
            oracion_scores.append(entailment_prob)

        mean_entailment = float(np.mean(oracion_scores)) if oracion_scores else 0.0
        scores_por_oracion[j] = oracion_scores
        scores_por_chunk[j]   = mean_entailment

        logger.debug("NLI chunk %d (%s): mean_entailment=%.4f",
                     j, chunk.get('source', '?'), mean_entailment)

    ## Chunk de soporte: el de mayor entailment medio
    chunk_soporte = max(scores_por_chunk, key=scores_por_chunk.get)

    return {
        'scores_por_chunk':   scores_por_chunk,
        'scores_por_oracion': scores_por_oracion,
        'chunk_soporte':      chunk_soporte,
        'n_oraciones':        n_oraciones,
    }


## =========================================================
## calcular_atribuciones_rag (funcion principal)
## =========================================================

def calcular_atribuciones_rag(generator, retriever, llm, tokenizer,
                               nli_model, nli_tokenizer, entailment_idx,
                               prediction_dict):
    """
    Pipeline completo de atribucion RAG para una imagen:
      1. Genera el informe de forma greedy (determinista).
      2. Calcula Shapley exacto sobre los 3 chunks.
      3. Calcula grounding NLI por chunk.

    Parametros
    ----------
    generator : ReportGenerator
    retriever : ReportRetriever
    llm : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    nli_model : AutoModelForSequenceClassification
    nli_tokenizer : AutoTokenizer
    entailment_idx : int
    prediction_dict : dict
        Salida de carga_rag.prediccion_a_dict_generador().

    Retorna
    -------
    dict con claves:
        'informe':        str
        'chunks':         list de dict (RAG_TOP_K chunks con text, source, page, ...)
        'shapley':        dict (shapley_values, v_subsets, informe_objetivo)
        'nli':            dict (scores_por_chunk, chunk_soporte, n_oraciones)
        'prediction':     dict (copia del prediction_dict)
    """
    ## Paso 1: generar informe greedy (determinista)
    logger.info("Generando informe greedy...")
    informe, chunks = generar_informe_greedy(generator, prediction_dict, retriever)
    logger.info("Informe generado (%d chars, %d chunks).", len(informe), len(chunks))

    ## Paso 2: Shapley exacto (8 subconjuntos)
    logger.info("Calculando Shapley exacto sobre %d chunks...", len(chunks))
    shapley = calcular_shapley_rag(
        generator=generator,
        llm=llm,
        tokenizer=tokenizer,
        prediction_dict=prediction_dict,
        chunks_recuperados=chunks,
        informe_objetivo=informe,
    )

    ## Paso 3: grounding NLI
    logger.info("Calculando grounding NLI...")
    nli = calcular_grounding_nli(
        informe_texto=informe,
        chunks_recuperados=chunks,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        entailment_idx=entailment_idx,
    )

    return {
        'informe':    informe,
        'chunks':     chunks,
        'shapley':    shapley,
        'nli':        nli,
        'prediction': prediction_dict,
    }


## =========================================================
## Persistencia
## =========================================================

## Longitud maxima del snippet de chunk para figuras cualitativas
_SNIPPET_MAX_CHARS = 300


def guardar_atribucion_rag(resultado, image_id, out_dir=None):
    """
    Guarda el resultado de calcular_atribuciones_rag() como JSON.

    El JSON incluye, para cada chunk:
      - source_file, page_number, score de similitud FAISS
      - snippet: primeros 300 caracteres del texto del chunk (para figuras cualitativas)
    Estos campos permiten las figuras de la tesis sin tener que recargar el indice FAISS.

    Parametros
    ----------
    resultado : dict
        Salida de calcular_atribuciones_rag().
    image_id : str
    out_dir : Path o None
        Por defecto, OUT_RAG de config_xai.
    """
    if out_dir is None:
        out_dir = OUT_RAG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta = out_dir / f"{image_id}_rag.json"

    ## Incluir source_file, page_number y snippet para figuras cualitativas de tesis.
    ## El texto completo no se guarda; el snippet es suficiente para las figuras.
    chunks_meta = []
    for c in resultado['chunks']:
        texto_completo = c.get('text', '')
        snippet = texto_completo[:_SNIPPET_MAX_CHARS]
        if len(texto_completo) > _SNIPPET_MAX_CHARS:
            snippet += '...'
        chunks_meta.append({
            'source_file':  c.get('source', ''),
            'page_number':  c.get('page', 0),
            'score':        round(float(c.get('score', 0.0)), 6),
            'snippet':      snippet,
        })

    ## int keys de shapley_values y nli_scores se convierten a str en JSON
    shapley_vals = {str(k): v for k, v in resultado['shapley']['shapley_values'].items()}
    nli_scores   = {str(k): v for k, v in resultado['nli']['scores_por_chunk'].items()}

    payload = {
        'image_id':            image_id,
        'informe':             resultado['informe'],
        'chunks_meta':         chunks_meta,
        'shapley_values':      shapley_vals,
        'v_subsets':           resultado['shapley']['v_subsets'],
        'eficiencia_error':    resultado['shapley'].get('eficiencia_error', None),
        'nli_scores':          nli_scores,
        'chunk_soporte':       resultado['nli']['chunk_soporte'],
        'n_oraciones':         resultado['nli']['n_oraciones'],
        'prediction':          resultado['prediction'],
    }

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Atribucion RAG guardada en %s", ruta)


def cargar_atribucion_rag(image_id, out_dir=None):
    """
    Carga el JSON guardado por guardar_atribucion_rag().

    Los keys de shapley_values y nli_scores se convierten de str a int.

    Retorna
    -------
    dict con las claves de la funcion guardar_atribucion_rag().

    Raises
    ------
    FileNotFoundError si el archivo no existe.
    """
    if out_dir is None:
        out_dir = OUT_RAG
    ruta = Path(out_dir) / f"{image_id}_rag.json"

    if not ruta.exists():
        raise FileNotFoundError(
            f"Atribucion RAG no encontrada: {ruta}. "
            "Ejecuta calcular_atribuciones_rag() primero."
        )

    with open(ruta, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    ## Restaurar keys enteras en los dicts de scores
    payload['shapley_values'] = {int(k): v for k, v in payload['shapley_values'].items()}
    payload['nli_scores']     = {int(k): v for k, v in payload['nli_scores'].items()}

    return payload
