## report_generator.py
## Generador de informes diagnosticos mamograficos
## Area 2: Razonamiento Multimodal
##
## Toma las predicciones del modelo MammoVLM (BI-RADS + densidad), recupera
## contexto relevante de la literatura medica (RAG) y genera un informe
## diagnostico estructurado en espanol, fundamentado en el sistema ACR BI-RADS.
##
## DISENO (rediseno para el modelo actual):
## El modelo MammoVLM predice de forma fiable BI-RADS (1-5) y densidad (A-D).
## El informe se construye EXCLUSIVAMENTE a partir de lo que el modelo predice;
## no se fabrican hallazgos morfologicos especificos (forma de masa, morfologia
## de calcificaciones) porque el modelo no los predice. Esto es una decision de
## seguridad clinica: un informe que no inventa hallazgos es preferible a uno que
## los alucina. Opcionalmente, si se dispone del score de malignidad del threshold
## tuning, se incorpora como senal de riesgo adicional.
##
## Flujo:
##   1. Recibe la prediccion del modelo (BI-RADS + densidad [+ score malignidad])
##   2. Construye query y recupera literatura relevante (ReportRetriever)
##   3. Construye prompt con la prediccion + contexto RAG (augment_prompt)
##   4. El LLM (Qwen2.5) genera el informe estructurado en espanol
##
## El informe sigue las secciones del ACR BI-RADS Atlas: indicacion, tecnica,
## composicion mamaria, hallazgos, impresion (con categoria BI-RADS justificada)
## y recomendaciones.

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


## Mapeo de indice de densidad (0-3) a la descripcion ACR estandar en espanol
DENSITY_DESCRIPTIONS_ES = {
    0: "mamas casi enteramente grasas (ACR A)",
    1: "densidades fibroglandulares dispersas (ACR B)",
    2: "tejido mamario heterogeneamente denso (ACR C)",
    3: "tejido mamario extremadamente denso (ACR D)",
}

## Nota clinica sobre la densidad (las categorias C y D reducen la sensibilidad
## de la mamografia y son un factor de riesgo independiente)
DENSITY_CLINICAL_NOTE_ES = {
    0: "",
    1: "",
    2: ("La densidad mamaria elevada puede disminuir la sensibilidad de la "
        "mamografia y constituye un factor de riesgo independiente."),
    3: ("La densidad mamaria muy elevada disminuye la sensibilidad de la "
        "mamografia y constituye un factor de riesgo independiente; puede "
        "considerarse estudio complementario."),
}


class PromptBuilder:
    ## Construye prompts estructurados para el LLM a partir de las predicciones

    def __init__(self, language: str = "es"):
        self.language = language

    def build_system_prompt(self) -> str:
        ## Prompt de sistema que define el rol del modelo
        if self.language == "es":
            return (
                "Usted es un radiologo especialista en mamografia. "
                "RESPONDA UNICA Y EXCLUSIVAMENTE EN ESPANOL. No use ningun otro "
                "idioma bajo ninguna circunstancia. "
                "Su tarea es redactar informes diagnosticos claros, precisos y "
                "fundamentados en la terminologia oficial del sistema ACR BI-RADS. "
                "Base su informe UNICAMENTE en la informacion proporcionada por el "
                "modelo (categoria BI-RADS y composicion mamaria). NO invente "
                "hallazgos morfologicos especificos (forma de masas, morfologia de "
                "calcificaciones, ubicaciones) que no esten en la informacion "
                "proporcionada. Use la literatura de referencia solo para "
                "fundamentar la terminologia y las recomendaciones clinicas. "
                "Redacte un informe completo y cierre cada seccion; no haga "
                "preguntas ni pida informacion adicional."
            )
        else:
            return (
                "You are a radiologist specializing in mammography. "
                "Write clear, precise diagnostic reports grounded in official ACR "
                "BI-RADS terminology. Base your report ONLY on the provided "
                "information (BI-RADS category and breast composition). Do not "
                "fabricate specific morphological findings not provided."
            )

    def build_findings_block(
        self,
        birads_pred: int,
        birads_confidence: float,
        density_description: str,
        density_note: str,
        recommendation: str,
        malignancy_score: Optional[float] = None,
    ) -> str:
        ##
        ## Construye el bloque de informacion del modelo para el prompt
        ##
        ## Solo incluye lo que el modelo predice de forma fiable:
        ## BI-RADS, densidad, recomendacion estandar y (opcional) score de riesgo.
        ##
        if self.language == "es":
            lines = ["INFORMACION PROPORCIONADA POR EL MODELO:"]
            lines.append(
                f"- Categoria BI-RADS: {birads_pred} (confianza del modelo: {birads_confidence:.2f})"
            )
            lines.append(f"- Composicion mamaria: {density_description}")
            if density_note:
                lines.append(f"- Nota sobre la densidad: {density_note}")
            if malignancy_score is not None:
                lines.append(
                    f"- Score de riesgo de malignidad estimado: {malignancy_score:.2f} "
                    f"(probabilidad agregada de categorias sospechosas BI-RADS 4-5)"
                )
            lines.append(f"- Recomendacion clinica estandar para BI-RADS {birads_pred}: {recommendation}")
            return "\n".join(lines)
        else:
            lines = [
                "MODEL-PROVIDED INFORMATION:",
                f"- BI-RADS category: {birads_pred} (confidence: {birads_confidence:.2f})",
                f"- Breast composition: {density_description}",
            ]
            if malignancy_score is not None:
                lines.append(f"- Estimated malignancy risk score: {malignancy_score:.2f}")
            lines.append(f"- Standard recommendation for BI-RADS {birads_pred}: {recommendation}")
            return "\n".join(lines)

    def build_instruction(self) -> str:
        ## Instruccion final para el LLM: estructura del informe ACR BI-RADS
        if self.language == "es":
            return (
                "Redacte un informe mamografico profesional estructurado en las "
                "siguientes secciones, usando terminologia ACR BI-RADS:\n"
                "1. TECNICA: indique que se realizo mamografia digital en "
                "proyecciones estandar.\n"
                "2. COMPOSICION MAMARIA: describa la composicion segun la categoria "
                "de densidad proporcionada.\n"
                "3. HALLAZGOS: describa los hallazgos de forma acorde a la categoria "
                "BI-RADS proporcionada, SIN inventar morfologias o ubicaciones "
                "especificas que no se hayan proporcionado. Si la categoria es "
                "BI-RADS 1 o 2, indique que no se identifican hallazgos sospechosos "
                "de malignidad. Si es BI-RADS 3, indique un hallazgo probablemente "
                "benigno. Si es BI-RADS 4 o 5, indique un hallazgo de aspecto "
                "sospechoso. Use una o dos frases; no divague ni pida mas datos.\n"
                "4. IMPRESION: indique la categoria BI-RADS con su justificacion "
                "clinica segun el nivel de sospecha.\n"
                "5. RECOMENDACION: indique la conducta clinica correspondiente a la "
                "categoria BI-RADS.\n"
                "Escriba TODO el informe en espanol, de forma concisa y profesional, "
                "sin fabricar informacion no proporcionada y sin usar otro idioma."
            )
        else:
            return (
                "Write a professional structured mammography report (technique, "
                "breast composition, findings, impression with BI-RADS "
                "justification, recommendation), using BI-RADS terminology and "
                "without fabricating information."
            )


class ReportGenerator:
    ## Generador de informes que integra las predicciones del modelo + RAG + LLM
    ##
    ## Puede operar en dos modos:
    ##   - Con LLM (modo principal): genera informe completo en lenguaje natural
    ##   - Sin LLM (template): genera informe estructurado por plantilla
    ##     (util para validar el pipeline sin cargar el LLM de 7B)

    def __init__(
        self,
        retriever=None,
        llm=None,
        tokenizer=None,
        language: str = "es",
        use_rag: bool = True,
        rag_top_k: int = 3,
    ):
        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer
        self.language = language
        self.use_rag = use_rag and retriever is not None
        self.rag_top_k = rag_top_k
        self.prompt_builder = PromptBuilder(language=language)

        ## Importar lexico para recomendaciones
        try:
            from medical_vocabulary import birads_lexicon
            self.lexicon = birads_lexicon
        except ImportError:
            self.lexicon = None

    def _resolve_density(self, prediction: Dict):
        ##
        ## Obtiene la descripcion y nota clinica de densidad a partir de la
        ## prediccion. Acepta 'density_pred' (indice 0-3) del modelo actual.
        ##
        density_idx = prediction.get("density_pred", None)
        if density_idx is None:
            return "no especificada", ""
        density_idx = int(density_idx)
        desc = DENSITY_DESCRIPTIONS_ES.get(density_idx, "no especificada")
        note = DENSITY_CLINICAL_NOTE_ES.get(density_idx, "")
        return desc, note

    def generate(
        self,
        prediction: Dict,
        max_new_tokens: int = 500,
    ) -> Dict:
        ##
        ## Genera un informe diagnostico a partir de una prediccion del modelo
        ##
        ## Parametros:
        ##   prediction: salida de MammoVLM.predict() para una imagen. Debe incluir
        ##     'birads_pred' (indice 0-4 o nivel 1-5) y 'birads_confidence'.
        ##     Opcionalmente 'density_pred' (0-3) y 'malignancy_score'.
        ##   max_new_tokens: longitud maxima del informe generado
        ##
        ## Retorna: dict con el informe, contexto RAG usado y metadatos
        ##
        ## Normalizar el BI-RADS a nivel clinico 1-5. El modelo entrega indice
        ## 0-4; si viene como 0-4 lo convertimos a 1-5 para el informe.
        birads_raw = int(prediction["birads_pred"])
        ## Heuristica: si el valor esta en 0-4 lo tratamos como indice y sumamos 1
        ## (el modelo MammoVLM entrega indices 0-4). Si ya viene 1-5, se respeta.
        if "birads_is_level" in prediction and prediction["birads_is_level"]:
            birads_level = birads_raw
        else:
            birads_level = birads_raw + 1

        birads_conf = float(prediction.get("birads_confidence", 0.0))
        malignancy_score = prediction.get("malignancy_score", None)

        ## Resolver densidad
        density_desc, density_note = self._resolve_density(prediction)

        ## Obtener recomendacion clinica del lexico (usa nivel 1-5)
        if self.lexicon:
            recommendation = self.lexicon.get_recommendation_for_birads(birads_level)
        else:
            recommendation = "Consultar guia clinica"

        ## Recuperar contexto RAG (sin hallazgos morfologicos; solo BI-RADS + densidad)
        retrieved_chunks = []
        if self.use_rag:
            query = self.retriever.build_query_from_findings(
                birads_pred=birads_level,
                findings=[],  ## el modelo no predice hallazgos morfologicos
                density=density_desc,
            )
            retrieved_chunks = self.retriever.retrieve(query, top_k=self.rag_top_k)

        ## Construir prompt
        system_prompt = self.prompt_builder.build_system_prompt()
        findings_block = self.prompt_builder.build_findings_block(
            birads_level, birads_conf, density_desc, density_note,
            recommendation, malignancy_score,
        )
        instruction = self.prompt_builder.build_instruction()

        ## Generar informe
        if self.llm is not None and self.tokenizer is not None:
            report_text = self._generate_with_llm(
                system_prompt, findings_block, instruction,
                retrieved_chunks, max_new_tokens
            )
        else:
            report_text = self._generate_with_template(
                birads_level, density_desc, density_note, recommendation, malignancy_score
            )

        return {
            "report": report_text,
            "birads_level": birads_level,
            "birads_confidence": birads_conf,
            "density_description": density_desc,
            "malignancy_score": malignancy_score,
            "recommendation": recommendation,
            "rag_chunks_used": len(retrieved_chunks),
            "rag_sources": [c.get("source", "") for c in retrieved_chunks],
        }

    def _generate_with_llm(
        self,
        system_prompt: str,
        findings_block: str,
        instruction: str,
        retrieved_chunks: List[Dict],
        max_new_tokens: int,
    ) -> str:
        ##
        ## Genera informe usando el LLM con contexto RAG
        ##
        import torch
        from rag import augment_prompt

        ## Construir prompt base en formato chat de Qwen
        user_content = f"{findings_block}\n\n{instruction}"
        base_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        ## Aumentar con contexto RAG (grounding en literatura)
        if retrieved_chunks:
            full_prompt = augment_prompt(base_prompt, retrieved_chunks, language=self.language)
        else:
            full_prompt = base_prompt

        ## Tokenizar y generar
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.llm.device)

        ## Generar con temperatura baja para minimizar fugas de idioma (Qwen
        ## tiende a saltar al chino bajo alta aleatoriedad). Si aun asi se detecta
        ## texto no latino, se regenera de forma determinista (greedy) una vez.
        report = self._run_generation(inputs, max_new_tokens, temperature=0.1, do_sample=True)

        if self._contains_non_latin(report):
            logger.warning("Fuga de idioma detectada; regenerando de forma determinista")
            report = self._run_generation(inputs, max_new_tokens, temperature=None, do_sample=False)
            ## Si tras regenerar aun hay texto no latino, truncar en ese punto
            if self._contains_non_latin(report):
                report = self._truncate_at_non_latin(report)

        return report.strip()

    def _run_generation(self, inputs, max_new_tokens, temperature, do_sample):
        ##
        ## Ejecuta una pasada de generacion del LLM
        ##
        import torch
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            ## Greedy: determinista, sin aleatoriedad (mas estable para el idioma)
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            generated = self.llm.generate(**inputs, **gen_kwargs)

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _contains_non_latin(text: str) -> bool:
        ##
        ## Detecta si el texto contiene caracteres CJK (chino, japones, coreano),
        ## senal de que el LLM se fugo de idioma. Rangos Unicode principales:
        ##   CJK Unified Ideographs: 4E00-9FFF
        ##   Hiragana/Katakana: 3040-30FF
        ##   Hangul: AC00-D7AF
        ##
        for ch in text:
            code = ord(ch)
            if (0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF
                    or 0xAC00 <= code <= 0xD7AF):
                return True
        return False

    @staticmethod
    def _truncate_at_non_latin(text: str) -> str:
        ##
        ## Trunca el texto en el primer caracter no latino (red de seguridad
        ## final si la regeneracion tampoco produjo texto limpio)
        ##
        for i, ch in enumerate(text):
            code = ord(ch)
            if (0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF
                    or 0xAC00 <= code <= 0xD7AF):
                return text[:i].strip()
        return text

    def _generate_with_template(
        self,
        birads_level: int,
        density_desc: str,
        density_note: str,
        recommendation: str,
        malignancy_score: Optional[float] = None,
    ) -> str:
        ##
        ## Genera informe estructurado por plantilla (sin LLM)
        ## Util para validar el pipeline sin cargar el modelo de 7B
        ##
        lines = []
        lines.append("INFORME MAMOGRAFICO")
        lines.append("")
        lines.append("TECNICA: Mamografia digital en proyecciones craneocaudal (CC) "
                     "y mediolateral oblicua (MLO).")
        lines.append("")
        lines.append(f"COMPOSICION MAMARIA: {density_desc}.")
        if density_note:
            lines.append(density_note)
        lines.append("")
        ## Hallazgos: descripcion acorde a la categoria, sin inventar morfologias
        if birads_level <= 2:
            hallazgo = ("No se identifican hallazgos sospechosos de malignidad en "
                        "el analisis automatizado.")
        elif birads_level == 3:
            hallazgo = ("Se identifica un hallazgo probablemente benigno que amerita "
                        "seguimiento a corto plazo.")
        else:
            hallazgo = ("Se identifica un hallazgo con caracteristicas sospechosas "
                        "que amerita evaluacion adicional.")
        lines.append(f"HALLAZGOS: {hallazgo}")
        if malignancy_score is not None:
            lines.append(f"Score de riesgo de malignidad estimado: {malignancy_score:.2f}.")
        lines.append("")
        lines.append(f"IMPRESION: Categoria BI-RADS {birads_level}.")
        lines.append("")
        lines.append(f"RECOMENDACION: {recommendation}.")
        return "\n".join(lines)


def load_llm_for_generation(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "auto",
    dtype: str = "bfloat16",
):
    ##
    ## Carga el LLM Qwen2.5 para generacion de informes
    ##
    ## Retorna: (model, tokenizer)
    ##
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Cargando LLM: %s", model_name)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    logger.info("LLM cargado correctamente")
    return model, tokenizer
