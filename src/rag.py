## rag.py
## Sistema RAG (Retrieval-Augmented Generation) para literatura medica
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Componentes:
##   1. CorpusIndexer: extrae texto de PDFs medicos, fragmenta y construye indice FAISS
##   2. ReportRetriever: recupera fragmentos relevantes dada una query
##   3. augment_prompt: inyecta contexto recuperado en el prompt del LLM
##
## Decisiones de diseno:
##   - PyMuPDF4LLM para extraccion de PDF (preserva estructura, tablas, captions)
##   - NeuML/pubmedbert-base-embeddings para embeddings (mean pooling, 768-dim)
##   - FAISS IndexFlatIP con vectores normalizados (cosine similarity)
##   - Chunks de ~400 tokens con 15% overlap
##   - Top-k=5 para recuperacion

import os
import json
import logging
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

## Constantes de configuracion RAG
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 5
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM = 768


class TextChunk:
    ## Fragmento de texto con metadatos de origen

    def __init__(
        self,
        text: str,
        source_file: str,
        page_number: int,
        chapter: str = "",
        section: str = "",
        content_type: str = "text",
    ):
        self.text = text
        self.source_file = source_file
        self.page_number = page_number
        self.chapter = chapter
        self.section = section
        self.content_type = content_type

    def get_contextualized_text(self) -> str:
        ##
        ## Retorna el texto con contexto prepend (header path)
        ## Esto mejora la calidad de recuperacion segun Anthropic Contextual Retrieval
        ##
        prefix_parts = []
        if self.source_file:
            prefix_parts.append(f"Source: {self.source_file}")
        if self.chapter:
            prefix_parts.append(f"Chapter: {self.chapter}")
        if self.section:
            prefix_parts.append(f"Section: {self.section}")
        if self.content_type != "text":
            prefix_parts.append(f"Type: {self.content_type}")

        prefix = " | ".join(prefix_parts)
        if prefix:
            return f"[{prefix}] {self.text}"
        return self.text

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chapter": self.chapter,
            "section": self.section,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TextChunk":
        return cls(**data)


class PDFExtractor:
    ## Extrae texto de PDFs medicos preservando estructura
    ## Usa PyMuPDF (fitz) para extraccion directa de texto embebido
    ## Si detecta un PDF escaneado (sin texto), aplica OCR con cache
    ##
    ## El OCR es lento pero se ejecuta una sola vez por documento:
    ## el resultado se cachea en disco como JSON para reutilizarlo

    def __init__(self, ocr_cache_dir: Optional[Path] = None, ocr_lang: str = "eng"):
        ##
        ## Parametros:
        ##   ocr_cache_dir: directorio para cachear texto OCR (default: data/ocr_cache)
        ##   ocr_lang: idioma para Tesseract ("eng" ingles, "spa" espanol, "eng+spa" ambos)
        ##
        self.ocr_cache_dir = ocr_cache_dir or Path("data/ocr_cache")
        self.ocr_lang = ocr_lang

    def _has_extractable_text(self, doc, sample_pages: int = 10) -> bool:
        ##
        ## Determina si un PDF tiene texto embebido extraible
        ## Muestrea las primeras paginas y verifica si hay texto
        ##
        total_pages = len(doc)
        pages_to_check = min(sample_pages, total_pages)
        text_chars = 0

        for page_num in range(pages_to_check):
            text = doc[page_num].get_text("text")
            text_chars += len(text.strip())

        ## Si el promedio de caracteres por pagina es muy bajo, es escaneado
        avg_chars = text_chars / max(1, pages_to_check)
        return avg_chars > 50

    def extract_from_pdf(self, pdf_path: str) -> List[TextChunk]:
        ##
        ## Extrae texto de un PDF y retorna lista de chunks con metadatos
        ## Detecta automaticamente si el PDF es escaneado y aplica OCR
        ##
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error("Archivo PDF no encontrado: %s", pdf_path)
            return []

        source_name = pdf_path.stem
        logger.info("Extrayendo texto de: %s", source_name)

        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF (fitz) no esta instalado. Ejecute: pip install PyMuPDF")
            return []

        doc = fitz.open(str(pdf_path))

        ## Detectar si tiene texto extraible o necesita OCR
        if self._has_extractable_text(doc):
            chunks = self._extract_text_direct(doc, source_name)
        else:
            logger.info("  %s parece escaneado, aplicando OCR (puede tardar)...", source_name)
            doc.close()
            chunks = self._extract_with_ocr(pdf_path, source_name)
            return chunks

        doc.close()
        return chunks

    def _extract_text_direct(self, doc, source_name: str) -> List[TextChunk]:
        ##
        ## Extraccion directa de texto embebido (rapida, sin OCR)
        ##
        total_pages = len(doc)
        chunks = []
        current_chapter = ""
        current_section = ""
        pages_with_text = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            if not text or len(text.strip()) < 30:
                continue

            pages_with_text += 1

            chapter, section = self._detect_headers(text, current_chapter, current_section)
            if chapter:
                current_chapter = chapter
            if section:
                current_section = section

            content_type = self._detect_content_type(text)
            page_chunks = self._split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < 50:
                    continue
                chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    source_file=source_name,
                    page_number=page_num + 1,
                    chapter=current_chapter,
                    section=current_section,
                    content_type=content_type,
                ))

        logger.info("  %s: %d chunks de %d paginas con texto (de %d totales)",
                    source_name, len(chunks), pages_with_text, total_pages)
        return chunks

    def _extract_with_ocr(self, pdf_path: Path, source_name: str, dpi: int = 200) -> List[TextChunk]:
        ##
        ## Extraccion con OCR para PDFs escaneados
        ## Cachea el resultado en disco para no repetir el proceso
        ##
        ## Parametros:
        ##   dpi: resolucion de renderizado (200 es buen balance calidad/velocidad)
        ##
        self.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.ocr_cache_dir / f"{source_name}_ocr.json"

        ## Intentar cargar desde cache
        if cache_path.exists():
            logger.info("  Cargando OCR desde cache: %s", cache_path.name)
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            chunks = [TextChunk.from_dict(d) for d in cached]
            logger.info("  %s: %d chunks cargados desde cache OCR", source_name, len(chunks))
            return chunks

        ## Ejecutar OCR pagina por pagina
        try:
            import fitz
            import pytesseract
            from PIL import Image
            import io
        except ImportError as e:
            logger.error("OCR requiere pytesseract y Pillow: pip install pytesseract Pillow")
            logger.error("Y el binario tesseract: sudo apt install tesseract-ocr")
            logger.error("Detalle: %s", str(e))
            return []

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        chunks = []
        current_chapter = ""
        current_section = ""

        for page_num in range(total_pages):
            page = doc[page_num]

            ## Renderizar pagina como imagen
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            ## Aplicar OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)

            if not text or len(text.strip()) < 30:
                continue

            chapter, section = self._detect_headers(text, current_chapter, current_section)
            if chapter:
                current_chapter = chapter
            if section:
                current_section = section

            page_chunks = self._split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < 50:
                    continue
                chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    source_file=source_name,
                    page_number=page_num + 1,
                    chapter=current_chapter,
                    section=current_section,
                    content_type="ocr_text",
                ))

            ## Log de progreso cada 20 paginas
            if (page_num + 1) % 20 == 0:
                logger.info("  OCR %s: %d/%d paginas procesadas",
                            source_name, page_num + 1, total_pages)

        doc.close()

        ## Guardar en cache
        chunks_data = [c.to_dict() for c in chunks]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info("  %s: %d chunks via OCR (cacheados en %s)",
                    source_name, len(chunks), cache_path.name)
        return chunks

    def _extract_fallback(self, pdf_path: Path, source_name: str) -> List[TextChunk]:
        ##
        ## Fallback usando PyMuPDF directo si pymupdf4llm no esta disponible
        ##
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            chunks = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                if not text.strip():
                    continue

                page_chunks = self._split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

                for chunk_text in page_chunks:
                    if len(chunk_text.strip()) < 50:
                        continue

                    chunks.append(TextChunk(
                        text=chunk_text.strip(),
                        source_file=source_name,
                        page_number=page_num + 1,
                        chapter="",
                        section="",
                        content_type="text",
                    ))

            doc.close()
            logger.info("  Extraidos %d chunks (fallback) de %s", len(chunks), source_name)
            return chunks

        except ImportError:
            logger.error("Ni pymupdf4llm ni PyMuPDF estan disponibles")
            return []

    def _detect_headers(
        self, text: str, current_chapter: str, current_section: str
    ) -> Tuple[str, str]:
        ##
        ## Detecta capitulos y secciones desde headers Markdown
        ## Retorna (chapter, section) detectados o strings vacios
        ##
        chapter = ""
        section = ""

        ## Buscar headers nivel 1 (# Chapter)
        h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if h1_match:
            chapter = h1_match.group(1).strip()[:100]

        ## Buscar headers nivel 2 (## Section)
        h2_match = re.search(r"^##\s+(.+)$", text, re.MULTILINE)
        if h2_match:
            section = h2_match.group(1).strip()[:100]

        return chapter, section

    def _detect_content_type(self, text: str) -> str:
        ##
        ## Detecta si el contenido es texto, tabla o caption de figura
        ##
        text_lower = text.lower()

        ## Detectar tablas (presencia de pipes de Markdown)
        if text.count("|") > 5:
            return "table"

        ## Detectar captions de figuras
        if re.search(r"(figure|fig\.|imagen|tabla)\s*\d+", text_lower):
            return "figure_caption"

        return "text"

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        ##
        ## Fragmenta texto en chunks de tamano fijo con overlap
        ## Intenta cortar en limites de oracion cuando es posible
        ##
        if not text.strip():
            return []

        ## Dividir en oraciones
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_length + sentence_words > chunk_size and current_chunk:
                ## Guardar chunk actual
                chunks.append(" ".join(current_chunk))

                ## Calcular overlap: mantener ultimas oraciones
                overlap_words = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_words += s_words

                current_chunk = overlap_sentences
                current_length = overlap_words

            current_chunk.append(sentence)
            current_length += sentence_words

        ## Ultimo chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class EmbeddingModel:
    ## Modelo de embeddings PubMedBERT para generar vectores de texto
    ## Usa mean pooling sobre los hidden states del modelo

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = "auto"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load(self):
        ## Carga el modelo y tokenizer
        if self.model is not None:
            return

        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info("Cargando modelo de embeddings: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Modelo de embeddings cargado en: %s", self.device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        ##
        ## Genera embeddings para una lista de textos
        ## Usa mean pooling sobre los hidden states
        ## Retorna array [n_texts, embedding_dim] normalizado
        ##
        self.load()
        import torch

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            ## Mean pooling con mascara de atencion
            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            ## Normalizar para cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _mean_pooling(self, token_embeddings, attention_mask):
        ## Mean pooling: promedio ponderado por mascara de atencion
        import torch

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class CorpusIndexer:
    ## Procesa PDFs de literatura medica y construye indice FAISS
    ##
    ## Flujo:
    ##   1. Extrae texto de cada PDF con PyMuPDF4LLM
    ##   2. Fragmenta en chunks con overlap
    ##   3. Genera embeddings con PubMedBERT
    ##   4. Construye indice FAISS (IndexFlatIP, cosine similarity)
    ##   5. Persiste indice y metadatos a disco

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index_dir: Optional[Path] = None,
        ocr_lang: str = "eng",
    ):
        self.index_dir = index_dir or Path("data/rag_index")
        ## El cache OCR se guarda junto al indice
        ocr_cache = self.index_dir.parent / "ocr_cache"
        self.extractor = PDFExtractor(ocr_cache_dir=ocr_cache, ocr_lang=ocr_lang)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunks: List[TextChunk] = []
        self.index = None

    def build_index(self, pdf_dir: str, force_rebuild: bool = False) -> int:
        ##
        ## Construye el indice FAISS a partir de los PDFs en un directorio
        ##
        ## Parametros:
        ##   pdf_dir: ruta al directorio con PDFs de literatura medica
        ##   force_rebuild: si True, reconstruye aunque exista cache
        ##
        ## Retorna: numero total de chunks indexados
        ##
        import faiss

        ## Verificar si ya existe un indice en cache
        if not force_rebuild and self._load_cached_index():
            logger.info("Indice cargado desde cache: %d chunks", len(self.chunks))
            return len(self.chunks)

        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.error("Directorio de PDFs no encontrado: %s", pdf_dir)
            return 0

        ## Buscar todos los PDFs
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error("No se encontraron archivos PDF en: %s", pdf_dir)
            return 0

        logger.info("Encontrados %d archivos PDF en %s", len(pdf_files), pdf_dir)

        ## Extraer texto de cada PDF
        self.chunks = []
        for pdf_file in pdf_files:
            file_chunks = self.extractor.extract_from_pdf(str(pdf_file))
            self.chunks.extend(file_chunks)

        logger.info("Total de chunks extraidos: %d", len(self.chunks))

        if len(self.chunks) == 0:
            logger.error("No se extrajeron chunks de los PDFs")
            return 0

        ## Generar embeddings
        logger.info("Generando embeddings con %s...", self.embedding_model.model_name)
        texts = [chunk.get_contextualized_text() for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts)

        ## Construir indice FAISS (IndexFlatIP con vectores normalizados = cosine sim)
        logger.info("Construyendo indice FAISS (dim=%d)...", embeddings.shape[1])
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

        logger.info("Indice FAISS construido con %d vectores", self.index.ntotal)

        ## Guardar en cache
        self._save_index(embeddings)

        return len(self.chunks)

    def _save_index(self, embeddings: np.ndarray):
        ## Guarda indice FAISS y metadatos a disco
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)

        ## Guardar indice FAISS
        faiss.write_index(self.index, str(self.index_dir / "faiss_index.bin"))

        ## Guardar metadatos de chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(self.index_dir / "chunks_metadata.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        ## Guardar embeddings para reconstruccion
        np.save(self.index_dir / "embeddings.npy", embeddings)

        logger.info("Indice guardado en: %s", self.index_dir)

    def _load_cached_index(self) -> bool:
        ## Intenta cargar indice y metadatos desde cache
        import faiss

        index_path = self.index_dir / "faiss_index.bin"
        chunks_path = self.index_dir / "chunks_metadata.json"

        if not index_path.exists() or not chunks_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(index_path))

            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [TextChunk.from_dict(d) for d in chunks_data]

            return True

        except Exception as e:
            logger.warning("Error cargando cache: %s", str(e))
            return False

    def get_index_stats(self) -> Dict:
        ## Retorna estadisticas del indice construido
        if not self.chunks:
            return {"status": "empty"}

        sources = {}
        for chunk in self.chunks:
            src = chunk.source_file
            if src not in sources:
                sources[src] = 0
            sources[src] += 1

        return {
            "status": "ready",
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": EMBEDDING_DIM,
            "chunks_per_source": sources,
        }


class ReportRetriever:
    ## Recupera fragmentos relevantes de literatura medica dada una query
    ##
    ## Usa el indice FAISS construido por CorpusIndexer para buscar
    ## los chunks mas similares a una query textual

    def __init__(
        self,
        corpus_indexer: CorpusIndexer,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = TOP_K,
    ):
        self.indexer = corpus_indexer
        self.embedding_model = embedding_model or corpus_indexer.embedding_model
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.3,
    ) -> List[Dict]:
        ##
        ## Recupera los fragmentos mas relevantes para una query
        ##
        ## Parametros:
        ##   query: texto de busqueda (hallazgos, BI-RADS, descriptores)
        ##   top_k: numero de resultados (default: self.top_k)
        ##   min_score: score minimo de similitud para incluir resultado
        ##
        ## Retorna: lista de diccionarios con texto, score y metadatos
        ##
        if self.indexer.index is None:
            logger.error("Indice FAISS no construido. Ejecute build_index primero.")
            return []

        k = top_k or self.top_k

        ## Generar embedding de la query
        query_embedding = self.embedding_model.encode([query])

        ## Buscar en FAISS
        scores, indices = self.indexer.index.search(
            query_embedding.astype(np.float32), k
        )

        ## Construir resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.indexer.chunks):
                continue
            if score < min_score:
                continue

            chunk = self.indexer.chunks[idx]
            results.append({
                "text": chunk.text,
                "score": float(score),
                "source": chunk.source_file,
                "page": chunk.page_number,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "content_type": chunk.content_type,
            })

        return results

    def build_query_from_findings(
        self,
        birads_pred: int,
        findings: List[str],
        density: str = "",
    ) -> str:
        ##
        ## Construye una query optimizada para el RAG a partir de
        ## los hallazgos predichos por el VLM
        ##
        ## Ejemplo de salida:
        ##   "mammography BI-RADS 4 spiculated mass heterogeneous dense breast biopsy"
        ##
        parts = ["mammography"]

        if birads_pred is not None:
            parts.append(f"BI-RADS {birads_pred}")

        if findings:
            parts.extend(findings)

        if density:
            parts.append(density)

        ## Agregar recomendacion segun BI-RADS
        if birads_pred is not None:
            if birads_pred <= 2:
                parts.append("benign routine screening")
            elif birads_pred == 3:
                parts.append("probably benign follow-up")
            elif birads_pred == 4:
                parts.append("suspicious biopsy recommended")
            elif birads_pred == 5:
                parts.append("highly suggestive malignancy immediate biopsy")

        return " ".join(parts)


def augment_prompt(
    base_prompt: str,
    retrieved_chunks: List[Dict],
    language: str = "es",
) -> str:
    ##
    ## Inyecta contexto de literatura medica en el prompt del LLM
    ##
    ## Parametros:
    ##   base_prompt: prompt original del VLM (build_prompt)
    ##   retrieved_chunks: fragmentos recuperados por ReportRetriever
    ##   language: idioma del informe final ("es" = espanol)
    ##
    ## El contexto se inserta antes de la instruccion del usuario
    ## con directivas de grounding para evitar alucinaciones
    ##

    if not retrieved_chunks:
        return base_prompt

    ## Construir bloque de contexto medico
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", 0)
        text = chunk.get("text", "")
        context_parts.append(
            f"[Referencia {i} - {source}, p.{page}]\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    ## Instrucciones de grounding segun idioma
    if language == "es":
        grounding_instruction = (
            "CONTEXTO DE LITERATURA MEDICA:\n"
            "Los siguientes fragmentos provienen de literatura medica de referencia "
            "(ACR BI-RADS Atlas, guias clinicas, textos de radiologia mamaria). "
            "Usa esta informacion UNICAMENTE como referencia para fundamentar tu "
            "analisis con terminologia clinica precisa. Los hallazgos visuales de "
            "la imagen tienen prioridad sobre el texto de referencia. No inventes "
            "hallazgos que no observes en la imagen.\n\n"
            f"{context_block}\n\n"
            "FIN DEL CONTEXTO DE REFERENCIA\n\n"
        )
    else:
        grounding_instruction = (
            "MEDICAL LITERATURE CONTEXT:\n"
            "The following excerpts come from reference medical literature "
            "(ACR BI-RADS Atlas, clinical guidelines, breast radiology textbooks). "
            "Use this information ONLY as reference to support your analysis with "
            "precise clinical terminology. Visual findings from the image take "
            "priority over reference text. Do not fabricate findings not observed "
            "in the image.\n\n"
            f"{context_block}\n\n"
            "END OF REFERENCE CONTEXT\n\n"
        )

    ## Insertar contexto en el prompt antes del bloque de usuario
    ## El prompt original tiene formato:
    ##   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...
    ##
    ## Insertamos el contexto despues del system y antes del user
    if "<|im_start|>user" in base_prompt:
        parts = base_prompt.split("<|im_start|>user")
        augmented = parts[0] + grounding_instruction + "<|im_start|>user" + parts[1]
    else:
        ## Fallback: prepend al inicio
        augmented = grounding_instruction + base_prompt

    return augmented


def create_rag_pipeline(
    pdf_dir: str,
    index_dir: Optional[str] = None,
    device: str = "auto",
    force_rebuild: bool = False,
) -> Tuple[CorpusIndexer, ReportRetriever]:
    ##
    ## Funcion de conveniencia para crear el pipeline RAG completo
    ##
    ## Parametros:
    ##   pdf_dir: ruta al directorio con PDFs de literatura medica
    ##   index_dir: ruta para guardar/cargar el indice FAISS
    ##   device: dispositivo para modelo de embeddings
    ##   force_rebuild: si True, reconstruye el indice
    ##
    ## Retorna: (indexer, retriever) listos para usar
    ##
    embedding_model = EmbeddingModel(device=device)

    idx_dir = Path(index_dir) if index_dir else Path("data/rag_index")
    indexer = CorpusIndexer(embedding_model=embedding_model, index_dir=idx_dir)

    n_chunks = indexer.build_index(pdf_dir, force_rebuild=force_rebuild)
    logger.info("Pipeline RAG construido: %d chunks indexados", n_chunks)

    retriever = ReportRetriever(corpus_indexer=indexer, embedding_model=embedding_model)

    return indexer, retriever
