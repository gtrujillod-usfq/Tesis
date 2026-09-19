## models.py
## Definicion y gestion de modelos para diagnostico mamografico con VLM.
## Version optimizada para NVIDIA H200 (141 GB HBM3e).
##
## Arquitectura principal:
##   MammoVLM: Vision Language Model compuesto por:
##     1. Encoder visual (CLIP ViT-L/14 / DINOv2-Large / ConvNeXt-Base)
##     2. Modulo de proyeccion multimodal (MLP 4 capas, BF16 nativo)
##     3. LLM base Qwen2.5-7B en precision BF16 completa (sin cuantizacion)
##        con fine-tuning via LoRA sobre capas de atencion
##
## Modelos adicionales:
##   MammoClassifier: clasificador puro de imagen (baseline de referencia)
##   VisualMoE:       selector de encoder segun densidad mamaria (3 experts activos)
##
## Decisiones de diseno H200:
##   - BF16 completo: el H200 ejecuta BF16 sin penalizacion de rendimiento.
##     Se elimina QLoRA porque con 141 GB de VRAM el modelo Qwen2.5-7B
##     cabe completo en BF16 (~14 GB) con amplio margen para batch grandes.
##   - Flash Attention 2: reduce la complejidad cuadratica de la atencion.
##     El H200 tiene ancho de banda HBM3e suficiente para aprovechar FA2.
##   - torch.compile(mode="max-autotune"): genera kernels CUDA optimizados
##     especificamente para la arquitectura Hopper (SM90).
##   - LoRA full BF16: se mantiene LoRA (no QLoRA) para controlar el numero
##     de parametros entrenables y conservar la capacidad de comparar
##     experimentos con distintos rangos r sin reentrenar el backbone.
##   - VisualMoE con todos los experts activos: a diferencia de la version
##     de bajo recurso donde los experts estaban congelados, aqui se
##     entrena el MoE completo porque la VRAM lo permite.
##   - TF32 habilitado para matmul: aprovecha el hardware tensor core del H200
##     con precision suficiente para entrenamiento de redes neuronales.

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

## Verificacion de Flash Attention 2 ANTES de importar transformers.
##
## Problema: transformers intenta importar flash_attn a nivel de modulo
## dentro de modeling_flash_attention_utils.py. Si el binario .so fue compilado
## contra una version de PyTorch distinta a la instalada actualmente, se
## produce un "undefined symbol" RuntimeError que aborta toda la importacion
## de transformers, incluso si el codigo propio ya tenia un try/except.
##
## Solucion: verificar si flash_attn carga correctamente ANTES de que
## transformers lo intente. Si falla, registrar sus submódulos como None en
## sys.modules. Cuando sys.modules[nombre] = None, cualquier intento posterior
## de "import flash_attn" lanza ImportError en lugar de RuntimeError,
## que es lo que transformers espera manejar con is_flash_attn_2_available().
##
## Alternativa de usuario si quiere Flash Attention 2 real:
##   pip uninstall flash-attn -y
##   pip install flash-attn --no-build-isolation
##   (recompila el .so contra el PyTorch actual, tarda ~10 min)

_FLASH_ATTN_AVAILABLE = False

try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("flash_attn")
    if _spec is not None:
        ## El paquete esta instalado en disco; verificar que el .so carga
        import flash_attn.flash_attn_interface as _fai_probe  # noqa: F401
        _FLASH_ATTN_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    ## ABI incompatible o .so corrupto: enmascarar todos los submódulos
    ## relevantes para que transformers los trate como no instalados
    _broken_mods = [
        "flash_attn",
        "flash_attn.bert_padding",
        "flash_attn.flash_attn_interface",
        "flash_attn_2_cuda",
    ]
    for _mod_name in _broken_mods:
        sys.modules[_mod_name] = None  # type: ignore[assignment]

    logging.getLogger("models").warning(
        "flash_attn instalado pero incompatible con PyTorch actual (ABI mismatch). "
        "Se usara attn_implementation='sdpa' (PyTorch nativo, casi identico en velocidad "
        "en H200). Para rehabilitar Flash Attention 2: "
        "pip uninstall flash-attn -y && pip install flash-attn --no-build-isolation"
    )

## Ahora es seguro importar transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT no disponible. LoRA desactivado. Instalar con: pip install peft")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm no disponible. Instalar con: pip install timm")

logger = logging.getLogger("models")

## Configuracion global de precision para H200 (Hopper SM90)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_float32_matmul_precision("high")

## Alias publico del flag interno para que el resto del modulo lo use
FLASH_ATTN_AVAILABLE = _FLASH_ATTN_AVAILABLE

## Dtype de computacion global para H200.
## BF16 es preferido sobre FP16 porque tiene rango dinamico equivalente a FP32
## (8 bits de exponente) lo que elimina los overflow tipicos de FP16 en LLMs.
COMPUTE_DTYPE = torch.bfloat16

## LLM base: Qwen2.5-7B-Instruct en BF16 ocupa ~14 GB de los 141 GB del H200.
## Con batch_size=64 y secuencias de 512 tokens el consumo total es ~40-60 GB,
## dejando amplio margen para los encoders visuales y gradientes.
## Para experimentos con mayor capacidad se puede escalar a Qwen2.5-72B (~144 GB).
DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"

## Encoders visuales disponibles y sus nombres en HuggingFace/timm.
## Se usan versiones Large/Base en lugar de las Tiny de la version de bajo recurso.
VISUAL_ENCODERS = {
    "clip":      "openai/clip-vit-large-patch14",   ## ViT-L/14: 1024 dim
    "dinov2":    "facebook/dinov2-large",             ## Large: 1024 dim (mejor que base)
    "convnext":  "convnext_base.fb_in22k",            ## Base: 1024 dim
    "resnet50":  "resnet50.a1_in1k",
}

## Dimension de salida de cada encoder
ENCODER_OUTPUT_DIM = {
    "clip":     1024,
    "dinov2":   1024,   ## Large (era 768 en base)
    "convnext": 1024,   ## Base (era 768 en tiny)
    "resnet50": 2048,
}


@dataclass
class ModelConfig:
    """
    Configuracion centralizada del modelo para H200.

    Cambios respecto a la version de bajo recurso:
      - use_qlora eliminado: se usa BF16 completo.
      - llm_name: Qwen2.5-7B en lugar de 3B.
      - use_flash_attention: habilita Flash Attention 2 en el LLM.
      - compile_model: aplica torch.compile con kernels Hopper optimizados.
      - encoders Large en lugar de Tiny/Base.
    """
    llm_name: str = DEFAULT_LLM
    visual_encoder: str = "clip"
    use_moe: bool = True               ## MoE activo con todos los experts entrenables
    image_embed_dim: int = 512         ## Aumentado de 256 a 512 (mas capacidad)
    projection_layers: int = 4
    projection_dropout: float = 0.1
    use_flash_attention: bool = True   ## Flash Attention 2 para el LLM
    compile_model: bool = True         ## torch.compile modo max-autotune
    lora_r: int = 64                   ## Rango LoRA mayor: mas capacidad con H200
    lora_alpha: int = 128              ## alpha = 2 * r es la heuristica estandar
    lora_dropout: float = 0.05
    lora_target_modules: list = None
    num_classes: int = 2
    max_new_tokens: int = 512          ## Mas tokens en generacion (antes 256)
    device: str = "auto"

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


def get_device(device_str: str = "auto") -> torch.device:
    """
    Selecciona el dispositivo de computo de forma inteligente.
    El orden de preferencia es: CUDA > MPS (Apple Silicon) > CPU.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Dispositivo: CUDA (%s)", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Dispositivo: MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Dispositivo: CPU (modo portabilidad)")
    else:
        device = torch.device(device_str)
        logger.info("Dispositivo forzado: %s", device)
    return device


class ProjectionMLP(nn.Module):
    """
    Modulo de proyeccion multimodal que alinea el espacio visual con el
    espacio textual del LLM.

    Arquitectura: Linear -> LayerNorm -> GELU -> Dropout (x n_layers)

    Basado en el modulo UMiCon de MammoVLM pero simplificado para
    entrenamiento en hardware limitado. La clave es la normalizacion por
    capa que estabiliza el entrenamiento cuando los features visuales
    y textuales tienen escalas muy diferentes.

    Parametros
    ----------
    input_dim : int
        Dimension de entrada (output del encoder visual).
    output_dim : int
        Dimension de salida (debe coincidir con embedding_dim del LLM).
    hidden_dim : int
        Dimension interna. Por defecto = max(input_dim, output_dim).
    n_layers : int
        Numero de bloques MLP.
    dropout : float
        Tasa de dropout para regularizacion.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)

        layers = []
        current_dim = input_dim
        for i in range(n_layers):
            next_dim = output_dim if i == n_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim

        ## Eliminamos la activacion y dropout de la ultima capa para
        ## que el vector de salida pueda tomar cualquier valor real
        ## (igual que los embeddings del tokenizer)
        del layers[-2:]   ## Eliminar ultimo GELU y Dropout

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Inicializacion de Xavier para convergencia estable."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class VisualEncoder(nn.Module):
    """
    Encoder visual que wrappea modelos preentrenados de HuggingFace o timm.
    Devuelve un vector de embedding por imagen (pooling del CLS token o
    global average pooling segun el modelo).

    Soporta congelacion parcial: solo se afina la ultima capa del encoder
    para preservar las representaciones aprendidas en preentrenamiento
    y reducir la cantidad de parametros entrenables.

    Parametros
    ----------
    encoder_name : str
        Clave en VISUAL_ENCODERS ("clip", "dinov2", "convnext", "resnet50").
    freeze_layers : int
        Numero de bloques a congelar desde el inicio. -1 = congelar todo.
        0 = entrenar todo (no recomendado con poco datos).
    """

    def __init__(self, encoder_name: str = "clip", freeze_layers: int = -1):
        super().__init__()

        self.encoder_name = encoder_name
        self.output_dim = ENCODER_OUTPUT_DIM.get(encoder_name, 768)

        if encoder_name == "clip":
            self.model = self._load_clip_encoder()
        elif encoder_name == "dinov2":
            self.model = self._load_dinov2_encoder()
        elif encoder_name in ("convnext", "resnet50"):
            self.model = self._load_timm_encoder(encoder_name)
        else:
            raise ValueError(f"Encoder no soportado: {encoder_name}. Opciones: {list(VISUAL_ENCODERS.keys())}")

        self._apply_freezing(freeze_layers)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Encoder '%s': %d/%d parametros entrenables (%.1f%%)",
            encoder_name, trainable, total, 100 * trainable / max(total, 1),
        )

    def _load_clip_encoder(self) -> nn.Module:
        """Carga el encoder visual de CLIP (ViT-L/14)."""
        try:
            from transformers import CLIPVisionModel
            model = CLIPVisionModel.from_pretrained(VISUAL_ENCODERS["clip"])
            self.output_dim = model.config.hidden_size
            return model
        except Exception as e:
            logger.error("Error cargando CLIP: %s. Verifique conexion a HuggingFace.", e)
            raise

    def _load_dinov2_encoder(self) -> nn.Module:
        """Carga DINOv2 con pooling de CLS token."""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(VISUAL_ENCODERS["dinov2"])
            self.output_dim = model.config.hidden_size
            return model
        except Exception as e:
            logger.error("Error cargando DINOv2: %s", e)
            raise

    def _load_timm_encoder(self, name: str) -> nn.Module:
        """Carga encoder de timm sin cabeza de clasificacion."""
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm requerido para ConvNeXt/ResNet. Instalar: pip install timm")
        model = timm.create_model(
            VISUAL_ENCODERS[name],
            pretrained=True,
            num_classes=0,   ## Sin clasificador final
            global_pool="avg",
        )
        self.output_dim = model.num_features
        return model

    def _apply_freezing(self, freeze_layers: int):
        """
        Congela capas del encoder para reducir memoria y evitar catastrophic forgetting.
        freeze_layers=-1 congela todo excepto la ultima capa de LayerNorm
        (util para transferencia a imagenes medicas muy distintas a ImageNet).
        """
        if freeze_layers == 0:
            return

        params = list(self.model.parameters())
        if freeze_layers == -1:
            ## Congelar todos los parametros
            for p in params:
                p.requires_grad = False
        else:
            ## Congelar los primeros 'freeze_layers' grupos de parametros
            n = len(params)
            freeze_up_to = min(freeze_layers * 12, n - 12)  ## Mantener al menos los ultimos 12 params
            for i, p in enumerate(params):
                if i < freeze_up_to:
                    p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extrae embedding visual de la imagen.

        Parametros
        ----------
        pixel_values : torch.Tensor
            Tensor de imagen [batch, 3, H, W] normalizado con stats ImageNet.

        Retorna
        -------
        torch.Tensor
            Embedding [batch, output_dim].
        """
        if self.encoder_name == "clip":
            outputs = self.model(pixel_values=pixel_values)
            ## CLS token del ultimo bloque transformer
            return outputs.pooler_output

        elif self.encoder_name == "dinov2":
            outputs = self.model(pixel_values=pixel_values)
            return outputs.pooler_output

        else:
            ## timm: retorna directamente el vector tras global average pooling
            return self.model(pixel_values)


class VisualMoE(nn.Module):
    """
    Mixture-of-Experts visual que selecciona el encoder mas adecuado
    segun la densidad mamaria de cada imagen.

    Motivacion (de MammoVLM):
      - Mamas de densidad A (grasa): CLIP captura mejor los patrones globales
      - Mamas de densidad B: DINOv2 da features mas robustos
      - Mamas de densidad C y D (densas): ConvNeXt tiene mejor resolucion local

    El selector es un ResNet-50 ligero que predice la densidad (A/B/C/D)
    y redirige el forward a uno de los tres experts.

    En inferencia sin GPU, el MoE puede desactivarse y usar solo un encoder
    (use_moe=False en ModelConfig) para ahorrar memoria.

    Parametros
    ----------
    output_dim : int
        Dimension de salida de todos los experts (deben coincidir).
    freeze_experts : bool
        Si True, los experts estan congelados y solo el selector se entrena.
    """

    ## Asignacion densidad -> encoder experto
    DENSITY_EXPERT = {0: "clip", 1: "dinov2", 2: "convnext", 3: "convnext"}

    def __init__(self, output_dim: int = 768, freeze_experts: bool = True):
        super().__init__()

        self.output_dim = output_dim

        ## Tres encoders especializados
        self.clip_encoder    = VisualEncoder("clip",     freeze_layers=-1 if freeze_experts else 0)
        self.dinov2_encoder  = VisualEncoder("dinov2",   freeze_layers=-1 if freeze_experts else 0)
        self.convnext_encoder = VisualEncoder("convnext", freeze_layers=-1 if freeze_experts else 0)

        ## Proyecciones individuales para unificar dimensiones de salida
        self.clip_proj = nn.Linear(self.clip_encoder.output_dim, output_dim)
        self.dinov2_proj = nn.Linear(self.dinov2_encoder.output_dim, output_dim)
        self.convnext_proj = nn.Linear(self.convnext_encoder.output_dim, output_dim)

        ## Selector de densidad (ResNet-50 ligero)
        if TIMM_AVAILABLE:
            self.density_selector = timm.create_model(
                VISUAL_ENCODERS["resnet50"],
                pretrained=True,
                num_classes=4,  ## A, B, C, D
            )
        else:
            ## Fallback: selector simple con una CNN ligera
            self.density_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(3 * 49, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
            )

        logger.info("VisualMoE inicializado con 3 encoders especializados")

    def forward(
        self,
        pixel_values: torch.Tensor,
        density_labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward del MoE.

        Si density_labels esta disponible (entrenamiento), lo usamos directamente.
        Si no (inferencia), el selector predice la densidad.

        Retorna (embeddings, density_logits):
            - embeddings:      [batch, output_dim]
            - density_logits:  [batch, 4] para supervision auxiliar
        """
        batch_size = pixel_values.shape[0]

        ## Prediccion de densidad
        density_logits = self.density_selector(pixel_values)
        predicted_density = density_logits.argmax(dim=-1)

        ## Usar etiquetas de densidad conocidas cuando esten disponibles
        if density_labels is not None:
            routing_density = density_labels
        else:
            routing_density = predicted_density

        embeddings = torch.zeros(batch_size, self.output_dim, device=pixel_values.device)

        ## Routing: procesar cada muestra con su encoder asignado
        for density_idx, encoder_name in self.DENSITY_EXPERT.items():
            mask = (routing_density == density_idx)
            if not mask.any():
                continue

            subset = pixel_values[mask]
            if encoder_name == "clip":
                feat = self.clip_proj(self.clip_encoder(subset))
            elif encoder_name == "dinov2":
                feat = self.dinov2_proj(self.dinov2_encoder(subset))
            else:
                feat = self.convnext_proj(self.convnext_encoder(subset))

            embeddings[mask] = feat

        return embeddings, density_logits


class MammoVLM(nn.Module):
    """
    Vision Language Model principal para diagnostico mamografico.

    Flujo de datos:
      imagen  -> encoder visual -> proyeccion -> [VIS_TOKEN]
      pregunta -> tokenizer -> embedding LLM -> [TXT_TOKENS]
      [VIS_TOKEN] + [TXT_TOKENS] -> LLM -> respuesta diagnostica

    La imagen se codifica como un unico token visual (o una secuencia corta)
    que se prepend a los tokens de texto. Esto permite que el LLM "vea" la
    imagen a traves del mecanismo de atencion sin modificar su arquitectura.

    Modos de uso:
      1. Clasificacion: predice BI-RADS (cabeza clasificadora)
      2. Generacion: genera reporte diagnostico en texto libre
      3. VQA: responde preguntas del usuario sobre la mamografia

    Parametros
    ----------
    config : ModelConfig
        Configuracion del modelo con todos los hiperparametros.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = get_device(config.device)

        ## 1. Encoder visual: MoE con experts Large, todos entrenables en H200
        if config.use_moe and TIMM_AVAILABLE:
            self.visual_encoder = VisualMoE(
                output_dim=config.image_embed_dim,
                freeze_experts=False,   ## H200: entrenar todos los experts
            )
            visual_out_dim = config.image_embed_dim
        else:
            self.visual_encoder = VisualEncoder(
                encoder_name=config.visual_encoder,
                freeze_layers=0,   ## H200: no congelar encoder desde el inicio
            )
            visual_out_dim = self.visual_encoder.output_dim

        ## 2. LLM en BF16 con Flash Attention 2 y LoRA
        self.tokenizer, self.llm = self._load_llm(config)
        llm_embed_dim = self.llm.config.hidden_size

        ## 3. Proyeccion visual -> espacio LLM
        self.projection = ProjectionMLP(
            input_dim=visual_out_dim,
            output_dim=llm_embed_dim,
            n_layers=config.projection_layers,
            dropout=config.projection_dropout,
        )

        ## 4. Cabeza de clasificacion
        self.classifier = nn.Sequential(
            nn.Linear(llm_embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.num_classes),
        )

        self._add_special_tokens()

        ## Convertir modulos propios a BF16 (el LLM ya esta en BF16 por device_map)
        self.visual_encoder.to(dtype=COMPUTE_DTYPE)
        self.projection.to(dtype=COMPUTE_DTYPE)
        self.classifier.to(dtype=COMPUTE_DTYPE)

        ## torch.compile: genera kernels CUDA especificos para SM90 (Hopper).
        ## mode="max-autotune" hace profiling de kernels al inicio (~5 min)
        ## y luego ejecuta a velocidad maxima durante el resto del entrenamiento.
        ## Se aplica solo al encoder y proyeccion, no al LLM (ya optimizado internamente).
        if config.compile_model and torch.cuda.is_available():
            logger.info("Compilando encoder y proyeccion con torch.compile max-autotune...")
            self.visual_encoder = torch.compile(
                self.visual_encoder, mode="max-autotune", fullgraph=False
            )
            self.projection = torch.compile(
                self.projection, mode="max-autotune", fullgraph=True
            )
            logger.info("Compilacion completada")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "MammoVLM (H200): %dM/%dM parametros entrenables (%.1f%%) | dtype=%s",
            trainable // 1_000_000, total // 1_000_000,
            100 * trainable / max(total, 1), COMPUTE_DTYPE,
        )

    def _load_llm(self, config: ModelConfig) -> tuple:
        """
        Carga el LLM en BF16 completo con Flash Attention 2.

        En H200 con 141 GB de VRAM no hay necesidad de cuantizacion.
        Qwen2.5-7B en BF16 ocupa ~14 GB, dejando ~127 GB para los encoders,
        gradientes, activaciones y el batch de imagenes.

        Flash Attention 2 reduce la complejidad de atencion de O(n^2) a O(n)
        en VRAM y acelera la inferencia ~2-3x en secuencias largas.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ## Seleccion de implementacion de atencion.
        ##
        ## Jerarquia de preferencia:
        ##   1. flash_attention_2: mas rapida, requiere flash-attn compilado correctamente.
        ##   2. sdpa: scaled_dot_product_attention nativo de PyTorch >= 2.0.
        ##      En H200 usa kernels cuDNN Flash-Attention internos, rendimiento
        ##      equivalente a flash_attention_2 sin dependencia externa.
        ##   3. eager: implementacion de referencia, solo para debugging.
        if config.use_flash_attention and FLASH_ATTN_AVAILABLE:
            attn_impl = "flash_attention_2"
            logger.info("Cargando LLM con Flash Attention 2 en BF16")
        else:
            attn_impl = "sdpa"
            logger.info(
                "Cargando LLM con attn_implementation='sdpa' (PyTorch nativo). "
                "Velocidad en H200 equivalente a Flash Attention 2. "
                "Flash Attention 2 no disponible: %s",
                "ABI mismatch" if not FLASH_ATTN_AVAILABLE else "desactivado en config",
            )

        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=COMPUTE_DTYPE,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        ## Aplicar LoRA en BF16 (sin cuantizacion de base)
        if PEFT_AVAILABLE:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            llm = get_peft_model(llm, lora_config)
            llm.print_trainable_parameters()
        else:
            logger.warning("PEFT no disponible. Entrenando el LLM completo sin LoRA.")

        return tokenizer, llm

    def _add_special_tokens(self):
        """
        Agrega token especial [IMG] para indicar la posicion del embedding
        visual en la secuencia de entrada del LLM.
        """
        special_tokens = {"additional_special_tokens": ["[IMG]", "[/IMG]"]}
        n_added = self.tokenizer.add_special_tokens(special_tokens)
        if n_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            logger.info("Tokens especiales agregados: [IMG], [/IMG]")

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        density_labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Codifica la imagen y proyecta al espacio del LLM.

        Retorna (visual_embeds, density_logits):
            - visual_embeds:  [batch, llm_embed_dim]
            - density_logits: [batch, 4] o None si no usa MoE
        """
        if isinstance(self.visual_encoder, VisualMoE):
            visual_features, density_logits = self.visual_encoder(
                pixel_values, density_labels
            )
        else:
            visual_features = self.visual_encoder(pixel_values)
            density_logits = None

        visual_embeds = self.projection(visual_features)
        return visual_embeds, density_logits

    def build_prompt(self, question: str = None) -> str:
        """
        Construye el prompt estandar para la tarea de diagnostico.
        El [IMG] marker indica donde se insertara el embedding visual.

        Para VQA: incluye la pregunta del usuario.
        Para clasificacion: usa prompt de diagnostico estandar.
        """
        if question is None:
            question = (
                "Analice esta mamografia y proporcione: "
                "1) categoria BI-RADS, "
                "2) descripcion de hallazgos, "
                "3) recomendacion clinica."
            )

        prompt = (
            f"<|im_start|>system\n"
            f"Usted es un asistente de radiologia especializado en mamografia. "
            f"Proporciona diagnosticos precisos, profesionales y clinicamente utiles.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[IMG]imagen mamografica[/IMG]\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def forward(
        self,
        image: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        density_labels: Optional[torch.Tensor] = None,
        mode: str = "classify",
    ) -> dict:
        """
        Forward pass del modelo.

        Parametros
        ----------
        image : torch.Tensor
            Batch de imagenes [batch, 3, H, W].
        input_ids : torch.Tensor, opcional
            Tokens del prompt [batch, seq_len].
        attention_mask : torch.Tensor, opcional
            Mascara de atencion [batch, seq_len].
        labels : torch.Tensor, opcional
            Etiquetas para la loss. Para clasificacion: [batch] con clase.
            Para generacion: [batch, seq_len] con tokens de respuesta.
        density_labels : torch.Tensor, opcional
            Etiquetas de densidad [batch] para supervision del selector MoE.
        mode : str
            "classify": prediccion de BI-RADS con clasificador.
            "generate":  generacion de reporte en texto libre.
            "both":      ambas salidas simultaneamente.

        Retorna
        -------
        dict con claves: loss, logits, generated_ids, density_logits
        """
        outputs = {}
        total_loss = torch.tensor(0.0, device=image.device)

        ## Codificar imagen
        visual_embeds, density_logits = self.encode_image(image, density_labels)
        outputs["density_logits"] = density_logits

        ## Loss auxiliar de densidad (cuando el selector MoE tiene supervision)
        if density_logits is not None and density_labels is not None:
            density_loss = F.cross_entropy(density_logits, density_labels)
            total_loss = total_loss + 0.1 * density_loss   ## Peso 0.1: tarea auxiliar
            outputs["density_loss"] = density_loss

        if mode in ("classify", "both"):
            ## Para clasificacion: pasar visual_embeds directamente al clasificador
            logits = self.classifier(visual_embeds)
            outputs["logits"] = logits

            if labels is not None:
                cls_loss = F.cross_entropy(logits, labels)
                total_loss = total_loss + cls_loss
                outputs["cls_loss"] = cls_loss

        if mode in ("generate", "both") and input_ids is not None:
            ## Para generacion: insertar visual_embeds en la secuencia de embeddings
            llm_inputs = self._prepare_llm_inputs(
                visual_embeds, input_ids, attention_mask
            )
            llm_out = self.llm(
                inputs_embeds=llm_inputs["inputs_embeds"],
                attention_mask=llm_inputs["attention_mask"],
                labels=labels if mode == "generate" else None,
            )
            if llm_out.loss is not None:
                total_loss = total_loss + llm_out.loss
                outputs["lm_loss"] = llm_out.loss

            outputs["lm_logits"] = llm_out.logits

        outputs["loss"] = total_loss
        return outputs

    def _prepare_llm_inputs(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> dict:
        """
        Construye los inputs del LLM insertando el embedding visual
        antes de los tokens de texto.

        El visual_embed se trata como un token adicional al inicio de la
        secuencia, concatenando los embeddings del texto que siguen.
        """
        ## Obtener embeddings de texto del LLM
        embed_layer = self.llm.get_input_embeddings()
        text_embeds = embed_layer(input_ids)   ## [batch, seq, dim]

        ## Expandir visual_embeds para tener dimension de secuencia
        vis_seq = visual_embeds.unsqueeze(1)   ## [batch, 1, dim]

        ## Concatenar: [VIS | TEXT...]
        combined_embeds = torch.cat([vis_seq, text_embeds], dim=1)

        ## Extender la mascara de atencion para el token visual
        if attention_mask is not None:
            vis_mask = torch.ones(
                attention_mask.shape[0], 1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_mask = torch.cat([vis_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        return {"inputs_embeds": combined_embeds, "attention_mask": combined_mask}

    @torch.no_grad()
    def generate_report(
        self,
        image: torch.Tensor,
        question: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,  ## Temperatura baja para diagnosticos mas deterministas
        do_sample: bool = False,
    ) -> list:
        """
        Genera un reporte diagnostico en texto libre para un batch de imagenes.

        Temperatura 0.1 (casi greedy): preferible en aplicaciones medicas donde
        la reproducibilidad es importante y no queremos variabilidad creativa.

        Retorna lista de strings con los reportes generados.
        """
        self.eval()

        prompt = self.build_prompt(question)
        tokenized = self.tokenizer(
            [prompt] * image.shape[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(image.device)

        visual_embeds, _ = self.encode_image(image)
        llm_inputs = self._prepare_llm_inputs(
            visual_embeds,
            tokenized["input_ids"],
            tokenized["attention_mask"],
        )

        generated = self.llm.generate(
            inputs_embeds=llm_inputs["inputs_embeds"],
            attention_mask=llm_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        reports = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return reports


class MammoClassifier(nn.Module):
    """
    Clasificador de mamografia basado solo en vision (sin LLM).
    Sirve como baseline de referencia para comparar con MammoVLM y
    para la fase inicial de entrenamiento antes de incorporar el LLM.

    Arquitectura: VisualEncoder -> GlobalAvgPool -> MLP -> Softmax

    Parametros
    ----------
    encoder_name : str
        Encoder visual a usar.
    num_classes : int
        Numero de clases (2 para binario, 7 para BI-RADS 0-6).
    freeze_encoder : bool
        Si True, el encoder esta congelado (solo se entrena la cabeza).
    """

    def __init__(
        self,
        encoder_name: str = "clip",
        num_classes: int = 2,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = VisualEncoder(
            encoder_name=encoder_name,
            freeze_layers=-1 if freeze_encoder else 0,
        )

        feat_dim = self.encoder.output_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes
        logger.info(
            "MammoClassifier: encoder=%s, clases=%d, encoder_congelado=%s",
            encoder_name, num_classes, freeze_encoder,
        )

    def forward(
        self,
        image: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parametros
        ----------
        image : torch.Tensor
            [batch, 3, H, W]
        labels : torch.Tensor, opcional
            Etiquetas de clase [batch] para calcular la loss.

        Retorna dict con 'logits' y opcionalmente 'loss'.
        """
        features = self.encoder(image)
        logits = self.head(features)

        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = F.cross_entropy(logits, labels)

        return outputs

    def unfreeze_encoder(self, last_n_layers: int = 2):
        """
        Descongela las ultimas n capas del encoder para fine-tuning.
        Util en una segunda fase de entrenamiento tras estabilizar la cabeza.
        """
        params = list(self.encoder.model.parameters())
        n = len(params)
        unfreeze_from = max(0, n - last_n_layers * 12)
        for i, p in enumerate(params):
            if i >= unfreeze_from:
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Encoder parcialmente descongelado: %d parametros entrenables", trainable)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    """
    Guarda checkpoint completo del modelo.

    Se guardan: pesos del modelo, estado del optimizer, epoca, metricas
    y configuracion para poder reanudar el entrenamiento exactamente.

    Retorna la ruta del archivo guardado.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ## Para modelos PEFT (QLoRA), solo guardar los adaptadores LoRA
    ## Los pesos del LLM base no se guardan (estan en HuggingFace Hub)
    if PEFT_AVAILABLE and isinstance(model, MammoVLM) and hasattr(model.llm, "save_pretrained"):
        lora_dir = save_path / f"lora_epoch_{epoch}"
        model.llm.save_pretrained(str(lora_dir))
        logger.info("Adaptadores LoRA guardados en: %s", lora_dir)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("llm.")  ## Excluir pesos del LLM base
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    ckpt_path = save_path / filename
    torch.save(checkpoint, ckpt_path)

    if is_best:
        best_path = save_path / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info("Mejor modelo guardado en: %s (metrics=%s)", best_path, metrics)

    logger.info("Checkpoint guardado: %s (epoch=%d)", ckpt_path, epoch)
    return str(ckpt_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
) -> dict:
    """
    Carga un checkpoint guardado con save_checkpoint.

    strict=False permite cargar checkpoints parciales (p.ej. cuando
    la arquitectura cambio ligeramente entre experimentos).

    Retorna el diccionario con epoch y metricas.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=strict
    )

    if missing:
        logger.warning("Pesos no encontrados en checkpoint: %d capas", len(missing))
    if unexpected:
        logger.warning("Pesos inesperados en checkpoint: %d capas", len(unexpected))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        "Checkpoint cargado desde '%s' (epoch=%d)",
        checkpoint_path, checkpoint.get("epoch", -1),
    )
    return {"epoch": checkpoint.get("epoch", 0), "metrics": checkpoint.get("metrics", {})}


def count_parameters(model: nn.Module) -> dict:
    """
    Cuenta los parametros del modelo separando entrenables de congelados.
    Util para reportar en el notebook y verificar que QLoRA funciona.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    stats = {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(100 * trainable / max(total, 1), 2),
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }
    logger.info(
        "Parametros del modelo: total=%.2fM, entrenables=%.2fM (%.1f%%)",
        stats["total_M"], stats["trainable_M"], stats["trainable_pct"],
    )
    return stats
