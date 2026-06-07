## models.py
## Arquitectura del MammoVLM V2
## Area 2: Razonamiento Multimodal con Mammo-CLIP + Dual-Head
##
## Arquitectura:
##   1. Encoder visual: Mammo-CLIP (EfficientNet-B5, especializado en mamografia, alta resolucion)
##   2. Dual-head paralelo:
##      - Head BI-RADS: clasificacion multiclase 0-5
##      - Heads de hallazgos: un head multiclase por cada slot del esquema
##   3. Generador de informes: LLM (Qwen2.5) con contexto RAG
##
## El diseño dual-head separa la prediccion estructurada (BI-RADS + hallazgos)
## de la generacion de texto, lo que da interpretabilidad: cada decision
## clinica es trazable a un head especifico.

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MammoCLIPEncoder(nn.Module):
    ## Encoder visual basado en Mammo-CLIP (EfficientNet-B5)
    ##
    ## Mammo-CLIP esta preentrenado especificamente en mamografia (pares
    ## mamograma-reporte de UPMC + dataset VinDr), a alta resolucion. A
    ## diferencia de BiomedCLIP (generalista, 224x224), Mammo-CLIP preserva
    ## mucho mejor las caracteristicas mamograficas finas como
    ## microcalcificaciones, porque trabaja a resolucion alta de forma nativa.
    ##
    ## El backbone es un EfficientNet-B5 (via timm), que produce features de
    ## 2048 dimensiones. EfficientNet, al ser CNN, acepta imagenes de alta
    ## resolucion con flexibilidad (no tiene tamano de entrada fijo como un ViT).
    ##
    ## Referencia: Ghosh et al., "Mammo-CLIP", MICCAI 2024.
    ## Licencia del checkpoint: CC BY-NC-SA 4.0 (uso academico no comercial).

    ## Constantes de normalizacion ImageNet (las que usa EfficientNet de timm)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        checkpoint_path: str,
        efficientnet_name: str = "efficientnet-b5",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        ##
        ## Parametros:
        ##   checkpoint_path: ruta al checkpoint .tar de Mammo-CLIP
        ##     (ej. b5-model-best-epoch-7.tar)
        ##   efficientnet_name: arquitectura del backbone en la libreria
        ##     efficientnet_pytorch (B5 -> 'efficientnet-b5'). IMPORTANTE:
        ##     Mammo-CLIP fue entrenado con la libreria efficientnet_pytorch
        ##     (lukemelas), NO con timm. Los nombres de las capas difieren,
        ##     por eso hay que usar la misma libreria para que los pesos carguen.
        ##   freeze_backbone: si True, congela el encoder
        ##   unfreeze_last_n_blocks: numero de bloques finales a descongelar
        ##     para fine-tuning parcial. 0 = encoder totalmente congelado.
        ##
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.efficientnet_name = efficientnet_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.backbone = None
        self.preprocess = None
        self._feature_dim = 2048  ## dimension de salida de EfficientNet-B5

    def load_backbone(self):
        ## Carga el EfficientNet-B5 y le aplica los pesos de Mammo-CLIP
        if self.backbone is not None:
            return

        from efficientnet_pytorch import EfficientNet

        ## Construir el backbone EfficientNet-B5 con efficientnet_pytorch
        ## (la misma libreria con la que se entreno Mammo-CLIP, para que los
        ## nombres de las capas coincidan con los del checkpoint).
        ## No se cargan pesos de ImageNet (los reemplazamos por los de Mammo-CLIP).
        logger.info("Construyendo backbone efficientnet_pytorch: %s", self.efficientnet_name)
        self.backbone = EfficientNet.from_name(self.efficientnet_name)

        ## Cargar los pesos de Mammo-CLIP desde el checkpoint .tar
        self._load_mammoclip_weights()

        ## Congelar / descongelar segun configuracion
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone Mammo-CLIP (EfficientNet-B5) congelado")

            if self.unfreeze_last_n_blocks > 0:
                self._unfreeze_last_blocks(self.unfreeze_last_n_blocks)

        logger.info("Mammo-CLIP cargado correctamente (feature_dim=%d)", self._feature_dim)

    def _load_mammoclip_weights(self):
        ##
        ## Carga los pesos del image encoder de Mammo-CLIP en el backbone
        ##
        ## El checkpoint .tar de Mammo-CLIP es un dict con la clave 'model',
        ## cuyo state_dict contiene el modelo CLIP completo. El image encoder
        ## (EfficientNet-B5) esta bajo el prefijo 'image_encoder.' y tiene 852
        ## claves (verificado con el script de inspeccion). El resto de claves
        ## (text_encoder, image_projection, text_projection, logit_scale) no se
        ## usan para clasificacion y se descartan.
        ##
        from pathlib import Path

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(
                "Checkpoint de Mammo-CLIP no encontrado en %s. "
                "El backbone usara inicializacion aleatoria.",
                ckpt_path,
            )
            return

        logger.info("Cargando pesos de Mammo-CLIP desde %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        ## Localizar el state_dict del modelo
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        ## Extraer las claves del image encoder (prefijo 'image_encoder.')
        ## y quitar ese prefijo para que coincidan con el backbone
        prefix = "image_encoder."
        backbone_keys = set(self.backbone.state_dict().keys())
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                if new_key in backbone_keys:
                    remapped[new_key] = v

        if len(remapped) == 0:
            logger.warning(
                "No se mapeo ninguna clave del image encoder. "
                "Revisar la estructura del checkpoint con el script de inspeccion."
            )
            return

        missing, unexpected = self.backbone.load_state_dict(remapped, strict=False)
        logger.info(
            "Pesos de Mammo-CLIP cargados: %d/%d claves del backbone EfficientNet-B5",
            len(remapped), len(backbone_keys),
        )
        if len(missing) > 0:
            logger.info("Claves del backbone sin cargar (primeras 5): %s", list(missing)[:5])

    def _unfreeze_last_blocks(self, n_blocks: int):
        ##
        ## Descongela los ultimos n_blocks bloques del EfficientNet
        ##
        ## En efficientnet_pytorch, los bloques estan en self.backbone._blocks
        ## (una lista de MBConvBlocks). Descongelamos los ultimos n_blocks,
        ## ademas de la cabeza convolucional final (_conv_head, _bn1).
        ##
        if not hasattr(self.backbone, "_blocks"):
            logger.warning("El backbone no tiene atributo '_blocks' para descongelar")
            return

        blocks = list(self.backbone._blocks)
        total = len(blocks)
        n_to_unfreeze = min(n_blocks, total)

        ## Descongelar los ultimos n_to_unfreeze bloques MBConv
        for block in blocks[-n_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

        ## Descongelar tambien las capas finales (conv_head + batchnorm)
        ## que vienen despues de los bloques en efficientnet_pytorch
        for attr in ["_conv_head", "_bn1"]:
            if hasattr(self.backbone, attr):
                module = getattr(self.backbone, attr)
                for param in module.parameters():
                    param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info("Descongelados %d/%d bloques finales del encoder (%d params entrenables)",
                    n_to_unfreeze, total, n_trainable)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        ##
        ## Extrae features visuales de las imagenes
        ##
        ## Parametros:
        ##   images: [batch, 3, H, W] imagenes preprocesadas (alta resolucion)
        ##
        ## Retorna: [batch, 2048] embeddings visuales
        ##
        ## En efficientnet_pytorch, extract_features da el mapa espacial de
        ## features [batch, 2048, h', w'] (sin la cabeza de clasificacion).
        ## Aplicamos global average pooling para obtener el vector [batch, 2048].
        ##
        if self.backbone is None:
            self.load_backbone()

        feature_map = self.backbone.extract_features(images)
        ## Global average pooling espacial: [batch, 2048, h', w'] -> [batch, 2048]
        pooled = F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)
        return pooled

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class ClassificationHead(nn.Module):
    ## Head de clasificacion multiclase generico
    ## Se usa tanto para BI-RADS como para cada slot de hallazgos

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MammoVLM(nn.Module):
    ## Modelo principal MammoVLM V2 con arquitectura dual-head
    ##
    ## Componentes:
    ##   - encoder: Mammo-CLIP (EfficientNet-B5) para features visuales
    ##   - birads_head: clasificacion BI-RADS (5 clases, BI-RADS 1-5)
    ##   - density_head: clasificacion de densidad (4 clases, DENSITY A-D)
    ##
    ## Version exp06: simplificada a dos tareas (BI-RADS + densidad) sobre VinDr.
    ## La generacion de informes se maneja por separado para mantener la
    ## separacion entre prediccion estructurada y texto.

    def __init__(
        self,
        checkpoint_path: str,
        efficientnet_name: str = "efficientnet-b5",
        num_birads_classes: int = 5,
        num_density_classes: int = 4,
        freeze_encoder: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        self.num_birads_classes = num_birads_classes
        self.num_density_classes = num_density_classes

        ## Encoder visual Mammo-CLIP (EfficientNet-B5, alta resolucion)
        ## unfreeze_last_n_blocks > 0 permite fine-tuning parcial del encoder
        self.encoder = MammoCLIPEncoder(
            checkpoint_path=checkpoint_path,
            efficientnet_name=efficientnet_name,
            freeze_backbone=freeze_encoder,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        ## Cargar backbone para conocer la dimension de features
        self.encoder.load_backbone()
        feat_dim = self.encoder.feature_dim

        ## Head de clasificacion BI-RADS (5 clases: BI-RADS 1-5)
        ## (sin attention pooling: el encoder de alta resolucion ya produce
        ## un vector de features que va directo a los heads)
        self.birads_head = ClassificationHead(
            input_dim=feat_dim,
            num_classes=num_birads_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        ## Head de clasificacion de densidad (4 clases: DENSITY A-D)
        self.density_head = ClassificationHead(
            input_dim=feat_dim,
            num_classes=num_density_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        logger.info(
            "MammoVLM inicializado: encoder=Mammo-CLIP (%s), BI-RADS head (%d clases), densidad head (%d clases)",
            efficientnet_name, num_birads_classes, num_density_classes
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        ##
        ## Forward pass del modelo
        ##
        ## Parametros:
        ##   images: [batch, 3, H, W] imagenes de alta resolucion preprocesadas
        ##
        ## Retorna: diccionario con logits
        ##   - "birads": [batch, num_birads_classes]
        ##   - "density": [batch, num_density_classes]
        ##
        ## Extraer features con el encoder Mammo-CLIP (alta resolucion)
        features = self.encoder(images)

        outputs = {
            "birads": self.birads_head(features),
            "density": self.density_head(features),
        }
        return outputs

    def predict(self, images: torch.Tensor) -> List[Dict]:
        ##
        ## Prediccion con softmax para BI-RADS y densidad
        ##
        ## Parametros:
        ##   images: [batch, 3, H, W] imagenes de alta resolucion
        ##
        ## Retorna por cada imagen del batch:
        ##   - birads_pred: indice BI-RADS predicho (0-4, equivale a BI-RADS 1-5)
        ##   - birads_confidence: confianza (max softmax)
        ##   - birads_probs: distribucion completa sobre las 5 clases
        ##   - density_pred: indice de densidad predicho (0-3, equivale a A-D)
        ##   - density_confidence: confianza de la densidad
        ##
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)

        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            ## BI-RADS
            birads_probs = F.softmax(outputs["birads"][i], dim=-1)
            birads_pred = int(torch.argmax(birads_probs).item())
            birads_conf = float(birads_probs[birads_pred].item())

            ## Densidad
            density_probs = F.softmax(outputs["density"][i], dim=-1)
            density_pred = int(torch.argmax(density_probs).item())
            density_conf = float(density_probs[density_pred].item())

            results.append({
                "birads_pred": birads_pred,
                "birads_confidence": birads_conf,
                "birads_probs": birads_probs.cpu().numpy().tolist(),
                "density_pred": density_pred,
                "density_confidence": density_conf,
                "density_probs": density_probs.cpu().numpy().tolist(),
            })

        return results

    def get_trainable_parameters(self) -> int:
        ## Cuenta parametros entrenables
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        ## Cuenta parametros totales
        return sum(p.numel() for p in self.parameters())


class FocalLoss(nn.Module):
    ## Focal Loss para clasificacion con desbalance severo de clases
    ##
    ## La focal loss modifica la cross-entropy para reducir la contribucion
    ## de los ejemplos faciles (bien clasificados, tipicamente la clase
    ## mayoritaria) y mantener el foco en los dificiles (clases minoritarias).
    ##
    ## Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    ##   - gamma: controla cuanto se reduce el peso de los ejemplos faciles
    ##            (gamma=0 equivale a cross-entropy; gamma=2 es el valor tipico)
    ##   - alpha: pesos por clase (como class weights), opcional
    ##
    ## Referencia: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    ##
    ## Soporta ignore_index=-100 para el masking de etiquetas ausentes.

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ##
        ## Parametros:
        ##   logits: [batch, num_classes] logits sin normalizar
        ##   targets: [batch] indices de clase verdaderos
        ##
        ## Retorna: focal loss promedio sobre las muestras validas
        ##
        ## Filtrar muestras enmascaradas (ignore_index)
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            ## Todo enmascarado: retornar 0 (sin contribucion al gradiente)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits[valid_mask]
        targets_valid = targets[valid_mask]

        ## log-probabilidades y probabilidades
        log_probs = F.log_softmax(logits_valid, dim=-1)
        probs = torch.exp(log_probs)

        ## Seleccionar la probabilidad y log-prob de la clase verdadera
        ## p_t = probabilidad asignada a la clase correcta
        target_log_probs = log_probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)

        ## Factor de modulacion focal: (1 - p_t)^gamma
        focal_factor = (1.0 - target_probs) ** self.gamma

        ## Loss base: -log(p_t) modulada por el factor focal
        loss = -focal_factor * target_log_probs

        ## Aplicar alpha (pesos por clase) si se proporciona
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets_valid]
            loss = alpha_t * loss

        return loss.mean()


class MultiTaskLoss(nn.Module):
    ## Loss multi-tarea para entrenar BI-RADS + densidad simultaneamente
    ##
    ## Combina:
    ##   - Loss para BI-RADS (tarea principal, peso mayor)
    ##   - Loss para densidad (tarea secundaria, peso menor)
    ##
    ## Permite class weights para manejar el desbalance de BI-RADS (los niveles
    ## 1-2 son mucho mas frecuentes que 4-5). Soporta Focal Loss para atacar
    ## el desbalance, y ignore_index=-100 para muestras de densidad ausente.

    def __init__(
        self,
        birads_weight: float = 1.0,
        density_weight: float = 0.5,
        birads_class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        num_birads_classes: int = 5,
        ## Loss ordinal hibrida (exp08): SORD + lambda*QWK para BI-RADS
        use_ordinal_loss: bool = False,
        ordinal_lambda_qwk: float = 0.3,
        ordinal_distance_power: float = 1.0,
        ## SORD asimetrico (exp09): penaliza mas el sub-diagnostico
        ordinal_undergrade_beta: float = 1.0,
    ):
        super().__init__()
        self.birads_weight = birads_weight
        self.density_weight = density_weight
        self.use_focal_loss = use_focal_loss
        self.use_ordinal_loss = use_ordinal_loss

        ## Loss para BI-RADS (tarea principal). Tres opciones, en orden de prioridad:
        ##   1. Ordinal hibrida (SORD + QWK): respeta la naturaleza ordinal de BI-RADS
        ##   2. Focal loss: ataca el desbalance tratando las clases como nominales
        ##   3. Cross-entropy: baseline
        if use_ordinal_loss:
            from ordinal_losses import HybridOrdinalLoss
            self.birads_loss = HybridOrdinalLoss(
                num_classes=num_birads_classes,
                lambda_qwk=ordinal_lambda_qwk,
                distance_power=ordinal_distance_power,
                undergrade_beta=ordinal_undergrade_beta,
                class_weights=birads_class_weights,
                ignore_index=-100,
            )
        elif use_focal_loss:
            ## La focal loss usa los class weights como factor alpha
            self.birads_loss = FocalLoss(
                gamma=focal_gamma, alpha=birads_class_weights, ignore_index=-100
            )
        else:
            self.birads_loss = nn.CrossEntropyLoss(
                weight=birads_class_weights, ignore_index=-100
            )

        ## Loss para densidad (tarea secundaria; siempre nominal, no ordinal)
        ## ignore_index=-100 enmascara las imagenes sin densidad valida
        if use_focal_loss and not use_ordinal_loss:
            self.density_loss = FocalLoss(gamma=focal_gamma, ignore_index=-100)
        else:
            self.density_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ##
        ## Calcula la loss combinada
        ##
        ## Parametros:
        ##   outputs: salida del modelo (logits por tarea)
        ##     - "birads": [batch, 5]
        ##     - "density": [batch, 4]
        ##   targets: etiquetas verdaderas
        ##     - "birads": [batch] indices BI-RADS (0-4)
        ##     - "density": [batch] indices densidad (0-3, o -100 si falta)
        ##
        ## Retorna: dict con loss total y por componente
        ##

        ## Loss BI-RADS (tarea principal)
        birads_loss = self.birads_loss(outputs["birads"], targets["birads"])
        if torch.isnan(birads_loss):
            birads_loss = torch.tensor(0.0, device=outputs["birads"].device)

        ## Loss densidad (tarea secundaria)
        ## Si todas las muestras del batch tienen densidad enmascarada (-100),
        ## la loss seria NaN; en ese caso se sustituye por 0
        density_target = targets["density"]
        valid_density = (density_target != -100).sum()
        if valid_density > 0:
            density_loss = self.density_loss(outputs["density"], density_target)
            if torch.isnan(density_loss):
                density_loss = torch.tensor(0.0, device=outputs["density"].device)
        else:
            density_loss = torch.tensor(0.0, device=outputs["density"].device)

        ## Loss total ponderada
        total_loss = (
            self.birads_weight * birads_loss +
            self.density_weight * density_loss
        )

        return {
            "total": total_loss,
            "birads": birads_loss,
            "density": density_loss,
        }
