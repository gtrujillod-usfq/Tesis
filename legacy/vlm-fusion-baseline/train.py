## train.py
## Loops de entrenamiento optimizados para NVIDIA H200 (141 GB HBM3e).
##
## Cambios respecto a la version de bajo recurso:
##
##   Batch size:
##     - Clasificador: 128 (era 16). El H200 puede manejar batches grandes
##       sin gradient accumulation, lo que simplifica el loop y mejora
##       la utilizacion del hardware.
##     - VLM: 32-64 (era 8). El LLM de 7B en BF16 ocupa ~14 GB, dejando
##       ~120 GB para batch, activaciones y gradientes.
##
##   Gradient accumulation:
##     - Clasificador: 1 (desactivado). Con batch 128 no es necesario.
##     - VLM: 2 como maximo, solo para estabilizar el gradiente del LLM,
##       no por limitacion de VRAM.
##
##   GradScaler (AMP):
##     - Eliminado para BF16. BF16 no sufre underflow como FP16, por lo que
##       el escalado de gradientes no aporta beneficio y agrega overhead.
##     - Se mantiene autocast(dtype=bfloat16) para que las operaciones
##       matriciales usen los tensor cores BF16 del H200.
##
##   Optimizer:
##     - Se agrega opcion de usar Fused AdamW (torch.optim.AdamW con fused=True).
##       La implementacion fusionada ejecuta todas las operaciones del paso de
##       optimizacion en un solo kernel CUDA, reduciendo el overhead de lanzamiento.
##
##   DataLoader:
##     - num_workers=16 (era 4): el H200 en servidor tiene muchos cores CPU.
##     - prefetch_factor=4: precargar 4 batches por worker para eliminar cuellos
##       de botella de I/O durante la GPU bound.
##     - persistent_workers=True: evita el overhead de relanzar workers.
##
##   Buenas practicas que se conservan:
##     - LR diferencial por modulo (cabeza > encoder > LLM)
##     - Early stopping con Youden J
##     - Checkpointing del mejor modelo
##     - Label smoothing

import os
import time
import logging
import math
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from evaluate import evaluate_model, print_metrics_summary, ClassificationEvaluator
from models import save_checkpoint, count_parameters, MammoClassifier, MammoVLM, COMPUTE_DTYPE

logger = logging.getLogger("train")


@dataclass
class TrainingConfig:
    """
    Configuracion de entrenamiento para H200.
    Los valores por defecto asumen un unico H200 con 141 GB de VRAM.
    """
    experiment_name: str = "mammo_vlm_v1_h200"
    output_dir: str = "outputs"

    ## Hiperparametros de training
    n_epochs: int = 30
    batch_size: int = 128              ## Era 16 en hardware limitado
    learning_rate: float = 2e-4        ## LR ligeramente mayor por batch mas grande
    weight_decay: float = 0.01
    warmup_steps: int = 200            ## Mas warmup por el LR mas alto
    gradient_accumulation_steps: int = 1  ## Desactivado para clasificador
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    ## Scheduler
    scheduler_type: str = "cosine"
    min_lr: float = 1e-6

    ## Early stopping
    early_stopping_patience: int = 7
    early_stopping_metric: str = "auc_roc"
    early_stopping_mode: str = "max"

    ## Precision: BF16 nativo en H200, sin GradScaler
    use_amp: bool = True
    amp_dtype: str = "bfloat16"        ## BF16 siempre en H200 (no FP16)
    use_fused_optimizer: bool = True   ## AdamW fusionado para H200

    ## Checkpointing
    save_every_n_epochs: int = 3       ## Mas frecuente (I/O rapida en H200)
    save_best_only: bool = True

    ## Fases de entrenamiento
    ## En H200 se acortan porque el encoder se puede entrenar desde el principio
    freeze_encoder_epochs: int = 2     ## Era 5: solo 2 epocas congelado en H200
    unfreeze_last_n_layers: int = 6    ## Descongelar mas capas (era 3)

    ## LR diferencial
    encoder_lr_factor: float = 0.1
    llm_lr_factor: float = 0.05

    ## Logging
    log_every_n_steps: int = 20        ## Mas frecuente por batches mas rapidos
    eval_every_n_epochs: int = 1

    ## DataLoader optimizado para servidor H200
    num_workers: int = 16              ## Era 4: aprovechar CPUs del servidor
    prefetch_factor: int = 4           ## Batches precargados por worker
    persistent_workers: bool = True    ## Reutilizar procesos worker

    ## Reproducibilidad
    random_seed: int = 42

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        logger.info(
            "TrainingConfig H200: batch_efectivo=%d, epochs=%d, lr=%.2e, workers=%d",
            self.effective_batch_size, self.n_epochs, self.learning_rate, self.num_workers,
        )


def set_seed(seed: int):
    """Fija semillas para reproducibilidad en todos los generadores de numeros aleatorios."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Semilla aleatoria fijada: %d", seed)


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> optim.Optimizer:
    """
    Construye el optimizer con learning rates diferenciales por componente.

    En H200 se agrega la opcion fused=True para AdamW:
    La implementacion fusionada ejecuta todo el paso de optimizacion en un
    solo kernel CUDA en lugar de lanzar un kernel por operacion, reduciendo
    el overhead de lanzamiento de ~30% en batches grandes.

    Requiere CUDA y parametros en GPU (siempre se cumple en H200).
    """
    param_groups = []

    classifier_params = []
    projection_params = []
    encoder_params = []
    llm_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "head" in name:
            classifier_params.append(param)
        elif "projection" in name:
            projection_params.append(param)
        elif "visual_encoder" in name or "encoder" in name:
            encoder_params.append(param)
        elif "llm" in name or "lora" in name:
            llm_params.append(param)
        else:
            other_params.append(param)

    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": config.learning_rate, "name": "classifier"})
    if projection_params:
        param_groups.append({"params": projection_params, "lr": config.learning_rate, "name": "projection"})
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": config.learning_rate * config.encoder_lr_factor,
            "name": "encoder",
        })
    if llm_params:
        param_groups.append({
            "params": llm_params,
            "lr": config.learning_rate * config.llm_lr_factor,
            "name": "llm",
        })
    if other_params:
        param_groups.append({"params": other_params, "lr": config.learning_rate, "name": "other"})

    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    ## Fused AdamW: disponible en PyTorch >= 2.0 con CUDA
    use_fused = (
        config.use_fused_optimizer
        and torch.cuda.is_available()
        and hasattr(optim, "AdamW")
    )

    try:
        optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
            fused=use_fused,
        )
        if use_fused:
            logger.info("Usando AdamW fusionado (fused=True) para H200")
    except TypeError:
        ## Fallback para versiones de PyTorch sin soporte de fused
        optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )

    for group in optimizer.param_groups:
        n_params = sum(p.numel() for p in group["params"])
        group_name = group.get("name", "unnamed")
        logger.info("Grupo optimizer '%s': %dK params, lr=%.2e", group_name, n_params // 1000, group["lr"])

    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    n_train_steps: int,
) -> optim.lr_scheduler._LRScheduler:
    """
    Construye el LR scheduler segun la configuracion.

    OneCycleLR: sube linealmente hasta el LR maximo y baja con coseno.
    Excelente para fine-tuning porque evita el calentamiento lento.

    CosineAnnealingLR: decaimiento suave de coseno desde LR hasta min_lr.
    Mas estable que OneCycle, preferido cuando el LR inicial ya es optimo.

    Linear: decaimiento lineal simple, util para debugging.
    """
    if config.scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=n_train_steps,
            pct_start=0.1,           ## 10% del entrenamiento para warmup
            div_factor=25,           ## LR_inicial = max_lr / 25
            final_div_factor=1e4,    ## LR_final = LR_inicial / 10000
            anneal_strategy="cos",
        )

    elif config.scheduler_type == "cosine":
        ## Cosine con warmup manual
        warmup_steps = config.warmup_steps

        def cosine_with_warmup(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, n_train_steps - warmup_steps))
            return max(config.min_lr / config.learning_rate,
                       0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

    elif config.scheduler_type == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_train_steps
        )

    else:
        ## Scheduler constante (sin cambio de LR)
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    logger.info("Scheduler: %s, pasos totales=%d", config.scheduler_type, n_train_steps)
    return scheduler


class EarlyStopping:
    """
    Detiene el entrenamiento cuando la metrica de validacion deja de mejorar.

    Parametros
    ----------
    patience : int
        Numero de epocas consecutivas sin mejora antes de detener.
    metric : str
        Nombre de la metrica a monitorear en el diccionario de metricas.
    mode : str
        "max" si la metrica debe maximizarse (AUC, accuracy), "min" si debe minimizarse (loss).
    min_delta : float
        Cambio minimo para considerar una mejora real (evita mejoras insignificantes).
    """

    def __init__(
        self,
        patience: int = 7,
        metric: str = "auc_roc",
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, metrics: dict, epoch: int) -> bool:
        """
        Evalua si el entrenamiento debe continuar.
        Retorna True si se ha mejorado (no detener), False si no.

        Actualiza el flag should_stop cuando se agota la paciencia.
        """
        current = metrics.get(self.metric, None)
        if current is None:
            ## Si la metrica no existe, intentar fallbacks
            fallbacks = ["accuracy", "val_loss", "f1"]
            for fb in fallbacks:
                if fb in metrics:
                    current = metrics[fb]
                    break

        if current is None:
            logger.warning("Metrica '%s' no encontrada en metricas de validacion.", self.metric)
            return True  ## Continuar por defecto si no hay metrica

        improved = (
            (self.mode == "max" and current > self.best_value + self.min_delta)
            or (self.mode == "min" and current < self.best_value - self.min_delta)
        )

        if improved:
            self.best_value = current
            self.counter = 0
            self.best_epoch = epoch
            logger.info(
                "Mejora en '%s': %.4f (epoch %d)",
                self.metric, current, epoch,
            )
            return True
        else:
            self.counter += 1
            logger.info(
                "Sin mejora en '%s': %.4f (mejor=%.4f, contador=%d/%d)",
                self.metric, current, self.best_value, self.counter, self.patience,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping activado en epoch %d. Mejor epoch: %d (%.4f=%.4f)",
                    epoch, self.best_epoch, self.metric, self.best_value,
                )
            return False


class MetricTracker:
    """
    Acumula y promedia metricas durante un epoch de entrenamiento.
    Evita acumular tensores en GPU al hacer .item() inmediatamente.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums: dict = {}
        self._counts: dict = {}

    def update(self, metrics: dict, n: int = 1):
        """Acumula valores escalares. n es el tamano del batch."""
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            if not math.isfinite(val):
                continue  ## Ignorar NaN/Inf sin abortar el entrenamiento
            self._sums[k] = self._sums.get(k, 0.0) + val * n
            self._counts[k] = self._counts.get(k, 0) + n

    def averages(self) -> dict:
        """Retorna promedios ponderados por numero de muestras."""
        return {
            k: round(self._sums[k] / max(self._counts[k], 1), 6)
            for k in self._sums
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
    criterion: Optional[nn.Module] = None,
    mode: str = "classify",
) -> dict:
    """
    Ejecuta un epoch completo de entrenamiento optimizado para H200.

    Diferencias respecto a la version de bajo recurso:
    - Sin GradScaler: BF16 no necesita escalado de gradientes.
    - autocast("cuda", dtype=bfloat16): usa los tensor cores BF16 del H200.
    - Gradient accumulation minimo (1 para clasificador, 2 para VLM).
    - num_workers y prefetch configurados en el DataLoader (no aqui).

    Parametros
    ----------
    model : nn.Module
    dataloader : DataLoader
    optimizer : optim.Optimizer
    scheduler : LRScheduler
    config : TrainingConfig
    device : torch.device
    epoch : int
    criterion : nn.Module, opcional
    mode : str
        "classify", "generate" o "both".
    """
    model.train()
    tracker = MetricTracker()
    optimizer.zero_grad(set_to_none=True)   ## set_to_none=True es mas eficiente que cero

    n_batches = len(dataloader)
    use_amp = config.use_amp and torch.cuda.is_available()

    with tqdm(dataloader, desc=f"Epoch {epoch:03d} [train]", leave=False) as pbar:
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True, dtype=COMPUTE_DTYPE)
            labels = batch["label"].to(device, non_blocking=True)

            density_labels = batch.get("density", None)
            if density_labels is not None:
                density_labels = density_labels.to(device, non_blocking=True)
                density_labels = density_labels.where(density_labels >= 0, other=torch.zeros_like(density_labels))

            input_ids = batch.get("report_input_ids", None)
            attention_mask = batch.get("report_attention_mask", None)
            if input_ids is not None:
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None

            ## Forward con autocast BF16.
            ## En H200 no se usa GradScaler porque BF16 tiene rango de exponente
            ## equivalente a FP32, eliminando el riesgo de underflow de gradientes
            ## que afecta a FP16 y que requeria el scaler.
            with autocast("cuda", dtype=COMPUTE_DTYPE, enabled=use_amp):
                try:
                    out = model(
                        image=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        density_labels=density_labels,
                        mode=mode,
                    )
                    loss = out.get("loss", None)

                    if loss is None and criterion is not None:
                        logits = out.get("logits", None)
                        if logits is not None:
                            loss = criterion(logits, labels)

                    if loss is None:
                        logger.warning("Modelo sin loss en step %d.", step)
                        continue

                except TypeError:
                    out = model(images, labels=labels)
                    loss = out.get("loss", None)
                    if loss is None and criterion is not None:
                        loss = criterion(out.get("logits"), labels)

            ## Gradient accumulation (generalmente 1 para clasificador en H200)
            loss_scaled = loss / config.gradient_accumulation_steps
            loss_scaled.backward()

            step_metrics = {"loss": loss}
            for k in ["cls_loss", "lm_loss", "density_loss"]:
                if k in out:
                    step_metrics[k] = out[k]
            tracker.update(step_metrics, n=len(images))

            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if (step + 1) % config.log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else config.learning_rate
                avg = tracker.averages()
                pbar.set_postfix({
                    "loss": f"{avg.get('loss', 0):.4f}",
                    "lr": f"{current_lr:.2e}",
                })

    return tracker.averages()


def train_classifier(
    model: MammoClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    """
    Loop de entrenamiento para MammoClassifier (Fases 1 y 2).

    Fase 1 (primeras freeze_encoder_epochs epocas):
      Solo la cabeza clasificadora se entrena con el encoder congelado.
      LR alto para aprender representaciones de BI-RADS desde features pre-calculados.

    Fase 2 (epocas restantes):
      Se descongela el encoder parcialmente y se reduce el LR del encoder
      para fine-tuning sin catastrophic forgetting.

    Retorna historia de entrenamiento (loss y metricas por epoca).
    """
    set_seed(config.random_seed)
    model = model.to(device)
    stats = count_parameters(model)
    logger.info("MammoClassifier: %.2fM params totales, %.2fM entrenables", stats["total_M"], stats["trainable_M"])

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = build_optimizer(model, config)
    n_steps = len(train_loader) * config.n_epochs
    scheduler = build_scheduler(optimizer, config, n_steps)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        metric=config.early_stopping_metric,
        mode=config.early_stopping_mode,
    )

    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "val_auc_roc": [], "val_sensitivity": [], "val_specificity": [],
        "learning_rates": [],
    }

    best_metrics = {}
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Iniciando entrenamiento de MammoClassifier por %d epocas", config.n_epochs)

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()

        ## Transicion a Fase 2: descongelar encoder
        if epoch == config.freeze_encoder_epochs + 1:
            model.unfreeze_encoder(last_n_layers=config.unfreeze_last_n_layers)
            ## Reconstruir optimizer con los nuevos parametros entrenables
            optimizer = build_optimizer(model, config)
            remaining_steps = len(train_loader) * (config.n_epochs - epoch + 1)
            scheduler = build_scheduler(optimizer, config, remaining_steps)
            logger.info("Fase 2 activada: encoder parcialmente descongelado en epoch %d", epoch)

        ## Epoch de entrenamiento
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            config, device, epoch,
            criterion=criterion, mode="classify",
        )

        ## Evaluacion en validacion
        if epoch % config.eval_every_n_epochs == 0:
            val_metrics, _, _ = evaluate_model(
                model, val_loader, device,
                num_classes=2,
                optimize_threshold=True,
                desc=f"Epoch {epoch:03d} [val]",
            )
        else:
            val_metrics = {}

        ## Registrar historia
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics.get("loss", 0))
        history["val_loss"].append(val_metrics.get("val_loss", 0))
        history["val_accuracy"].append(val_metrics.get("accuracy", 0))
        history["val_auc_roc"].append(val_metrics.get("auc_roc", 0))
        history["val_sensitivity"].append(val_metrics.get("sensitivity", 0))
        history["val_specificity"].append(val_metrics.get("specificity", 0))
        history["learning_rates"].append(current_lr)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %03d/%03d | loss=%.4f | auc=%.4f | sens=%.4f | spec=%.4f | lr=%.2e | %.1fs",
            epoch, config.n_epochs,
            train_metrics.get("loss", 0),
            val_metrics.get("auc_roc", 0),
            val_metrics.get("sensitivity", 0),
            val_metrics.get("specificity", 0),
            current_lr, elapsed,
        )

        ## Checkpoint
        is_best = early_stopping.step(val_metrics, epoch)
        if is_best:
            best_metrics = val_metrics.copy()

        if is_best or (epoch % config.save_every_n_epochs == 0):
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(output_dir),
                filename=f"checkpoint_epoch_{epoch:03d}.pt",
                is_best=is_best,
            )

        if early_stopping.should_stop:
            logger.info("Entrenamiento detenido en epoch %d por early stopping.", epoch)
            break

    logger.info("Entrenamiento finalizado. Mejor epoch: %d", early_stopping.best_epoch)
    logger.info("Mejores metricas de validacion:")
    print_metrics_summary(best_metrics)

    history["best_epoch"] = early_stopping.best_epoch
    history["best_metrics"] = best_metrics
    return history


def train_vlm(
    model: MammoVLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    """
    Loop de entrenamiento para MammoVLM (Fase 3: fine-tuning multimodal).

    Esta fase es mas costosa y delicada que la clasificacion pura porque:
    - El LLM tiene millones de parametros (aunque solo entrenamos LoRA)
    - La loss de lenguaje y de clasificacion se combinan
    - El gradiente puede ser inestable sin clipping agresivo

    Estrategia:
    1. Congelar encoder visual completamente (ya entrenado en Fases 1 y 2)
    2. Entrenar solo proyeccion + cabeza clasificadora + LoRA del LLM
    3. Usar LR muy bajo para LoRA (1e-5 o menos) y LR normal para proyeccion
    4. Alternar entre batches con/sin reporte para entrenamiento balanceado
    """
    set_seed(config.random_seed)
    model = model.to(device)
    stats = count_parameters(model)
    logger.info("MammoVLM: %.2fM params totales, %.2fM entrenables (%.1f%%)",
                stats["total_M"], stats["trainable_M"], stats["trainable_pct"])

    optimizer = build_optimizer(model, config)
    n_steps = len(train_loader) * config.n_epochs
    scheduler = build_scheduler(optimizer, config, n_steps)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        metric=config.early_stopping_metric,
        mode=config.early_stopping_mode,
    )

    history = {
        "train_loss": [], "train_cls_loss": [], "train_lm_loss": [],
        "val_accuracy": [], "val_auc_roc": [], "val_sensitivity": [],
        "val_specificity": [], "learning_rates": [],
    }

    best_metrics = {}
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Iniciando entrenamiento de MammoVLM por %d epocas", config.n_epochs)

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()

        ## Determinar modo de entrenamiento:
        ## Si hay reportes en el batch se usa "both", sino "classify"
        ## Esto se maneja dentro de train_one_epoch al revisar input_ids
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            config, device, epoch,
            mode="both",
        )

        ## Evaluacion
        if epoch % config.eval_every_n_epochs == 0:
            ## Evaluacion de clasificacion (siempre)
            val_metrics, _, _ = evaluate_model(
                model, val_loader, device,
                num_classes=model.config.num_classes,
                compute_reports=False,  ## Generacion de reportes es costosa; activar segun recursos
                optimize_threshold=True,
                desc=f"Epoch {epoch:03d} [val]",
            )
        else:
            val_metrics = {}

        ## Registrar historia
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics.get("loss", 0))
        history["train_cls_loss"].append(train_metrics.get("cls_loss", 0))
        history["train_lm_loss"].append(train_metrics.get("lm_loss", 0))
        history["val_accuracy"].append(val_metrics.get("accuracy", 0))
        history["val_auc_roc"].append(val_metrics.get("auc_roc", 0))
        history["val_sensitivity"].append(val_metrics.get("sensitivity", 0))
        history["val_specificity"].append(val_metrics.get("specificity", 0))
        history["learning_rates"].append(current_lr)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %03d/%03d | loss=%.4f | cls=%.4f | lm=%.4f | auc=%.4f | sens=%.4f | spec=%.4f | %.1fs",
            epoch, config.n_epochs,
            train_metrics.get("loss", 0),
            train_metrics.get("cls_loss", 0),
            train_metrics.get("lm_loss", 0),
            val_metrics.get("auc_roc", 0),
            val_metrics.get("sensitivity", 0),
            val_metrics.get("specificity", 0),
            elapsed,
        )

        is_best = early_stopping.step(val_metrics, epoch)
        if is_best:
            best_metrics = val_metrics.copy()

        if is_best or (epoch % config.save_every_n_epochs == 0):
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(output_dir),
                filename=f"vlm_epoch_{epoch:03d}.pt",
                is_best=is_best,
            )

        if early_stopping.should_stop:
            logger.info("Entrenamiento VLM detenido en epoch %d.", epoch)
            break

    history["best_epoch"] = early_stopping.best_epoch
    history["best_metrics"] = best_metrics
    return history


def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    """
    Orquestador de alto nivel que selecciona el loop correcto segun el tipo de modelo.
    Punto de entrada unico desde el notebook principal.

    Retorna el historial de entrenamiento.
    """
    model_type = type(model).__name__
    logger.info("Iniciando pipeline de entrenamiento para modelo: %s", model_type)

    if isinstance(model, MammoVLM):
        history = train_vlm(model, train_loader, val_loader, config, device)
    elif isinstance(model, MammoClassifier):
        history = train_classifier(model, train_loader, val_loader, config, device)
    else:
        ## Fallback para cualquier otro modelo con interfaz compatible
        logger.info("Usando loop de clasificacion generico para modelo: %s", model_type)
        history = train_classifier(model, train_loader, val_loader, config, device)

    return history
