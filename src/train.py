## train.py
## Entrenamiento del MammoVLM V2 (exp06) sobre VinDr-Mammo
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Estrategia de entrenamiento (exp06):
##   - Una sola etapa (no por etapas como en versiones anteriores)
##   - Un solo dataset: VinDr-Mammo (BI-RADS 1-5 + densidad A-D)
##   - Encoder Mammo-CLIP (EfficientNet-B5, alta resolucion)
##   - Split oficial de VinDr (columna 'split': training/test)
##   - El test oficial de VinDr se reserva para la evaluacion final
##   - La validacion se separa del conjunto de entrenamiento (15%)
##
## El encoder puede estar congelado (solo se entrenan los heads) o con
## descongelamiento parcial de los ultimos bloques (fine-tuning).

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingConfig:
    ## Configuracion del entrenamiento de exp06 (VinDr, una etapa)

    def __init__(
        self,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 16,
        num_workers: int = 4,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.15,
        random_seed: int = 42,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        ## Resolucion de entrada (nativa de Mammo-CLIP: 1520x912)
        image_height: int = 1520,
        image_width: int = 912,
        ## Limite de muestras (None = sin limite). Util para prueba de humo.
        max_samples: Optional[int] = None,
        ## Reanudar desde el ultimo checkpoint si existe
        resume: bool = False,
        ## Numero de bloques finales del encoder a descongelar (0 = congelado)
        unfreeze_last_n_blocks: int = 0,
        ## Usar focal loss en lugar de cross-entropy (mejor para desbalance)
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        ## Potencia para amplificar los class weights (1.0 = estandar, >1 = agresivo)
        weight_power: float = 1.0,
        ## Loss ordinal hibrida (exp08): SORD + lambda*QWK para BI-RADS
        use_ordinal_loss: bool = False,
        ordinal_lambda_qwk: float = 0.3,
        ordinal_distance_power: float = 1.0,
        ## SORD asimetrico (exp09): factor de penalizacion del sub-diagnostico
        ordinal_undergrade_beta: float = 1.0,
        ## Usar oversampling de clases minoritarias (WeightedRandomSampler)
        use_oversampling: bool = False,
        ## Pesos de las tareas en la loss multi-tarea
        birads_weight: float = 1.0,
        density_weight: float = 0.5,
        ## Directorio fijo para el test set reservado
        test_set_dir: Optional[str] = None,
        ## Usar el split oficial de VinDr (columna 'split') para el test set
        use_official_split: bool = True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.random_seed = random_seed
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.max_samples = max_samples
        self.resume = resume
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.weight_power = weight_power
        self.use_ordinal_loss = use_ordinal_loss
        self.ordinal_lambda_qwk = ordinal_lambda_qwk
        self.ordinal_distance_power = ordinal_distance_power
        self.ordinal_undergrade_beta = ordinal_undergrade_beta
        self.use_oversampling = use_oversampling
        self.birads_weight = birads_weight
        self.density_weight = density_weight
        self.test_set_dir = test_set_dir
        self.use_official_split = use_official_split


## Numero de clases BI-RADS en VinDr (1-5, mapeadas a indices 0-4)
NUM_BIRADS_CLASSES = 5


def compute_birads_class_weights(records, num_classes: int = NUM_BIRADS_CLASSES, weight_power: float = 1.0):
    ##
    ## Calcula pesos de clase inversamente proporcionales a la frecuencia
    ## para compensar el desbalance de BI-RADS (1-2 mucho mas frecuentes)
    ##
    ## Los BI-RADS de VinDr (1-5) se mapean a indices 0-4 (birads - 1).
    ##
    ## weight_power amplifica el sesgo hacia clases minoritarias:
    ##   1.0 = pesos inversos a la frecuencia (estandar)
    ##   >1.0 = mas agresivo (las minoritarias pesan aun mas)
    ##   Ejemplo: con power=1.5, un peso de 4.0 se vuelve 4.0^1.5 = 8.0
    ## Se renormaliza para que el promedio de pesos sea ~1 (estabilidad numerica).
    ##
    import torch

    counts = np.zeros(num_classes)
    for b in records["birads"]:
        idx = int(b) - 1  ## BI-RADS 1-5 -> indice 0-4
        if 0 <= idx < num_classes:
            counts[idx] += 1

    ## Evitar division por cero
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)

    ## Amplificar el sesgo hacia minoritarias si weight_power > 1
    if weight_power != 1.0:
        weights = np.power(weights, weight_power)
        ## Renormalizar para mantener el promedio de pesos cercano a 1
        weights = weights * (num_classes / weights.sum())

    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(records, num_classes: int = NUM_BIRADS_CLASSES):
    ##
    ## Construye un WeightedRandomSampler para oversampling de clases minoritarias
    ##
    ## Ajusta la probabilidad de muestreo de cada registro de forma inversamente
    ## proporcional a la frecuencia de su clase BI-RADS, balanceando los batches
    ## sin duplicar datos en memoria.
    ##
    import torch
    from torch.utils.data import WeightedRandomSampler

    birads_idx = [int(b) - 1 for b in records["birads"]]
    counts = np.zeros(num_classes)
    for b in birads_idx:
        if 0 <= b < num_classes:
            counts[b] += 1
    counts = np.maximum(counts, 1)

    class_sample_weights = 1.0 / counts
    sample_weights = np.array([
        class_sample_weights[b] if 0 <= b < num_classes else 0.0
        for b in birads_idx
    ])

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def separate_test_set_vindr(records, config, test_set_path):
    ##
    ## Separa el test set de VinDr y lo guarda a disco para reproducibilidad
    ##
    ## Si config.use_official_split es True, usa la columna 'split' de VinDr
    ## (training/test), que es el split oficial del dataset (mas reproducible
    ## y citable). Si no, crea un split estratificado por BI-RADS.
    ##
    ## Si el test set ya existe en disco, lo carga (reproducibilidad entre
    ## ejecuciones). Si no, lo crea y lo guarda.
    ##
    ## Retorna: (train_val_df, test_df)
    ##
    import pandas as pd

    test_set_path = Path(test_set_path)

    ## Si ya existe un test set guardado, usarlo (reproducibilidad)
    if test_set_path.exists():
        test_df = pd.read_csv(test_set_path)
        test_paths = set(test_df["image_path"].tolist())
        train_val_df = records[~records["image_path"].isin(test_paths)].reset_index(drop=True)
        logger.info("Test set cargado desde disco: %d muestras (train_val: %d)",
                    len(test_df), len(train_val_df))
        return train_val_df, test_df

    ## Crear el test set
    if config.use_official_split and "split" in records.columns:
        ## Usar el split oficial de VinDr (columna 'split')
        ## Los valores tipicos son 'training' y 'test'
        split_lower = records["split"].astype(str).str.lower()
        test_df = records[split_lower == "test"].reset_index(drop=True)
        train_val_df = records[split_lower != "test"].reset_index(drop=True)
        logger.info("Usando split oficial de VinDr: %d test, %d train_val",
                    len(test_df), len(train_val_df))
    else:
        ## Split estratificado propio por BI-RADS
        records = records.copy().reset_index(drop=True)
        test_indices = []
        for birads_val in records["birads"].unique():
            strata_idx = records.index[records["birads"] == birads_val].tolist()
            n_test = max(1, int(len(strata_idx) * config.val_fraction))
            rng = np.random.RandomState(config.random_seed)
            chosen = rng.choice(strata_idx, size=min(n_test, len(strata_idx)), replace=False)
            test_indices.extend(chosen.tolist())
        test_df = records.loc[test_indices].reset_index(drop=True)
        train_val_df = records.drop(index=test_indices).reset_index(drop=True)
        logger.info("Split estratificado propio: %d test, %d train_val",
                    len(test_df), len(train_val_df))

    ## Guardar el test set a disco
    test_set_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_set_path, index=False)
    logger.info("Test set guardado: %s", test_set_path)

    return train_val_df, test_df


def split_train_val(records, val_fraction: float, seed: int):
    ##
    ## Divide los registros en train/val de forma estratificada por BI-RADS
    ##
    from sklearn.model_selection import train_test_split

    try:
        train_df, val_df = train_test_split(
            records,
            test_size=val_fraction,
            random_state=seed,
            stratify=records["birads"],
        )
    except (ValueError, ImportError):
        ## Fallback sin estratificacion si sklearn falla o hay clases muy raras
        shuffled = records.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(shuffled) * val_fraction)
        val_df = shuffled.iloc[:n_val]
        train_df = shuffled.iloc[n_val:]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def subsample_stratified(records, max_samples: int, seed: int):
    ##
    ## Submuestrea los registros manteniendo la proporcion de clases BI-RADS
    ## Util para pruebas de humo: reduce el dataset pero conserva todas las clases
    ##
    if len(records) <= max_samples:
        return records

    records = records.copy().reset_index(drop=True)
    birads_idx = records["birads"].apply(lambda b: int(b) - 1)
    frac = max_samples / len(records)

    selected_indices = []
    for clase in birads_idx.unique():
        clase_indices = records.index[birads_idx == clase].tolist()
        n_clase = max(1, int(len(clase_indices) * frac))
        rng = np.random.RandomState(seed)
        chosen = rng.choice(clase_indices, size=min(n_clase, len(clase_indices)), replace=False)
        selected_indices.extend(chosen.tolist())

    return records.loc[selected_indices].reset_index(drop=True)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    ##
    ## Entrena una epoca y retorna las losses promedio
    ##
    import torch

    model.train()
    total_loss = 0.0
    total_birads = 0.0
    total_density = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = {
            "birads": batch["birads"].to(device),
            "density": batch["density"].to(device),
        }

        outputs = model(images)
        losses = loss_fn(outputs, targets)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        total_birads += losses["birads"].item()
        total_density += losses["density"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "birads_loss": total_birads / max(1, n_batches),
        "density_loss": total_density / max(1, n_batches),
    }


def evaluate(model, loader, loss_fn, device):
    ##
    ## Evalua el modelo en validacion. Retorna loss y accuracy BI-RADS.
    ##
    import torch

    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = {
                "birads": batch["birads"].to(device),
                "density": batch["density"].to(device),
            }

            outputs = model(images)
            losses = loss_fn(outputs, targets)
            total_loss += losses["total"].item()
            n_batches += 1

            preds = torch.argmax(outputs["birads"], dim=1)
            valid = targets["birads"] != -100
            correct += ((preds == targets["birads"]) & valid).sum().item()
            total += valid.sum().item()

    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_birads_acc": correct / max(1, total),
    }


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir):
    ##
    ## Guarda un checkpoint del modelo (y una copia 'latest' para reanudacion)
    ##
    import torch

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    ckpt_path = ckpt_dir / f"mammovlm_epoch{epoch}.pt"
    torch.save(payload, ckpt_path)

    latest_path = ckpt_dir / "mammovlm_latest.pt"
    torch.save(payload, latest_path)

    logger.info("Checkpoint guardado: %s", ckpt_path.name)
    return ckpt_path


def load_checkpoint_if_exists(model, optimizer, checkpoint_dir, device):
    ##
    ## Carga el checkpoint 'latest' si existe, para reanudar.
    ## Retorna: epoca desde la cual reanudar (0 si no hay checkpoint)
    ##
    import torch

    latest_path = Path(checkpoint_dir) / "mammovlm_latest.pt"
    if not latest_path.exists():
        return 0

    try:
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resume_epoch = ckpt.get("epoch", 0)
        logger.info("Reanudando desde epoca %d", resume_epoch)
        return resume_epoch
    except Exception as e:
        logger.warning("No se pudo cargar checkpoint: %s", e)
        return 0


def train_mammovlm(
    model,
    dataset_root: str,
    config: Optional[TrainingConfig] = None,
    epoch_callback=None,
):
    ##
    ## Entrenamiento completo del MammoVLM sobre VinDr (una sola etapa)
    ##
    ## Parametros:
    ##   model: instancia de MammoVLM (ya simplificado: BI-RADS + densidad)
    ##   dataset_root: ruta a la carpeta de VinDr-Mammo
    ##   config: TrainingConfig (usa defaults si None)
    ##   epoch_callback: funcion(stage_num, history) para notificar progreso
    ##
    import torch
    from torch.utils.data import DataLoader
    from data_loading import load_vindr_records, MammoCLIPTransform, MammoDataset
    from models import MultiTaskLoss

    if config is None:
        config = TrainingConfig()

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("=" * 70)
    print("ENTRENAMIENTO MammoVLM V2 (exp06) - VinDr + Mammo-CLIP")
    print("=" * 70)
    print(f"Dispositivo: {device}")
    print(f"Parametros entrenables: {model.get_trainable_parameters():,}")
    print(f"Resolucion de entrada: {config.image_height}x{config.image_width}")

    ## Directorio del test set reservado
    if getattr(config, "test_set_dir", None):
        test_dir = Path(config.test_set_dir)
    else:
        test_dir = Path(config.checkpoint_dir).parent / "test_sets"
    logger.info("Directorio de test sets: %s", test_dir)

    ## ===== Cargar registros de VinDr =====
    records = load_vindr_records(dataset_root)
    if len(records) == 0:
        print("ERROR: no se cargaron registros de VinDr")
        return {}

    ## Separar el test set fijo (split oficial de VinDr)
    if config.max_samples is None:
        records, test_df = separate_test_set_vindr(
            records, config, test_dir / "test_set_vindr.csv",
        )
        print(f"  Test set reservado: {len(test_df)} muestras")

    ## Submuestreo para prueba de humo
    if config.max_samples is not None:
        n_antes = len(records)
        records = subsample_stratified(records, config.max_samples, config.random_seed)
        print(f"PRUEBA DE HUMO: reducido de {n_antes} a {len(records)} registros")

    ## Split train/val
    train_df, val_df = split_train_val(records, config.val_fraction, config.random_seed)
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}")

    ## Transforms (alta resolucion). Augment solo en train.
    train_transform = MammoCLIPTransform(
        height=config.image_height, width=config.image_width,
        augment=True, use_clahe=True,
    )
    val_transform = MammoCLIPTransform(
        height=config.image_height, width=config.image_width,
        augment=False, use_clahe=True,
    )

    train_ds = MammoDataset(train_df, train_transform, augment=True)
    val_ds = MammoDataset(val_df, val_transform, augment=False)

    ## Oversampling opcional (WeightedRandomSampler reemplaza al shuffle)
    if config.use_oversampling:
        train_sampler = build_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, sampler=train_sampler,
            num_workers=config.num_workers, pin_memory=True,
        )
        logger.info("Oversampling activado (WeightedRandomSampler)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    ## Class weights para BI-RADS (compensar desbalance)
    ## weight_power amplifica el sesgo hacia minoritarias (exp07+)
    class_weights = compute_birads_class_weights(
        train_df, weight_power=config.weight_power
    ).to(device)
    logger.info("Class weights BI-RADS (power=%.1f): %s",
                config.weight_power, [round(w, 3) for w in class_weights.cpu().tolist()])

    ## Loss multi-tarea (BI-RADS + densidad)
    loss_fn = MultiTaskLoss(
        birads_weight=config.birads_weight,
        density_weight=config.density_weight,
        birads_class_weights=class_weights,
        use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
        num_birads_classes=NUM_BIRADS_CLASSES,
        use_ordinal_loss=config.use_ordinal_loss,
        ordinal_lambda_qwk=config.ordinal_lambda_qwk,
        ordinal_distance_power=config.ordinal_distance_power,
        ordinal_undergrade_beta=config.ordinal_undergrade_beta,
    )
    if config.use_ordinal_loss:
        logger.info(
            "Usando Loss Ordinal hibrida: SORD + %.2f*QWK (distance_power=%.1f, undergrade_beta=%.1f)",
            config.ordinal_lambda_qwk, config.ordinal_distance_power, config.ordinal_undergrade_beta,
        )
    elif config.use_focal_loss:
        logger.info("Usando Focal Loss (gamma=%.1f) para manejar desbalance", config.focal_gamma)

    ## Mover la loss al device. Es necesario porque la loss ordinal tiene
    ## buffers internos (soft_labels, weight_matrix) que deben estar en el mismo
    ## device que los tensores de datos para poder indexarlos.
    loss_fn = loss_fn.to(device)

    ## Optimizador con LR diferenciado:
    ## los heads (nuevos) usan el LR completo; el encoder descongelado (si lo hay)
    ## usa LR/10 para no destruir las features preentrenadas de Mammo-CLIP
    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    param_groups = [{"params": head_params, "lr": config.lr}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config.lr * 0.1})
        logger.info("Encoder con %d params entrenables (LR=%.1e), heads LR=%.1e",
                    sum(p.numel() for p in encoder_params), config.lr * 0.1, config.lr)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    ## Reanudacion
    start_epoch = 1
    if config.resume:
        resumed = load_checkpoint_if_exists(model, optimizer, config.checkpoint_dir, device)
        if resumed > 0:
            start_epoch = resumed + 1
            for _ in range(resumed):
                scheduler.step()
            print(f"  Reanudando desde epoca {start_epoch}")

    ## Loop de entrenamiento
    best_val_loss = float("inf")
    history = []

    if start_epoch > config.epochs:
        print(f"  Entrenamiento ya completado ({config.epochs} epocas)")
        return {"stage1": history}

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoca {epoch}/{config.epochs} ({elapsed:.0f}s): "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['val_loss']:.4f}  "
              f"val_birads_acc={val_metrics['val_birads_acc']:.4f}", flush=True)

        history.append({**train_metrics, **val_metrics, "epoch": epoch})

        ## Notificar progreso (para monitoreo en background)
        ## Se mantiene la firma (stage_num, history) por compatibilidad con el
        ## sistema de tracking; en exp06 hay una sola etapa, asi que stage_num=1
        if epoch_callback is not None:
            epoch_callback(1, history)

        ## Guardar mejor checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, config.checkpoint_dir)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    ## Se devuelve bajo la clave 'stage1' por compatibilidad con el codigo de
    ## monitoreo y registro que espera esa estructura
    return {"stage1": history}
