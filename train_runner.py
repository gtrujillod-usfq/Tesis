## train_runner.py
## Runner de entrenamiento en background para el MammoVLM V2
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Este script ejecuta el entrenamiento completo como un proceso independiente
## del notebook. Se lanza con nohup para que sobreviva al cierre de la sesion SSH.
##
## Escribe su progreso en:
##   - <output_dir>/training_status.json : estado legible por el notebook
##   - <output_dir>/training.log         : log completo de ejecucion
##
## Uso:
##   nohup python train_runner.py > /dev/null 2>&1 &
##
## El notebook lee training_status.json para monitorear sin interferir.

import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime


def write_status(status_path, status_dict):
    ##
    ## Escribe el estado actual del entrenamiento a un JSON
    ## El notebook lee este archivo para monitorear el progreso
    ##
    status_dict["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = str(status_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, ensure_ascii=False, indent=2)
    ## Escritura atomica: renombrar evita que el notebook lea un JSON a medias
    Path(tmp).replace(status_path)


def main():
    ##
    ## Punto de entrada del runner. Lee la configuracion desde un JSON,
    ## construye el modelo y lanza el entrenamiento por etapas.
    ##
    ## La configuracion se pasa por un archivo train_runner_config.json
    ## generado por el notebook, con las rutas y parametros necesarios.
    ##

    ## Localizar el directorio del proyecto (donde esta este script o src/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(project_root))

    ## Leer configuracion del runner
    config_path = project_root / "train_runner_config.json"
    if not config_path.exists():
        print(f"ERROR: no se encontro {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        runner_cfg = json.load(f)

    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    output_dir = resolve(runner_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "training_status.json"
    log_path = output_dir / "training.log"

    ## Configurar logging a archivo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger("train_runner")

    ## Estado inicial
    write_status(status_path, {
        "state": "starting",
        "stage": None,
        "message": "Inicializando entrenamiento",
        "history": {},
    })

    try:
        import torch
        from models import MammoVLM
        from data_loading import MammoCLIPTransform
        from train import TrainingConfig, train_mammovlm

        logger.info("Construyendo modelo MammoVLM (exp06: Mammo-CLIP + VinDr)...")
        write_status(status_path, {
            "state": "loading_model",
            "message": "Cargando Mammo-CLIP (EfficientNet-B5) y construyendo heads",
            "history": {},
        })

        tc = runner_cfg["training_config"]

        model = MammoVLM(
            checkpoint_path=str(resolve(runner_cfg["checkpoint_path"])),
            efficientnet_name=runner_cfg.get("efficientnet_name", "efficientnet-b5"),
            num_birads_classes=tc.get("num_birads_classes", 5),
            num_density_classes=tc.get("num_density_classes", 4),
            freeze_encoder=True,
            unfreeze_last_n_blocks=tc.get("unfreeze_last_n_blocks", 0),
        )

        ## Configuracion de entrenamiento (exp06: una sola etapa, VinDr)
        config = TrainingConfig(
            epochs=tc["epochs"],
            lr=tc["lr"],
            batch_size=tc["batch_size"],
            num_workers=tc["num_workers"],
            weight_decay=tc.get("weight_decay", 1e-4),
            val_fraction=tc["val_fraction"],
            random_seed=tc["random_seed"],
            checkpoint_dir=str(resolve(tc["checkpoint_dir"])),
            device=tc.get("device", "auto"),
            image_height=tc.get("image_height", 1520),
            image_width=tc.get("image_width", 912),
            max_samples=tc.get("max_samples", None),
            resume=tc.get("resume", True),
            unfreeze_last_n_blocks=tc.get("unfreeze_last_n_blocks", 0),
            use_focal_loss=tc.get("use_focal_loss", False),
            focal_gamma=tc.get("focal_gamma", 2.0),
            weight_power=tc.get("weight_power", 1.0),
            use_ordinal_loss=tc.get("use_ordinal_loss", False),
            ordinal_lambda_qwk=tc.get("ordinal_lambda_qwk", 0.3),
            ordinal_distance_power=tc.get("ordinal_distance_power", 1.0),
            ordinal_undergrade_beta=tc.get("ordinal_undergrade_beta", 1.0),
            use_oversampling=tc.get("use_oversampling", False),
            birads_weight=tc.get("birads_weight", 1.0),
            density_weight=tc.get("density_weight", 0.5),
            test_set_dir=str(resolve(tc["test_set_dir"])) if tc.get("test_set_dir") else None,
            use_official_split=tc.get("use_official_split", True),
        )

        ## Ruta del dataset VinDr
        dataset_root = str(resolve(runner_cfg["dataset_root"]))

        write_status(status_path, {
            "state": "training",
            "message": "Entrenamiento en progreso",
            "history": {},
        })

        logger.info("Iniciando entrenamiento (una etapa, VinDr)...")

        ## Marca de tiempo de inicio del entrenamiento
        training_start_time = time.time()

        ## Callback que actualiza el estado despues de cada epoca
        accumulated_history = {}

        def on_epoch_end(stage_num, stage_history):
            key = f"stage{stage_num}"
            accumulated_history[key] = stage_history
            write_status(status_path, {
                "state": "training",
                "stage": stage_num,
                "message": f"Entrenando epoca {stage_history[-1]['epoch']}",
                "history": dict(accumulated_history),
            })

        ## Ejecutar entrenamiento (nueva firma: dataset_root, sin schema)
        history = train_mammovlm(
            model=model,
            dataset_root=dataset_root,
            config=config,
            epoch_callback=on_epoch_end,
        )

        ## Calcular duracion total del entrenamiento
        training_duration_s = time.time() - training_start_time
        duration_hours = training_duration_s / 3600.0
        duration_str = f"{int(training_duration_s // 3600)}h {int((training_duration_s % 3600) // 60)}m {int(training_duration_s % 60)}s"
        logger.info("Duracion total del entrenamiento: %s (%.2f horas)", duration_str, duration_hours)

        ## Guardar el modelo final
        final_path = output_dir / "mammovlm_final.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "history": history,
            "training_duration_s": training_duration_s,
            "training_duration_str": duration_str,
        }, final_path)
        logger.info("Modelo final guardado: %s", final_path)

        ## Guardar historial de entrenamiento (incluye la duracion)
        history_path = output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "training_duration_s": training_duration_s,
                "training_duration_str": duration_str,
            }, f, ensure_ascii=False, indent=2)

        write_status(status_path, {
            "state": "completed",
            "message": "Entrenamiento completado exitosamente",
            "final_model": str(final_path),
            "training_duration_s": training_duration_s,
            "training_duration_str": duration_str,
            "history": history,
        })
        logger.info("ENTRENAMIENTO COMPLETADO en %s", duration_str)

    except Exception as e:
        ## Registrar el error en el estado para que el notebook lo vea
        error_trace = traceback.format_exc()
        logger.error("Error en entrenamiento: %s", error_trace)
        write_status(status_path, {
            "state": "error",
            "message": f"Error: {str(e)}",
            "traceback": error_trace,
            "history": {},
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
