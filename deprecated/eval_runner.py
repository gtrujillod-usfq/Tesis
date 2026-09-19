## eval_runner.py
## Evaluacion en background del MammoVLM V2 (exp06) sobre el test set de VinDr
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## Proposito: evaluar el modelo entrenado como proceso independiente (igual que
## train_runner.py), para correr en GPU sin depender del kernel del notebook.
## Esto resuelve el caso en que el kernel del notebook se reinicio sin acceso a
## CUDA tras cerrar la sesion remota.
##
## Maneja correctamente la numeracion de exp06: BI-RADS 1-5 mapeado a indices 0-4.
## El corte benigno/maligno se ajusta a esa numeracion:
##   benigno    = BI-RADS 1, 2, 3 = indices 0, 1, 2
##   maligno    = BI-RADS 4, 5     = indices 3, 4
##
## Uso (desde una terminal del H200 con GPU, NO desde el notebook):
##   cd /home/gtrujillod/Tesis
##   nohup python3 eval_runner.py > outputs/training_run/eval.out 2>&1 &
##
## Escribe el resultado en:
##   outputs/training_run/eval_status.json  : estado legible por el notebook
##   results/test_set_evaluation_<timestamp>.json : reporte completo de metricas

import sys
import json
import logging
from pathlib import Path
from datetime import datetime


def write_status(status_path, status_dict):
    ## Escritura atomica del estado (el notebook lee este archivo)
    status_dict["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = str(status_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(status_path)


def main():
    ## Localizar el proyecto y agregar src al path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(project_root))

    ## Leer la configuracion del runner de evaluacion
    config_path = project_root / "eval_runner_config.json"
    if not config_path.exists():
        print(f"ERROR: no se encontro {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "eval_status.json"
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    ## Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "eval.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("eval_runner")

    write_status(status_path, {"state": "starting", "message": "Iniciando evaluacion"})

    try:
        import torch
        import numpy as np
        import pandas as pd
        from torch.utils.data import DataLoader

        from models import MammoVLM
        from data_loading import MammoCLIPTransform, MammoDataset
        from medical_metrics import MedicalMetricsReport, ClinicalSeverityMetrics

        ## Verificar GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Dispositivo: %s (CUDA disponible: %s)", device, torch.cuda.is_available())
        if device == "cpu":
            logger.warning("CUDA no disponible; la evaluacion en CPU sera lenta.")

        tc = cfg["model_config"]

        ## Construir el modelo y cargar los pesos entrenados
        logger.info("Construyendo modelo y cargando pesos finales...")
        model = MammoVLM(
            checkpoint_path=cfg["checkpoint_path"],
            efficientnet_name=tc.get("efficientnet_name", "efficientnet-b5"),
            num_birads_classes=tc.get("num_birads_classes", 5),
            num_density_classes=tc.get("num_density_classes", 4),
            freeze_encoder=True,
            unfreeze_last_n_blocks=0,
        )
        final_model_path = Path(cfg["final_model_path"])
        checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        logger.info("Modelo cargado desde %s", final_model_path)

        ## Cargar el test set
        test_path = Path(cfg["test_set_path"])
        test_records = pd.read_csv(test_path)
        logger.info("Test set: %d muestras", len(test_records))

        write_status(status_path, {
            "state": "evaluating",
            "message": f"Evaluando {len(test_records)} muestras en {device}",
        })

        ## Transform de evaluacion (alta resolucion, sin augmentation)
        eval_transform = MammoCLIPTransform(
            height=tc.get("image_height", 1520),
            width=tc.get("image_width", 912),
            augment=False, use_clahe=True,
        )
        test_ds = MammoDataset(test_records, eval_transform, augment=False)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.get("batch_size", 8),
            shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True,
        )

        ## Recolectar predicciones
        y_true, y_pred, y_probs = [], [], []
        n_done = 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                probs = torch.softmax(outputs["birads"], dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(batch["birads"].numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                y_probs.extend(probs.cpu().numpy().tolist())

                n_done += len(images)
                if n_done % 400 == 0:
                    logger.info("Procesadas %d/%d imagenes", n_done, len(test_records))
                    write_status(status_path, {
                        "state": "evaluating",
                        "message": f"Procesadas {n_done}/{len(test_records)} imagenes",
                    })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        logger.info("Predicciones recolectadas: %d", len(y_true))

        ## Calcular metricas de Area 3 con 5 clases
        ## El corte benigno/maligno se infiere automaticamente de num_classes=5
        ## (benigno = BI-RADS 1-3 = indices 0-2; maligno = BI-RADS 4-5 = indices 3-4)
        report_gen = MedicalMetricsReport(num_classes=5)

        test_report = report_gen.compute_full_report(y_true, y_pred, y_probs)
        summary = report_gen.generate_summary(test_report)
        logger.info("\n%s", summary)

        ## Guardar el reporte completo
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = results_dir / f"test_set_evaluation_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)
        logger.info("Reporte guardado: %s", report_path)

        write_status(status_path, {
            "state": "completed",
            "message": "Evaluacion completada",
            "report_path": str(report_path),
            "summary": summary,
        })
        logger.info("EVALUACION COMPLETADA")

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error("Error en evaluacion: %s", error_trace)
        write_status(status_path, {
            "state": "error",
            "message": f"Error: {str(e)}",
            "traceback": error_trace,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
