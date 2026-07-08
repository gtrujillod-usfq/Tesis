## profile_computational_cost.py
## Profiling ligero de costo computacional del pipeline MammoVLM (exp08)
## No entrena ni modifica pesos: solo carga el checkpoint congelado y mide.
## Modulo de vision: MammoVLM (encoder Mammo-CLIP EfficientNet-B5 + dual head)
## Modulo de lenguaje: Qwen2.5-7B-Instruct (generador de informes) + retrieval RAG
## Pipeline completo: imagen -> escalares -> retrieval RAG -> informe generado

import os

## Fijar la GPU ANTES de importar torch: el servidor tiene 4x H200 compartidas
## con otros procesos: se usa la GPU 1, casi libre al momento de correr esto
## (531 MiB / 143771 MiB ocupados segun nvidia-smi), para no interferir con
## otros jobs en las GPU 0, 2 y 3.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
import json
import time
import hashlib
import subprocess
import warnings
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import torch

TESIS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TESIS_ROOT / "src"))

from utils import count_parameters
from models import MammoVLM
from data_loading import MammoCLIPTransform, load_image_as_pil
from rag import create_rag_pipeline
from report_generator import ReportGenerator, load_llm_for_generation

## ============================================================
## Constantes congeladas (alineadas con XAI/xai/config_xai.py,
## unica fuente de verdad de rutas y arquitectura de exp08)
## ============================================================

EXPERIMENT_FINAL = "exp08_ordinal_sord_qwk_descongelado"
MAMMOCLIP_CKPT = str(TESIS_ROOT / "models" / "mammo_clip_b5.tar")
EXP08_CKPT = str(TESIS_ROOT / "outputs" / "experiments" / EXPERIMENT_FINAL / "model.pt")
TEST_CSV = TESIS_ROOT / "outputs" / "test_sets" / "test_set_vindr.csv"
RAG_INDEX_DIR = TESIS_ROOT / "data" / "rag_index"
LITERATURE_DIR = TESIS_ROOT / "Libros"
OUT_DIR = TESIS_ROOT / "results"

## Resolucion de entrada del encoder, confirmada en config_xai.py
IMAGE_HEIGHT = 1520
IMAGE_WIDTH = 912
## Bloques finales descongelados del encoder en exp08 (config_xai.py)
UNFREEZE_LAST_N_BLOCKS = 2

QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_DTYPE = "bfloat16"
RAG_TOP_K = 3

SEED = 42
N_FORWARD_LATENCY = 60   ## >=50 pasadas pedidas para la latencia del encoder
N_WARMUP = 10
N_PIPELINE_CASES = 15    ## dentro del rango 10-20 pedido para LLM/pipeline

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Este profiling requiere GPU; no se detecto CUDA."


def git_hash() -> str:
    ## Hash del commit HEAD para trazabilidad del resultado
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(TESIS_ROOT)
        ).decode().strip()
    except Exception:
        return "N/A"


def file_sha256(path: str) -> str:
    ## Hash del checkpoint para confirmar que se midio exp08 y no otro experimento
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def mb(n_bytes) -> float:
    ## Bytes a megabytes con 2 decimales
    return round(n_bytes / (1024 ** 2), 2)


CKPT_SHA256 = file_sha256(EXP08_CKPT)
COMMIT_HASH = git_hash()
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print("=" * 70)
print("PROFILING DE COSTO COMPUTACIONAL - exp08_ordinal_sord_qwk_descongelado")
print("=" * 70)
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Checkpoint exp08: {EXP08_CKPT}")
print(f"SHA256 checkpoint: {CKPT_SHA256}")
print(f"Commit HEAD: {COMMIT_HASH}")

results = {
    "metadata": {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": TIMESTAMP,
        "commit": COMMIT_HASH,
        "seed": SEED,
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "checkpoint_path": EXP08_CKPT,
        "checkpoint_sha256": CKPT_SHA256,
        "mammoclip_checkpoint_path": MAMMOCLIP_CKPT,
        "test_csv_path": str(TEST_CSV),
        "rag_index_dir": str(RAG_INDEX_DIR),
        "vision_input_shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
        "qwen_model_id": QWEN_MODEL_ID,
        "qwen_dtype": QWEN_DTYPE,
        "rag_top_k": RAG_TOP_K,
    }
}

## ============================================================
## 1. MODULO DE VISION (clasificador exp08)
## ============================================================
print()
print("=" * 70)
print("1. MODULO DE VISION")
print("=" * 70)

## Arquitectura identica a la usada en main.ipynb (celda 7.2) para cargar
## exp08 en inferencia: encoder congelado salvo los 2 bloques finales
vlm_model = MammoVLM(
    checkpoint_path=MAMMOCLIP_CKPT,
    efficientnet_name="efficientnet-b5",
    num_birads_classes=5,
    num_density_classes=4,
    freeze_encoder=True,
    unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
    hidden_dim=256,
    dropout=0.2,
)
ckpt = torch.load(EXP08_CKPT, map_location="cpu", weights_only=False)
missing, unexpected = vlm_model.load_state_dict(ckpt["model_state_dict"], strict=True)
assert not missing and not unexpected, f"claves no cargadas: missing={missing} unexpected={unexpected}"
vlm_model = vlm_model.to(DEVICE)
vlm_model.eval()
print("Checkpoint exp08 cargado correctamente (strict=True, sin claves faltantes)")

## --- 1.1 Conteo de parametros (funcion ya existente en src/utils.py) ---
param_counts = count_parameters(vlm_model)
print(f"Parametros totales:     {param_counts['total']:,}")
print(f"Parametros entrenables: {param_counts['trainable']:,} "
      f"({param_counts['trainable_percentage']:.2f}%)")
print(f"Parametros congelados:  {param_counts['frozen']:,}")

## --- Imagen real del test set VinDr para las mediciones ---
test_df = pd.read_csv(TEST_CSV)
sample_row = test_df.iloc[0]
transform_eval = MammoCLIPTransform(
    height=IMAGE_HEIGHT, width=IMAGE_WIDTH, augment=False, use_clahe=True
)
pil_img = load_image_as_pil(sample_row["image_path"])
img_tensor = transform_eval(pil_img).unsqueeze(0).to(DEVICE)
assert tuple(img_tensor.shape) == (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
print(f"Imagen de prueba: {sample_row['image_path']}")

## --- 1.2 VRAM pico de un forward pass ---
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
with torch.no_grad():
    _ = vlm_model(img_tensor)
torch.cuda.synchronize()
vram_forward_bytes = torch.cuda.max_memory_allocated()
print(f"VRAM pico forward pass (1,3,{IMAGE_HEIGHT},{IMAGE_WIDTH}): {mb(vram_forward_bytes)} MB")

## --- 1.3 GFLOPs del encoder + cabezas con ptflops ---
gflops_info = {"library": "ptflops", "gflops": None, "macs": None, "error": None}
try:
    from ptflops import get_model_complexity_info

    with torch.no_grad():
        macs, _params_ptflops = get_model_complexity_info(
            vlm_model,
            (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    ## 1 MAC (multiplicacion-acumulacion) equivale a 2 FLOPs
    gflops_info["macs"] = int(macs)
    gflops_info["gflops"] = round(2 * macs / 1e9, 3)
    print(f"GFLOPs (ptflops, MACs*2, entrada 1x3x{IMAGE_HEIGHT}x{IMAGE_WIDTH}): "
          f"{gflops_info['gflops']}")
except Exception as e:
    gflops_info["error"] = str(e)
    print(f"No se pudo calcular GFLOPs con ptflops: {e}")

## --- 1.4 Latencia por imagen: media y desviacion sobre >=50 forward passes ---
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = vlm_model(img_tensor)
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(N_FORWARD_LATENCY):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = vlm_model(img_tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

latencies_ms = np.array(latencies_ms)
vision_latency = {
    "n_forward_passes": N_FORWARD_LATENCY,
    "n_warmup": N_WARMUP,
    "mean_ms": float(latencies_ms.mean()),
    "std_ms": float(latencies_ms.std()),
    "min_ms": float(latencies_ms.min()),
    "max_ms": float(latencies_ms.max()),
}
print(f"Latencia por imagen: {vision_latency['mean_ms']:.2f} +/- "
      f"{vision_latency['std_ms']:.2f} ms (n={N_FORWARD_LATENCY}, warmup={N_WARMUP})")

results["vision_module"] = {
    "architecture": "MammoCLIP EfficientNet-B5 (encoder) + dual head BI-RADS/densidad",
    "checkpoint": EXP08_CKPT,
    "checkpoint_sha256": CKPT_SHA256,
    "test_image_used": str(sample_row["image_path"]),
    "parameters": param_counts,
    "vram_peak_forward_pass_isolated": {
        "note": "medido con SOLO el modulo de vision cargado en GPU (antes de cargar el LLM)",
        "bytes": int(vram_forward_bytes),
        "mb": mb(vram_forward_bytes),
        "input_shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
    },
    "gflops": gflops_info,
    "latency_per_image": vision_latency,
}

## ============================================================
## 2 y 3. MODULO DE LENGUAJE (Qwen2.5-7B + RAG) y PIPELINE COMPLETO
## ============================================================
print()
print("=" * 70)
print("2. MODULO DE LENGUAJE + 3. PIPELINE COMPLETO")
print("=" * 70)

try:
    ## --- Retriever RAG: carga el indice ya construido, no lo reconstruye ---
    print("Cargando indice RAG desde cache (force_rebuild=False)...")
    rag_indexer, rag_retriever = create_rag_pipeline(
        pdf_dir=str(LITERATURE_DIR),
        index_dir=str(RAG_INDEX_DIR),
        device="auto",
        force_rebuild=False,
    )
    rag_chunks_total = rag_indexer.get_index_stats().get("total_chunks", 0)
    print(f"  RAG listo: {rag_chunks_total} chunks indexados")

    ## --- Cargar LLM Qwen2.5-7B-Instruct (ya cacheado localmente en HF hub) ---
    print("Cargando LLM Qwen2.5-7B-Instruct...")
    t_load0 = time.perf_counter()
    llm_model, llm_tokenizer = load_llm_for_generation(
        model_name=QWEN_MODEL_ID, device="auto", dtype=QWEN_DTYPE,
    )
    t_load1 = time.perf_counter()
    llm_load_seconds = round(t_load1 - t_load0, 1)
    print(f"  LLM cargado en {llm_load_seconds}s")

    ## Parametros del LLM: misma funcion count_parameters(), reutilizada sobre
    ## el objeto AutoModelForCausalLM ya cargado. El desglose entrenable/congelado
    ## no es semanticamente relevante aqui porque el LLM solo se usa en inferencia
    ## (nunca se llama .backward() en este pipeline), pero se reporta tal cual
    ## lo entrega la funcion existente.
    llm_param_counts = count_parameters(llm_model)
    print(f"Parametros LLM (accesibles del objeto cargado): {llm_param_counts['total']:,}")

    report_generator = ReportGenerator(
        retriever=rag_retriever, llm=llm_model, tokenizer=llm_tokenizer,
        language="es", use_rag=True, rag_top_k=RAG_TOP_K,
    )

    ## --- Seleccionar casos reales, estratificados por BI-RADS (1-5) ---
    casos_por_clase = []
    for b in [1, 2, 3, 4, 5]:
        subset = test_df[test_df["birads"] == b]
        n_tomar = min(3, len(subset))
        if n_tomar > 0:
            casos_por_clase.append(subset.sample(n_tomar, random_state=SEED))
    casos_df = pd.concat(casos_por_clase).head(N_PIPELINE_CASES).reset_index(drop=True)
    print(f"Casos seleccionados para modulo de lenguaje / pipeline completo: {len(casos_df)}")

    per_case_records = []
    for i, row in casos_df.iterrows():
        ## Ventana 1: imagen -> preprocesamiento -> encoder -> escalares
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        pil_img_i = load_image_as_pil(row["image_path"])
        img_tensor_i = transform_eval(pil_img_i).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs_i = vlm_model(img_tensor_i)
            birads_probs_i = torch.softmax(outputs_i["birads"][0], dim=-1)
            density_probs_i = torch.softmax(outputs_i["density"][0], dim=-1)
        birads_idx = int(torch.argmax(birads_probs_i).item())
        density_idx = int(torch.argmax(density_probs_i).item())
        birads_conf = float(birads_probs_i[birads_idx].item())
        malignancy_score = float(birads_probs_i[3].item() + birads_probs_i[4].item())

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        vram_after_vision = torch.cuda.max_memory_allocated()

        prediction = {
            "birads_pred": birads_idx,
            "birads_confidence": birads_conf,
            "density_pred": density_idx,
            "malignancy_score": malignancy_score,
        }

        ## Ventana 2: retrieval RAG + generate() del LLM, aislada para medir
        ## el costo del modulo de lenguaje en especifico
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1b = time.perf_counter()
        result = report_generator.generate(prediction, max_new_tokens=500)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        vram_llm_only_peak = torch.cuda.max_memory_allocated()

        record = {
            "image_path": str(row["image_path"]),
            "birads_real": int(row["birads"]),
            "birads_predicho": birads_idx + 1,
            "vision_latency_ms": (t1 - t0) * 1000.0,
            "llm_rag_latency_ms": (t2 - t1b) * 1000.0,
            "end_to_end_latency_ms": (t2 - t0) * 1000.0,
            "vram_peak_after_vision_mb": mb(vram_after_vision),
            "vram_peak_llm_only_mb": mb(vram_llm_only_peak),
            "vram_peak_pipeline_mb": mb(max(vram_after_vision, vram_llm_only_peak)),
            "rag_chunks_used": result["rag_chunks_used"],
            "n_report_chars": len(result["report"]),
        }
        per_case_records.append(record)
        print(f"  caso {i + 1}/{len(casos_df)}  BI-RADS real={row['birads']} "
              f"pred={birads_idx + 1}  e2e={record['end_to_end_latency_ms']:.0f} ms  "
              f"VRAM_pico={record['vram_peak_pipeline_mb']:.0f} MB")

    df_cases = pd.DataFrame(per_case_records)

    llm_latency_stats = {
        "n_cases": len(df_cases),
        "mean_ms": float(df_cases["llm_rag_latency_ms"].mean()),
        "std_ms": float(df_cases["llm_rag_latency_ms"].std()),
        "min_ms": float(df_cases["llm_rag_latency_ms"].min()),
        "max_ms": float(df_cases["llm_rag_latency_ms"].max()),
    }
    llm_vram_stats = {
        "note": "VRAM pico medida en una ventana propia alrededor de report_generator.generate() (retrieval RAG + generacion LLM), con vlm_model ya residente en GPU",
        "mean_mb": float(df_cases["vram_peak_llm_only_mb"].mean()),
        "max_mb": float(df_cases["vram_peak_llm_only_mb"].max()),
    }

    pipeline_latency_stats = {
        "n_cases": len(df_cases),
        "mean_ms": float(df_cases["end_to_end_latency_ms"].mean()),
        "std_ms": float(df_cases["end_to_end_latency_ms"].std()),
        "min_ms": float(df_cases["end_to_end_latency_ms"].min()),
        "max_ms": float(df_cases["end_to_end_latency_ms"].max()),
    }
    pipeline_vram_stats = {
        "note": "maximo entre el pico tras la fase de vision y el pico durante generate(), por caso; incluye los pesos residentes de vlm_model + Qwen2.5-7B + modelo de embeddings RAG, cargados simultaneamente (igual que en el pipeline real de main.ipynb celdas 7.2-7.4)",
        "mean_mb": float(df_cases["vram_peak_pipeline_mb"].mean()),
        "max_mb": float(df_cases["vram_peak_pipeline_mb"].max()),
    }

    results["language_module"] = {
        "model_id": QWEN_MODEL_ID,
        "dtype": QWEN_DTYPE,
        "load_time_seconds": llm_load_seconds,
        "parameters": llm_param_counts,
        "parameters_note": "conteo real via count_parameters() sobre el objeto AutoModelForCausalLM ya cargado en memoria",
        "rag_top_k": RAG_TOP_K,
        "rag_index_chunks": rag_chunks_total,
        "latency_per_report": llm_latency_stats,
        "vram_peak": llm_vram_stats,
    }

    results["full_pipeline"] = {
        "description": "imagen -> preprocesamiento -> encoder+heads -> argmax -> retrieval RAG -> generate() LLM -> informe",
        "n_cases": len(df_cases),
        "latency_end_to_end": pipeline_latency_stats,
        "vram_peak_end_to_end": pipeline_vram_stats,
        "per_case_detail": per_case_records,
    }

except Exception as e:
    ## Si algo falla en la carga del LLM/RAG, se conserva igual el profiling
    ## del modulo de vision ya medido, con el error explicito de lo que fallo
    print(f"ERROR en modulo de lenguaje / pipeline completo: {e}")
    results["language_module"] = {"error": str(e)}
    results["full_pipeline"] = {"error": str(e)}

## ============================================================
## Tamano en disco de los modelos (proxy de tamano de modelo)
## ============================================================
results["model_sizes_on_disk"] = {
    "exp08_checkpoint_mb": mb(Path(EXP08_CKPT).stat().st_size),
    "mammoclip_backbone_checkpoint_mb": mb(Path(MAMMOCLIP_CKPT).stat().st_size),
}

## ============================================================
## Guardar JSON y tabla resumen
## ============================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"computational_cost_{EXPERIMENT_FINAL}_{TIMESTAMP}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print()
print("=" * 70)
print("TABLA RESUMEN")
print("=" * 70)
v = results["vision_module"]
print(f"{'Modulo de vision':30s} params_totales={v['parameters']['total']:>12,}  "
      f"params_entrenables={v['parameters']['trainable']:>10,}  "
      f"VRAM_fwd={v['vram_peak_forward_pass_isolated']['mb']:>8.1f} MB  "
      f"GFLOPs={v['gflops']['gflops']}  "
      f"latencia={v['latency_per_image']['mean_ms']:.1f}+/-{v['latency_per_image']['std_ms']:.1f} ms")

if "error" not in results["language_module"]:
    l = results["language_module"]
    p = results["full_pipeline"]
    print(f"{'Modulo de lenguaje':30s} params_totales={l['parameters']['total']:>12,}  "
          f"VRAM_generate={l['vram_peak']['mean_mb']:>8.1f} MB  "
          f"latencia={l['latency_per_report']['mean_ms']:.1f}+/-{l['latency_per_report']['std_ms']:.1f} ms  "
          f"(n={l['latency_per_report']['n_cases']})")
    print(f"{'Pipeline completo':30s} VRAM_e2e={p['vram_peak_end_to_end']['mean_mb']:>8.1f} MB  "
          f"latencia_e2e={p['latency_end_to_end']['mean_ms']:.1f}+/-{p['latency_end_to_end']['std_ms']:.1f} ms  "
          f"(n={p['n_cases']})")
else:
    print("Modulo de lenguaje / pipeline completo: FALLO (ver 'error' en el JSON)")

print()
print(f"JSON guardado en: {out_path}")
