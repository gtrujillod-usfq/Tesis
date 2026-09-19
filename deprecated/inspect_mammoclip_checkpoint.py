## inspect_mammoclip_checkpoint.py
## Script de inspeccion del checkpoint de Mammo-CLIP (EfficientNet-B5)
##
## Proposito: antes de integrar Mammo-CLIP en el modelo completo, este script
## carga SOLO el checkpoint .tar y verifica que los pesos del image encoder
## se pueden mapear correctamente al EfficientNet-B5 de timm. Aisla el riesgo
## de la estructura del checkpoint antes de construir el resto del pipeline.
##
## Uso en el H200:
##   1. Descargar el checkpoint desde Hugging Face (ver instrucciones abajo)
##   2. python inspect_mammoclip_checkpoint.py --checkpoint /ruta/al/b5-model-best-epoch-7.tar
##
## Para descargar el checkpoint (ejecutar una vez en el H200):
##   pip install huggingface_hub
##   python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='shawn24/Mammo-CLIP', filename='Pre-trained-checkpoints/b5-model-best-epoch-7.tar'))"
##   (el comando imprime la ruta local donde quedo descargado el .tar)

import argparse
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path: str, timm_model_name: str = "tf_efficientnet_b5"):
    ##
    ## Inspecciona la estructura del checkpoint y cuantifica el mapeo de pesos
    ## hacia el backbone EfficientNet-B5 de timm
    ##
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"ERROR: no se encontro el checkpoint en {ckpt_path}")
        return

    print("=" * 70)
    print("INSPECCION DEL CHECKPOINT DE MAMMO-CLIP")
    print("=" * 70)
    print(f"  Archivo: {ckpt_path}")
    print(f"  Tamano: {ckpt_path.stat().st_size / 1e6:.1f} MB")
    print()

    ## Cargar el checkpoint
    ## weights_only=False porque el .tar puede contener objetos de configuracion
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ## Paso 1: identificar la estructura de nivel superior
    print("PASO 1: ESTRUCTURA DE NIVEL SUPERIOR")
    print("-" * 70)
    if isinstance(ckpt, dict):
        print(f"  El checkpoint es un dict con {len(ckpt)} claves de nivel superior:")
        for k in ckpt.keys():
            v = ckpt[k]
            tipo = type(v).__name__
            if isinstance(v, dict):
                print(f"    '{k}': dict con {len(v)} claves")
            elif torch.is_tensor(v):
                print(f"    '{k}': tensor {tuple(v.shape)}")
            else:
                print(f"    '{k}': {tipo} = {repr(v)[:60]}")
    else:
        print(f"  El checkpoint NO es un dict, es: {type(ckpt).__name__}")
    print()

    ## Paso 2: localizar el state_dict
    print("PASO 2: LOCALIZAR EL STATE_DICT DEL MODELO")
    print("-" * 70)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print("  state_dict encontrado en la clave 'model'")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("  state_dict encontrado en la clave 'state_dict'")
    else:
        state_dict = ckpt
        print("  Se usa el checkpoint directamente como state_dict")

    if not isinstance(state_dict, dict):
        print(f"  ERROR: el state_dict no es un dict, es {type(state_dict).__name__}")
        return
    print(f"  Total de claves en el state_dict: {len(state_dict)}")
    print()

    ## Paso 3: mostrar una muestra de las claves para ver los prefijos
    print("PASO 3: MUESTRA DE CLAVES DEL STATE_DICT")
    print("-" * 70)
    all_keys = list(state_dict.keys())
    print("  Primeras 15 claves:")
    for k in all_keys[:15]:
        shape = tuple(state_dict[k].shape) if torch.is_tensor(state_dict[k]) else "?"
        print(f"    {k}  {shape}")
    print()

    ## Identificar los prefijos unicos (primer y segundo nivel)
    prefijos_nivel1 = {}
    for k in all_keys:
        p = k.split(".")[0]
        prefijos_nivel1[p] = prefijos_nivel1.get(p, 0) + 1
    print("  Prefijos de primer nivel (y cuantas claves tiene cada uno):")
    for p, c in sorted(prefijos_nivel1.items(), key=lambda x: -x[1]):
        print(f"    '{p}.': {c} claves")
    print()

    ## Paso 4: construir el backbone timm y medir el mapeo
    print("PASO 4: MAPEO HACIA EL BACKBONE EFFICIENTNET-B5 (timm)")
    print("-" * 70)
    import timm
    backbone = timm.create_model(timm_model_name, pretrained=False, num_classes=0)
    backbone_keys = set(backbone.state_dict().keys())
    print(f"  El backbone timm '{timm_model_name}' tiene {len(backbone_keys)} claves")
    print()

    ## Probar los mismos prefijos candidatos que usa el modelo
    candidate_prefixes = [
        "image_encoder.image_encoder.",
        "image_encoder.encoder.",
        "image_encoder.",
        "img_encoder.",
        "visual.",
        "",  ## sin prefijo (carga directa)
    ]

    print("  Probando prefijos candidatos para el mapeo:")
    mejor_prefijo = None
    mejor_matches = 0
    for prefix in candidate_prefixes:
        n_matches = 0
        for k in all_keys:
            if prefix == "":
                if k in backbone_keys:
                    n_matches += 1
            else:
                if k.startswith(prefix) and k[len(prefix):] in backbone_keys:
                    n_matches += 1
        etiqueta = f"'{prefix}'" if prefix else "(sin prefijo)"
        print(f"    {etiqueta}: {n_matches}/{len(backbone_keys)} claves coinciden")
        if n_matches > mejor_matches:
            mejor_matches = n_matches
            mejor_prefijo = prefix
    print()

    ## Paso 5: veredicto
    print("PASO 5: VEREDICTO")
    print("-" * 70)
    pct = 100.0 * mejor_matches / len(backbone_keys) if backbone_keys else 0
    if mejor_matches == 0:
        print("  PROBLEMA: ningun prefijo mapeo claves al backbone.")
        print("  Hay que revisar manualmente las claves de arriba (Paso 3) y")
        print("  ajustar la lista de prefijos candidatos en models.py.")
    else:
        etiqueta = f"'{mejor_prefijo}'" if mejor_prefijo else "(sin prefijo)"
        print(f"  Mejor prefijo: {etiqueta}")
        print(f"  Mapea {mejor_matches}/{len(backbone_keys)} claves ({pct:.1f}%)")
        if pct >= 95:
            print("  EXCELENTE: el mapeo es casi completo. El encoder cargara bien.")
        elif pct >= 70:
            print("  ACEPTABLE: la mayoria de pesos mapean. Revisar las claves")
            print("  faltantes por si son criticas (conv_stem, blocks, etc.).")
        else:
            print("  ATENCION: mapeo parcial. Puede que falten capas importantes.")
            print("  Revisar las claves del Paso 3 para ajustar el prefijo.")

        ## Mostrar que claves del backbone NO se mapearon
        if mejor_prefijo is not None:
            remapped = set()
            for k in all_keys:
                if mejor_prefijo == "":
                    if k in backbone_keys:
                        remapped.add(k)
                else:
                    if k.startswith(mejor_prefijo) and k[len(mejor_prefijo):] in backbone_keys:
                        remapped.add(k[len(mejor_prefijo):])
            faltantes = backbone_keys - remapped
            if faltantes:
                print()
                print(f"  Claves del backbone sin mapear ({len(faltantes)}), primeras 10:")
                for k in list(faltantes)[:10]:
                    print(f"    {k}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspeccionar checkpoint de Mammo-CLIP")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Ruta al checkpoint .tar de Mammo-CLIP (ej. b5-model-best-epoch-7.tar)",
    )
    parser.add_argument(
        "--timm-model", default="tf_efficientnet_b5",
        help="Nombre del modelo timm del backbone (default: tf_efficientnet_b5)",
    )
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint, args.timm_model)
