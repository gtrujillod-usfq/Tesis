## verify_encoder_load.py
## Verificacion de la carga del encoder Mammo-CLIP con el checkpoint REAL
##
## A diferencia del script de inspeccion (que solo mira la estructura), este
## script construye el MammoCLIPEncoder real y carga los pesos del checkpoint,
## confirmando que el mapeo funciona de extremo a extremo y que el encoder
## produce features de la dimension esperada.
##
## Uso en el H200:
##   python verify_encoder_load.py --checkpoint /home/gtrujillod/Tesis/models/mammo_clip_b5.tar

import argparse
import sys
from pathlib import Path

import torch

## Agregar src al path para importar el encoder
sys.path.insert(0, str(Path(__file__).parent / "src"))


def verify(checkpoint_path: str):
    from models import MammoCLIPEncoder

    print("=" * 70)
    print("VERIFICACION DE CARGA DEL ENCODER MAMMO-CLIP")
    print("=" * 70)

    ## Construir el encoder y cargar los pesos del checkpoint real
    encoder = MammoCLIPEncoder(
        checkpoint_path=checkpoint_path,
        efficientnet_name="efficientnet-b5",
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
    )
    encoder.load_backbone()
    encoder.eval()

    ## Verificar la dimension de features con una imagen de prueba
    print()
    print("Probando forward con una imagen de alta resolucion (1, 3, 512, 512)...")
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        feats = encoder(x)
    print(f"  Features de salida: {tuple(feats.shape)}")
    print(f"  Dimension esperada: 2048")

    if feats.shape[-1] == 2048:
        print()
        print("  RESULTADO: el encoder carga y produce features de 2048 dims.")
        print("  Listo para integrar en el entrenamiento.")
    else:
        print()
        print(f"  ATENCION: dimension inesperada ({feats.shape[-1]}).")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificar carga del encoder Mammo-CLIP")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .tar")
    args = parser.parse_args()
    verify(args.checkpoint)
