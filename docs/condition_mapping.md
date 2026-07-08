# Equivalencia de nomenclatura: condiciones de la tesis vs. experimentos del repo

La tesis se refiere a las nueve corridas del clasificador como **condiciones
experimentales C1–C9**. El repositorio (directorios en `outputs/experiments/`,
configs, notebooks y código) sigue usando los nombres originales **exp01–exp09**
y **no se renombra nada en el repo**: este documento es el único punto de
verdad para traducir entre ambas nomenclaturas.

| Etiqueta en la tesis | Directorio real en `outputs/experiments/` |
|---|---|
| C1 | `exp01_baseline_encoder_congelado` |
| C2 | `exp02_encoder_descongelado_3bloques` |
| C3 | `exp03_focal_loss_encoder_congelado` |
| C4 | `exp04_focal_loss_oversampling` |
| C5 | `exp05_focal_loss_sin_dmid` |
| C6 | `exp06_mammoclip_vindr` |
| C7 | `exp07_focal_gamma3_weights_agresivos` |
| C8 | `exp08_ordinal_sord_qwk_descongelado` |
| C9 | `exp09_asymmetric_sord_weighted` |

**Nota**: la tesis usa las etiquetas `C1`–`C9`; el repositorio usa `exp01`–`exp09`.
Ambas nomenclaturas designan exactamente las mismas corridas — no se debe
renombrar directorios, configs ni código para hacerlas coincidir.

- **C8 (`exp08_ordinal_sord_qwk_descongelado`) es la condición definitiva**: el
  modelo de clasificación adoptado como final del proyecto.
- **C9 (`exp09_asymmetric_sord_weighted`) es una regresión descartada**: su
  diseño (SORD con penalización asimétrica β=2.0) colapsó en el conjunto de
  test pese a resultados comparables a C8 en validación, y no se usa en
  ningún artefacto final del proyecto.
