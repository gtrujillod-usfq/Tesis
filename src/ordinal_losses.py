## ordinal_losses.py
## Funciones de perdida ordinales para clasificacion BI-RADS (exp08)
## Tesis de maestria: Diagnostico Mamografico Asistido por IA
##
## BI-RADS es una escala ordinal (el riesgo progresa de 1 a 5), no nominal.
## La cross-entropy y la focal loss tratan las clases como independientes y
## penalizan igual confundir BR1 con BR2 que BR1 con BR5. Estas losses ordinales
## reorganizan los errores hacia la diagonal (errores leves en vez de graves),
## lo que sube el Quadratic Weighted Kappa y reduce los falsos negativos graves.
##
## Este modulo implementa:
##   - SORDLoss: Soft ORDinal labels (Diaz & Marathe, CVPR 2019). Convierte la
##     etiqueta dura en una distribucion suave por distancia y entrena con KL.
##   - QWKLoss: version diferenciable del Quadratic Weighted Kappa, para empujar
##     directamente hacia esa metrica.
##   - HybridOrdinalLoss: combina SORD + lambda*QWK (con lambda pequeno). El
##     componente SORD da estabilidad; el QWK afina hacia la metrica. Se usa
##     lambda pequeno para evitar la solucion degenerada de "columna-cero" del
##     QWK puro bajo desbalance severo.

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_soft_labels(num_classes: int, distance_power: float = 1.0,
                      undergrade_beta: float = 1.0) -> torch.Tensor:
    ##
    ## Construye la matriz de soft labels ordinales [num_classes, num_classes]
    ##
    ## Para cada clase verdadera t, la fila t es una distribucion sobre todas las
    ## clases, donde la masa decae con la distancia ordinal entre t e i:
    ##   soft[t, i] = softmax(-distancia(t, i)^distance_power)
    ##
    ## Ejemplo (5 clases, distance_power=1, simetrico): para t=2 (BR3) la fila es
    ## aproximadamente [0.05, 0.15, 0.60, 0.15, 0.05]: la mayor masa en la clase
    ## verdadera, menos en las vecinas, casi nada en las lejanas.
    ##
    ## distance_power: 1.0 = penalizacion lineal (estandar); 2.0 = cuadratica
    ##   (concentra mas la masa en la clase verdadera y vecinas inmediatas).
    ##
    ## undergrade_beta (exp09): factor de penalizacion ASIMETRICA del
    ##   sub-diagnostico. La distancia hacia las clases INFERIORES a la verdadera
    ##   (i < t, es decir, predecir un riesgo menor: el falso negativo clinico)
    ##   se multiplica por beta. Con beta > 1, las soft labels ponen menos masa en
    ##   las clases inferiores, sesgando la distribucion hacia arriba y obligando
    ##   al modelo a ser mas preventivo (sube la sensibilidad de malignos).
    ##   beta = 1.0 recupera el SORD simetrico clasico.
    ##
    idx = torch.arange(num_classes, dtype=torch.float32)
    ## Distancia con signo: t (filas) menos i (columnas)
    ## signed[t, i] = t - i. Positivo cuando i < t (clase candidata por debajo).
    signed = idx.unsqueeze(1) - idx.unsqueeze(0)  ## [num_classes, num_classes]
    dist = torch.abs(signed)

    ## Aplicar la penalizacion asimetrica: cuando la clase candidata i esta por
    ## DEBAJO de la verdadera t (signed > 0, i < t), multiplicar la distancia por
    ## beta. Esto penaliza mas que las soft labels asignen masa a clases inferiores
    ## (sub-diagnostico), empujando la distribucion hacia arriba.
    if undergrade_beta != 1.0:
        undergrade_mask = (signed > 0).float()  ## 1 donde i < t
        dist = dist * (1.0 + (undergrade_beta - 1.0) * undergrade_mask)

    ## Penalizacion por distancia y softmax por fila
    penalty = -torch.pow(dist, distance_power)
    soft_labels = F.softmax(penalty, dim=1)
    return soft_labels


class SORDLoss(nn.Module):
    ## Soft ORDinal labels loss (Diaz & Marathe, CVPR 2019)
    ##
    ## En lugar de etiquetas one-hot, usa una distribucion suave por distancia
    ## ordinal y entrena con divergencia KL (equivalente a cross-entropy con
    ## etiquetas suaves). Esto guia el gradiente para que los errores caigan
    ## cerca de la diagonal.
    ##
    ## Soporta class weights (suaves, recomendado) e ignore_index para muestras
    ## sin etiqueta valida.

    def __init__(
        self,
        num_classes: int,
        distance_power: float = 1.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = -100,
        undergrade_beta: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        ## Matriz de soft labels precalculada (se mueve al device en el forward)
        ## undergrade_beta > 1 penaliza el sub-diagnostico (exp09)
        self.register_buffer(
            "soft_labels",
            build_soft_labels(num_classes, distance_power, undergrade_beta),
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ##
        ## logits: [batch, num_classes]
        ## targets: [batch] indices de clase verdadera (0..num_classes-1) o ignore_index
        ##
        ## Filtrar las muestras validas (distintas de ignore_index)
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            ## Todo el batch enmascarado: loss cero que no rompe el gradiente
            return logits.sum() * 0.0

        logits_v = logits[valid_mask]
        targets_v = targets[valid_mask]

        ## Distribucion suave objetivo para cada muestra: [n_valid, num_classes]
        target_dist = self.soft_labels[targets_v]

        ## Log-probabilidades predichas
        log_probs = F.log_softmax(logits_v, dim=1)

        ## Cross-entropy con etiquetas suaves: -sum(target_dist * log_probs) por muestra
        per_sample = -(target_dist * log_probs).sum(dim=1)  ## [n_valid]

        ## Ponderar por class weights (segun la clase verdadera) si se proveen
        if self.class_weights is not None:
            w = self.class_weights[targets_v]
            per_sample = per_sample * w
            return per_sample.sum() / w.sum()

        return per_sample.mean()


class QWKLoss(nn.Module):
    ## Quadratic Weighted Kappa loss (version diferenciable)
    ##
    ## Optimiza directamente una version suave del QWK, que penaliza los errores
    ## proporcionalmente al cuadrado de la distancia ordinal. Retorna (1 - QWK)
    ## para que minimizar la loss equivalga a maximizar el kappa.
    ##
    ## ADVERTENCIA: bajo desbalance severo, el QWK puro puede llevar a soluciones
    ## degeneradas (una clase rara nunca se predice). Por eso se usa SIEMPRE
    ## combinada con SORD/CE y con peso lambda pequeno (ver HybridOrdinalLoss).

    def __init__(self, num_classes: int, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        ## Matriz de pesos cuadraticos W[i,j] = (i - j)^2 / (N-1)^2
        idx = torch.arange(num_classes, dtype=torch.float32)
        w = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
        w = w / ((num_classes - 1) ** 2)
        self.register_buffer("weight_matrix", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ##
        ## logits: [batch, num_classes]
        ## targets: [batch] indices de clase verdadera o ignore_index
        ##
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits_v = logits[valid_mask]
        targets_v = targets[valid_mask]
        n = logits_v.shape[0]
        device = logits_v.device

        ## Probabilidades predichas (soft)
        probs = F.softmax(logits_v, dim=1)  ## [n, C]

        ## One-hot de las etiquetas verdaderas
        true_onehot = F.one_hot(targets_v, num_classes=self.num_classes).float()  ## [n, C]

        W = self.weight_matrix.to(device)  ## [C, C]

        ## Numerador: observado ponderado. Para cada muestra, el costo esperado
        ## segun la distancia a su clase verdadera: sum_j probs[:,j] * W[true, j]
        ## O_w = sum sobre muestras de (true_onehot @ W) . probs
        numerator = (true_onehot @ W * probs).sum()

        ## Denominador: esperado bajo independencia de los histogramas marginales
        hist_true = true_onehot.sum(dim=0)   ## [C]
        hist_pred = probs.sum(dim=0)         ## [C]
        expected = (hist_true.unsqueeze(1) * hist_pred.unsqueeze(0)) * W  ## [C, C]
        denominator = expected.sum() / n

        ## QWK suave = 1 - O_w / E_w ; loss = 1 - QWK = O_w / E_w
        ## (minimizar O_w/E_w equivale a maximizar el kappa)
        eps = 1e-7
        loss = numerator / (denominator + eps)
        return loss


class HybridOrdinalLoss(nn.Module):
    ## Loss hibrida: SORD + lambda * QWK
    ##
    ## Combina dos componentes complementarios:
    ##   - SORD: aporta la senal de clasificacion ordinal estable (etiquetas
    ##     suaves por distancia). Maneja el aprendizaje base.
    ##   - QWK: empuja directamente hacia el Quadratic Weighted Kappa.
    ##
    ## El peso lambda es pequeno (ej. 0.3) para que el QWK afine sin dominar ni
    ## causar la solucion degenerada de columna-cero bajo desbalance.
    ##
    ## Esta combinacion sigue la evidencia del hibrido MCE&WK (Litvinov et al.,
    ## J Clin Med 2026), que fue el mejor en test para clasificacion BI-RADS.

    def __init__(
        self,
        num_classes: int,
        lambda_qwk: float = 0.3,
        distance_power: float = 1.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = -100,
        undergrade_beta: float = 1.0,
    ):
        super().__init__()
        self.lambda_qwk = lambda_qwk
        self.sord = SORDLoss(
            num_classes=num_classes,
            distance_power=distance_power,
            class_weights=class_weights,
            ignore_index=ignore_index,
            undergrade_beta=undergrade_beta,
        )
        self.qwk = QWKLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sord_loss = self.sord(logits, targets)
        qwk_loss = self.qwk(logits, targets)
        return sord_loss + self.lambda_qwk * qwk_loss
