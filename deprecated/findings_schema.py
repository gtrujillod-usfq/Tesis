## findings_schema.py
## Esquema de hallazgos radiologicos alineado al lexico ACR BI-RADS
## Area 2: Razonamiento Multimodal
##
## Define la estructura de slots de hallazgos que el modelo debe predecir.
## Cada slot corresponde a una categoria del lexico BI-RADS (mass_shape,
## mass_margin, calc_morphology, etc.) y tiene un conjunto cerrado de valores
## posibles mas un valor "ausente" cuando el hallazgo no esta presente.
##
## Esta alineacion garantiza coherencia entre Area 2 (prediccion de hallazgos)
## y Area 4 (evaluacion contra reportes CDD-CESM).

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FindingSlot:
    ## Un slot de hallazgo con sus valores posibles
    ## name: nombre del slot (corresponde a categoria BI-RADS)
    ## values: lista de valores posibles (el indice 0 siempre es "ausente")
    ## birads_category: categoria del lexico BI-RADS a la que mapea
    name: str
    values: List[str]
    birads_category: str
    description: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.values)

    def value_to_index(self, value: str) -> int:
        ## Convierte un valor a su indice de clase
        value_lower = value.lower().strip()
        for i, v in enumerate(self.values):
            if v.lower() == value_lower:
                return i
        return 0  ## ausente por defecto

    def index_to_value(self, index: int) -> str:
        ## Convierte indice de clase a valor
        if 0 <= index < len(self.values):
            return self.values[index]
        return self.values[0]


class FindingsSchema:
    ## Esquema completo de hallazgos para mamografia alineado a BI-RADS
    ##
    ## El modelo predice un valor para cada slot. Los slots con valor
    ## "ausente" indican que ese hallazgo no esta presente en la imagen.
    ##
    ## Esta es una formulacion multi-task: cada slot es una sub-tarea de
    ## clasificacion multiclase independiente, todas compartiendo el mismo
    ## encoder visual (BiomedCLIP).

    def __init__(self):
        self._build_schema()

    def _build_schema(self):
        ## Define todos los slots de hallazgos

        self.slots: List[FindingSlot] = [
            ## Composicion mamaria (siempre presente, 4 categorias ACR)
            FindingSlot(
                name="breast_density",
                values=[
                    "almost entirely fatty",
                    "scattered fibroglandular",
                    "heterogeneously dense",
                    "extremely dense",
                ],
                birads_category="density",
                description="Composicion del tejido mamario (ACR a-d)",
            ),

            ## Presencia y forma de masa
            FindingSlot(
                name="mass_shape",
                values=["ausente", "oval", "round", "irregular"],
                birads_category="mass_shape",
                description="Forma de masa si esta presente",
            ),

            ## Margenes de masa
            FindingSlot(
                name="mass_margin",
                values=[
                    "ausente",
                    "circumscribed",
                    "obscured",
                    "microlobulated",
                    "indistinct",
                    "spiculated",
                ],
                birads_category="mass_margin",
                description="Margenes de masa si esta presente",
            ),

            ## Densidad de masa
            FindingSlot(
                name="mass_density",
                values=[
                    "ausente",
                    "high density",
                    "equal density",
                    "low density",
                    "fat-containing",
                ],
                birads_category="mass_density",
                description="Densidad radiologica de la masa",
            ),

            ## Morfologia de calcificaciones
            FindingSlot(
                name="calc_morphology",
                values=[
                    "ausente",
                    "amorphous",
                    "coarse heterogeneous",
                    "fine pleomorphic",
                    "fine linear",
                    "round calcifications",
                    "vascular calcifications",
                    "coarse popcorn",
                ],
                birads_category="calc_morphology",
                description="Morfologia de calcificaciones si estan presentes",
            ),

            ## Distribucion de calcificaciones
            FindingSlot(
                name="calc_distribution",
                values=[
                    "ausente",
                    "diffuse",
                    "regional",
                    "grouped",
                    "linear",
                    "segmental",
                ],
                birads_category="calc_distribution",
                description="Distribucion de calcificaciones",
            ),

            ## Asimetria
            FindingSlot(
                name="asymmetry",
                values=[
                    "ausente",
                    "asymmetry",
                    "focal asymmetry",
                    "global asymmetry",
                ],
                birads_category="asymmetry",
                description="Tipo de asimetria si esta presente",
            ),

            ## Distorsion arquitectural
            FindingSlot(
                name="architectural_distortion",
                values=["ausente", "architectural distortion"],
                birads_category="distortion",
                description="Presencia de distorsion arquitectural",
            ),
        ]

        ## Indice rapido por nombre
        self.slots_by_name: Dict[str, FindingSlot] = {
            slot.name: slot for slot in self.slots
        }

    def get_slot(self, name: str) -> Optional[FindingSlot]:
        return self.slots_by_name.get(name)

    def get_slot_names(self) -> List[str]:
        return [slot.name for slot in self.slots]

    def get_total_heads(self) -> int:
        ## Numero total de heads de clasificacion (uno por slot)
        return len(self.slots)

    def get_head_dimensions(self) -> Dict[str, int]:
        ## Diccionario nombre_slot -> numero de clases
        ## Define la arquitectura de los heads del modelo
        return {slot.name: slot.num_classes for slot in self.slots}

    def decode_predictions(self, predictions: Dict[str, int]) -> Dict[str, str]:
        ##
        ## Convierte predicciones (indices) a valores legibles
        ## Filtra los slots con valor "ausente"
        ##
        ## Parametros:
        ##   predictions: dict nombre_slot -> indice de clase predicho
        ##
        ## Retorna: dict nombre_slot -> valor (solo hallazgos presentes)
        ##
        decoded = {}
        for slot_name, pred_idx in predictions.items():
            slot = self.get_slot(slot_name)
            if slot is None:
                continue
            value = slot.index_to_value(pred_idx)
            ## Incluir density siempre, otros solo si no son "ausente"
            if slot_name == "breast_density" or value != "ausente":
                decoded[slot_name] = value
        return decoded

    def findings_to_text(self, decoded: Dict[str, str], language: str = "en") -> str:
        ##
        ## Convierte hallazgos decodificados a descripcion textual
        ## que sirve como query para RAG o entrada para el LLM
        ##
        parts = []

        if "breast_density" in decoded:
            parts.append(f"breast composition {decoded['breast_density']}")

        if "mass_shape" in decoded or "mass_margin" in decoded:
            mass_desc = "mass"
            if "mass_shape" in decoded:
                mass_desc = f"{decoded['mass_shape']} {mass_desc}"
            if "mass_margin" in decoded:
                mass_desc = f"{mass_desc} with {decoded['mass_margin']} margins"
            if "mass_density" in decoded:
                mass_desc = f"{mass_desc} {decoded['mass_density']}"
            parts.append(mass_desc)

        if "calc_morphology" in decoded:
            calc_desc = decoded["calc_morphology"]
            if "calc_distribution" in decoded:
                calc_desc = f"{calc_desc} {decoded['calc_distribution']} distribution"
            parts.append(calc_desc)

        if "asymmetry" in decoded:
            parts.append(decoded["asymmetry"])

        if "architectural_distortion" in decoded:
            parts.append(decoded["architectural_distortion"])

        return ", ".join(parts) if parts else "no significant findings"

    def get_findings_list(self, decoded: Dict[str, str]) -> List[str]:
        ##
        ## Retorna lista plana de hallazgos presentes (para metricas)
        ## Excluye breast_density que siempre esta presente
        ##
        findings = []
        for slot_name, value in decoded.items():
            if slot_name == "breast_density":
                continue
            if value != "ausente":
                findings.append(value)
        return findings


## Instancia global del esquema
findings_schema = FindingsSchema()
