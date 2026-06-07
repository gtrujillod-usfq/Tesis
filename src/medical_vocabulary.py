## medical_vocabulary.py
## Vocabulario medico controlado basado en ACR BI-RADS Atlas 5ta edicion
## y actualizaciones BI-RADS v2025
##
## Este modulo define la taxonomia oficial de descriptores mamograficos
## para validar que las explicaciones del VLM usen terminologia clinica
## correcta. Se usa como referencia en las 4 areas del proyecto.

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import re


@dataclass
class BIRADSDescriptor:
    ## Descriptor individual del lexico BI-RADS
    canonical: str
    synonyms: List[str] = field(default_factory=list)
    category: str = ""
    risk_level: str = ""

    def matches(self, text: str) -> bool:
        ## Verifica si el descriptor aparece en un texto dado
        text_lower = text.lower()
        if self.canonical.lower() in text_lower:
            return True
        for syn in self.synonyms:
            if syn.lower() in text_lower:
                return True
        return False


class BIRADSLexicon:
    ## Lexico completo BI-RADS para mamografia
    ## Basado en ACR BI-RADS Atlas 5ta edicion y v2025
    
    def __init__(self):
        self._build_lexicon()
    
    def _build_lexicon(self):
        ## Construye todas las categorias del lexico BI-RADS
        
        ## Forma de masa
        self.mass_shape = [
            BIRADSDescriptor("oval", ["ovoid", "elliptical"], "mass_shape", "benign"),
            BIRADSDescriptor("round", ["circular", "spherical"], "mass_shape", "benign"),
            BIRADSDescriptor("irregular", ["lobulated", "angular"], "mass_shape", "suspicious"),
        ]
        
        ## Margenes de masa
        self.mass_margin = [
            BIRADSDescriptor("circumscribed", ["well-defined", "sharply defined", "well defined"], "mass_margin", "benign"),
            BIRADSDescriptor("obscured", ["partially obscured", "partially hidden"], "mass_margin", "indeterminate"),
            BIRADSDescriptor("microlobulated", ["micro-lobulated", "microlobular"], "mass_margin", "suspicious"),
            BIRADSDescriptor("indistinct", ["ill-defined", "poorly defined", "ill defined"], "mass_margin", "suspicious"),
            BIRADSDescriptor("spiculated", ["spiculate", "stellate", "star-shaped"], "mass_margin", "malignant"),
        ]
        
        ## Densidad de masa
        self.mass_density = [
            BIRADSDescriptor("high density", ["hyperdense", "dense mass"], "mass_density", "suspicious"),
            BIRADSDescriptor("equal density", ["isodense", "iso-dense"], "mass_density", "indeterminate"),
            BIRADSDescriptor("low density", ["hypodense", "low-density"], "mass_density", "benign"),
            BIRADSDescriptor("fat-containing", ["fat density", "radiolucent", "lipoma"], "mass_density", "benign"),
        ]
        
        ## Morfologia de calcificaciones (sospechosas)
        self.calc_morphology_suspicious = [
            BIRADSDescriptor("amorphous", ["amorphous calcifications", "indistinct calcifications"], "calc_morphology", "suspicious"),
            BIRADSDescriptor("coarse heterogeneous", ["coarse irregular", "heterogeneous coarse"], "calc_morphology", "suspicious"),
            BIRADSDescriptor("fine pleomorphic", ["pleomorphic calcifications", "fine irregular"], "calc_morphology", "malignant"),
            BIRADSDescriptor("fine linear", ["fine linear branching", "casting calcifications", "linear branching"], "calc_morphology", "malignant"),
        ]
        
        ## Morfologia de calcificaciones (tipicamente benignas)
        self.calc_morphology_benign = [
            BIRADSDescriptor("skin calcifications", ["dermal", "cutaneous calcifications"], "calc_morphology", "benign"),
            BIRADSDescriptor("vascular calcifications", ["arterial calcifications", "vessel wall"], "calc_morphology", "benign"),
            BIRADSDescriptor("coarse popcorn", ["popcorn calcifications", "fibroadenoma calcifications"], "calc_morphology", "benign"),
            BIRADSDescriptor("large rod-like", ["secretory calcifications", "rod-like"], "calc_morphology", "benign"),
            BIRADSDescriptor("round calcifications", ["punctate calcifications", "spherical calcifications"], "calc_morphology", "benign"),
            BIRADSDescriptor("rim calcifications", ["eggshell", "lucent-centered"], "calc_morphology", "benign"),
            BIRADSDescriptor("dystrophic", ["dystrophic calcifications"], "calc_morphology", "benign"),
            BIRADSDescriptor("milk of calcium", ["sedimented calcifications", "teacup"], "calc_morphology", "benign"),
            BIRADSDescriptor("suture calcifications", ["suture"], "calc_morphology", "benign"),
        ]
        
        ## Distribucion de calcificaciones
        self.calc_distribution = [
            BIRADSDescriptor("diffuse", ["scattered", "bilateral diffuse"], "calc_distribution", "benign"),
            BIRADSDescriptor("regional", ["regional distribution", "large area"], "calc_distribution", "indeterminate"),
            BIRADSDescriptor("grouped", ["clustered", "cluster of calcifications"], "calc_distribution", "suspicious"),
            BIRADSDescriptor("linear", ["linear distribution", "line pattern"], "calc_distribution", "malignant"),
            BIRADSDescriptor("segmental", ["segmental distribution", "ductal distribution"], "calc_distribution", "malignant"),
        ]
        
        ## Tipos de asimetria
        self.asymmetry = [
            BIRADSDescriptor("asymmetry", ["one-view asymmetry", "single view"], "asymmetry", "indeterminate"),
            BIRADSDescriptor("focal asymmetry", ["focal", "localized asymmetry"], "asymmetry", "indeterminate"),
            BIRADSDescriptor("global asymmetry", ["large area asymmetry", "generalized"], "asymmetry", "benign"),
        ]
        
        ## Distorsion arquitectural
        self.architectural_distortion = [
            BIRADSDescriptor("architectural distortion", ["distortion", "parenchymal distortion", "stromal distortion"], "distortion", "suspicious"),
        ]
        
        ## Hallazgos asociados
        self.associated_features = [
            BIRADSDescriptor("skin retraction", ["retraccion cutanea", "skin dimpling"], "associated", "suspicious"),
            BIRADSDescriptor("nipple retraction", ["retraccion del pezon", "inverted nipple"], "associated", "suspicious"),
            BIRADSDescriptor("skin thickening", ["engrosamiento cutaneo", "thickened skin"], "associated", "suspicious"),
            BIRADSDescriptor("trabecular thickening", ["trabecular", "Cooper ligament thickening"], "associated", "suspicious"),
            BIRADSDescriptor("axillary adenopathy", ["axillary lymphadenopathy", "enlarged axillary nodes"], "associated", "suspicious"),
        ]
        
        ## Composicion mamaria (densidad)
        self.breast_density = [
            BIRADSDescriptor("almost entirely fatty", ["fatty", "density a", "ACR a"], "density", "low_risk"),
            BIRADSDescriptor("scattered fibroglandular", ["scattered", "density b", "ACR b"], "density", "low_risk"),
            BIRADSDescriptor("heterogeneously dense", ["heterogeneous", "density c", "ACR c"], "density", "high_risk"),
            BIRADSDescriptor("extremely dense", ["very dense", "density d", "ACR d"], "density", "high_risk"),
        ]
        
        ## Categorias de evaluacion BI-RADS
        self.assessment_categories = {
            0: {
                "name": "Incomplete",
                "description": "Need additional imaging evaluation",
                "recommendation": "Additional imaging needed",
                "keywords_en": ["incomplete", "additional imaging", "recall", "further evaluation"],
                "keywords_es": ["incompleto", "evaluacion adicional", "imagenes adicionales", "requiere"],
            },
            1: {
                "name": "Negative",
                "description": "No significant abnormality",
                "recommendation": "Routine screening",
                "keywords_en": ["negative", "normal", "no findings", "unremarkable"],
                "keywords_es": ["negativo", "normal", "sin hallazgos", "sin anomalias"],
            },
            2: {
                "name": "Benign",
                "description": "Definitely benign finding",
                "recommendation": "Routine screening",
                "keywords_en": ["benign", "definitely benign", "non-malignant", "cyst", "fibroadenoma"],
                "keywords_es": ["benigno", "definitivamente benigno", "no maligno", "quiste", "fibroadenoma"],
            },
            3: {
                "name": "Probably Benign",
                "description": "Finding with very low probability of malignancy",
                "recommendation": "Short-interval follow-up (6 months)",
                "keywords_en": ["probably benign", "low suspicion", "short interval", "follow-up"],
                "keywords_es": ["probablemente benigno", "baja sospecha", "seguimiento", "corto plazo"],
            },
            4: {
                "name": "Suspicious",
                "description": "Finding with moderate to high probability of malignancy",
                "recommendation": "Tissue diagnosis (biopsy)",
                "keywords_en": ["suspicious", "biopsy", "tissue diagnosis", "moderate suspicion"],
                "keywords_es": ["sospechoso", "biopsia", "diagnostico tisular", "sospecha moderada"],
            },
            5: {
                "name": "Highly Suggestive of Malignancy",
                "description": "Finding with very high probability of malignancy",
                "recommendation": "Immediate biopsy and appropriate action",
                "keywords_en": ["highly suggestive", "malignancy", "high probability", "classic malignant"],
                "keywords_es": ["altamente sugestivo", "malignidad", "alta probabilidad", "maligno"],
            },
        }
    
    def get_all_descriptors(self) -> List[BIRADSDescriptor]:
        ## Retorna todos los descriptores del lexico como lista plana
        all_descriptors = []
        for attr_name in [
            "mass_shape", "mass_margin", "mass_density",
            "calc_morphology_suspicious", "calc_morphology_benign",
            "calc_distribution", "asymmetry",
            "architectural_distortion", "associated_features",
            "breast_density"
        ]:
            all_descriptors.extend(getattr(self, attr_name))
        return all_descriptors
    
    def get_descriptors_by_category(self, category: str) -> List[BIRADSDescriptor]:
        ## Retorna descriptores filtrados por categoria
        return [d for d in self.get_all_descriptors() if d.category == category]
    
    def get_keywords_for_birads(self, birads_level: int, language: str = "es") -> List[str]:
        ##
        ## Retorna palabras clave para un nivel BI-RADS dado
        ## language: "en" para ingles, "es" para espanol
        ##
        category = self.assessment_categories.get(birads_level, {})
        key = f"keywords_{language}"
        return category.get(key, [])
    
    def validate_descriptors_in_text(self, text: str) -> Dict[str, List[str]]:
        ##
        ## Busca todos los descriptores BI-RADS mencionados en un texto
        ## Retorna diccionario con categorias y descriptores encontrados
        ##
        found = {}
        for descriptor in self.get_all_descriptors():
            if descriptor.matches(text):
                category = descriptor.category
                if category not in found:
                    found[category] = []
                found[category].append(descriptor.canonical)
        return found
    
    def get_recommendation_for_birads(self, birads_level: int) -> str:
        ## Retorna la recomendacion clinica oficial para un nivel BI-RADS
        category = self.assessment_categories.get(birads_level, {})
        return category.get("recommendation", "Unknown")
    
    def compute_terminology_adherence(self, text: str) -> float:
        ##
        ## Calcula el porcentaje de adherencia terminologica (metrica CRUDA)
        ## Es decir, cuantos terminos usados en el texto corresponden
        ## a descriptores oficiales del lexico BI-RADS
        ##
        ## NOTA: cuenta TODOS los descriptores, incluidas morfologias (forma de
        ## masa, calcificaciones). Para un modelo que solo predice BI-RADS +
        ## densidad, esta metrica sera baja por diseno (no genera morfologias).
        ## Para una evaluacion justa de ese modelo, ver
        ## compute_terminology_adherence_pertinent.
        ##
        found_descriptors = self.validate_descriptors_in_text(text)
        total_found = sum(len(v) for v in found_descriptors.values())
        
        ## Heuristica: un buen informe deberia mencionar al menos
        ## 3 descriptores BI-RADS (densidad, hallazgo, recomendacion)
        if total_found == 0:
            return 0.0
        elif total_found >= 5:
            return 1.0
        else:
            return total_found / 5.0

    def compute_terminology_adherence_pertinent(self, text: str, birads_level: int) -> float:
        ##
        ## Adherencia terminologica AJUSTADA a lo que el modelo puede producir
        ##
        ## Evalua solo los elementos terminologicos pertinentes para un modelo
        ## que predice BI-RADS + densidad (no morfologias). Son tres componentes,
        ## cada uno aporta 1/3:
        ##   1. Densidad ACR: el texto menciona una categoria de composicion
        ##      mamaria valida (fatty / scattered / heterogeneously dense /
        ##      extremely dense, o su equivalente en espanol).
        ##   2. Categoria BI-RADS: el texto menciona explicitamente la categoria.
        ##   3. Recomendacion coherente: el texto incluye la conducta clinica
        ##      acorde al nivel BI-RADS.
        ##
        ## A diferencia de la metrica cruda, NO penaliza la ausencia de
        ## descriptores morfologicos (que el modelo no predice por diseno).
        ##
        text_lower = text.lower()
        score = 0.0

        ## 1. Densidad ACR (busca los descriptores de breast_density del lexico)
        density_found = any(d.matches(text) for d in self.breast_density)
        ## Tambien aceptar la forma "ACR A/B/C/D" o "densidad ..."
        if not density_found:
            density_found = bool(re.search(r"acr\s*[a-d]|densidad", text_lower))
        if density_found:
            score += 1.0 / 3.0

        ## 2. Categoria BI-RADS mencionada explicitamente
        if re.search(r"bi[\s\-]?rads", text_lower):
            score += 1.0 / 3.0

        ## 3. Recomendacion coherente con el nivel
        rec_coherence = self.validate_recommendation_coherence(birads_level, text)
        score += (1.0 / 3.0) * rec_coherence

        return score
    
    def validate_recommendation_coherence(self, birads_pred: int, text: str) -> float:
        ##
        ## Valida que la recomendacion en el texto sea coherente
        ## con el nivel BI-RADS predicho
        ##
        text_lower = text.lower()
        
        ## Definir recomendaciones esperadas por nivel
        expected_actions = {
            0: ["adicional", "complementar", "additional", "recall"],
            1: ["rutina", "screening", "control", "anual", "routine"],
            2: ["rutina", "screening", "control", "anual", "routine"],
            3: ["seguimiento", "corto plazo", "6 meses", "follow-up", "short interval"],
            4: ["biopsia", "diagnostico tisular", "biopsy", "tissue"],
            5: ["biopsia", "inmediata", "urgente", "biopsy", "immediate", "urgent"],
        }
        
        actions = expected_actions.get(birads_pred, [])
        if not actions:
            return 0.5
        
        found = sum(1 for action in actions if action in text_lower)
        return min(1.0, found / max(1, len(actions) * 0.3))


## Instancia global del lexico
birads_lexicon = BIRADSLexicon()
