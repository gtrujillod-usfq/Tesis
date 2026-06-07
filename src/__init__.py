## __init__.py
## Modulo src del proyecto MammoVLM V2

from .config import MammoVLMConfig, DatasetConfig, ModelConfig, TrainingConfig, EvaluationConfig, config
from .utils import (
    setup_logging, save_json, load_json, get_timestamp,
    count_parameters, get_device, normalize_birads,
    birads_to_risk_category, AverageMeter, print_separator, print_section,
)
from .coherence_metrics import (
    PredictionRecord, CoherenceValidator, CalibrationAnalyzer,
    ConsistencyAnalyzer, AlignmentMetricsCollector,
)
from .medical_vocabulary import BIRADSDescriptor, BIRADSLexicon, birads_lexicon
from .rag import (
    TextChunk, PDFExtractor, EmbeddingModel, CorpusIndexer,
    ReportRetriever, augment_prompt, create_rag_pipeline,
)
## Modelo (exp08): encoder Mammo-CLIP (EfficientNet-B5) + dual-head BI-RADS/densidad
from .models import (
    MammoCLIPEncoder, ClassificationHead, MammoVLM, MultiTaskLoss, FocalLoss,
)
from .report_generator import (
    PromptBuilder, ReportGenerator, load_llm_for_generation,
)
from .medical_metrics import (
    BIRADSClassificationMetrics, ClinicalSeverityMetrics,
    InterRaterReliability, ClinicalErrorAnalysis, MedicalMetricsReport,
)
from .label_normalization import (
    normalize_birads, normalize_birads_list,
    generate_remap_summary, get_clinical_explanation,
    BIRADS_MIN, BIRADS_MAX, BIRADS_REMAP,
)
## Carga de datos (exp06): solo VinDr, alta resolucion
from .data_loading import (
    MammoCLIPTransform, load_image_as_pil, MammoDataset, apply_clahe,
    load_vindr_records, birads_to_index, index_to_birads, density_to_index,
)
from .train import (
    TrainingConfig, train_mammovlm,
    compute_birads_class_weights, build_weighted_sampler, split_train_val,
    train_one_epoch, evaluate, save_checkpoint,
    separate_test_set_vindr, load_checkpoint_if_exists, subsample_stratified,
)
from .experiment_tracker import ExperimentTracker

__version__ = "2.0.0"
__author__ = "Geovanny Enrique Trujillo Delgado"
