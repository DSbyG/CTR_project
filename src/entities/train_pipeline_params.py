import yaml
import logging
import sys

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from src.entities.split_params import SplittingParams
from src.entities.feature_params import FeatureParams
from src.entities.train_params import TrainingParams

# Настройка логирования
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Путь к конфигурационному файлу
PATH = "configs/train_config.yaml"

# Определение дата-классов для конфигурации
@dataclass
class TrainingPipelineParams:
    output_model_path: str
    output_transformer_path: str
    output_ctr_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    input_data_path: str = field(default="data/raw/sampled_train_50k.csv")
    input_preprocessed_data_path: str = field(default="data/raw/sampled_preprocessed_train_50k.csv")
    use_mlflow: bool = False
    mlflow_experiment: str = "inference_demo"

# Создание схемы для валидации и десериализации
TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)

def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        logger.debug("Содержимое конфигурационного файла: %s", config_dict)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        logger.info("Схема успешно загружена: %s", schema)
        return schema

if __name__ == "__main__":
    try:
        params = read_training_pipeline_params(PATH)
        logger.debug(f"Параметры: {params}")
    except Exception as exc:
        logger.error(f"Ошибка при чтении параметров конвейера обучения: {exc}")