from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    
@dataclass
class PrepareCallbackConfig:
    root_dir: Path
    checkpoint_dir: Path 
    tensorboard_dir: Path 
    
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    training_data : Path
    validation_data : Path
    test_data : Path
    
@dataclass
class EvaluationConfig:
    path_of_model: str
    test_data: str
    all_params: dict
    params_image_size: List[int]
    params_batch_size: int
