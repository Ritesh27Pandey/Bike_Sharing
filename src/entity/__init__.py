from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    local_data_path: Path
    train_data_path: Path
    test_data_path: Path
    pickel_file_path: str
    train_arr_path: Path
    test_arr_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    pickel_file_path: str
    train_arr_path: str
    test_arr_path: str


@dataclass(frozen=True)
class ModelPredictionConfig:
    model_pickel_file_path: str
    preprocessor_pickel_file_path: str
