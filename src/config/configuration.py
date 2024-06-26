from pathlib import Path
from src.utils.common import *
from src.constants  import *
from src.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelPredictionConfig




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH
        ):
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_files,
            train_data_path = config.train_data_path,
            test_data_path =config.test_data_path
        )

        return data_ingestion_config
    

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            local_data_path=config.local_data_path,
            train_data_path=config.train_data_path,
            test_data_path = config.test_data_path,
            pickel_file_path=config.pickel_file_path,
            train_arr_path= config.train_arr_path,
            test_arr_path = config.test_arr_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config=self.config.model_trainer
        create_directories([config.root_dir])

        model_training_config=ModelTrainerConfig(
            root_dir=config.root_dir,
            pickel_file_path=config.pickel_file_path,
            train_arr_path= config.train_arr_path,
            test_arr_path = config.test_arr_path
        )

        return model_training_config
    
    def prediction_config(self) -> ModelPredictionConfig:
        config=self.config.predicting
        model_prediction_config= ModelPredictionConfig(
            preprocessor_pickel_file_path=config.preprocessor_pickel_file_path,
            model_pickel_file_path=config.model_pickel_file_path
        )
        return model_prediction_config