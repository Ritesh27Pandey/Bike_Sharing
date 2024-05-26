from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.logger import logging
from src.pipeline.training_pipeline import ModelTrainerTrainingPipeline
from src.pipeline.perdiction_pipeline import  PredictionPipeline,CustomData





STAGE_NAME="Data Ingestion Stage"
try:
    logging.info(f">> stage {STAGE_NAME} started")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f'>> stage {STAGE_NAME} completed')
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME="Data Transformation Stage"
try:
    logging.info(f">> stage {STAGE_NAME} started")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logging.info(f'>> stage {STAGE_NAME} completed')
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME="Model Trainer Stage"
try:
    logging.info(f">> stage {STAGE_NAME} started")
    data_transformation = ModelTrainerTrainingPipeline()
    data_transformation.main()
    logging.info(f'>> stage {STAGE_NAME} completed')
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME="Prediction pipeline"
try:
    logging.info(f">> stage {STAGE_NAME} started")
    model_prediction = PredictionPipeline()
    custom_data_obj=CustomData(12,32,32,32,32,"Summer","Holiday","Yes", 2,3,2024)
    custom_data=custom_data_obj.get_data_as_dataframe()
    prediction=int(model_prediction.predict(custom_data))
    print(prediction)
    logging.info(f'>> stage {STAGE_NAME} completed')
    logging.info(f"predicting result {prediction}")
except Exception as e:
    logging.exception(e)
    raise e