from src.components.data_transformation import DataTransformation
from src.config.configuration import DataTransformationConfig
from src.config.configuration import ConfigurationManager
# def data_transformation_pipe()   
#     data_trans_obj=DataTransformation(DataTransformationConfig)
#     data_trans_obj.initiate_data_transformation()


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()


    