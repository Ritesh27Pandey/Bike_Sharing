import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils.common import load_object
import pandas as pd
from src.config.configuration import ConfigurationManager




class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            config = ConfigurationManager()
            prediction_config = config.prediction_config()
            preprocessor_file_path=prediction_config.preprocessor_pickel_file_path
            model_file_path=prediction_config.model_pickel_file_path
            preprocessor_obj=load_object(preprocessor_file_path)
            model_obj=load_object(model_file_path)

            data_scale=preprocessor_obj.transform(features)
            pred=model_obj.predict(data_scale)
            return pred
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
                 Hour:float,
                 Temperature:int,
                 Humidity:int,
                 Windspeed :float,
                 Visibility:int,
                 Seasons:str,
                 Holiday:str,
                 FunctioningDay:str,
                 Day:int,
                 Month:int,
                 Year:int
                 ):

        
        self.Hour=Hour
        self.Temperature=Temperature
        self.Humidity=Humidity
        self.Windspeed=Windspeed
        self.Visibility=Visibility
        self.Seasons=Seasons
        self.Holiday = Holiday
        self.FunctioningDay = FunctioningDay
        self.Day = Day
        self.Month=Month
        self.Year=Year


# Hour,Temperature(°C),Humidity(%),Wind speed (m/s),Visibility (10m),Seasons,Holiday,Functioning Day,Day,Month,Year

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Hour':[self.Hour],
                'Temperature(캜)':[self.Temperature],
                'Humidity(%)':[self.Humidity],
                'Wind speed (m/s)':[self.Windspeed],
                'Visibility (10m)':[self.Visibility],
                'Seasons':[self.Seasons],
                'Holiday':[self.Holiday],
                'Functioning Day':[self.FunctioningDay],
                'Day':[self.Day],
                'Month':[self.Month],
                'Year':[self.Year]   
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
