import os
import urllib.request as request
from src.logger import logging
from src.entity import DataIngestionConfig
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from src.exception import CustomException


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename,header=request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file 
            )
            logging.info(f"{filename} download! with following info: \n{header}")
        else:
            logging.info(f"file already exists of size: {Path(self.config.local_data_file)}")


    def train_test_split_data(self):
        try:
            logging.info("train test split has started in data ingestion")
            raw_data=self.config.local_data_file
            raw_data=pd.read_csv(raw_data)
            train_set,test_set=train_test_split(raw_data,test_size=0.30,random_state=42)

            train_set.to_csv(self.config.train_data_path, index= False)
            test_set.to_csv(self.config.test_data_path, index= False)

            logging.info('data split done successfull')
        except Exception as e:
            raise CustomException (e,sys)




