# basic import
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from src.utils.common import save_object
from src.exception import CustomException
# from sklearn.metrics import mean_absolute_error


from src.logger import logging
from dataclasses import dataclass
from src.utils.common import evaluate_model
import sys
import os

from src.config.configuration import ConfigurationManager
from src.entity import *

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config

    def initiate_model_training(self):
        try:    
            logging.info('model training has started')

            train_arr=np.loadtxt(self.config.train_arr_path, dtype="int")
            test_arr=np.loadtxt(self.config.test_arr_path, dtype="int")

            logging.info(f"train data shape and test data shape{train_arr.shape},{test_arr.shape}")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                # logging.info(f'train_arr{X_train.shape}'),
                train_arr[:,-1],
                # logging.info(f'train_arr{y_train.shape}'),
                test_arr[:,:-1],
                # logging.info(f'train_arr{X_test.shape}'),
                test_arr[:,-1]
                # logging.info(f'train_arr{y_test.shape}'),
            )
            models={
                # "LogisticRegression":LogisticRegression(),
                # "DecisionTreeRegressor":DecisionTreeRegressor(),
                # "RandomForestRegressor":RandomForestRegressor(),
                "Xgboost": GradientBoostingRegressor()
            }
            logging.info(f'X_train_arr{X_train.shape}')
            logging.info(f'y_train_arr{y_train.shape}')
            logging.info(f'X_test_arr{X_test.shape}')
            logging.info(f'y_test_arr{y_test.shape}')


            model_report=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            # print(confusion_matrix)
            print("\n==========================================================================")
            logging.info(f'model report :{model_report}')
            # logging.info(f'confusion matrix :{confusion_matrix}')

            # to get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            print(f'best model name :{best_model_name} and model r2 score is :{best_model_score}')
            print("\n==========================================================================\n")
            logging.info(f'best model name :{best_model_name} and model accuracy is :{best_model_score}')
            

            best_model=models[best_model_name]

            save_object(
                path= self.config.pickel_file_path,
                obj=best_model

            )
        except Exception as e:
            logging.info("error occured at model training")
            raise CustomException(e,sys)

