import os
from box.exceptions import BoxValueError
import yaml
from logger import logging
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score



"""
reads yaml file and returns
Args:
    path_to_yaml(str): path like input

Raise:
    ValueError: if file is empty
"""
@ensure_annotations  #help to ensure datatype mention in def correctly 
def read_yaml(path_to_yaml:Path)->ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            logging.info(f"yaml file :{path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('yaml file is empty')
    except Exception as e:
        raise (e,sys)
    


@ensure_annotations
def create_directories(path_to_directories: list,verbos=True):
    # creating list of directories if dir already exists then it will cancelled
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbos:
            logging.info(f"created directory at: {path}")




@ensure_annotations
def get_size(path: Path)-> str:
    size_in_kb=round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} kb"


@ensure_annotations
def save_object(path: str, obj):
    try:
        dir_path = os.path.dirname(path)

        os.makedirs(dir_path, exist_ok=True)

        with open(path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

@ensure_annotations
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report:dict = {}
        # report_confusion_matrix:dict={}
        for i in range(len(models)):
            model = list(models.values())[i]
            # train models
            model.fit(X_train,y_train)

            # pridction testing data
            y_test_pred=model.predict(X_test)

            # get confusion matrix and precision, recall, f score 
            test_model_score=r2_score(y_test,y_test_pred)
            logging.info(f'model name{list(models.keys())[i]} having score {test_model_score}')
            

            report[list(models.keys())[i]]=test_model_score
            return report
    except Exception as e:
        raise CustomException(e,sys)
    

    
@ensure_annotations
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
        

