import os
from src.logger import logging
from src.entity import DataTransformationConfig
from src.config.configuration import ConfigurationManager
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder,StandardScaler, OneHotEncoder
import os
from src.utils.common import save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config

    # def convert_to_datetime(self):
    #     if "train" in str(self.config.train_data_path):
    #         train_data_path=self.config.train_data_path
    #         train_data_frame=pd.read_csv(train_data_path)
    #         train_data_frame["Date"]= pd.to_datetime(train_data_frame["Date"])
    #         return train_data_frame["Date"]
    #     else:
    #         test_data_path=self.config.test_data_path
    #         test_data_frame=pd.read_csv(test_data_path)
    #         test_data_frame["Date"]= pd.to_datetime(test_data_frame["Date"])
    #         return test_data_frame["Date"]

    def get_data_transformation_obj(self):
        categorical_col=["Seasons","Holiday","Functioning Day"]
        numerical_col=["Day","Month","Year","Hour","Temperature(캜)","Humidity(%)","Wind speed (m/s)","Visibility (10m)"]
        # date_col=["Date"]


        seasons_cat=['Winter', 'Spring', 'Summer', 'Autumn']

        holiday_cat=["Holiday","No Holiday"]

        functioning_cat=["Yes","No"]
        # date_cat=["Day","Month","Year",]
        # date column transformation
        
        # date_transformer=FunctionTransformer(self.convert_to_datetime(Date_col))

        # numerical transformation pipeline
        num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
        ) 
        # catogerical transformationn pipeline
        cat_pipeline = Pipeline(
        steps=[
            ("onehotencoder", OneHotEncoder(categories=[seasons_cat, holiday_cat, functioning_cat])),
            ("scaler", StandardScaler(with_mean=False))
        ]
        )
        # date_pipeline=Pipeline(
        #     steps=["date",date_transformer()
        #     ]
        # )

        # getting preprocessor object
        preprocessor=ColumnTransformer([
            ("num_pipeline",num_pipeline,numerical_col),
            ("cat_pipeline",cat_pipeline,categorical_col)],
            remainder='passthrough' 
            # ("date_transformation",date_pipeline)
        )

        logging.info("column transformation pipeline has started")
        return preprocessor

    def initiate_data_transformation(self):
        # reading dataset
        try:
            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

            logging.info(f'checking shape of data{train_df.shape,test_df.shape}')

            logging.info("read train and test data completes")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("obtaining preprocessor object")

            preprocessor_obj=self.get_data_transformation_obj()

            target_column_name="Rented Bike Count"
            # drop_columns=[target_column_name]
            # Day=train_df["Date"].day
            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## transforming using preprocessor object

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)
            logging.info("applied preprocessing object on train and test data")

            # concatnating train and test arr
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            np.savetxt(self.config.train_arr_path,train_arr,fmt="%d")
            np.savetxt(self.config.test_arr_path,test_arr,fmt="%d")

            save_object(
                path=self.config.pickel_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)




