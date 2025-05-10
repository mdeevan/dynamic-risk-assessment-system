import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import inspect
import sys
import yaml
import logging
import argparse


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

from lib import utilities

class Diagnostics():

    def __init__(self):


        try:
            with open("./config/config.yaml") as stream:
                self.cfg = yaml.safe_load(stream)

        except Exception as err:
            self.cfg = None
            logging.error(f"FATAL: Error initialization configuration %s", err)


        self.parent_folder   = "./"
        self.model_name      = self.cfg["prod_deployment"]["output_model_name"]
        self.model_path_name = self.cfg["prod_deployment"]["prod_deployment_path"]
        self.test_data_path  = self.cfg['scoring']['test_data_path']
        self.test_data_name  = "testdata.csv"
        self.num_features    = self.cfg['num_features']


        try:
            self.model = utilities.load_model(p_model_file_name = self.model_name,
                                              p_parent_folder   = self.parent_folder,
                                              p_model_path_name = self.model_path_name)
        except Exception as err:
            self.model = None
            logging.error(f"Error loading Model %s", err)

        # try:
        #     df = utilities.load_dataset(p_parent_folder = self.parent_folder,
        #                                 p_data_folder   = self.data_folder,
        #                                 p_data_files    = self.data_files)
        # except Exception as err:
        #     logging.error(f"Error loading data %s", err)



        # self.diagnostic_instance = self.__get_diagnosic_instance()


    def find_null_values(self, p_data_path: str = "") -> str:
        # ---------------------------------
        
        data_file = p_data_path if (p_data_path != "" ) else ( 
                    utilities.get_filename(p_filename=self.test_data_name, 
                                           p_parent_folder=self.parent_folder,
                                           p_path=self.test_data_path)
        )

        df = utilities.read_file(data_file)
        null_values = df.isna().sum() 
        rtn = pd.DataFrame(null_values).T.to_json(index=False)

        return rtn
        
    def capture_statistics(self, p_data_path: str = "") -> str:
        # ---------------------------------
        
        data_file = p_data_path if (p_data_path != "" ) else ( 
                    utilities.get_filename(p_filename=self.test_data_name, 
                                           p_parent_folder=self.parent_folder,
                                           p_path=self.test_data_path)
        )

        df = utilities.read_file(data_file)
        agg_values = (df[self.num_features]
                      .agg((['mean','median','std']))
                      .T.reset_index()).to_json()
        
        rtn = agg_values
        # to_csv(outfile, index=False)

        return rtn
        
    def make_predictions(self, p_data_path: str = "") -> str:
        # ---------------------------------
        
        data_file = p_data_path if (p_data_path != "" ) else ( 
                    utilities.get_filename(p_filename=self.test_data_name, 
                                           p_parent_folder=self.parent_folder,
                                           p_path=self.test_data_path)
        )

        df = utilities.read_file(data_file)
        rtn = None
        try:
            y_pred = None
            if df is not None:
                X = df[self.num_features]
                y = X.pop('exited')            

                y_pred = self.model.predict(X)

                rtn = pd.DataFrame(zip(y, y_pred.tolist()), 
                             columns=['target','predicted']
                             ).to_json()

        except Exception as err:
            logging.error(f"%s: diagnostic error making prediction %s", func_name, err)
            raise

        return rtn
        
    



###############Load config.json and get path variables


if __name__ == '__main__':

    diagnostics = Diagnostics()
    nv = diagnostics.find_null_values()
    stat = diagnostics.capture_statistics()
    predict = diagnostics.make_predictions()
    
    print(nv)
    print(stat)
    print(predict)


