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
import subprocess
import timeit


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

from lib import utilities
from data_ingestion.ingestion import Ingest_Data
from training.training import Train_Model

class Diagnostics():

    def __init__(self):


        try:
            with open("./config/config.yaml") as stream:
                self.cfg = yaml.safe_load(stream)

        except Exception as err:
            self.cfg = None
            logging.error(f"FATAL: Error initialization configuration %s", err)


        self.parent_folder   = "./"
        self.model_name      = self.cfg["training"]["output_model_name"]
        self.model_path_name = self.cfg["prod_deployment"]["prod_deployment_path"]

        self.data_folder     = self.cfg['diagnostics']['data_folder']
        self.data_files      = self.cfg['diagnostics']['data_files']
        self.ingested_filename= self.cfg['ingestion']['ingested_filename']

        self.test_data_path  = self.cfg['scoring']['test_data_path']
        self.test_data_name  = "testdata.csv"
        self.num_features    = self.cfg['num_features']
        self.lr_params       = self.cfg['logistic_regression_params']

        outfile_path = self.cfg['training']['output_model_path']
        outfile_name = self.cfg['diagnostics']['apicallstxt_file']
        confusion_matrix_file = self.cfg['diagnostics']['confusion_matrix_file']

        self.outfile = utilities.get_filename(outfile_name,
                                              p_parent_folder="",
                                              p_path=outfile_path)

        self.confusion_matrix_file = utilities.get_filename(confusion_matrix_file,
                                                            p_parent_folder="",
                                                            p_path=outfile_path)


        try:
            self.model = utilities.load_model(p_model_file_name = self.model_name,
                                              p_parent_folder   = self.parent_folder,
                                              p_model_path_name = self.model_path_name)
        except Exception as err:
            self.model = None
            logging.error(f"Error loading Model %s", err)


        try:
            filename = utilities.get_filename(p_filename=self.test_data_name, 
                                              p_parent_folder=self.parent_folder,
                                              p_path=self.test_data_path)
            
            self.df = utilities.read_file(filename)

        except Exception as err:
            self.df = None
            logging.error(f"Error loading test df : %s", err)


    def find_null_values(self, p_data_path: str = "") -> str:
        # ---------------------------------
        
        if (p_data_path == ""):
            df = self.df
        else:
            df = utilities.read_file(p_data_path)


        null_values = df.isna().sum() 
        rtn = pd.DataFrame(null_values).T.to_json(index=False)

        return rtn
        
    def capture_statistics(self, p_data_path: str = "") -> str:
        # ---------------------------------
        
        # data_file = p_data_path if (p_data_path != "" ) else ( 
        #             utilities.get_filename(p_filename=self.test_data_name, 
        #                                    p_parent_folder=self.parent_folder,
        #                                    p_path=self.test_data_path)
        # )

        # df = utilities.read_file(data_file)

        if (p_data_path == ""):
            df = self.df
        else:
            df = utilities.read_file(p_data_path)

        agg_values = (df[self.num_features]
                      .agg((['mean','median','std']))
                      .T.reset_index()).to_json()
        
        rtn = agg_values
        # to_csv(outfile, index=False)

        return rtn
        
    def make_predictions(self, p_data_path: str = "") -> str:
        # ---------------------------------
        
        print("inside make predictions")
        # data_file = p_data_path if (p_data_path != "" ) else ( 
        #             utilities.get_filename(p_filename=self.test_data_name, 
        #                                    p_parent_folder=self.parent_folder,
        #                                    p_path=self.test_data_path)
        # )

        # df = utilities.read_file(data_file)

        if (p_data_path == ""):
            df = self.df
        else:
            df = utilities.read_file(p_data_path)



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
        
    def dependencies_status(self ):
        print("inside pip_outdated")

        command = ["pip", "list","--outdated", "--format", "json"]

        result = subprocess.run(command,  text=True, capture_output=True)#, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logger.info("Error running command %s", command)
        else:
            logger.info("\nSuccess running command %s", command)

        return result.stdout

                                # ingestion_path ,
                                # ingestion_filename ,
                                # out_path ,
                                # out_file ,
                                # ingested_files_log ,
                                # mlflow_logging ,
                                # parent_folder 

    def __timing_ingestion_command(self):

        ingest_data = Ingest_Data(
            p_ingestion_path     = self.data_folder,
            p_ingestion_filename = self.data_files,
            p_out_path = "temp",
            p_out_file = self.ingested_filename,
            p_ingested_files_log = "templog",
            p_mlflow_logging = False,
            p_parent_folder = "./"
        )

        ingest_data.process_files()

    def timing_ingestion(self, p_iterations=10):  

        logging.info("inside time_ingestion")

        t = timeit.Timer(self.__timing_ingestion_command )
        execution_time = t.timeit(p_iterations)
        logging.debug(f"INGESTION execution time with {p_iterations} iterations : {execution_time}")

        return execution_time

    def __timing_training_command(self):
        train_model = Train_Model(
            p_ingested_data_path = "temp", # from ingested timing method
            p_ingestion_filename = self.ingested_filename ,
            p_out_path           = "temp" ,
            p_out_model          = self.model_name ,
            p_parent_folder      = "./"  ,
            p_num_features       = self.num_features ,
            p_lr_params          = self.lr_params[0] ,
            p_mlflow_logging     = False ,
        )

        train_model.train_model()

    def timing_training(self, p_iterations=10):  

        logging.info("inside timing_training")

        t = timeit.Timer(self.__timing_training_command )
        execution_time = t.timeit(p_iterations)
        logging.debug(f"TRAINING execution time with {p_iterations} iterations : {execution_time}")

        return execution_time


if __name__ == '__main__':

    diagnostics = Diagnostics()

    nv = diagnostics.find_null_values()
    stat = diagnostics.capture_statistics()
    predict = diagnostics.make_predictions()
    result = diagnostics.dependencies_status()
    time_ingestion = diagnostics.timing_ingestion(10)
    train_ingestion = diagnostics.timing_training(10)

    diagnostics_list =['Null Values', 'Statistics', 'Prediction', 
                       'Dependencies', 'Time to Ingest data', 
                       'training ingested data']
    responses =[nv, stat, predict, result, time_ingestion, train_ingestion]

    # with open('apireturns_diagnostics.txt', "w") as f:
    with open(diagnostics.outfile, "w") as f:
        f.write("Diagnostics \n")
        for idx, response in enumerate(responses):
            f.write("\n ------------------------------------- \n")
            f.write(f"result from {diagnostics_list[idx]}  \n")

            f.write(str(response))
            f.write("\n")

    
    print("\n--------------\n null values \n")
    print(nv)
    print("\n--------------\n statistics \n")
    print(stat)
    print("\n--------------\n predication \n")
    print(predict)
    print("\n--------------\n dependencies \n")
    print(result )

    print("\n--------------\n ingestion timing \n")
    print(time_ingestion )

    print("\n--------------\n training timing \n")
    print(train_ingestion )


