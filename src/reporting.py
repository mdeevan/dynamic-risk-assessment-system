import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
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

from diagnostics.diagnostics import Diagnostics
from lib import utilities

class Reporting():

    def __init__(self):
        try:
            with open("../config/config.yaml") as stream:
                self.cfg = yaml.safe_load(stream)

        except Exception as err:
            logging.error(f"Error initialization %s", err)

        self.diagnostic_instance = self.__get_diagnosic_instance()


    def __get_diagnosic_instance(self):
        parser = argparse.ArgumentParser(description="time ingestion")

        parser.add_argument("--model_path_name", type=str, 
                            default=reporting.cfg["prod_deployment"]["prod_deployment_path"])

        parser.add_argument("--model_file_name", type=str, 
                            default=reporting.cfg["prod_deployment"]["prod_deployment_path"])

        parser.add_argument("--data_folder", type=str, 
                            default=reporting.cfg['scoring']['test_data_path'])

        parser.add_argument("--data_files", type=str, default=None)

        parser.add_argument("--report_folder", type=str, 
                            default=reporting.cfg['scoring']['report_folder'])

        parser.add_argument("--prediction_output", type=str, default=None)
        parser.add_argument("--score_filename"   , type=str, default=None)
        parser.add_argument("--timing_filename"  , type=str, default=None)
        parser.add_argument("--mlflow_logging"   , type=str, default=None)
        parser.add_argument("--temp_folder"      , type=str, default=None)
        parser.add_argument("--num_features"     , type=str, default=None)
        parser.add_argument("--lr_params"        , type=str, default=None)


        args = parser.parse_args([]) # Pass an empty list for non-command-line usage

        diagnostic = Diagnostics(args)
        return diagnostic

    def generate_confusion_matrix(self):
        pass


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    pass



if __name__ == '__main__':
    score_model()

    reporting = Reporting()

    


