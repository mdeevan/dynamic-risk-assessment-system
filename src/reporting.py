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
            with open("./config/config.yaml") as stream:
                self.cfg = yaml.safe_load(stream)

        except Exception as err:
            logging.error(f"Error initialization %s", err)

        self.diagnostic_instance = self.__get_diagnosic_instance()


    def __get_diagnosic_instance(self):
        parser = argparse.ArgumentParser(description="time ingestion")

        parser.add_argument("--model_path_name", type=str, 
                            default=self.cfg["prod_deployment"]["prod_deployment_path"])

        parser.add_argument("--model_file_name", type=str, 
                            default=self.cfg["prod_deployment"]["output_model_name"])

        parser.add_argument("--data_folder", type=str, 
                            default=self.cfg['scoring']['test_data_path'])

        parser.add_argument("--data_files", type=str, default="[*]")
        parser.add_argument("--ingested_file", type=str, default="[*]")

        parser.add_argument("--report_folder"    , type=str, default="temp")
        parser.add_argument("--prediction_output", type=str, default="temp_predict")
        # parser.add_argument("--score_filename"   , type=str, default=None)
        parser.add_argument("--timing_filename"  , type=str, default="temp_timing")
        parser.add_argument("--mlflow_logging"   , type=str, default=False)
        parser.add_argument("--temp_folder"      , type=str, default="temp")
        parser.add_argument("--num_features"     , type=str, 
                            default=str(self.cfg['num_features']))
        parser.add_argument("--lr_params"        , type=str, default=None)
        parser.add_argument("--parent_folder"    , type=str, default="./")


        args = parser.parse_args([]) # Pass an empty list for non-command-line usage

        diagnostic = Diagnostics(args)
        return diagnostic

    def generate_confusion_matrix(self):
        pred, _, _ = self.diagnostic_instance.run_diagnostics(['prediction'])

        predict_path = utilities.get_filename(p_filename     =self.diagnostic_instance.prediction_output,
                                              p_parent_folder=self.diagnostic_instance.parent_folder,
                                              p_path         =self.diagnostic_instance.report_folder
                                            )

        with open(predict_path, "r") as stream:
            predict_output = stream.read()

        print(f"predict_output : {predict_output}" )


        pass


###############Load config.json and get path variables


if __name__ == '__main__':

    reporting = Reporting()
    reporting.generate_confusion_matrix()

    


