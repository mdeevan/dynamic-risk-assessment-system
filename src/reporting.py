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

sys.path.append("../")
from diagnostics import Diagnostics
from lib import utilities

class Reporting():

    def __init__(self):
        # try:
        #     with open("./config/config.yaml") as stream:
        #         self.cfg = yaml.safe_load(stream)

        # except Exception as err:
        #     logging.error(f"Error initialization %s", err)

        # self.diagnostic_instance = self.__get_diagnosic_instance()

        self.diagnostic_instance = Diagnostics()


    # def __get_diagnosic_instance(self):
    #     parser = argparse.ArgumentParser(description="time ingestion")

    #     parser.add_argument("--model_path_name", type=str, 
    #                         default=self.cfg["prod_deployment"]["prod_deployment_path"])

    #     parser.add_argument("--model_file_name", type=str, 
    #                         default=self.cfg["prod_deployment"]["output_model_name"])

    #     parser.add_argument("--data_folder", type=str, 
    #                         default=self.cfg['scoring']['test_data_path'])

    #     parser.add_argument("--data_files", type=str, default="[*]")
    #     parser.add_argument("--ingested_file", type=str, default="[*]")

    #     parser.add_argument("--report_folder"    , type=str, default="temp")
    #     parser.add_argument("--prediction_output", type=str, default="temp_predict")
    #     # parser.add_argument("--score_filename"   , type=str, default=None)
    #     parser.add_argument("--timing_filename"  , type=str, default="temp_timing")
    #     parser.add_argument("--mlflow_logging"   , type=str, default=False)
    #     parser.add_argument("--temp_folder"      , type=str, default="temp")
    #     parser.add_argument("--num_features"     , type=str, 
    #                         default=str(self.cfg['num_features']))
    #     parser.add_argument("--lr_params"        , type=str, default=None)
    #     parser.add_argument("--parent_folder"    , type=str, default="./")


    #     args = parser.parse_args([]) # Pass an empty list for non-command-line usage

    #     diagnostic = Diagnostics(args)
    #     return diagnostic

    def generate_confusion_matrix(self):
        # pred, _, _ = self.diagnostic_instance.run_diagnostics(['prediction'])

        # predict_path = utilities.get_filename(p_filename     =self.diagnostic_instance.prediction_output,
        #                                       p_parent_folder=self.diagnostic_instance.parent_folder,
        #                                       p_path         =self.diagnostic_instance.report_folder
        #                                     )

        # df = utilities.read_file(predict_path)

        pred = self.diagnostic_instance.make_predictions()

        df = pd.read_json(pred)

        # https://www.kaggle.com/code/agungor2/various-confusion-matrix-plots

        plt.figure(figsize = (6,4))

        data = confusion_matrix(df['target'], df['predicted'])

        df_cm = pd.DataFrame(data, 
                             columns=np.unique(df['predicted']), 
                             index = np.unique(df['target']))

        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 12})# font size
        plt.savefig("confusion_matrix.png")

        # out_path = utilities.get_filename(p_filename     ="confusionmatrix.png",
        #                                   p_parent_folder= ".\\.\\" ,
        #                                   p_path         =".\\"
        #                                     )

        # out_path = "./confusionmatrix.png"
        # print(f"outpath : {out_path}")
        # plt.savefig(out_path)
        



###############Load config.json and get path variables


if __name__ == '__main__':

    reporting = Reporting()
    reporting.generate_confusion_matrix()

    


