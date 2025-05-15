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
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# sys.path.append("../")
from diagnostics import Diagnostics
from lib import utilities

class Reporting():

    def __init__(self):
        self.diagnostic_instance = Diagnostics()

    def generate_confusion_matrix(self):

        pred = self.diagnostic_instance.make_predictions()

        df = pd.read_json(StringIO(pred))

        # https://www.kaggle.com/code/agungor2/various-confusion-matrix-plots

        plt.figure(figsize = (6,4))

        data = confusion_matrix(df['target'], df['predicted'])

        df_cm = pd.DataFrame(data, 
                             columns=np.unique(df['predicted']), 
                             index = np.unique(df['target']))

        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 12})# font size
        plt.savefig(self.diagnostic_instance.confusion_matrix_file)

        

if __name__ == '__main__':

    reporting = Reporting()
    reporting.generate_confusion_matrix()

    


