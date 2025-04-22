#!/usr/bin/env python
"""
ingest data from input folder, combine and save to an output folder
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse
import logging
# import wandb
import dagshub
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

mlflow.autolog()

class Ingest_Data():

    def __init__(self, args):
        self.in_folder = args.in_folder
        self.in_file = args.in_file
        self.out_folder = args.out_folder
        self.out_file = args.out_file

    def process_files(self) -> pd.DataFrame:

        if (self._in_file == "*"):
            files = [f for f in listdir(self.in_folder) if isfile(join(self.in_folder, f))]
            df = pd.DataFrame()
            for file in files:
                filename = join(self.in_folder, file)
                df_new = self.__read_file(filename)
                df = pd.concat([df, df_new], axis=1)
        else:
            filename = join(self.in_folder, self.in_file)
            df = self.read_file(filename)

        return df

    def __read_file(filename:str) -> pd.DataFrame:
        return pd.read_csv(filename)

def go(args):

    # run = wandb.init(job_type="data_ingestion")
    # run.config.update(args)
    
    dagshub.init(repo_owner='mdeevan', repo_name='dynamic-risk-assessment-system', mlflow=True)

    with mlflow.start_run():
        ingest_data = Ingest_Data(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ingest data from the input folder")


    parser.add_argument(
        "--in_path", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "-- in_filename", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "-- out_path", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "-- out_filename", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
