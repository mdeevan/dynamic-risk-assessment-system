#!/usr/bin/env python
"""
ingest data from input folder, combine and save to an output folder
"""
import os
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

# mlflow.autolog()

class Ingest_Data():

    def __init__(self, args):
        self.in_path = args.in_path
        self.in_file = args.in_file
        self.out_path = args.out_path
        self.out_file = args.out_file

    def process_files(self) -> int:

        try:
            if (self.in_file == "*"):
                print(f"self.infile {self.in_file}")
                files = [f for f in listdir(self.in_path) if isfile(join(self.in_path, f))]
                df = pd.DataFrame()
                for file in files:
                    filename = join(self.in_path, file)

                    print(f"filename : {filename}")
                    df_new = self.__read_file(filename)
                    df = pd.concat([df, df_new], axis=1)
            else:
                filename = join(self.in_path, self.in_file)
                df = self.read_file(filename)
        except Exception as err:
            print("error reading file %s", err)


        out = join(self.out_path, self.out_file)
        try:
            df.to_csv(out)
        except Exception as err:
            print("error %s in creating outfile %s", err, out)
            raise

        return True

    def __read_file(filename:str) -> pd.DataFrame:
        return pd.read_csv(filename)

def go(args):

    # run = wandb.init(job_type="data_ingestion")
    # run.config.update(args)

    ingest_data = Ingest_Data(args)

    dagshub.init(repo_owner='mdeevan', 
                 repo_name='dynamic-risk-assessment-system', 
                 mlflow=True)

    print("beore mlflow_start_run")
    
    # mlflow.set_tracking_uri("https://dagshub.com/mdeevan/dynamic-risk-assessment-system.mlflow")
    # mlflow.
    # os.environ['MLFLOW_TRACKING_USERNAME'] = 'mdeevan'
    # os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cffbcbe17a5519468e0cff1f2a2fc472c527c9d3'


curl -u mdeevan:cffbcbe17a5519468e0cff1f2a2fc472c527c9d3 https://dagshub.com/api/v1/user

# Add this to your training script (train.py)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Env vars:", {k: v for k, v in os.environ.items() if "MLFLOW" in k or "DAGSHUB" in k})

    with mlflow.start_run():
        print("inside mlflow_start_run")
        print(f"inside go and in scope of mlflow.start_run")
        ingest_data.process_files(args)

        mlflow.log_param("out_filename", args.out_file)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################





if __name__ == "__main__":

    print("inside run_data_ingestion.py")

    parser = argparse.ArgumentParser(description="ingest data from the input folder")


    parser.add_argument(
        "--in_path", 
        type=str, 
        help='path to the data file for ingestion',  
        required=True
    )

    parser.add_argument(
        "--in_file", 
        type=str,
        help='filename to ingest, asterisk (*) means all files in the folder', 
        required=True
    )

    parser.add_argument(
        "--out_path", 
        type=str ,
        help="path where the processed file to be stored",
        required=True
    )

    parser.add_argument(
        "--out_file", 
        type=str,
        help='name of the outfile',
        required=True
    )


    args = parser.parse_args()

    go(args)
