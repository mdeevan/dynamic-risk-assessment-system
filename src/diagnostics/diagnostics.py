#!/usr/bin/env python
"""
Make predictions on test data with the newly created model to diagnose problem and evaluate results
"""
import argparse
import logging
import dagshub
import mlflow
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Diagnostics():

    def __init__(self, args):
        self.model_path_name = args.model_path_name
        self.model_file_name = args.model_file_name
        self.data_path_name =  args.data_path_name
        self.mlflow_logging = args.mlflow_logging


    def __get_filename(self, p_filename:str, p_path:str=None) -> str:
        '''
        Form and return a filename
        Input:
                    p_filename : str - filename 
            p_path : str - path where the filename is stored/created

        return:
            None
        '''

        path = self.in_path if (p_path is None) else p_path

        filename = os.path.join(self.parent_folder, path, p_filename)
        logger.info(f"_-get-filename : {filename}")
        return filename


    def __read_file(self, filename:str) -> pd.DataFrame:
        '''
        read csv into panda framework

        INPUT:
            filename : csv files to read
        RETURN:
            pd.DataFrme : panda dataframe
        '''
        return pd.read_csv(filename)


    def run_diagnostics(self):

        # load model

        # load dataset

        # make predictions

        return True


def go(args):

    diagnostics = Diagnostics(args)


    if Diagnostics.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")
            
            try:
                path = diagnostics.run_diagnostics()

                mlflow.log_param("out_filename", path)
                mlflow.log_artifact(path)

            except Exception as err:
                logger.error(f"Train Model Error %s", err)
                return False
    else:
        try: 
            logger.info("training without logging")
            path = diagnostics.run_diagnostics()

        except Exception as err:
            logger.error(f"Train Model - w/o logging Error %s", err)
            return False




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="perform diagnostic by testing model")


    parser.add_argument(
        "--model_path", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "-- model_name", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
