#!/usr/bin/env python
"""
Score model metrics on provided test data
"""
import argparse
import logging
import dagshub
import mlflow
import pandas as pd
import os
from sklearn import metrics
import inspect



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Score_Model():

    def __init__(self, args):
        self.model_path_name = args.model_path_name
        self.report_folder =  args.report_folder
        self.prediction_output =  args.prediction_output
        self.score_filename =  args.score_filename
        self.mlflow_logging = args.mlflow_logging
        self.parent_folder = "../../"

    def __get_filename(self, p_filename:str, p_path:str=None) -> str:
        '''
        Form and return a filename
        Input:
                    p_filename : str - filename 
            p_path : str - path where the filename is stored/created

        return:
            None
        '''

        filename = os.path.join(self.parent_folder, p_path, p_filename)
        logger.info(f"_-get-filename : {filename}")
        return filename


    def __make_dir(self, p_parent:str, p_folder:str) -> bool:
        '''
        make a folder, if it doesn;'t already exists

        INPUT:
            p_parent: str : parent folder path
            p_folder: str : folder name to check and create
        RETURN:
            bool : created or failed in creating the folder
        '''

        parent = self.parent_folder if p_parent is None else p_parent
        folder = self.report_folder if p_folder is None else p_folder

        try:
            folder_name = os.path.join(parent, folder)

            if (not os.path.isdir(folder_name)):
                os.mkdir(folder_name)

        except Exception as err:
            logging.info(f"Error creating folder : %s", folder_name)
            raise
            # return False

        return True

    def __read_file(self, filename:str) -> pd.DataFrame:
        '''
        read csv into panda framework

        INPUT:
            filename : csv files to read
        RETURN:
            pd.DataFrme : panda dataframe
        '''
        return pd.read_csv(filename)


    def run_model_scoring(self) -> float:

        func_name = inspect.currentframe().f_code.co_name


        logging.info("Loading predictions ")
        try:

            filename = self.__get_filename(p_filename=self.prediction_output,
                                           p_path=self.report_folder)


            print(f"filename : {filename}")
            df = self.__read_file(filename)

        
            f1_score = metrics.f1_score(df['predicted'], df['target'])

            outfile = self.__get_filename(p_path=self.model_path_name,
                                           p_filename=self.score_filename )

            with open(outfile, 'w+') as f:
                f.write(str(f1_score) + '\n')

        except Exception as err:
            logging.error(f"%s: error running model scoring %s", func_name, err)
            raise

        return f1_score


def go(args):

    score_model = Score_Model(args)

    if score_model.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")
            
            try:
                f1_score = score_model.run_model_scoring()

                mlflow.log_metric('fi-score', value=f1_score)

            except Exception as err:
                logger.error(f"Error running model scoring %s", err)
                return False
    else:
        try: 
            logger.info("training without logging")
            f1_score = score_model.run_model_scoring()

            mlflow.log_metric('fi-score', value=f1_score)

        except Exception as err:
            logger.error(f"Error running model scoring w/o logging %s", err)
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="scoring the model")


    parser.add_argument(
        "--model_path_name", 
        type=str,
        help="path where model is stored",
        required=True
    )

    parser.add_argument(
        "--report_folder", 
        type=str,
        help="folder where the predictions were stored after training",
        required=True
    )
    parser.add_argument(
        "--prediction_output", 
        type=str,
        help="prediction filename ",
        required=True
    )
    parser.add_argument(
        "--score_filename", 
        type=str,
        help="filename to store the model scoring - f1 score" ,
        required=True
    )
    parser.add_argument(
        "--mlflow_logging", 
        type=bool,
        help='mlflow logging enable or disabled',
        required=False
    )

    args = parser.parse_args()

    go(args)
