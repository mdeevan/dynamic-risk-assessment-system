#!/usr/bin/env python
"""
Make predictions on test data with the newly created model to diagnose problem and evaluate results
"""
import os
import sys
import argparse
import logging
import dagshub
import mlflow
import pandas as pd
import pickle
import ast
import inspect
import timeit

sys.path.append("../")
# from  ../../data_ingestion import Ingest_Data()
from data_ingestion.ingestion import Ingest_Data
from lib import utilities

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Diagnostics():

    def __init__(self, args):
            
        self.model_path_name   = args.model_path_name
        self.model_file_name   = args.model_file_name
        self.data_folder       = args.data_folder
        self.data_files        = args.data_files
        self.report_folder     = args.report_folder
        self.prediction_output = args.prediction_output
        self.score_filename    = args.score_filename
        self.mlflow_logging    = args.mlflow_logging
        self.parent_folder     = "../../"

        num_features = ast.literal_eval(args.num_features) 
        self.num_features = num_features


    # def __get_filename(self, p_filename:str, p_path:str=None) -> str:
    #     '''
    #     Form and return a filename
    #     Input:
    #                 p_filename : str - filename 
    #         p_path : str - path where the filename is stored/created

    #     return:
    #         None
    #     '''

    #     path = self.data_path_name if (p_path is None) else p_path

    #     filename = os.path.join(self.parent_folder, path, p_filename)
    #     logger.info(f"_-get-filename : {filename}")
    #     return filename

    # def __make_dir(self, p_parent:str, p_folder:str) -> bool:

    #     parent = self.parent_folder if p_parent is None else p_parent
    #     folder = self.report_folder if p_folder is None else p_folder

    #     try:
    #         folder_name = os.path.join(parent, folder)
    #         os.mkdir(folder_name)

    #     except Exception as err:
    #         logging.info(f"folder already exists : %s", folder_name)
    #         raise

    # def __read_file(self, filename:str) -> pd.DataFrame:
    #     '''
    #     read csv into panda framework

    #     INPUT:
    #         filename : csv files to read
    #     RETURN:
    #         pd.DataFrme : panda dataframe
    #     '''
    #     return pd.read_csv(filename)


    def run_diagnostics(self) -> str:
        

# https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function
        # func_name = inspect.currentframe().f_back.f_code.co_name
        func_name = inspect.currentframe().f_code.co_name

        # load model
        # load model
        logging.info("Loading deployed model")
        # model_loc = self.__get_filename(self.model_file_name, self.model_path_name)
        model_loc = utilities.get_filename(self.model_file_name, self.model_path_name)
        try:

            logging.info(f"Loading deployed model %s", model_loc)
            file = open(model_loc, 'rb')
            model = pickle.load(file)

        except Exception as err:
            logging.error(f"%s: error loading model %s", func_name, err)
            raise
            logging.error(f"%s: error loading model %s", func_name, err)

        # load dataset
        logging.info("Loading test data")
        try:

            df = pd.DataFrame()
            test_data_folder = os.path.join(self.parent_folder, self.data_folder)
            print(f"Diagnostic test data folder: {test_data_folder} , parent={self.parent_folder}, data folder={self.data_folder}")

            # Process all files in the data folder 
            # alternate is to process a single file as configured in config.yaml
            if self.data_files == "*":
                # files = [f for f in os.listdir(test_data_folder) 
                #         if os.path.isfile(self.__get_filename(f))]
                files = [f for f in os.listdir(test_data_folder) 
                        if os.path.isfile(utilities.get_filename(f))]
            else:
                files = [self.data_files]

            for file in files:
                # filename = self.__get_filename(file)            
                filename = utilities.get_filename(file)            

                # df_new = self.__read_file(filename)
                df_new = utilities.read_file(filename)
                df = pd.concat([df, df_new], axis=0)         
   
        except Exception as err:
            logging.error(f"%s: error diagnostic reading test data %s", func_name, err)
            raise

        # make predictions

        logging.info("Making predictions")
        try:
            y_pred = None
            if df is not None:
                X = df[self.num_features]
                y = X.pop('exited')            

                # print(X.head())
                y_pred = model.predict(X)

                # _ = self.__make_dir(self.parent_folder,
                #                     self.report_folder)
                _ = utilities.make_dir(self.parent_folder,
                                    self.report_folder)

                # predict_output = self.__get_filename(p_path=self.report_folder, 
                #                                      p_filename=self.prediction_output)
                predict_output = utilities.get_filename(p_path=self.report_folder,
                                                        p_filename=self.prediction_output)

                # save prediction result in the same folder as the folder
                pd.DataFrame(zip(y, y_pred.tolist()), 
                             columns=['target','predicted']
                             ).to_csv(predict_output, index=False)

        except Exception as err:
            logging.error(f"%s: diagnostic error making prediction %s", func_name, err)
            raise

        return predict_output
    

    def __time_ingestion(self, p_iterations: int = 1):
        parser = argparse.ArgumentParser(description="time ingestion")

        parser.add_argument("--ingestion_path", type=str, default=self.data_folder)
        parser.add_argument("--ingestion_filename", type=str, default=self.data_files)
        parser.add_argument("--out_path", type=str, default="temp")
        parser.add_argument("--out_file", type=str, default="temp")
        parser.add_argument("--ingested_files_log", type=str, default="templog")
        parser.add_argument("--mlflow_logging", type=str, default=False)
        parser.add_argument("--diagnostic", type=str, default=True)

        test_ingestion = Ingest_Data(parser.parse_args())

        path = test_ingestion.process_files()

        execution_time = timeit.timeit(test_ingestion.process_files(), number=p_iterations)

        return execution_time


    def __time_training(self, p_iterations: int = 1):
        pass



def go(args):

    diagnostics = Diagnostics(args)

    if diagnostics.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")
            
            try:
                path = diagnostics.run_diagnostics()
                print(f"y_pred : %s", path)

                mlflow.log_artifact(path)

            except Exception as err:
                logger.error(f"Error running diagnostics %s", err)
                return False
    else:
        try: 
            logger.info("training without logging")
            path = diagnostics.run_diagnostics()
            mlflow.log_artifact(path)

        except Exception as err:
            logger.error(f"Error running diagnostics w/o logging %s", err)
            return False
        
    
    ingestion_time = diagnostics.__time_ingestion(10)
    logging.info(f"Ingestion time : {ingestion_time:.6f} seconds")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="perform diagnostic by testing model")


    parser.add_argument(
        "--model_path_name", 
        type=str,
        help="path where model is stored",
        required=True
    )

    parser.add_argument(
        "--model_file_name", 
        type=str,
        help="name of the model, stored under model_file_name",
        required=True
    )
    parser.add_argument(
        "--data_folder", 
        type=str,
        help="path where data is stored ",
        required=True
    )
    parser.add_argument(
        "--data_files", 
        type=str,
        help="Files to process ",
        required=True
    )
    parser.add_argument(
        "--report_folder", 
        type=str,
        help="folder for reports and results ",
        required=True
    )
    parser.add_argument(
        "--prediction_output", 
        type=str,
        help="output from predictions ",
        required=True
    )
    parser.add_argument(
        "--score_filename", 
        type=str,
        help="filename to store the score ",
        required=True
    )
    parser.add_argument(
        "--num_features", 
        type=str,
        help='modeling parameters',
        required=False
    )
    parser.add_argument(
        "--mlflow_logging", 
        type=bool,
        help='mlflow logging enable or disabled',
        required=False
    )

    args = parser.parse_args()

    go(args)
