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
from datetime import datetime
import subprocess

sys.path.append("../")
from data_ingestion.ingestion import Ingest_Data
from training.training import Train_Model

from lib import utilities

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Diagnostics():

    def __init__(self, args):

        print(f"diagnostics args : {args}")
            
        self.model_path_name   = args.model_path_name
        self.model_file_name   = args.model_file_name
        self.data_folder       = args.data_folder
        self.data_files        = args.data_files
        self.ingested_file     = args.ingested_file
        self.report_folder     = args.report_folder
        self.prediction_output = args.prediction_output
        self.score_filename    = args.score_filename
        self.timing_filename   = args.timing_filename
        self.mlflow_logging    = args.mlflow_logging
        self.temp_folder       = args.temp_folder
        self.parent_folder     = "../../"
        self.arg_num_features  = args.num_features
        self.arg_lr_params     = args.lr_params
        # self.lr_params         = ast.literal_eval(args.lr_params.replace("'None'", 'None'))


        num_features = ast.literal_eval(args.num_features) 
        self.num_features = num_features

    # def __load_model(self):
    #     # https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function

    #     func_name = inspect.currentframe().f_code.co_name

    #     # load model
    #     logging.info("Loading deployed model")
    #     # model_loc = self.__get_filename(self.model_file_name, self.model_path_name)
    #     model_loc = utilities.get_filename(p_filename = self.model_file_name,
    #                                        p_parent_folder=self.parent_folder,
    #                                        p_path =self.model_path_name)

    #     logging.info("\nafter calling utitlies \n")
    #     try:

    #         logging.info(f"Loading deployed model %s", model_loc)
    #         file = open(model_loc, 'rb')
    #         model = pickle.load(file)

    #     except Exception as err:
    #         logging.error(f"%s: error loading model %s", func_name, err)
    #         raise

    #     return model

    # def __load_dataset(self) -> pd.DataFrame:
    #     logging.info("Loading test data")
    #     try:

    #         df = pd.DataFrame()
    #         test_data_folder = os.path.join(self.parent_folder, self.data_folder)
    #         logging.debug(f"Diagnostic test data folder: {test_data_folder} , \
    #                                 parent={self.parent_folder}, \
    #                                 data folder={self.data_folder}")

    #         # Process all files in the data folder 
    #         # alternate is to process a single file as configured in config.yaml
    #         if self.data_files == "*":
    #             # files = [f for f in os.listdir(test_data_folder) 
    #             #         if os.path.isfile(self.__get_filename(f))]
    #             files = [f for f in os.listdir(test_data_folder) 
    #                         if os.path.isfile(utilities.get_filename(p_filename=f,
    #                                                                  p_parent_folder=self.parent_folder,
    #                                                                  p_path=self.data_folder
    #                                                                  ))]
    #         else:
    #             files = [self.data_files]

    #         for file in files:
    #             # filename = self.__get_filename(file)            
    #             filename = utilities.get_filename(p_filename=file,
    #                                               p_parent_folder=self.parent_folder,
    #                                               p_path=self.data_folder
    #                                               )            

    #             # df_new = self.__read_file(filename)
    #             df_new = utilities.read_file(filename)
    #             df = pd.concat([df, df_new], axis=0)         
   
    #     except Exception as err:
    #         logging.error(f"%s: error diagnostic reading test data %s", func_name, err)
    #         raise

        # load dataset

    def __find_null_values(self, df) -> str:
        # ---------------------------------
        # Find an record null values in each of the columns
        outfile = utilities.get_filename(p_filename="null_values.csv",
                                         p_parent_folder=self.parent_folder,
                                         p_path=self.report_folder)
        null_values = df.isna().sum() 
        pd.DataFrame(null_values).T.to_csv(outfile, index=False)

        return null_values

    def __capture_statistics(self, p_df:pd.DataFrame, p_return_type:str="df"):
        # ---------------------------------
        # Capture statistics of numeric columns (mean, median, std)
        outfile = utilities.get_filename(p_filename="statistics.csv",
                                         p_parent_folder=self.parent_folder,
                                         p_path=self.report_folder)
        
        print(f"STAT: outfile : {outfile}")
        agg_values = (p_df[self.num_features].agg((['mean','median','std']))
                                .T.reset_index())
        agg_values.to_csv(outfile, index=False) # common behavior

        if p_return_type == "csv":
            stats_value = agg_values.to_csv()
        elif p_return_type=="md":
            stats_value = agg_values.to_markdown()
        elif p_return_type=="html":
            stats_value = agg_values.to_html()
        else:
            stats_value = agg_values 

        return stats_value


    def __pip_outdated(self, p_filename):
        # p_parent:str, 
        #                p_path_name:str, 
        #                p_filename: str ):
        # # ---------------------------------
        # Capture outdated installed packages
        print("inside pip_outdated")
        # print(f"{p_parent}")
        # outfile = utilities.get_filename(p_filename=p_filename,
        #                                  p_parent_folder=p_parent,
        #                                  p_path=p_path_name)

        outfile = utilities.get_filename(p_filename=p_filename,
                                         p_parent_folder=self.parent_folder,
                                         p_path=self.report_folder)

        command = ["pip", "list","--outdated"]
        print(f"command : {command}")

        with open(outfile, "w") as f:
            result = subprocess.run(command, stdout=f, text=True, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logger.info("Error generating %s", outfile)

        return result.returncode

    def __make_predictions(self, df:pd.DataFrame, model):
        # ---------------------------------
        # make predictions
        func_name = inspect.currentframe().f_code.co_name


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
                predict_output = utilities.get_filename(p_filename=self.prediction_output,
                                                        p_parent_folder=self.parent_folder,
                                                        p_path=self.report_folder
                                                        )

                # save prediction result in the same folder as the folder
                pd.DataFrame(zip(y, y_pred.tolist()), 
                             columns=['target','predicted']
                             ).to_csv(predict_output, index=False)

        except Exception as err:
            logging.error(f"%s: diagnostic error making prediction %s", func_name, err)
            raise

        return predict_output
    
    def run_diagnostics(self, p_diag_list:list=['all']) -> str:
        # https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function
        # func_name = inspect.currentframe().f_back.f_code.co_name
        func_name = inspect.currentframe().f_code.co_name

        diag_list = ['prediction','null', 'stat', 'timing'] if "all" in p_diag_list   else p_diag_list

        model = utilities.load_model(p_model_file_name = self.model_file_name,
                                     p_parent_folder   = self.parent_folder,
                                     p_model_path_name = self.model_path_name)

        df = utilities.load_dataset(p_parent_folder = self.parent_folder,
                                    p_data_folder   = self.data_folder,
                                    p_data_files    = self.data_files)
        

        print(f"p_diag_list : {diag_list}")
        pred_path = ""
        if "prediction" in diag_list:
            print('call prediction')
            pred_path= self.__make_predictions(df, model)

        null_value = ''
        if "null" in diag_list:
            print('call null')
            null_value = self.__find_null_values(df)

        stats_value = ''
        if "stat" in diag_list:
            print('call stat')
            stats_value = self.__capture_statistics(p_df=df,  p_return_type="df")


        if "timing" in diag_list:
            ingestion_time = self._time_ingestion(10)
            training_time  = self._time_training(10)
            
            logging.info(f"Ingestion time : {ingestion_time:.6f} seconds")
            logging.info(f"Training time  : {training_time:.6f} seconds")


            outfile = utilities.get_filename(p_filename=self.timing_filename ,
                                            p_parent_folder=self.parent_folder,
                                            p_path=self.report_folder )

            with open(outfile, 'w+') as f:
                exec_date = datetime.now().strftime('%m/%d/%Y %H:%M:%S')

                f.write("dte, process, execute time (secs)\n")
                f.write(f"{exec_date},ingestion,{ingestion_time}\n")
                f.write(f"{exec_date},training ,{training_time}\n")



        print("  pip_outdated")
        print(f"{self.parent_folder}")

        self.__pip_outdated(p_filename  ="outdated_packages.txt")
        # self.__pip_outdated(p_parent    = self.parent_folder,
        #                     p_path_name = self.report_folder,
        #                     p_filename  ="outdated_packages.txt")

        return pred_path, null_value, stats_value


    def __time_ingestion_setup(self):
        '''
        setup class instance to test the ingestion execution
        the parameters used as arguments are class attributes
        INGEST_DATA: its a python module that has class definition to perform ingestion
        PROCESS_DATA: is a class method to perform the ingestion
        
        INPUT:
            None
        OUTPUT:
            None
        '''
        logging.debug("Inside time_ingestion")

        parser = argparse.ArgumentParser(description="time ingestion")

        parser.add_argument("--ingestion_path", type=str, default=self.data_folder)
        parser.add_argument("--ingestion_filename", type=str, default=self.data_files)
        parser.add_argument("--out_path", type=str, default=self.temp_folder)
        parser.add_argument("--out_file", type=str, default=self.ingested_file)
        parser.add_argument("--ingested_files_log", type=str, default="templog")
        parser.add_argument("--mlflow_logging", type=str, default=False)
        parser.add_argument("--diagnostic", type=str, default=True)

        args = parser.parse_args([]) # Pass an empty list for non-command-line usage

        time_ingestion = Ingest_Data(args)
        time_ingestion.process_files()

    def __time_training_setup(self):
        '''
        setup class instance to test the ingestion execution
        the parameters used as arguments are class attributes
        INGEST_DATA: its a python module that has class definition to perform ingestion
        PROCESS_DATA: is a class method to perform the ingestion
        
        INPUT:
            None
        OUTPUT:
            None
        '''
        logging.debug("Inside time_ingestion")

        parser = argparse.ArgumentParser(description="time ingestion")

        parser.add_argument("--ingested_data_path", type=str, default=self.temp_folder)
        parser.add_argument("--ingestion_filename", type=str, default=self.ingested_file)
        parser.add_argument("--out_path", type=str, default=self.temp_folder)
        parser.add_argument("--out_model", type=str, default=self.model_file_name)
        parser.add_argument("--num_features", type=str, default=self.arg_num_features)
        parser.add_argument("--lr_params", type=str, default=self.arg_lr_params)
        parser.add_argument("--mlflow_logging", type=str, default=False)
        parser.add_argument("--diagnostic", type=str, default=True)

        args = parser.parse_args([]) # Pass an empty list for non-command-line usage

        time_training = Train_Model(args)
        time_training.train_model()

    def _time_ingestion(self, p_iterations: int = 1) -> float:
        '''
        class method to test the execution time of class ingestion using timeit.
        INPUT:
            p_iterations: INT: number of iterations to perform intesting performance
        OUTPUT:
            execution time : FLOAT - execution time in seconds

        '''

        logging.info("inside time_ingestion")
        t = timeit.Timer(self.__time_ingestion_setup)
        execution_time = t.timeit(p_iterations)
        logging.debug(f"INGESTION execution time with {p_iterations} iterations : {execution_time}")

        return execution_time

    def _time_training(self, p_iterations: int = 1) -> float:
        '''
        class method to test the execution time of class ingestion using timeit.
        INPUT:
            p_iterations: INT: number of iterations to perform intesting performance
        OUTPUT:
            execution time : FLOAT - execution time in seconds

        '''

        logging.info("inside time_training")
        t = timeit.Timer(self.__time_training_setup)
        execution_time = t.timeit(p_iterations)
        logging.debug(f"Training execution time with {p_iterations} iterations : {execution_time}")

        return execution_time

def go(args):

    diagnostics = Diagnostics(args)

    print(f"\nargs : {args}\n")
    if diagnostics.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")
            
            try:
                pred_path, null_value, stats_value = diagnostics.run_diagnostics()
                print(f"y_pred : %s", pred_path)

                mlflow.log_artifact(pred_path)
                # mlflow.log_artifact(null_value)
                # mlflow.log_artifact(stats_value)

            except Exception as err:
                logger.error(f"Error running diagnostics %s", err)
                return False
    else:
        try: 
            logger.info("training without logging")
            pred_path, null_value, stats_value  = diagnostics.run_diagnostics()
            mlflow.log_artifact(pred_path)

        except Exception as err:
            logger.error(f"Error running diagnostics w/o logging %s", err)
            return False
        
    
    # ingestion_time = diagnostics._time_ingestion(10)
    # training_time = diagnostics._time_training(10)
    
    # logging.info(f"Ingestion time : {ingestion_time:.6f} seconds")
    # logging.info(f"Training time  : {training_time:.6f} seconds")


    # outfile = utilities.get_filename(p_filename=diagnostics.timing_filename,
    #                                  p_parent_folder=diagnostics.parent_folder,
    #                                  p_path=diagnostics.report_folder)

    # with open(outfile, 'w+') as f:
    #     exec_date = datetime.now().strftime('%m/%d/%Y %H:%M:%S')

    #     f.write("dte, process, execute time (secs)\n")
    #     f.write(f"{exec_date},ingestion,{ingestion_time}\n")
    #     f.write(f"{exec_date},training ,{training_time}\n")
    


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
        "--ingested_file", 
        type=str,
        help="processed file resulting from input data files ",
        required=True
    )
    parser.add_argument(
        "--report_folder", 
        type=str,
        help="folder for reports and results ",
        required=True
    )
    parser.add_argument(
        "--temp_folder", 
        type=str,
        help="folder for diagnostics output - temporary ",
        required=False,
        default="temp"
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
        "--timing_filename", 
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
        "--lr_params", 
        type=str,
        help='logistic regression model tuning parameters',
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
