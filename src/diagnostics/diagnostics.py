#!/usr/bin/env python
"""
Make predictions on test data with the newly created model to diagnose problem and evaluate results
"""
import os
import argparse
import logging
import dagshub
import mlflow
import pandas as pd
import pickle
import ast
import inspect

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Diagnostics():

    def __init__(self, args):
        self.model_path_name = args.model_path_name
        self.model_file_name = args.model_file_name
        self.data_path_name =  args.data_path_name
        self.report_folder =  args.report_folder
        self.test_prediction_output =  args.test_prediction_output
        self.mlflow_logging = args.mlflow_logging
        self.parent_folder = "../../"

        num_features = ast.literal_eval(args.num_features) 
        self.num_features = num_features


    def __get_filename(self, p_filename:str, p_path:str=None) -> str:
        '''
        Form and return a filename
        Input:
                    p_filename : str - filename 
            p_path : str - path where the filename is stored/created

        return:
            None
        '''

        path = self.data_path_name if (p_path is None) else p_path

        filename = os.path.join(self.parent_folder, path, p_filename)
        logger.info(f"_-get-filename : {filename}")
        return filename


    def __make_dir(self, p_parent:str, p_folder:str) -> bool:

        parent = self.parent_folder if p_parent is None else p_parent
        folder = self.report_folder if p_folder is None else p_folder

        try:
            folder_name = os.path.join(parent, folder)
            os.mkdir(folder_name)

        except Exception as err:
            logging.info(f"folder already exists : %s", folder_name)
            raise

    def __read_file(self, filename:str) -> pd.DataFrame:
        '''
        read csv into panda framework

        INPUT:
            filename : csv files to read
        RETURN:
            pd.DataFrme : panda dataframe
        '''
        return pd.read_csv(filename)


    def run_diagnostics(self) -> str:

# https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function
        # func_name = inspect.currentframe().f_back.f_code.co_name
        func_name = inspect.currentframe().f_code.co_name

        # load model
        # load model
        logging.info("Loading deployed model")
        model_loc = self.__get_filename(self.model_file_name, self.model_path_name)
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
            test_data_folder = os.path.join(self.parent_folder, self.data_path_name)
            print(f"test data folder: {test_data_folder} , parent={self.parent_folder}, data folder={self.data_path_name}")
            files = [f for f in os.listdir(test_data_folder) if os.path.isfile(self.__get_filename(f))]

            for file in files:
                filename = self.__get_filename(file)            

                df_new = self.__read_file(filename)
                df = pd.concat([df, df_new], axis=0)         
   
        except Exception as err:
            logging.error(f"%s: error reading test data %s", func_name, err)
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

                _ = self.__make_dir(self.parent_folder,
                                    self.report_folder)

                predict_output = self.__get_filename(p_path=self.report_folder, 
                                                     p_filename=self.test_prediction_output)

                # save prediction result in the same folder as the folder
                pd.DataFrame(zip(y, y_pred.tolist()), 
                             columns=['target','predicted']
                             ).to_csv(predict_output, index=False)

        except Exception as err:
            logging.error(f"%s: error making prediction %s", func_name, err)
            raise

        return predict_output


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
        "--data_path_name", 
        type=str,
        help="path where data is stored ",
        required=True
    )
    parser.add_argument(
        "--report_folder", 
        type=str,
        help="folder for reports and results ",
        required=True
    )
    parser.add_argument(
        "--test_prediction_output", 
        type=str,
        help="output from predictions ",
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
