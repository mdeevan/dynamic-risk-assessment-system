#!/usr/bin/env python
"""
deploy the trainined model, test data and score in production folder
"""
import argparse
import logging
import dagshub
import mlflow
import inspect
import os
import shutil


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# (trainedmodel.pkl), 
# your model score (
#     latestscore.txt), and a record of your ingested data (
#         ingestedfiles.txt

class Production_Deployment():

    def __init__(self, args):
        self.model_path_name      = args.model_path_name
        self.output_model_name    = args.output_model_name
        self.score_filename       =  args.score_filename

        self.ingested_data_path   = args.ingested_data_path
        self.ingested_filename    = args.ingested_filename
        self.ingested_files_log   = args.ingested_files_log

        self.prod_deployment_path =  args.prod_deployment_path

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


    def __make_dir(self, p_parent:str, p_folder:str) -> str:
        '''
        make a folder, if it doesn;'t already exists

        INPUT:
            p_parent: str : parent folder path
            p_folder: str : folder name to check and create
        RETURN:
            bool : created or failed in creating the folder
        '''

        # parent = self.parent_folder if p_parent is None else p_parent
        # folder = self.report_folder if p_folder is None else p_folder

        try:
            folder_name = os.path.join(p_parent, p_folder)

            if (not os.path.isdir(folder_name)):
                os.mkdir(folder_name)

        except Exception as err:
            logging.info(f"Error creating folder : %s", folder_name)
            raise
            # return False

        return folder_name



    def run_model_deploy(self) -> float:
        func_name = inspect.currentframe().f_code.co_name

        prod_folder = self.__make_dir(self.parent_folder,
                                      self.prod_deployment_path)

        files_to_copy = []
        model_file = self.__get_filename(p_path=self.model_path_name,
                                         p_filename=self.output_model_name )

        score_file = self.__get_filename(p_path=self.model_path_name, 
                                         p_filename=self.score_filename )

        ingested_file = self.__get_filename(p_path=self.ingested_data_path,
                                            p_filename=self.ingested_filename )

        ingested_files_log = self.__get_filename(p_path=self.ingested_data_path,
                                            p_filename=self.ingested_files_log )

        files_to_copy.append(model_file)
        files_to_copy.append(score_file)
        files_to_copy.append(ingested_file)
        files_to_copy.append(ingested_files_log)

        try:
            for file in files_to_copy:
                shutil.copy(file, prod_folder)

        except Exception as err:
            logging.error(f"%s : error copying %s", func_name, err)
            raise

        return True




def go(args):

    production_deployment = Production_Deployment(args)

    if production_deployment.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")
            
            try:
                _ = production_deployment.run_model_deploy()
                # print(f"y_pred : %s", path)

            except Exception as err:
                logger.error(f"Error running model deployment %s", err)
                return False
    else:
        try: 
            logger.info("production deployment without logging")
            _ = production_deployment.run_model_deploy()

        except Exception as err:
            logger.error(f"Error running production deployment w/o logging %s", err)
            return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="deploy the trained model, data and score in production folder")


    parser.add_argument(
        "--model_path_name" , 
        type=str,
        help="path where model is stored",
        required=True
    )

    parser.add_argument(
        "--output_model_name", 
        type=str,
        help="name of the model",
        required=True
    )

    parser.add_argument(
        "--score_filename", 
        type=str,
        help="score filename",
        required=True
    )

    parser.add_argument(
        "--ingested_data_path", 
        type=str,
        help="path where data used for training is stored",
        required=True
    )

    parser.add_argument(
        "--ingested_filename", 
        type=str,
        help="filename of the file used in training",
        required=True
    )

    parser.add_argument(
        "--ingested_files_log", 
        type=str,
        help="log file with the list of files ingested",
        required=True
    )

    parser.add_argument(
        "--prod_deployment_path", 
        type=str,
        help="production deployment path",
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
