#!/usr/bin/env python
"""
deploy the trainined model, test data and score in production folder
"""
import sys
import argparse
import logging
import inspect
import shutil
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

sys.path.append("../")
from lib import utilities


class Production_Deployment:
    """
    class with implementation methods for production deployment
    """

    def __init__(self, args):
        self.model_path_name = args.model_path_name
        self.output_model_name = args.output_model_name
        self.score_filename = args.score_filename

        self.ingested_data_path = args.ingested_data_path
        self.ingested_filename = args.ingested_filename
        self.ingested_files_log = args.ingested_files_log

        self.prod_deployment_path = args.prod_deployment_path

        self.mlflow_logging = args.mlflow_logging
        self.parent_folder = "../../"

    def run_model_deploy(self) -> float:
        """
        method to perform deployment
        """
        func_name = inspect.currentframe().f_code.co_name

        prod_folder = utilities.make_dir(self.parent_folder, self.prod_deployment_path)

        files_to_copy = []
        model_file = utilities.get_filename(
            p_filename=self.output_model_name,
            p_parent_folder=self.parent_folder,
            p_path=self.model_path_name,
        )

        score_file = utilities.get_filename(
            p_filename=self.score_filename,
            p_parent_folder=self.parent_folder,
            p_path=self.model_path_name,
        )

        ingested_file = utilities.get_filename(
            p_filename=self.ingested_filename,
            p_parent_folder=self.parent_folder,
            p_path=self.ingested_data_path,
        )

        ingested_files_log = utilities.get_filename(
            p_filename=self.ingested_files_log,
            p_parent_folder=self.parent_folder,
            p_path=self.ingested_data_path,
        )

        files_to_copy.append(model_file)
        files_to_copy.append(score_file)
        files_to_copy.append(ingested_file)
        files_to_copy.append(ingested_files_log)

        logger.debug("files to copy : %s", files_to_copy)
        try:
            for file in files_to_copy:
                shutil.copy(file, prod_folder)

        except Exception as err:
            logger.error("%s : error copying %s", func_name, err)
            raise

        return True


def go(args):
    """'
    main routine to execute deployment
    """

    production_deployment = Production_Deployment(args)

    if production_deployment.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print("inside go and in scope of mlflow.start_run")

            try:
                _ = production_deployment.run_model_deploy()
                # print(f"y_pred : %s", path)

            except Exception as err:
                logger.error("Error running model deployment %s", err)
                return False
    else:
        try:
            logger.info("production deployment without logging")
            _ = production_deployment.run_model_deploy()

        except Exception as err:
            logger.error("Error running production deployment w/o logging %s", err)
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="deploy the trained model, data and score in production folder"
    )

    parser.add_argument(
        "--model_path_name", type=str, help="path where model is stored", required=True
    )

    parser.add_argument(
        "--output_model_name", type=str, help="name of the model", required=True
    )

    parser.add_argument(
        "--score_filename", type=str, help="score filename", required=True
    )

    parser.add_argument(
        "--ingested_data_path",
        type=str,
        help="path where data used for training is stored",
        required=True,
    )

    parser.add_argument(
        "--ingested_filename",
        type=str,
        help="filename of the file used in training",
        required=True,
    )

    parser.add_argument(
        "--ingested_files_log",
        type=str,
        help="log file with the list of files ingested",
        required=True,
    )

    parser.add_argument(
        "--prod_deployment_path",
        type=str,
        help="production deployment path",
        required=True,
    )
    parser.add_argument(
        "--mlflow_logging",
        type=bool,
        help="mlflow logging enable or disabled",
        required=False,
    )

    cmd_args = parser.parse_args()

    go(cmd_args)
