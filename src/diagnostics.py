"""
diagnostic.py
contains the diagnostic functions
"""

import logging
import subprocess
import timeit
import pandas as pd
import yaml
from lib import utilities
from data_ingestion.ingestion import Ingest_Data
from training.training import Train_Model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Diagnostics():
    """
    Diagnostic class, encapsulating the diagnostic functions

    """

    def __init__(self):
        """
        initialize the object
        read the config.yaml and initialize the object variables
        load the model and data file
        """
        try:
            with open("./config/config.yaml", encoding="utf-8") as stream:
                self.cfg = yaml.safe_load(stream)

        except Exception as err:
            self.cfg = None
            logging.error("FATAL: Error initialization configuration %s", err)

        self.parent_folder = "./"
        self.model_name = self.cfg["training"]["output_model_name"]
        self.model_path_name = self.cfg["prod_deployment"]["prod_deployment_path"]

        # self.data_folder     = self.cfg['diagnostics']['data_folder']
        # self.data_files      = self.cfg['diagnostics']['data_files']
        self.data_folder = self.cfg["ingestion"]["ingestion_path"]
        self.data_files = self.cfg["ingestion"]["ingestion_filename"]
        self.ingested_filename = self.cfg["ingestion"]["ingested_filename"]

        self.test_data_path = self.cfg["scoring"]["test_data_path"]
        self.test_data_name = "testdata.csv"
        self.num_features = self.cfg["num_features"]
        self.lr_params = self.cfg["logistic_regression_params"]

        outfile_path = self.cfg["training"]["output_model_path"]
        outfile_name = self.cfg["diagnostics"]["apicallstxt_file"]
        confusion_matrix_file = self.cfg["diagnostics"]["confusion_matrix_file"]

        self.outfile = utilities.get_filename(
            outfile_name, p_parent_folder="", p_path=outfile_path
        )

        self.confusion_matrix_file = utilities.get_filename(
            confusion_matrix_file, p_parent_folder="", p_path=outfile_path
        )

        try:
            self.model = utilities.load_model(
                p_model_file_name=self.model_name,
                p_parent_folder=self.parent_folder,
                p_model_path_name=self.model_path_name,
            )
        except Exception as err:
            self.model = None
            logging.error("Error loading Model %s", err)

        try:
            filename = utilities.get_filename(
                p_filename=self.test_data_name,
                p_parent_folder=self.parent_folder,
                p_path=self.test_data_path,
            )

            self.df = utilities.read_file(filename)

        except Exception as err:
            self.df = None
            logging.error("Error loading test df : %s", err)

    def find_null_values(self, p_data_path: str = "") -> str:
        """
        find the null value in the data
        RETURNS:
            dataframe
        """

        if p_data_path == "":
            df = self.df
        else:
            df = utilities.read_file(p_data_path)

        null_values = df.isna().sum()
        rtn = pd.DataFrame(null_values).T.to_json(index=False)

        return rtn

    def capture_statistics(self, p_data_path: str = "") -> str:
        """
        capture statistics
        RETURNS:
            json containing mean, median, standard deviation
        """

        if p_data_path == "":
            df = self.df
        else:
            df = utilities.read_file(p_data_path)

        agg_values = (
            df[self.num_features].agg((["mean", "median", "std"])).T.reset_index()
        ).to_json()

        rtn = agg_values

        return rtn

    def make_predictions(self, p_data_path: str = "") -> str:
        """
        make prediction on the test data
        RETURNS:
            target vs predictions
        """

        logger.debug("inside make predictions")

        if p_data_path == "":
            df = self.df
        else:
            df = utilities.read_file(p_data_path)

        rtn = None
        try:
            y_pred = None
            if df is not None:
                X = df[self.num_features]
                y = X.pop("exited")

                y_pred = self.model.predict(X)

                rtn = pd.DataFrame(
                    zip(y, y_pred.tolist()), columns=["target", "predicted"]
                ).to_json()

        except Exception as err:
            logging.error("diagnostic error making prediction %s", err)
            raise

        return rtn

    def dependencies_status(self):
        """
        check the outdated dependencies, to help decide which ones should be updated
        a version conflict can break the program
        the report helps make an informed decision

        """
        logger.debug("inside pip_outdated")

        command = ["pip", "list", "--outdated", "--format", "json"]

        result = subprocess.run(
            command, text=True, capture_output=True
        )  # , stderr=subprocess.PIPE)

        if result.returncode != 0:
            logger.info("Error running command %s", command)
        else:
            logger.info("\nSuccess running command %s", command)

        return result.stdout

    def __timing_ingestion_command(self):
        """
        a private method to setup the timing ingestion command
        this is then used inwith the timeit in calculating ingestion
        """

        ingest_data = Ingest_Data(
            p_ingestion_path=self.data_folder,
            p_ingestion_filename=self.data_files,
            p_out_path="temp",
            p_out_file=self.ingested_filename,
            p_ingested_files_log="templog",
            p_mlflow_logging=False,
            p_parent_folder="./",
        )

        ingest_data.process_files()

    def timing_ingestion(self, p_iterations=10):
        """
        calculate the ingestion timing by using timeit object
        """

        logging.info("inside time_ingestion")

        t = timeit.Timer(self.__timing_ingestion_command)
        execution_time = t.timeit(p_iterations)
        logging.debug(
            "INGESTION execution time with %s iterations : %s",
            p_iterations,
            execution_time,
        )

        return execution_time

    def __timing_training_command(self):
        """
        a private method to setup the timing training command
        this is then used inwith the timeit in calculating training
        """
        train_model = Train_Model(
            p_ingested_data_path="temp",  # from ingested timing method
            p_ingestion_filename=self.ingested_filename,
            p_out_path="temp",
            p_out_model=self.model_name,
            p_parent_folder="./",
            p_num_features=self.num_features,
            p_lr_params=self.lr_params[0],
            p_mlflow_logging=False,
        )

        train_model.train_model()

    def timing_training(self, p_iterations=10):
        """
        calculate the time it takes to train the model by using timeit object
        """

        logging.info("inside timing_training")

        t = timeit.Timer(self.__timing_training_command)
        execution_time = t.timeit(p_iterations)
        logging.debug(
            "TRAINING execution time with %s iterations : %s",
            p_iterations,
            execution_time,
        )

        return execution_time


if __name__ == "__main__":
    diagnostics = Diagnostics()

    nv = diagnostics.find_null_values()
    stat = diagnostics.capture_statistics()
    predict = diagnostics.make_predictions()
    result = diagnostics.dependencies_status()
    time_ingestion = diagnostics.timing_ingestion(10)
    train_ingestion = diagnostics.timing_training(10)

    diagnostics_list = [
        "Null Values",
        "Statistics",
        "Prediction",
        "Dependencies",
        "Time to Ingest data",
        "training ingested data",
    ]
    responses = [nv, stat, predict, result, time_ingestion, train_ingestion]

    # with open('apireturns_diagnostics.txt', "w") as f:
    with open(diagnostics.outfile, "w", encoding="utf-8") as f:
        f.write("Diagnostics \n")
        for idx, response in enumerate(responses):
            f.write("\n ------------------------------------- \n")
            f.write(f"result from {diagnostics_list[idx]}  \n")

            f.write(str(response))
            f.write("\n")

    logger.debug("\n--------------\n null values \n")
    logger.debug(nv)
    logger.debug("\n--------------\n statistics \n")
    logger.debug(stat)
    logger.debug("\n--------------\n predication \n")
    logger.debug(predict)
    logger.debug("\n--------------\n dependencies \n")
    logger.debug(result)

    logger.debug("\n--------------\n ingestion timing \n")
    logger.debug(time_ingestion)

    logger.debug("\n--------------\n training timing \n")
    logger.debug(train_ingestion)
