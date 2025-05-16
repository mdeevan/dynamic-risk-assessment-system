"""
Execute the full pipeline that involves
    - check if the new data is available
    -   ingest, if there is new data,
    -   calculate the score
    -   if score is worst,
    -      train, score and deploy the new model
"""

import os
import subprocess
import logging
import yaml

from lib import utilities

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Fullprocess:
    """
    Define the class to encapsulate the logic
    """

    def __init__(self):

        try:
            with open("../config/config.yaml", encoding="utf-8") as stream:
                cfg = yaml.safe_load(stream)

        except Exception as err:
            cfg = None
            logger.error("FATAL: Error initialization configuration %s", err)

        # ingested files from the last execution
        self.parent_folder = "../"
        self.prod_folder = cfg["prod_deployment"]["prod_deployment_path"]
        self.ingested_files = cfg["ingestion"]["ingested_files_log"]

        # files to ingest as part of new execution
        self.input_file_path = cfg["ingestion"]["ingestion_path"]
        self.input_file_name = cfg["ingestion"]["ingestion_filename"]

        # previous model score file
        self.model_folder = cfg["training"]["output_model_path"]
        self.score_filename = cfg["scoring"]["score_filename"]

    def read_ingested_files(self) -> list:
        """
        read the textfile that contains the files ingested the last time,
        stored in csv file
        RETURNS:
            list of ingested files
        """
        logger.debug("inside read_ingested_files")
        ingested_file_path = utilities.get_filename(
            self.ingested_files,
            p_parent_folder=self.parent_folder,
            p_path=self.prod_folder,
        )

        logger.debug("ingested file path %s ", ingested_file_path)

        df = utilities.read_file(ingested_file_path)
        ingested_files_list = df.file.to_list()

        return ingested_files_list

    def read_input_files(self, ingested_files_list: list) -> bool:
        """
        read the files in the input folder

        RETURNS:
            list of the files ingested
        """

        if "*" in self.input_file_name:
            input_folder = os.path.join(
                self.parent_folder,
                self.input_file_path,
            )

            logger.debug("input folder : %s", input_folder)
            files = [
                f
                for f in os.listdir(input_folder)
                if os.path.isfile(
                    utilities.get_filename(
                        p_filename=f,
                        p_parent_folder=self.parent_folder,
                        p_path=self.input_file_path,
                    )
                )
            ]
        else:
            files = self.input_file_name

        new_files = False
        for file in files:
            if file not in ingested_files_list:
                new_files = True

        logger.debug("new_files : %s: files : %s", new_files, files)

        return new_files

    def get_score(self, source: str = "deployed") -> float:
        """
        fetch the F1 score, depending on the source
        prod-deployment-folder, contains the score of the production model
        new, contains the score of the newly ingested input files

        RETURNS:
            float : Score

        """

        score_folder = self.prod_folder if source == "deployed" else self.model_folder

        fn = utilities.get_filename(
            self.score_filename, p_parent_folder=self.parent_folder, p_path=score_folder
        )

        score = -1
        try:
            with open(fn, "r", encoding="utf-8") as f:
                score = f.read()

            logger.debug("{source} score : %s", score)

        except FileNotFoundError as err:
            logger.debug("error : %s", err)
            score = 999.9

        return score

    def process_new_files(self):
        """
        ingest and score the new files
        it uses MLFlow to capture the parameters and output for later comparison

        RETURNS:
            float : F1 score
        """

        command = ["sh", "src/ingest_score.sh"]
        logger.debug("command : %s", command)
        subprocess.run(command, cwd="../")

        new_score = self.get_score("new")

        return new_score

    def process_train_deploy(self):
        """
        train , score and deploy the model to production
        """
        command = ["sh", "src/train_deploy.sh"]

        subprocess.run(command, cwd="../")

        return True

    def run_diagnostics_reporting(self):
        """
        run diagnostics and reporting to capture reports from the lastest deployment
        diagnostics include null values, statistics, ingestion, training time, package versions
        report include confusion matrix
        """

        commands = [["python", "diagnostics.py"], ["python", "reporting.py"]]

        for command in commands:
            subprocess.run(command)

        return True


def execute_fullprocess():
    """
    initialize the fullprocess object and execute depending up if new files are present,
    model has drifted (i.e., score is worse), then train and deploy a new mode.
    """

    fullprocess = Fullprocess()

    ##################Check and read new data
    # first, read ingestedfiles.txt
    logger.debug("read ingested files")
    ingested_files_list = fullprocess.read_ingested_files()

    # second, determine whether the source data folder has files
    # that aren't listed in ingestedfiles.txt
    ##################Deciding whether to proceed, part 1

    logger.debug("read new files : %s", ingested_files_list)
    new_files_exist = fullprocess.read_input_files(ingested_files_list)

    # if you found new data, you should proceed. otherwise, do end the process here
    new_score = -1.0
    deployed_score = -1.0
    if new_files_exist:
        new_score = fullprocess.process_new_files()

        logger.debug("new score : %s, type : %s", new_score, type(new_score))

    ##################Checking for model drift
    # check whether the score from the deployed model is different
    #  from the score from the model that uses the newest ingested data
    deployed_score = fullprocess.get_score("deployed")

    logger.debug(
        "deployed Score : %s , type : %s", deployed_score, type(deployed_score)
    )

    ##################Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here

    model_drift = float(new_score) > float(deployed_score)
    if new_files_exist and model_drift:
        ##################Re-deployment
        # if you found evidence for model drift, re-run the deployment.py script

        logger.debug("model drifted ")
        logger.debug("   training and deploying model ")

        fullprocess.process_train_deploy()

        ##################Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model

        logger.debug("   running diagnostic and reporting ")
        fullprocess.run_diagnostics_reporting()


if __name__ == "__main__":
    os.chdir("./src")
    execute_fullprocess()
