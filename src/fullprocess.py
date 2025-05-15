

# import o_training
# import o_scoring
# import deployment
# import o_diagnostics
# import reporting

import yaml
import logging
import pandas as pd
import os
import subprocess

from lib import utilities


class Fullprocess():

    def __init__(self):

        try:
            with open("./config/config.yaml") as stream:
                cfg = yaml.safe_load(stream)

        except Exception as err:
            cfg = None
            logging.error(f"FATAL: Error initialization configuration %s", err)

        # ingested files from the last execution
        self.parent_folder = ""
        self.prod_folder = cfg['prod_deployment']['prod_deployment_path']
        self.ingested_files = cfg['ingestion']['ingested_files_log']

        # files to ingest as part of new execution
        self.input_file_path = cfg['ingestion']['ingestion_path']
        self.input_file_name = cfg['ingestion']['ingestion_filename']

        # previous model score file
        self.model_folder = cfg['training']['output_model_path']
        self.score_filename = cfg['scoring']['score_filename']


    def read_ingested_files(self) ->list:
        ingested_file_path = utilities.get_filename(self.ingested_files, 
                                                    p_path=self.prod_folder)
        df = utilities.read_file(ingested_file_path)
        ingested_files_list = df.file.to_list()

        return ingested_files_list

    def read_input_files(self, ingested_files_list : list) -> bool:

        if ("*" in self.input_file_name):
            input_folder = os.path.join(self.parent_folder,
                                        self.input_file_path,
                                        )

            print(f"input folder : {input_folder}")
            files = [f for f in os.listdir(input_folder) 
                        if os.path.isfile(utilities.get_filename(p_filename= f,
                                                                p_parent_folder=self.parent_folder,
                                                                p_path=self.input_file_path
                                                                ))]
        else:
            files = self.input_file_name

        new_files = False
        for file in files:
            if file not in ingested_files_list:
                new_files = True

        print(f"new_files : {new_files}")

        return new_files 
    

    def get_score(self, source:str='deployed') -> float:

        score_folder = self.prod_folder if source=='deployed' else self.model_folder

        fn = utilities.get_filename(self.score_filename, 
                                    p_parent_folder=self.parent_folder,
                                    p_path=score_folder)

        score = -1
        with open(fn, "r") as f:
            score = f.read()

        print(f"{source} score : {score}")

        return score


    def process_new_files(self):
        command = ['mlflow','main',".", "-P", "steps='ingestion,scoring'"]
        subprocess.run(command, cwd="../")

        new_score = self.get_score('new')

        return new_score

    def process_train_deploy(self):
        command = ['mlflow','main',".", "-P", "steps='training,scoring,deployment'"]
        subprocess.run(command, cwd="../")

        return True

    def run_diagnostics_reporting(self):
        commands = [['python','src/diagnostics.py' ],
                    ['python','src/reporting.py' ]
                    ]


        for command in commands:
            subprocess.run(command)

        return True


def execute_fullprocess():
    fullprocess = Fullprocess()

    ##################Check and read new data
    #first, read ingestedfiles.txt
    ingested_files_list = fullprocess.read_ingested_files()


    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    ##################Deciding whether to proceed, part 1

    new_files_exist = fullprocess.read_input_files(ingested_files_list)

    #if you found new data, you should proceed. otherwise, do end the process here
    new_score = -1.0
    deployed_score = -1.0
    if new_files_exist:
        new_score = fullprocess.process_new_files()

        print(f"new score : {new_score}, type : {type(new_score)}")


    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    deployed_score = fullprocess.get_score('deployed')

    print(f"deployed Score : {deployed_score} , type : {type(deployed_score)}")

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here

    model_drift = float(new_score) < float(deployed_score)
    if (new_files_exist and model_drift) :
        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script

        print("model drifted ")
        print("   training and deploying model ")

        fullprocess.process_train_deploy()

        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model

        print("   running diagnostic and reporting ")
        fullprocess.run_diagnostics_reporting()


if __name__ == "__main__":
    execute_fullprocess()

