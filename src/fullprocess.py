

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


try:
    with open("./config/config.yaml") as stream:
        cfg = yaml.safe_load(stream)

except Exception as err:
    cfg = None
    logging.error(f"FATAL: Error initialization configuration %s", err)


# ingested files from the last execution
parent_folder = ""
prod_folder = cfg['prod_deployment']['prod_deployment_path']
ingested_files = cfg['ingestion']['ingested_files_log']

# files to ingest as part of new execution
input_file_path = cfg['ingestion']['ingestion_path']
input_file_name = cfg['ingestion']['ingestion_filename']

# previous model score file
score_folder = cfg['training']['output_model_path']
score_filename = cfg['scoring']['score_filename']


def read_ingested_files() ->list:
    ingested_file_path = utilities.get_filename(ingested_files, p_path=prod_folder)
    df = utilities.read_file(ingested_file_path)
    ingested_files_list = df.file.to_list()

    return ingested_files_list




##################Check and read new data
#first, read ingestedfiles.txt
ingested_files_list = read_ingested_files()


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
if ("*" in input_file_name):
    input_folder = os.path.join(parent_folder,
                                input_file_path,
                                )

    print(f"input folder : {input_folder}")
    files = [f for f in os.listdir(input_folder) 
                if os.path.isfile(utilities.get_filename(p_filename= f,
                                                         p_parent_folder=parent_folder,
                                                         p_path=input_file_path
                                                        ))]

else:
    files = input_file_name

new_files = False
for file in files:
    if file not in ingested_files_list:
        new_files = True


print(f"new_files : {new_files}")
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if new_files:
    command = ['mlflow','main',"."]
    subprocess.run(command, cwd="../")



##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
fn = utilities.get_filename(score_filename, 
                            p_parent_folder=parent_folder,
                            p_path=score_folder)

original_score = 0
with open(fn, "r") as f:
    original_score = f.read()

print(f"original score : {original_score}")


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







