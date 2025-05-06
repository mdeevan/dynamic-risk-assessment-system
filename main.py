import os
import json

import mlflow
import dagshub
import tempfile
import hydra
from omegaconf import DictConfig
from datetime import datetime

_steps = [
    "ingestion",
    "training",
    "scoring",
    "deployment",
    "diagnostics",
    "reporting",
]


def __run_ingestion(filename, cfg):
    return mlflow.run(
        uri=filename,
        entry_point="main",
        # version='main',
        # env_manager="virtualenv",
        env_manager="conda",
        parameters={
            "ingestion_path"    : cfg["ingestion"]["ingestion_path"],  
            "ingestion_filename": cfg["ingestion"]["ingestion_filename"],
            "out_path"          : cfg["ingestion"]["ingested_data_path"],
            "out_file"          : cfg["ingestion"]["ingested_filename"],
            "ingested_files_log": cfg["ingestion"]["ingested_files_log"],
            "mlflow_logging"    : cfg["main"]["mlflow_logging"]
            # "modeling": cfg["modeling"]
        },
    )

def __run_training(filename, cfg):
    return mlflow.run(
        uri=filename,
        entry_point="main",
        env_manager="conda",
        parameters={
            #  out path and outfile are where the ingested file is stored, 
            # from previous 'ingestion' step

            "ingested_data_path": cfg["ingestion"]["ingested_data_path"],
            "ingestion_filename": cfg["ingestion"]["ingested_filename"],
            "out_path": cfg["training"]["output_model_path"],
            "out_model": cfg["prod_deployment"]["output_model_name"],
            "num_features": cfg["num_features"],
            "lr_params": cfg["logistic_regression_params"][0],
            "mlflow_logging": cfg["main"]["mlflow_logging"]
        },
    )

def __run_scoring_model(filename, cfg):
    return mlflow.run(
        uri=filename,
        entry_point="main",
        env_manager="conda",
        parameters={
            #  out path and outfile are where the ingested file is stored, 
            # from previous 'ingestion' step

            "model_path_name": cfg["training"]["output_model_path"],
            "report_folder": cfg["scoring"]["report_folder"],
            "prediction_output": cfg["scoring"]["prediction_output"],
            "score_filename": cfg["scoring"]["score_filename"],
            "mlflow_logging": cfg["main"]["mlflow_logging"]
        },
    )

def __run_production_deployment(filename, cfg):
    return mlflow.run(
        uri=filename,
        entry_point="main",
        env_manager="conda",
        parameters={
            "model_path_name"     : cfg["training"]["output_model_path"],
            "output_model_name"   : cfg["prod_deployment"]["output_model_name"],
            "score_filename"      : cfg["scoring"]["score_filename"],
            "ingested_data_path"  : cfg["ingestion"]["ingested_data_path"],
            "ingested_filename"   : cfg["ingestion"]["ingested_filename"],
            "ingested_files_log"  : cfg["ingestion"]["ingested_files_log"],
            "prod_deployment_path": cfg["prod_deployment"]["prod_deployment_path"],
            "mlflow_logging"      : cfg["main"]["mlflow_logging"]
        },
    )

def __run_diagnostics(filename, cfg):
    return mlflow.run(
        uri=filename,
        entry_point="main",
        env_manager="conda",
        parameters={
            "model_path_name"  : cfg["diagnostics"]["prod_deployment_path"],
            "model_file_name"  : cfg["diagnostics"]["output_model_name"],
            "data_folder"      : cfg["diagnostics"]["data_folder"],
            "data_files"       : cfg["diagnostics"]["data_files"],
            "report_folder"    : cfg["diagnostics"]["report_folder"],
            "prediction_output": cfg["diagnostics"]["prediction_output"],
            "score_filename"   : cfg["diagnostics"]["score_filename"],
            "num_features"     : cfg["num_features"],
            "lr_params"        : cfg["logistic_regression_params"][0],
            "mlflow_logging"   : cfg["main"]["mlflow_logging"]
        },
    )

# TO-OD - mlflow_logging flag is coming in as boolean and argparse consider it positive
# when the flag is passed, and ignores the values pssed to the attribute
# Change the type to string OR
# refactor and replace argparse with click, which provides additional types


# This automatically reads in the configuration
@hydra.main(config_path="config", config_name="config")
def go(cfg: DictConfig):

    dagshub.init(
        repo_owner="mdeevan", 
        repo_name="dynamic-risk-assessment-system", 
        mlflow=True
    )
    exec_date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    mlflow.set_experiment(f"{exec_date}")

    print("beore mlflow_start_run")

    print(f'steps= {cfg["main"]["steps"]} \
          \nmlflow logging : {cfg["main"]["mlflow_logging"]}')

    # Steps to execute
    steps_par = cfg["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps



    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        ##############################
        #########  ingestion  ########
        ##############################
        if "ingestion" in active_steps:
            print(f"inside main.py ingestion ")
            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "data_ingestion"
            )

            print(
                f'cfg["ingestion"]["output_filename"] :{cfg["ingestion"]["output_filename"]}'
            )
            print(f"filename : {filename}")

            _ = __run_ingestion(filename, cfg)

        ##############################
        #########  Training   ########
        ##############################
        if "training" in active_steps:
            print(f"inside main.py training ")
            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "training"
            )

            print(
                f'cfg["ingestion"]["output_filename"] :{cfg["ingestion"]["output_filename"]}'
            )
            print(f"filename : {filename}")

            _ = __run_training(filename, cfg)



        #################################
        #########  Score Model   ########
        #################################
        if "scoring" in active_steps:
            print(f"inside main.py scoring ")
            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "scoring_model"
            )

            print(
                f'cfg["ingestion"]["output_filename"] :{cfg["ingestion"]["output_filename"]}'
            )
            print(f"filename : {filename}")

            _ = __run_scoring_model(filename, cfg)


        ###########################################
        #########  Production deployment   ########
        ###########################################
        if "deployment" in active_steps:
            print(f"inside main.py production deployment ")
            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "deployment"
            )

            print(
                f'cfg["ingestion"]["output_filename"] :{cfg["ingestion"]["output_filename"]}'
            )
            print(f"filename : {filename}")

            _ = __run_production_deployment(filename, cfg)


        #################################
        #########  Diagnostics   ########
        #################################
        if "diagnostics" in active_steps:
            print(f"inside main.py diagnostics ")
            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "diagnostics"
            )

            print(
                f'cfg["ingestion"]["output_filename"] :{cfg["ingestion"]["output_filename"]}'
            )
            print(f"filename : {filename}")

            _ = __run_diagnostics(filename, cfg)





if __name__ == "__main__":
    print("inside go")

    # dagshub.init(
    #     repo_owner="mdeevan", 
    #     repo_name="dynamic-risk-assessment-system", 
    #     mlflow=True
    # )

    # print("beore mlflow_start_run")

    # # mlflow.set_tracking_uri("https://dagshub.com/mdeevan/dynamic-risk-assessment-system.mlflow")
    # mlflow.set_experiment("my_experiment")

    print("Tracking URI:", mlflow.get_tracking_uri())
    print(
        "Env vars:",
        {k: v for k, v in os.environ.items() if "MLFLOW" in k or "DAGSHUB" in k},
    )

    go()
