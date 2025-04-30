import os
import json

import mlflow
import dagshub
import tempfile
import hydra
from omegaconf import DictConfig

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
            "in_path": cfg["ingestion"]["ingestion_path"],  
            "in_file": cfg["ingestion"]["ingestion_filename"],
            "out_path": cfg["ingestion"]["ingested_data_path"],
            "out_file": cfg["ingestion"]["ingested_filename"],
            "mlflow_logging": cfg["main"]["mlflow_logging"]
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

            "in_path": cfg["ingestion"]["ingested_data_path"],
            "in_file": cfg["ingestion"]["ingested_filename"],
            "out_path": cfg["ingestion"]["prod_deployment_path"],
            "out_model": cfg["ingestion"]["output_model_name"],
            "num_features": cfg["num_features"],
            "lr_params": cfg["logistic_regression_params"][0],
            "mlflow_logging": cfg["main"]["mlflow_logging"]
        },
    )

def __run_diagnostics(filename, cfg):
    return mlflow.run(
        uri=filename,
        entry_point="main",
        env_manager="conda",
        parameters={
            "model_path_name": cfg["ingestion"]["prod_deployment_path"],
            "model_file_name": cfg["ingestion"]["output_model_name"],
            "data_path_name" : cfg["ingestion"]["test_data_path"],
            "test_prediction_output" : cfg["ingestion"]["test_prediction_output"],
            "num_features": cfg["num_features"],
            "mlflow_logging": cfg["main"]["mlflow_logging"]
        },
    )

# TO-OD - mlflow_logging flag is coming in as boolean and argparse consider it positive
# when the flag is passed, and ignores the values pssed to the attribute
# Change the type to string OR
# refactor and replace argparse with click, which provides additional types


# This automatically reads in the configuration
@hydra.main(config_path="config", config_name="config")
def go(cfg: DictConfig):


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

            # _ = mlflow.run(
            #     uri=filename,
            #     entry_point="main",
            #     # version='main',
            #     # env_manager="virtualenv",
            #     env_manager="conda",
            #     parameters={
            #         "in_path": cfg["ingestion"]["ingestion_path"],  
            #         "in_file": cfg["ingestion"]["ingestion_filename"],
            #         "out_path": cfg["ingestion"]["ingested_data_path"],
            #         "out_file": cfg["ingestion"]["ingested_filename"],
            #         "mlflow_logging": cfg["main"]["mlflow_logging"]
            #         # "modeling": cfg["modeling"]
            #     },
            # )

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

            # _ = mlflow.run(
            #     uri=filename,
            #     entry_point="main",
            #     env_manager="conda",
            #     parameters={
            #         #  out path and outfile are where the ingested file is stored, 
            #         # from previous 'ingestion' step

            #         "in_path": cfg["ingestion"]["ingested_data_path"],
            #         "in_file": cfg["ingestion"]["ingested_filename"],
            #         "out_path": cfg["ingestion"]["prod_deployment_path"],
            #         "out_model": cfg["ingestion"]["output_model_name"],
            #         "num_features": cfg["num_features"],
            #         "lr_params": cfg["logistic_regression_params"][0],
            #         "mlflow_logging": cfg["main"]["mlflow_logging"]
            #     },
            # )


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




        if "basic_cleaning" in active_steps:
            ##################
            # Perform Basic Clearning
            ##################

            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "basic_cleaning"
            )
            _ = mlflow.run(
                uri=filename,
                entry_point="main",
                # version = "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Cleaned data: outliers removed, and review type chnanged",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################

            filename = os.path.join(hydra.utils.get_original_cwd(), "src", "data_check")
            _ = mlflow.run(
                uri=filename,
                entry_point="main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            filename = f"{config['main']['components_repository']}/train_val_test_split"

            # filename = os.path.join(hydra.utils.get_original_cwd(), 'components', 'train_val_test_split')
            _ = mlflow.run(
                uri=filename,
                entry_point="main",
                version="main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(
                    dict(config["modeling"]["random_forest"].items()), fp
                )  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################
            filename = os.path.join(
                hydra.utils.get_original_cwd(), "src", "train_random_forest"
            )
            _ = mlflow.run(
                uri=filename,
                entry_point="main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                },
            )

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            filename = (
                f"{config['main']['components_repository']}/test_regression_model"
            )
            _ = mlflow.run(
                uri=filename,
                entry_point="main",
                version="main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    print("inside go")

    dagshub.init(
        repo_owner="mdeevan", repo_name="dynamic-risk-assessment-system", mlflow=True
    )

    print("beore mlflow_start_run")

    # mlflow.set_tracking_uri("https://dagshub.com/mdeevan/dynamic-risk-assessment-system.mlflow")
    # mlflow.set_experiment("my_experiment")

    print("Tracking URI:", mlflow.get_tracking_uri())
    print(
        "Env vars:",
        {k: v for k, v in os.environ.items() if "MLFLOW" in k or "DAGSHUB" in k},
    )

    go()
