import json

import mlflow
import tempfile
import os
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


# This automatically reads in the configuration
@hydra.main(config_path="config", config_name='config')
def go(cfg: DictConfig):

     # Steps to execute
    steps_par = cfg['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "ingestion" in active_steps:
            print(f"inside main.py ingestion ")
            filename = os.path.join(hydra.utils.get_original_cwd(), 'src', 'data_ingestion')
            _ = mlflow.run(
                uri = filename,
                entry_point = "main",
                # version='main',
                env_manager="virtualenv",
                parameters={
                    "in_path":  cfg["ingestion"]["ingestion_path"],
                    "in_file":  cfg["ingestion"]["ingestion_filename"],
                    "out_path": cfg["ingestion"]["output_path"],
                    "out_file": cfg["ingestion"]["output_filename"]
                },
            )

        if "basic_cleaning" in active_steps:
            ##################
            # Perform Basic Clearning  
            ##################

            filename = os.path.join(hydra.utils.get_original_cwd(), 'src', 'basic_cleaning')
            _ = mlflow.run(
                uri = filename, 
                entry_point = "main",
                # version = "main",
                env_manager = "conda",
                parameters = {
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description" : "Cleaned data: outliers removed, and review type chnanged",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                }
            )

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################

            filename = os.path.join(hydra.utils.get_original_cwd(), 'src', 'data_check')
            _ = mlflow.run(
                uri = filename,
                entry_point = "main",
                env_manager = "conda",
                parameters = {
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                }
            )

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            filename= f"{config['main']['components_repository']}/train_val_test_split"

            # filename = os.path.join(hydra.utils.get_original_cwd(), 'components', 'train_val_test_split')
            _ = mlflow.run(
                uri = filename,
                entry_point = "main",
                version = "main",
                env_manager = "conda",
                parameters = {
                    "input": "clean_sample.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                }
            )


        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################
            filename = os.path.join(hydra.utils.get_original_cwd(), "src", 'train_random_forest')
            _ = mlflow.run(
                uri = filename,
                entry_point = "main",
                env_manager = "conda",
                parameters = {
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size" : config['modeling']['val_size'],
                    "random_seed" : config['modeling']['random_seed'],
                    "stratify_by" : config['modeling']['stratify_by'],
                    "rf_config" : rf_config,
                    "max_tfidf_features" : config['modeling']['max_tfidf_features'],
                    "output_artifact" : "random_forest_export"
                }
            )



        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################
            
            filename = f"{config['main']['components_repository']}/test_regression_model"
            _ = mlflow.run(
                uri = filename,
                entry_point = "main",
                version = "main",
                parameters = {
                    "mlflow_model" : "random_forest_export:prod",
                     "test_dataset": "test_data.csv:latest" 
                }
            )



if __name__ == "__main__":
    print('inside go')
    go()