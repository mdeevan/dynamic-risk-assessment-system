#!/usr/bin/env python
"""
Make predictions on test data with the newly created model to diagnose problem and evaluate results
"""
import argparse
import logging
import dagshub
import mlflow


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with mlflow.start_run():
        print("inside mlflow_start_run")
        print(f"inside go and in scope of mlflow.start_run")
        
        try:

    ######################
    # YOUR CODE HERE     #
    ######################

            mlflow.log_param("out_filename", args.out_file)
            mlflow.log_artifact(path)

        except Exception as err:
            logger.error("Error   %s", err)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="perform diagnostic by testing model")


    parser.add_argument(
        "--model_path", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "-- model_name", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
