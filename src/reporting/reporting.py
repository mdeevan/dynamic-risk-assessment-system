#!/usr/bin/env python
"""
generate reports
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

    parser = argparse.ArgumentParser(description="generate reports")


    parser.add_argument(
        "--parameter1", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--parameter2", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
