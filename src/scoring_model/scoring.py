#!/usr/bin/env python
"""
Score model metrics on provided test data
"""
import sys
import ast
import argparse
import inspect
import logging
import mlflow
import pandas as pd
from sklearn import metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

sys.path.append("../")

from lib import utilities


class Score_Model:

    def __init__(
        self,
        p_ingested_data_path,
        p_ingested_file_name,
        p_num_features,
        p_model_path_name,
        p_model_file_name,
        p_prod_deployment_path,
        p_report_folder,
        p_prediction_output,
        p_score_filename,
        p_mlflow_logging,
        p_parent_folder,
    ):

        self.ingested_data_path = p_ingested_data_path
        self.ingested_file_name = p_ingested_file_name
        self.num_features = p_num_features
        self.model_path_name = p_model_path_name
        self.model_file_name = p_model_file_name
        self.prod_deployment_path = p_prod_deployment_path
        self.report_folder = p_report_folder
        self.prediction_output = p_prediction_output
        self.score_filename = p_score_filename
        self.mlflow_logging = p_mlflow_logging
        self.parent_folder = p_parent_folder  # "../../"

    def make_predictions(self) -> str:
        func_name = inspect.currentframe().f_code.co_name

        model = utilities.load_model(
            p_model_file_name=self.model_file_name,
            p_parent_folder=self.parent_folder,
            p_model_path_name=self.prod_deployment_path,
        )

        input_file = utilities.get_filename(
            p_filename=self.ingested_file_name,
            p_parent_folder=self.parent_folder,
            p_path=self.ingested_data_path,
        )

        predict_output = utilities.get_filename(
            p_filename=self.prediction_output,
            p_parent_folder=self.parent_folder,
            p_path=self.model_path_name,
        )

        print(f"predict output : {predict_output}")

        df = utilities.read_file(input_file)

        try:
            y_pred = None
            if df is not None:
                X = df[self.num_features]
                y = X.pop("exited")

                y_pred = model.predict(X)

                pd.DataFrame(
                    zip(y, y_pred.tolist()), columns=["target", "predicted"]
                ).to_csv(predict_output, index=False)

        except Exception as err:
            logger.error(f"%s: scoring: error making prediction %s", func_name, err)
            raise

        return predict_output

    def run_model_scoring(self) -> float:

        func_name = inspect.currentframe().f_code.co_name

        logger.info("Loading predictions ")
        try:

            filename = self.make_predictions()
            print(f"{func_name} - predict output : {filename}")

            # filename = utilities.get_filename(p_filename=self.prediction_output,
            #                                   p_parent_folder=self.parent_folder,
            #                                   p_path=self.model_path_name)

            logger.debug(f"filename : {filename}")

            df = utilities.read_file(filename)

            f1_score = metrics.f1_score(df["predicted"], df["target"])

            outfile = utilities.get_filename(
                p_filename=self.score_filename,
                p_parent_folder=self.parent_folder,
                p_path=self.model_path_name,
            )

            with open(outfile, "w+") as f:
                f.write(str(f1_score) + "\n")

        except Exception as err:
            logger.error(f"%s: error running model scoring %s", func_name, err)
            raise

        # print(f"f1_score : {f1_score}, type : {type(f1_score)}")
        return f1_score


def go(args):

    score_model = Score_Model(
        p_ingested_data_path=args.ingested_data_path,
        p_ingested_file_name=args.ingested_file_name,
        p_num_features=ast.literal_eval(args.num_features),
        p_model_path_name=args.model_path_name,
        p_model_file_name=args.model_file_name,
        p_prod_deployment_path=args.prod_deployment_path,
        p_report_folder=args.report_folder,
        p_prediction_output=args.prediction_output,
        p_score_filename=args.score_filename,
        p_mlflow_logging=args.mlflow_logging,
        p_parent_folder="../../",
    )

    if score_model.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")

            try:
                f1_score = score_model.run_model_scoring()

                mlflow.log_metric("fi-score", value=f1_score)

            except Exception as err:
                logger.error(f"Error running model scoring %s", err)
                return False
    else:
        try:
            logger.info("training without logging")
            f1_score = score_model.run_model_scoring()

            mlflow.log_metric("fi-score", value=f1_score)

        except Exception as err:
            logger.error("Error running model scoring w/o logging %s", err)
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="scoring the model")

    parser.add_argument(
        "--ingested_data_path",
        type=str,
        help="path containined the ingested datafile ",
        required=True,
    )

    parser.add_argument(
        "--ingested_file_name", type=str, help="ingested filename ", required=True
    )

    parser.add_argument(
        "--num_features", type=str, help="ingested filename ", required=True
    )

    parser.add_argument("--model_path_name", type=str, help="model name", required=True)

    parser.add_argument("--model_file_name", type=str, help="model name", required=True)

    parser.add_argument(
        "--prod_deployment_path",
        type=str,
        help="production deployment folder",
        required=True,
    )

    parser.add_argument(
        "--report_folder",
        type=str,
        help="folder where the predictions were stored after training",
        required=True,
    )
    parser.add_argument(
        "--prediction_output", type=str, help="prediction filename ", required=True
    )
    parser.add_argument(
        "--score_filename",
        type=str,
        help="filename to store the model scoring - f1 score",
        required=True,
    )
    parser.add_argument(
        "--mlflow_logging",
        type=bool,
        help="mlflow logging enable or disabled",
        required=False,
    )

    args = parser.parse_args()

    go(args)
