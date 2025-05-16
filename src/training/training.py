#!/usr/bin/env python
"""
train the model based on the finaldata.csv created in the previous step of the pipeline
"""
import sys
import ast
import logging
import argparse
import pickle
from sklearn.linear_model import LogisticRegression
import mlflow

sys.path.append("../")
from lib import utilities


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Train_Model():
    '''
    class describing the training model
    '''

    def __init__(
        self,
        p_ingested_data_path,
        p_ingestion_filename,
        p_out_path,
        p_out_model,
        p_parent_folder,
        p_num_features,
        p_lr_params,
        p_mlflow_logging,
    ):

        self.ingested_data_path = p_ingested_data_path
        self.ingestion_filename = p_ingestion_filename
        self.out_path = p_out_path
        self.out_model = p_out_model
        self.parent_folder = p_parent_folder
        self.mlflow_logging = p_mlflow_logging

        self.num_features = p_num_features

        logger.debug("p_num_featues : \n{p_num_features}")
        logger.debug("p_lr_params : \n{p_lr_params}")

        self.C = p_lr_params["C"]
        # self.class_weight  = p_lr_params['class_weight']
        self.dual = p_lr_params["dual"]
        self.fit_intercept = p_lr_params["fit_intercept"]
        self.intercept_scaling = p_lr_params["intercept_scaling"]
        self.l1_ratio = p_lr_params["l1_ratio"]
        self.max_iter = p_lr_params["max_iter"]
        # self.multi_class = p_lr_params['multi_class']
        self.n_jobs = p_lr_params["n_jobs"]
        self.penalty = p_lr_params["penalty"]
        self.random_state = p_lr_params["random_state"]
        self.solver = p_lr_params["solver"]
        self.tol = p_lr_params["tol"]
        self.verbose = p_lr_params["verbose"]
        self.warm_start = p_lr_params["warm_start"]

        # with open("params.yml") as stream:
        #     try:
        #         cfg = yaml.safe_load(stream)

        #         self.num_features = cfg['num_features']

        #     except yaml.YAMLError as exc:
        #         print(exc)

    def train_model(self) -> str:
        """
        read in the files, combine, drop duplicates and save the file

        INPUT:
            uses instance level variables

        RETURN:
            path of the output file
        """

        logging.debug("Train model class method")
        try:
            # filename = self.__get_filename(self.ingestion_filename)
            filename = utilities.get_filename(
                p_filename=self.ingestion_filename,
                p_parent_folder=self.parent_folder,
                p_path=self.ingested_data_path,
            )
            # df = self.__read_file(filename)
            logging.debug("training.py filename : %s", filename)
            df = utilities.read_file(filename)

        except Exception as err:
            logger.info("error reading file %s", err)
            raise

        logging.debug("filename : %s", filename)
        try:
            lr = LogisticRegression(
                C=self.C,
                # class_weight =  self.class_weight ,
                dual=self.dual,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                penalty=self.penalty,
                random_state=self.random_state,
                solver=self.solver,
                tol=self.tol,
                verbose=self.verbose,
                warm_start=self.warm_start,
            )
        except Exception as err:
            logger.error("LR Error %s", err)

        try:
            logging.debug(
                "num_features = %s \n %s",
                type(self.num_features), self.num_features
            )
            X = df[self.num_features]
            y = X.pop("exited")

        except Exception as err:
            logger.error("dataframe error : %s", err)
            raise

        logging.debug("y : ", y.head())

        try:

            logging.debug("fitting model")
            model = lr.fit(X, y)

        except Exception as err:
            logger.error("error model training : %s", err)
            raise

        logging.debug("\ncreate directory\n")
        try:

            # folder_path = os.path.join(self.parent_folder, self.out_path)
            # print(f"folder_path : {folder_path}")
            # os.mkdir(folder_path)

            utilities.make_dir(self.parent_folder, self.out_path)

        except Exception as err:
            # folder exists so we don't need to abort processing
            logger.error("train_model: error creating directory : %s ", err)
            raise

        # out = self.__get_filename(self.out_model, self.out_path)
        out = utilities.get_filename(
            p_filename=self.out_model,
            p_parent_folder=self.parent_folder,
            p_path=self.out_path,
        )

        # print(f"out = {out}")
        logging.debug("training: Saving model : %s", out)
        try:
            pickle.dump(model, open(out, "wb"))

        except Exception as err:
            logging.debug("error %s in creating outfile %s", err, out)
            raise

        return out


def go(args):
    '''
        main routine to invoke the training 
    '''

    logging.debug("\nInside traingin .go")
    logging.debug("num_features : %s", args.num_features)
    logging.debug("lr_params : %s", {args.lr_params})

    logging.debug("\n ")

    train_model = Train_Model(
        p_ingested_data_path=args.ingested_data_path,
        p_ingestion_filename=args.ingestion_filename,
        p_out_path=args.out_path,
        p_out_model=args.out_model,
        p_parent_folder="../../",
        p_num_features=ast.literal_eval(args.num_features),
        p_lr_params=ast.literal_eval(args.lr_params.replace("'None'", "None")),
        p_mlflow_logging=args.mlflow_logging,
    )

    logging.debug("args.mlflow_logging : %s", train_model.mlflow_logging)
    if train_model.mlflow_logging:
        with mlflow.start_run():
            logging.debug("inside mlflow_start_run")
            logging.debug("inside go and in scope of mlflow.start_run")

            try:
                path = train_model.train_model()

                mlflow.log_param("out_filename", path)
                mlflow.log_artifact(path)

            except Exception as err:
                logger.error("Train Model Error %s", err)
                return False
    else:
        try:
            logger.info("training without logging")
            path = train_model.train_model()

        except Exception as err:
            logger.error("Train Model - w/o logging Error %s", err)
            return False

    return True


if __name__ == "__main__":

    logging.info("inside training.py")

    logging.debug("\n\n\n")
    for arg in sys.argv:
        logging.debug(arg)

    logging.debug("\n\n\n")
    parser = argparse.ArgumentParser(description="model training")

    parser.add_argument(
        "--ingested_data_path",
        type=str,
        help="path to the source data file",
        required=True,
    )

    parser.add_argument(
        "--ingestion_filename",
        type=str,
        help="source data file to process",
        required=True,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        help="path where the model will be stored",
        required=True,
    )

    parser.add_argument(
        "--out_model", type=str, help="model name to use in saving", required=True
    )

    parser.add_argument(
        "--num_features", type=str, help="modeling parameters", required=False
    )

    parser.add_argument(
        "--lr_params",
        type=str,
        help="logistic regression model tuning parameters",
        required=False,
    )

    parser.add_argument(
        "--mlflow_logging",
        type=bool,
        help="mlflow logging enable or disabled",
        required=False,
    )
    parser.add_argument(
        "--diagnostic",
        type=bool,
        help="is ingestion for diagnostics?",
        default=False,
        required=False,
    )

    cmd_args = parser.parse_args()

    logging.debug("inside training main -> %s", cmd_args)
    logging.debug("training.py args = %s", cmd_args)

    go(cmd_args)
