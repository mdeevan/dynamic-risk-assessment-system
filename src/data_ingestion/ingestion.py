#!/usr/bin/env python
"""
ingest data from input folder, combine and save to an output folder
"""
import os
import sys
import logging
import argparse
from datetime import datetime

import pandas as pd

# import dagshub
import mlflow

sys.path.append("../")
from lib import utilities

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# mlflow.autolog()


class Ingest_Data:
    """
    create ingest data calls to encapture the data ingestion
    """

    def __init__(
        self,
        p_ingestion_path,
        p_ingestion_filename,
        p_out_path,
        p_out_file,
        p_ingested_files_log,
        p_mlflow_logging,
        p_parent_folder="../../",
    ):
        """
        initialize the object instance
        """

        self.ingestion_path = p_ingestion_path
        self.ingestion_filename = p_ingestion_filename
        self.out_path = p_out_path
        self.out_file = p_out_file

        self.ingested_files_log = p_ingested_files_log
        self.mlflow_logging = p_mlflow_logging

        self.parent_folder = p_parent_folder

    def process_files(self) -> str:
        """
        read in the files, combine, drop duplicates and save the file

        INPUT:
            uses instance level variables

                    RETURN:
            path of the output file
        """

        df = pd.DataFrame()
        # parent_folder = "../"
        files = []
        try:
            if "*" in self.ingestion_filename:
                logging.debug(
                    "\nrun-data-ingestion: self.infile %s", self.ingestion_filename
                )

                source_folder = os.path.join(self.parent_folder, self.ingestion_path)

                # files = [f for f in os.listdir(source_folder)
                # if os.path.isfile(self.__get_filename(f))]
                files = [
                    f
                    for f in os.listdir(source_folder)
                    if os.path.isfile(
                        utilities.get_filename(
                            p_filename=f,
                            p_parent_folder=self.parent_folder,
                            p_path=self.ingestion_path,
                        )
                    )
                ]

                logging.debug("filename : %s", files)

                for file in files:
                    # filename = self.__get_filename(file)
                    filename = utilities.get_filename(
                        p_filename=file,
                        p_parent_folder=self.parent_folder,
                        p_path=self.ingestion_path,
                    )

                    logger.debug("run-data-ingestion: filename : %s", filename)
                    # df_new = self.__read_file(filename)
                    df_new = utilities.read_file(filename)
                    df = pd.concat([df, df_new], axis=0)
            else:
                files.append(self.ingestion_filename)

                logger.debug(
                    "run-data-ingestion: filename : %s", self.ingestion_filename
                )
                # filename = self.__get_filename(self.ingestion_filename)
                filename = utilities.get_filename(
                    p_filename=self.ingestion_filename,
                    p_parent_folder=self.parent_folder,
                    p_path=self.ingestion_path,
                )

                # df = self.__read_file(filename)
                df = utilities.read_file(filename)

        except Exception as err:
            logger.error("error reading file %s", err)
            raise

        try:
            # os.mkdir(os.path.join(self.parent_folder, self.out_path))
            utilities.make_dir(self.parent_folder, self.out_path)

        except Exception as err:
            # folder exists so we don't need to abort processing
            logger.info("error creating directory : %s ", err)

        # out = self.__get_filename(self.out_file, self.out_path)
        out = utilities.get_filename(
            p_filename=self.out_file,
            p_parent_folder=self.parent_folder,
            p_path=self.out_path,
        )

        logger.debug("run-data-ingestion: out filename %s", out)
        try:
            df = df.drop_duplicates().reset_index()
            df.to_csv(out, index=False)

        except Exception as err:
            logger.error("error %s in creating outfile %s", err, out)
            raise

        logger.debug("Saving ingested metadata")
        # outlog_file = self.__get_filename(self.ingested_files_log, self.out_path)
        outlog_file = utilities.get_filename(
            p_filename=self.ingested_files_log,
            p_parent_folder=self.parent_folder,
            p_path=self.out_path,
        )
        with open(outlog_file, "w", encoding="utf-8") as f:
            f.write("date, folder, file\n")
            exec_date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            # f.write(f"Ingestion date: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}\n")
            for file in files:
                # path, file = self.__get_filename(file).rsplit("/", 1)
                filename = utilities.get_filename(
                    p_filename=file,
                    p_parent_folder=self.parent_folder,
                    p_path=self.out_path,
                )
                path, file = filename.rsplit("/", 1)
                # f.write(self.__get_filename(file)+"\n")

                f.write(f"{exec_date},{path},{file}\n")

        return out


def go(args):
    """
    main routine to execute
    """

    p_ingestion_path = args.ingestion_path
    p_ingestion_filename = args.ingestion_filename
    p_out_path = args.out_path
    p_out_file = args.out_file
    p_ingested_files_log = args.ingested_files_log
    p_mlflow_logging = args.mlflow_logging
    p_parent_folder = "../../"

    ingest_data = Ingest_Data(
        p_ingestion_path,
        p_ingestion_filename,
        p_out_path,
        p_out_file,
        p_ingested_files_log,
        p_mlflow_logging,
        p_parent_folder,
    )

    # print("Tracking URI:", mlflow.get_tracking_uri())
    # print("Env vars:", {k: v for k, v in os.environ.items() if "MLFLOW" in k or "DAGSHUB" in k})

    # mlflow.set_experiment("data-ingestion")
    if args.mlflow_logging:
        with mlflow.start_run():
            logger.debug("inside mlflow_start_run")
            logger.debug("inside go and in scope of mlflow.start_run")

            try:

                path = ingest_data.process_files()

                mlflow.log_param("out_filename", args.out_file)
                mlflow.log_artifact(path)

            except Exception as err:
                logger.error("Error ingesting data %s", err)
    else:
        try:
            logger.info("ingestion without logging")
            path = ingest_data.process_files()

        except Exception as err:
            logger.error("Ingestion - w/o logging Error %s", err)
            return False


if __name__ == "__main__":

    logger.info("inside run_data_ingestion.py")

    parser = argparse.ArgumentParser(description="ingest data from the input folder")

    parser.add_argument(
        "--ingestion_path",
        type=str,
        help="path to the data file for ingestion",
        required=True,
    )

    parser.add_argument(
        "--ingestion_filename",
        type=str,
        help="filename to ingest, asterisk (*) means all files in the folder",
        required=True,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        help="path where the processed file to be stored",
        required=True,
    )

    parser.add_argument(
        "--out_file", type=str, help="name of the outfile", required=True
    )

    parser.add_argument(
        "--ingested_files_log",
        type=str,
        help="log of the files ingested",
        required=True,
    )

    parser.add_argument(
        "--mlflow_logging",
        type=bool,
        help="mlflow logging enable or disabled",
        required=True,
    )
    parser.add_argument(
        "--diagnostic",
        type=bool,
        help="is ingestion for diagnostics?",
        default=False,
        required=False,
    )

    args = parser.parse_args()

    go(args)
