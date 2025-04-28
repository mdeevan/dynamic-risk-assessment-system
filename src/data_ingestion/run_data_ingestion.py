#!/usr/bin/env python
"""
ingest data from input folder, combine and save to an output folder
"""
import os
import pandas as pd
import argparse
import logging
import dagshub
import mlflow
from datetime import datetime

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# mlflow.autolog()

class Ingest_Data():

    def __init__(self, args):
        self.in_path = args.in_path
        self.in_file = args.in_file
        self.out_path = args.out_path
        self.out_file = args.out_file
        self.parent_folder = "../../"

    def __get_filename(self, p_filename:str, p_path:str=None) -> None:
        '''
        Form and return a filename
        Input:
                    p_filename : str - filename 
            p_path : str - path where the filename is stored/created

        return:
            None
        '''

        path = self.in_path if (p_path is None) else p_path

        filename = os.path.join(self.parent_folder, path, p_filename)
        print(f"_-get-filename : {filename}")
        return filename

    def __read_file(self, filename:str) -> pd.DataFrame:
        '''
        read csv into panda framework

        INPUT:
            filename : csv files to read
        RETURN:
            pd.DataFrme : panda dataframe
        '''
        return pd.read_csv(filename)

    def process_files(self) -> str:
        '''
        read in the files, combine, drop duplicates and save the file

        INPUT:
            uses instance level variables

                    RETURN:
            path of the output file
        '''

        df = pd.DataFrame()
        # parent_folder = "../"
        files = []
        try:
            if (self.in_file == "*"):
                print(f"\nrun-data-ingestion: self.infile {self.in_file}")

                source_folder = os.path.join(self.parent_folder, self.in_path)

                files = [f for f in os.listdir(source_folder) if os.path.isfile(self.__get_filename(f))]

                print(f"filename : {files}")

                for file in files:
                    filename = self.__get_filename(file)

                    print(f"run-data-ingestion: filename : {filename}")
                    df_new = self.__read_file(filename)
                    df = pd.concat([df, df_new], axis=0)
            else:
                files.append(self.in_file)

                print(f"run-data-ingestion: filename : {self.in_file}")
                filename = self.__get_filename(self.in_file)

                df = self.__read_file(filename)
    
        except Exception as err:
            print("error reading file %s", err)
            raise

        try:
            os.mkdir(os.path.join(self.parent_folder, self.out_path))

        except Exception as err:
            # folder exists so we don't need to abort processing
            logger.info("error creating directory : %s ", err)


        out = self.__get_filename(self.out_file, self.out_path)

        print(f"run-data-ingestion: out filename {out}")
        try:
            df= df.drop_duplicates().reset_index()
            df.to_csv(out, index=False)

        except Exception as err:
            print("error %s in creating outfile %s", err, out)
            raise


        logging.info("Saving ingested metadata")
        outlog_file = self.__get_filename("ingestedfiles.txt", self.out_path)
        with open(outlog_file, "w") as f:
            f.write(f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            for file in files:
                f.write(self.__get_filename(file)+"\n")


        return out


def go(args):

    ingest_data = Ingest_Data(args)

    # print("Tracking URI:", mlflow.get_tracking_uri())
    # print("Env vars:", {k: v for k, v in os.environ.items() if "MLFLOW" in k or "DAGSHUB" in k})

    # mlflow.set_experiment("data-ingestion")
    if args.mlflow_logging:
        with mlflow.start_run():
            print("inside mlflow_start_run")
            print(f"inside go and in scope of mlflow.start_run")
            
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
            logger.error(f"Ingestion - w/o logging Error %s", err)
            return False




if __name__ == "__main__":

    print("inside run_data_ingestion.py")

    parser = argparse.ArgumentParser(description="ingest data from the input folder")

    parser.add_argument(
        "--in_path", 
        type=str, 
        help='path to the data file for ingestion',  
        required=True
    )

    parser.add_argument(
        "--in_file", 
        type=str,
        help='filename to ingest, asterisk (*) means all files in the folder', 
        required=True
    )

    parser.add_argument(
        "--out_path", 
        type=str ,
        help="path where the processed file to be stored",
        required=True
    )

    parser.add_argument(
        "--out_file", 
        type=str,
        help='name of the outfile',
        required=True
    )

    parser.add_argument(
        "--mlflow_logging", 
        type=bool,
        help='mlflow logging enable or disabled',
        required=True
    )

    args = parser.parse_args()

    go(args)
