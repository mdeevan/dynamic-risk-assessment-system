#!/usr/bin/env python
"""
train the model based on the finaldata.csv created in the previous step of the pipeline
"""
import argparse
import logging
import os
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Ingest_Data():

    def __init__(self, args):
        self.in_path = args.in_path
        self.in_file = args.in_file
        self.out_path = args.out_path
        self.out_model = args.out_model
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

    def train_model(self) -> str:
        '''
        read in the files, combine, drop duplicates and save the file

        INPUT:
            uses instance level variables

        RETURN:
            path of the output file
        '''

        try:
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


    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    return True;

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="model training")

    parser.add_argument(
        "--in_path", 
        type=str,
        help="path to the source data file",
        required=True
    )

    parser.add_argument(
        "--in_file", 
        type=str,
        help="source data file to process",
        required=True
    )

    parser.add_argument(
        "--out_path", 
        type=str,
        help="path where the model will be stored",
        required=True
    )

    parser.add_argument(
        "--out_model", 
        type=str,
        help="model name to use in saving",
        required=True
    )

    parser.add_argument(
        "--numeric_cols", 
        type=str,
        help="columns with numeric datatypes",
        required=True
    )


    args = parser.parse_args()

    go(args)
