#!/usr/bin/env python
"""
train the model based on the finaldata.csv created in the previous step of the pipeline
"""
import argparse
import logging
import os
import pandas as pd
import yaml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Ingest_Data():

    def __init__(self, args):
        self.in_path = args.in_path
        self.in_file = args.in_file
        self.out_path = args.out_path
        self.out_model = args.out_model
        self.parent_folder = "../../"
        self.num_features =""

        lrp=args[0]
        self.C  = lrp.C 
        self.class_weight  = lrp.class_weight 
        self.dual  = lrp.dual 
        self.fit_intercept  = lrp.fit_intercept 
        self.intercept_scaling = lrp.intercept_scaling  
        self.l1_ratio  = lrp.l1_ratio 
        self.max_iter  = lrp.max_iter 
        self.multi_class = lrp.multi_class  
        self.n_jobs   = lrp.n_jobs  
        self.penalty  = lrp.penalty 
        self.random_state   = lrp.random_state  
        self.solver   = lrp.solver  
        self.tol   = lrp.tol  
        self.verbose  = lrp.verbose 
        self.warm_start  = lrp.warm_start 


        with open("params.yml") as stream:
            try:
                cfg = yaml.safe_load(stream)

                self.num_features = cfg['num_features']

            except yaml.YAMLError as exc:
                print(exc)


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
        logger.info(f"_-get-filename : {filename}")
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


    def train_model(self) -> bool:
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
            logger.info(f"error reading file %s", err)
            raise

        lr = LogisticRegression(self.C ,
                                self.class_weight , 
                                self.dual , 
                                self.fit_intercept ,
                                self.intercept_scaling , 
                                self.l1_ratio , 
                                self.max_iter ,
                                self.multi_class , 
                                self.n_jobs , 
                                self.penalty ,
                                self.random_state , 
                                self.solver , 
                                self.tol , 
                                self.verbose ,
                                self.warm_start )

        X = df[self.num_features]
        y = X.pop('exited')

        model = lr.fit(X, y)

        try:
            os.mkdir(os.path.join(self.parent_folder, self.out_path))

        except Exception as err:
            # folder exists so we don't need to abort processing
            logger.info(f"train_model: error creating directory : %s ", err)



        out = self.__get_filename(self.out_file, self.out_path)


        print(f"training: out filename {out}")
        try:
            pickle.dump(model, open(out, 'wb'))  

        except Exception as err:
            print("error %s in creating outfile %s", err, out)
            raise


#       save model on the filesystem 
        logging.info("training: Saving model metadata")
        outmodel_file = self.__get_filename(self.out_model, self.out_path)



        return True


def go(args):


    print("\nInside traingin .go")
    print(f"num_features : {args.num_features}")
    print(f"lr_params : {args.lr_params}")

    print("\n ")

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    return True;

if __name__ == "__main__":

    print("inside training.py")
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
        "--num_features", 
        type=str,
        help='modeling parameters',
        required=False
    )

    parser.add_argument(
        "--lr_params", 
        type=str,
        help='logistic regression model tuning parameters',
        required=False
    )


    args = parser.parse_args()

    go(args)
