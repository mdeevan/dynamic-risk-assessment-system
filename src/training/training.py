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
import mlflow
import json
import ast
import sys

sys.path.append("../")
from lib import utilities


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Train_Model():

    def __init__(self, args):
        self.ingested_data_path = args.ingested_data_path
        self.ingestion_filename = args.ingestion_filename
        self.out_path = args.out_path
        self.out_model = args.out_model
        self.parent_folder = "../../"
        self.num_features =""
        self.mlflow_logging = args.mlflow_logging

        logging.debug("\n training.py -> incoming:")
        logging.debug(f"args.num_features :{type(args.num_features)} -> {args.num_features}")
        logging.debug("\n")
        logging.debug(f"args.lr_params : {type(args.lr_params)} -> {args.lr_params}")
        logging.debug("\n")

        logging.debug("\n training.py -> conversion:")
        
        # num_features is a list parameter with string values but passed in as string.
        # convering the string into a list with ast.literal_eval 

        logging.debug(f"training.py args : \n{args}")

        num_features = ast.literal_eval(args.num_features) 
        logging.debug(f"args.num_features :{type(num_features)} -> {num_features}")
        logging.debug("\n")

        # lr_params is a dictionary of mixed types but passed in as string.
        # convering the string into a list with ast.literal_eval 
        lr_params = ast.literal_eval(args.lr_params.replace("'None'", 'None'))
        logging.debug(f"args.lr_params : {type(lr_params)} -> {lr_params}")
        logging.debug("\n")

        # nf=args.num_features.replace("'", '"')
        # print (f"type of nf: {type(nf)}")
        # print(f"nf: {nf}")

        # # nf = nf.replace("[","").replace("]","")
        # self.num_features = json.loads(nf)

        # print (f"type of num_features {type(self.num_features)}")
        # print(f"num_features {self.num_features}")

        # # print(f"args : {args}")
        # # lr_params = args.lr_params
        # # print(f"lr_params {lr_params}")
        # # print(f"lr_params-type {type(lr_params)}")

        # lrp = json.loads( args.lr_params.replace("'", '"' ).replace("True",'"True"').replace("False",'"False"') )
        # # lrp=args.lr_params
        # print(f"args -lrp {lrp}")

        # print(f"lrp type {type(lrp)}")
        # print(f"args -lrp - C{lrp['C']}")

        self.num_features = num_features
        
        self.C  = lr_params['C'] 
        self.class_weight  = lr_params['class_weight'] 
        self.dual  = lr_params['dual'] 
        self.fit_intercept  = lr_params['fit_intercept'] 
        self.intercept_scaling = lr_params['intercept_scaling']  
        self.l1_ratio  = lr_params['l1_ratio'] 
        self.max_iter  = lr_params['max_iter'] 
        # self.multi_class = lr_params['multi_class']  
        self.n_jobs   = lr_params['n_jobs']  
        self.penalty  = lr_params['penalty'] 
        self.random_state   = lr_params['random_state']  
        self.solver   = lr_params['solver']  
        self.tol   = lr_params['tol']  
        self.verbose  = lr_params['verbose'] 
        self.warm_start  = lr_params['warm_start'] 


        # with open("params.yml") as stream:
        #     try:
        #         cfg = yaml.safe_load(stream)

        #         self.num_features = cfg['num_features']

        #     except yaml.YAMLError as exc:
        #         print(exc)


    # def __get_filename(self, p_filename:str, p_path:str=None) -> str:
    #     '''
    #     Form and return a filename
    #     Input:
    #                 p_filename : str - filename 
    #         p_path : str - path where the filename is stored/created

    #     return:
    #         None
    #     '''

    #     path = self.ingested_data_path if (p_path is None) else p_path

    #     filename = os.path.join(self.parent_folder, path, p_filename)
    #     logger.info(f"_-get-filename : {filename}")
    #     return filename


    # def __read_file(self, filename:str) -> pd.DataFrame:
    #     '''
    #     read csv into panda framework

    #     INPUT:
    #         filename : csv files to read
    #     RETURN:
    #         pd.DataFrme : panda dataframe
    #     '''
    #     return pd.read_csv(filename)


    def train_model(self) -> str:
        '''
        read in the files, combine, drop duplicates and save the file

        INPUT:
            uses instance level variables

        RETURN:
            path of the output file
        '''

        logging.debug("Train model class method")
        try:
            # filename = self.__get_filename(self.ingestion_filename)
            filename = utilities.get_filename(p_filename=self.ingestion_filename,
                                              p_parent_folder=self.parent_folder,
                                              p_path=self.ingested_data_path)
            # df = self.__read_file(filename)
            logging.debug(f"training.py filename : {filename}")
            df = utilities.read_file(filename)
    
        except Exception as err:
            logger.info(f"error reading file %s", err)
            raise

        logging.debug(f"filename : {filename}")
        try:
            lr = LogisticRegression(C = self.C ,
                                    class_weight =  self.class_weight , 
                                    dual =  self.dual , 
                                    fit_intercept = self.fit_intercept ,
                                    intercept_scaling = self.intercept_scaling , 
                                    l1_ratio =  self.l1_ratio , 
                                    max_iter = self.max_iter ,
                                    n_jobs =  self.n_jobs , 
                                    penalty = self.penalty ,
                                    random_state =  self.random_state , 
                                    solver =  self.solver , 
                                    tol =  self.tol , 
                                    verbose = self.verbose ,
                                    warm_start = self.warm_start )
        except Exception as err:
            logger.error(f"LR Error : {lr}")

        try:
            logging.debug(f"num_features = {type(self.num_features)} \n {self.num_features}")
            X = df[self.num_features]
            y = X.pop('exited')

        except Exception as err:
            logger.error(f"dataframe error : {err}")
            raise

        logging.debug(f"y : {y.head()}")

        try:

            logging.debug("fitting model")
            model = lr.fit(X, y)    
        
        except Exception as err:
            logger.error(f"error model training : {err}")
            raise

        logging.debug("\ncreate directory\n")
        try:

            # folder_path = os.path.join(self.parent_folder, self.out_path)
            # print(f"folder_path : {folder_path}")
            # os.mkdir(folder_path)

            utilities.make_dir(self.parent_folder, self.out_path)

        except Exception as err:
            # folder exists so we don't need to abort processing
            logger.error(f"train_model: error creating directory : %s ", err)
            raise

        # out = self.__get_filename(self.out_model, self.out_path)
        out = utilities.get_filename(p_filename=self.out_model,
                                     p_parent_folder=self.parent_folder,
                                     p_path=self.out_path)

        # print(f"out = {out}")
        logging.debug(f"training: Saving model : {out}")
        try:
            pickle.dump(model, open(out, 'wb'))  

        except Exception as err:
            logging.debug(f"error %s in creating outfile %s", err, out)
            raise

        return out


def go(args):


    logging.debug("\nInside traingin .go")
    logging.debug(f"num_features : {args.num_features}")
    logging.debug(f"lr_params : {args.lr_params}")

    logging.debug("\n ")


    train_model = Train_Model(args)
    

    logging.debug(f"args.mlflow_logging : {train_model.mlflow_logging}")
    if train_model.mlflow_logging:
        with mlflow.start_run():
            logging.debug("inside mlflow_start_run")
            logging.debug(f"inside go and in scope of mlflow.start_run")
            
            try:
                path = train_model.train_model()

                mlflow.log_param("out_filename", path)
                mlflow.log_artifact(path)

            except Exception as err:
                logger.error(f"Train Model Error %s", err)
                return False
    else:
        try: 
            logger.info("training without logging")
            path = train_model.train_model()

        except Exception as err:
            logger.error(f"Train Model - w/o logging Error %s", err)
            return False


    return True;


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
        required=True
    )

    parser.add_argument(
        "--ingestion_filename", 
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

    parser.add_argument(
        "--mlflow_logging", 
        type=bool,
        help='mlflow logging enable or disabled',
        required=False
    )
    parser.add_argument(
        "--diagnostic", 
        type=bool,
        help='is ingestion for diagnostics?',
        default=False,
        required=False
    )


    args = parser.parse_args()

    logging.debug(f"inside training main -> {args}")
    logging.debug(f"training.py args = {args}")

    go(args)
