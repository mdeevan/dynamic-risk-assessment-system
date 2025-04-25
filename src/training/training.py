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

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Train_Model():

    def __init__(self, args):
        self.in_path = args.in_path
        self.in_file = args.in_file
        self.out_path = args.out_path
        self.out_model = args.out_model
        self.parent_folder = "../../"
        self.num_features =""


        nf=args.num_features.replace("'", '"')
        print (f"type of nf: {type(nf)}")
        print(f"nf: {nf}")

        # nf = nf.replace("[","").replace("]","")
        num_features = json.loads(nf)

        print (f"type of num_features {type(num_features)}")
        print(f"num_features {num_features}")

        # print(f"args : {args}")
        # lr_params = args.lr_params
        # print(f"lr_params {lr_params}")
        # print(f"lr_params-type {type(lr_params)}")

        lrp = json.loads( args.lr_params.replace("'", '"' ).replace("True",'"True"').replace("False",'"False"') )
        # lrp=args.lr_params
        print(f"args -lrp {lrp}")

        print(f"lrp type {type(lrp)}")
        print(f"args -lrp - C{lrp['C']}")
        self.C  = lrp['C'] 

        self.class_weight  = lrp['class_weight'] 
        self.dual  = lrp['dual'] 
        self.fit_intercept  = lrp['fit_intercept'] 
        self.intercept_scaling = lrp['intercept_scaling']  
        self.l1_ratio  = lrp['l1_ratio'] 
        self.max_iter  = lrp['max_iter'] 
        self.multi_class = lrp['multi_class']  
        self.n_jobs   = lrp['n_jobs']  
        self.penalty  = lrp['penalty'] 
        self.random_state   = lrp['random_state']  
        self.solver   = lrp['solver']  
        self.tol   = lrp['tol']  
        self.verbose  = lrp['verbose'] 
        self.warm_start  = lrp['warm_start'] 


        # with open("params.yml") as stream:
        #     try:
        #         cfg = yaml.safe_load(stream)

        #         self.num_features = cfg['num_features']

        #     except yaml.YAMLError as exc:
        #         print(exc)


    def __get_filename(self, p_filename:str, p_path:str=None) -> str:
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


    def train_model(self) -> str:
        '''
        read in the files, combine, drop duplicates and save the file

        INPUT:
            uses instance level variables

        RETURN:
            path of the output file
        '''

        print("Train model class method")
        try:
            filename = self.__get_filename(self.in_file)
            df = self.__read_file(filename)
    
        except Exception as err:
            logger.info(f"error reading file %s", err)
            raise

        print(f"filename : {filename}")
        try:
            lr = LogisticRegression(C = self.C ,
                                    class_weight =  self.class_weight , 
                                    dual =  self.dual , 
                                    fit_intercept = self.fit_intercept ,
                                    intercept_scaling = self.intercept_scaling , 
                                    l1_ratio =  self.l1_ratio , 
                                    max_iter = self.max_iter ,
                                    multi_class =  self.multi_class , 
                                    n_jobs =  self.n_jobs , 
                                    penalty = self.penalty ,
                                    random_state =  self.random_state , 
                                    solver =  self.solver , 
                                    tol =  self.tol , 
                                    verbose = self.verbose ,
                                    warm_start = self.warm_start )
        except Exception as err:
            print(f"LR Error : {lr}")

        try:
            print(f"num_features = {type(self.num_features)} \ n{self.num_features}")
            X = df[self.num_features]
            y = X.pop('exited')

        except Exception as err:
            print(f"dataframe error : {err}")
            raise

        print(f"y : {y.head()}")

        print("fitting model")
        model = lr.fit(X, y)

        print("\ncreate directory\n")
        try:
            filename = self.parent_folder, self.out_path
            print(f"filename : {filename}")
            os.mkdir(os.path.join(filename))

        except Exception as err:
            # folder exists so we don't need to abort processing
            logger.info(f"train_model: error creating directory : %s ", err)
            raise


        out = self.__get_filename(self.out_model, self.out_path)


        logging.info(f"training: Saving model : {out}")
        try:
            pickle.dump(model, open(out, 'wb'))  

        except Exception as err:
            print(f"error %s in creating outfile %s", err, out)
            raise


        return out


def go(args):


    print("\nInside traingin .go")
    print(f"num_features : {args.num_features}")
    print(f"lr_params : {args.lr_params}")

    print("\n ")


    train_model = Train_Model(args)
    
    with mlflow.start_run():
        print("inside mlflow_start_run")
        print(f"inside go and in scope of mlflow.start_run")
        
        try:
            path = train_model.train_model()

            mlflow.log_param("out_filename", path)
            mlflow.log_artifact(path)

        except Exception as err:
            logger.error(f"Train Model Error %s", err)
            return False


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
