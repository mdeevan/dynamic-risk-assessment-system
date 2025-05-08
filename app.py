
import os
import sys

import logging
from typing import Optional

import argparse
import yaml

from contextlib import asynccontextmanager
from fastapi import FastAPI

from pydantic import BaseModel, Field
import pandas as pd
import joblib

# import uvicorn
# import dvc.api

from src.diagnostics.diagnostics import Diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def fastAPI_startup():
    try:
        with open("./config/config.yaml") as stream:
            cfg = yaml.safe_load(stream)

    except Exception as err:
        logging.error(f"Error initialization %s", err)


    parser = argparse.ArgumentParser(description="diagnostic")

    parser.add_argument("--model_path_name", type=str, 
                        default=cfg["prod_deployment"]["prod_deployment_path"])

    parser.add_argument("--model_file_name", type=str, 
                        default=cfg["prod_deployment"]["output_model_name"])

    parser.add_argument("--data_folder", type=str, 
                        default=cfg['scoring']['test_data_path'])

    parser.add_argument("--data_files", type=str, default="[*]")
    parser.add_argument("--ingested_file", type=str, default="[*]")

    parser.add_argument("--report_folder"    , type=str, default="temp")
    parser.add_argument("--prediction_output", type=str, default="temp_predict")
    # parser.add_argument("--score_filename"   , type=str, default=None)
    parser.add_argument("--timing_filename"  , type=str, default="temp_timing")
    parser.add_argument("--mlflow_logging"   , type=str, default=False)
    parser.add_argument("--temp_folder"      , type=str, default="temp")
    parser.add_argument("--num_features"     , type=str, 
                        default=str(cfg['num_features']))
    
    parser.add_argument("--lr_params"        , type=str, default=None)
    parser.add_argument("--parent_folder"    , type=str, default="./")


    args = parser.parse_args([]) # Pass an empty list for non-command-line usage

    diagnostic = Diagnostics(args)
    return diagnostic


    diagnostic_instance = self.__get_diagnosic_instance()

    pass



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["answer_to_everything"] = fastAPI_startup
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

@app.on_event("startup")
async def startup_event():
    """
    capture parameters and setup the environment variables
    expensive function, to be done only at startup
    """
    global model, encoder, lb, cat_features

    params = dvc.api.params_show()
    model_path = params["model"]["model_path"]
    model_name = params["model"]["model_name"]
    encoder_name = params["model"]["encoder_name"]
    lb_name = params["model"]["lb_name"]
    cat_features = params["cat_features"]

    # census_obj = cls.Census()
    model = joblib.load(os.path.join(model_path, model_name))
    encoder = joblib.load(os.path.join(model_path, encoder_name))
    lb = joblib.load(os.path.join(model_path, lb_name))


# Home site with welcome message - GET request
@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {
        "message": "Welcome to FastAPI interface to dynamic risk assessment system"
    }

@app.get("/scoring")
async def scoring():
    pass

@app.get("/summarystats")
async def summary_stats():
    pass

@app.get("/diagnostics")
async def diagnostics():
    pass


# POST request to /predict site. Used to validate model with sample census data
@app.post("/predict")
async def predict(input_params: CensusData) -> str:
    """
    POST request that will provide sample census data and expect a prediction

    Output:
        Salary value as,  >50K or <=50K
    """

    # Read data sent as POST
    print(f"input  = {input_params}, input type = {type(input_params)}")
    input_data = input_params.dict(by_alias=True)
    # print(f"input data \n {input_data}")
    # print (f"model type \n{type(model)}")

    input_df = pd.DataFrame(input_data, index=[0])
    print(f"input df \n {input_df}")
    # logger.info(f"Input data: {input_df}")

    census_obj = cls.Census()
    pred = census_obj.execute_inference(
        model=model, encoder=encoder, lb=lb, df=input_df
    )
    # Process the data
    pred = str(lb.inverse_transform(pred)[0])
    response = {"Salary prediction": pred}
    logger.info("Pred %s and its type %s", pred, type(pred))
    # response = pred

    logger.info("Prediction: %s", response)

    # return response
    try:
        return response
    except Exception as e:
        # raise HTTPException(status_code=422, detail=e.errors())
        logger.info("Exception : %s", e)