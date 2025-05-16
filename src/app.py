"""
FastAPI - to expose API for consumption
"""

import os

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI


# uvicorn app:app --app-dir src

from diagnostics import Diagnostics


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def get_diagnostic_object():
    """
    create and return the diagnostic object that contains the reporting functions
    INPUT:
        none
    RETURN:
        diagnostic object

    """
    logger.debug("fastAPI startup\n")
    diagnostic = Diagnostics()

    return diagnostic


global_vars = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan function to initialize objects / vars at
    the start of the application, and release when execution ends
    """

    logger.debug("inside lifespan ")

    global_vars["diagnostic_obj"] = get_diagnostic_object()
    yield
    # Clean up the memory
    global_vars.clear()


app = FastAPI(lifespan=lifespan)


# Home site with welcome message - GET request
@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {"message": "Welcome to FastAPI interface to dynamic risk assessment system"}


@app.get("/diagnostics")
async def get_diagnostics():
    """
    get all the diagnostics in a single call,
    INPUT:
        None
    RETURNS:
        dictionary of dictionary, where each entry is output of one
        diagnostic function
    """
    stat = global_vars["diagnostic_obj"].capture_statistics()
    nv = global_vars["diagnostic_obj"].find_null_values()
    ingestion_time = global_vars["diagnostic_obj"].timing_ingestion()
    training_time = global_vars["diagnostic_obj"].timing_training()
    dependencies = global_vars["diagnostic_obj"].dependencies_status()

    results = {}
    results["stat"] = stat
    results["null_values"] = nv
    results["time_ingestion"] = ingestion_time
    results["time_training"] = training_time
    results["dependencies"] = dependencies

    return results


@app.get("/statistics")
async def statistics():
    """
    Capture the statistic (mean, median and std. deviation) of the ingested data
    RETURN:
        statistics dictionary
    """
    stat = global_vars["diagnostic_obj"].capture_statistics()

    return stat


@app.get("/null_values")
async def null_values():
    """
    capture nul values in the ingested dataset
    RETURN:
        dictionary with null values for each of the columns in dataframe
    """
    nv = global_vars["diagnostic_obj"].find_null_values()

    return nv


@app.get("/time_ingestion")
async def time_ingestion(p_iterations: int = 10):
    """
    calculate the time it takes to ingest the data
    INPUT:
        INT : number of iteration : 10 - number of cycles to calculate the time
    RETURN:
        INT : time in seconds to ingested the data
    """
    ingestion_time = global_vars["diagnostic_obj"].timing_ingestion(p_iterations)

    return ingestion_time


@app.get("/time_training")
async def time_training(p_iterations: int = 10):
    """
    calculate the time it takes to train the model
    INPUT:
        INT : number of iteration : 10 - number of cycles to calculate the time
    RETURN:
        INT : time in seconds to ingested the data
    """
    _ = global_vars["diagnostic_obj"].timing_ingestion(1)

    # calling ingestion to use generated ingesteddata in temp folder
    # which is subsequently used by training
    training_time = global_vars["diagnostic_obj"].timing_training(p_iterations)

    return training_time


@app.get("/prediction")
async def get_predictions():
    """
    Make prediction over the test data, as configured in the config.yaml

    RETURNS:
        table of actual vs model predictions
    """
    predict = global_vars["diagnostic_obj"].make_predictions()

    return predict


@app.get("/dependencies")
async def get_dependencies():
    """
    Check the outdated version of the intstalled libraries

    RETURNS:
        table as json of current, latest and the version difference
    """

    dependencies = global_vars["diagnostic_obj"].dependencies_status()

    return dependencies


@app.get("/model_score")
async def get_model_score():
    """
    calcuate the F1 score of the model performance

    RETURN:
        float : F1 Score
    """

    filename = os.path.join(
        global_vars["diagnostic_obj"].model_path_name, "latestscore.txt"
    )

    with open(filename, "r", encoding="utf-8") as f:
        score = f.read()

    logger.debug(filename)
    logger.debug(score)

    return score
