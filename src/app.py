
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

import uvicorn
# import dvc.api


# uvicorn app:app --app-dir src



# from src.diagnostics.diagnostics import Diagnostics
# sys.path.append("../")
from diagnostics import Diagnostics


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def get_diagnostic_object():
    print("fastAPI startup\n")
    # try:
    #     with open("./config/config.yaml") as stream:
    #         cfg = yaml.safe_load(stream)

    # except Exception as err:
    #     logging.error(f"Error initialization %s", err)


    # diagnostic = diagnostics.Diagnostics()
    diagnostic =  Diagnostics()

    return diagnostic


global_vars = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("inside lifespan ")

    global_vars["diagnostic_obj"] = get_diagnostic_object()
    yield
    # Clean up the ML models and release the resources
    global_vars.clear()


app = FastAPI(lifespan=lifespan)

# Home site with welcome message - GET request
@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {
        "message": "Welcome to FastAPI interface to dynamic risk assessment system"
    }



@app.get("/diagnostics")
async def get_diagnostics():
    stat = global_vars['diagnostic_obj'].capture_statistics()
    nv = global_vars['diagnostic_obj'].find_null_values()
    time_ingestion = global_vars['diagnostic_obj'].timing_ingestion()
    time_training  = global_vars['diagnostic_obj'].timing_training()
    dependencies = global_vars['diagnostic_obj'].dependencies_status()

    results = {}
    results['null_values']    = nv
    results['time_ingestion'] = time_ingestion
    results['time_training']  = time_training
    results['dependencies']   = dependencies

    return results


@app.get("/statistics")
async def statistics():
    stat = global_vars['diagnostic_obj'].capture_statistics()

    return stat

@app.get("/null_values")
async def null_values():
    nv = global_vars['diagnostic_obj'].find_null_values()

    return nv

@app.get("/time_ingestion")
async def time_ingestion(p_iterations: int = 10):
    time_ingestion = global_vars['diagnostic_obj'].timing_ingestion(p_iterations)

    return time_ingestion

@app.get("/time_training")
async def time_training(p_iterations: int = 10):
    time_ingestion = global_vars['diagnostic_obj'].timing_ingestion(1)

    # calling ingestion to use generated ingesteddata in temp folder
    # which is subsequently used by training
    time_training  = global_vars['diagnostic_obj'].timing_training(p_iterations)

    return time_training


@app.get("/prediction")
async def get_predictions():
    predict = global_vars['diagnostic_obj'].make_predictions()

    return predict

@app.get("/dependencies")
async def get_dependencies():
    dependencies = global_vars['diagnostic_obj'].dependencies_status()

    return dependencies

@app.get("/model_score")
async def get_model_score():

    filename = os.path.join(global_vars['diagnostic_obj'].model_path_name,
                            "latestscore.txt")
    
    with open(filename, "r") as f:
        score = f.read()

    print(filename)
    print(score)

    return score

