"""
apicalls.py
calls each of the API deployed on FastAPI and capture the output in
apireturns.txt
"""

import logging
import requests
import yaml

from src.lib import utilities

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# the api app (fastapi) is in src folder. so use the following to run server
# uvicorn app:app --app-dir src

# once running following will be the URL
URL = "http://127.0.0.1:8000"


try:
    with open("./config/config.yaml", encoding="utf-8") as stream:
        CFG = yaml.safe_load(stream)

        outfile_path = CFG["training"]["output_model_path"]
        outfile_name = CFG["diagnostics"]["apicallstxt_file"]

        outfile = utilities.get_filename(
            outfile_name, p_parent_folder="", p_path=outfile_path
        )

except (FileNotFoundError, PermissionError) as err:
    CFG = None
    logger.error("FATAL: Error initialization configuration %s", err)


apis = ["prediction", "model_score", "statistics", "diagnostics"]

responses = []
for api in apis:
    print("URL : ", URL + f"/{api}")
    responses.append(requests.get(URL + f"/{api}", timeout=10))

# write the responses to your workspace


# with open('apireturns2.txt', "w") as f:
with open(outfile, "w", encoding="utf-8") as f:
    f.write("API Responses\n")
    for idx, response in enumerate(responses):
        f.write("\n ------------------------------------- \n")
        f.write(f"result from {apis[idx]} : status : {response}\n")

        f.write(str(response.json()))
        f.write("\n")
